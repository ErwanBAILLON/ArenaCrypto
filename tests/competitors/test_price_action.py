"""Price action family: levels, breakouts, Fibonacci pullbacks, stops, caps, presets, both universes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.competitors.price_action import PriceAction, presets
from arena.core.snapshot import Snapshot
from arena.core.types import Target
from arena.explain.reasons import explain_target

FLAT_VOL = 1000.0


def _frame(sym: str, idx: pd.DatetimeIndex, close, open_=None, high=None, low=None, volume=None) -> pd.DataFrame:
    close = np.asarray(close, float)
    n = len(close)
    open_ = np.concatenate([[close[0]], close[:-1]]) if open_ is None else np.asarray(open_, float)
    high = np.maximum(open_, close) * 1.002 if high is None else np.asarray(high, float)
    low = np.minimum(open_, close) * 0.998 if low is None else np.asarray(low, float)
    volume = np.full(n, FLAT_VOL) if volume is None else np.asarray(volume, float)
    return pd.DataFrame(
        {"symbol": sym, "ts": idx[:n], "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def _index(n: int, bar_hours: int) -> pd.DatetimeIndex:
    # daily bars are stamped at 00:00 UTC (the rebalance hour); hourly bars start at 01:00 like the fixtures
    start = "2024-01-01T00:00:00Z" if bar_hours == 24 else "2024-01-01T01:00:00Z"
    return pd.date_range(start, periods=n, freq=f"{bar_hours}h", tz="UTC")


def _range(n: int, period: int, level: float = 100.0, amp: float = 5.0) -> np.ndarray:
    """Deterministic trading range: a sine between level-amp and level+amp (clear swing highs/lows)."""
    return level + amp * np.sin(2 * np.pi * np.arange(n) / period)


def _breakout_series(sym: str, bar_hours: int, direction: int = 1, extra_rows: list[dict] | None = None):
    """Flat range then one wide bar closing beyond the range on 3× volume (mirrored for direction=-1)."""
    per_day = 24 // bar_hours
    n_range = 130 * per_day
    period = 40 * per_day
    idx = _index(n_range + 1 + len(extra_rows or []), bar_hours)
    close = _range(n_range, period)
    f = _frame(sym, idx, close)
    if direction == 1:
        last = {"open": 104.0, "high": 110.5, "low": 103.5, "close": 110.0, "volume": 3 * FLAT_VOL}
    else:
        last = {"open": 96.0, "high": 96.5, "low": 89.5, "close": 90.0, "volume": 3 * FLAT_VOL}
    rows = [last] + list(extra_rows or [])
    tail = pd.DataFrame(rows)
    tail.insert(0, "ts", idx[n_range : n_range + len(rows)])
    tail.insert(0, "symbol", sym)
    return pd.concat([f, tail], ignore_index=True)


def _snap(frames: list[pd.DataFrame], bar_hours: int, ts=None) -> Snapshot:
    c = pd.concat(frames, ignore_index=True)
    syms = list(dict.fromkeys(c["symbol"]))
    return Snapshot.from_long(ts if ts is not None else c["ts"].max(), syms, c, bar_hours=bar_hours)


# ------------------------------------------------------------------ breakouts


@pytest.mark.parametrize("bar_hours", [24, 1])
def test_breakout_on_volume_goes_long(bar_hours):
    c = _breakout_series("SPY", bar_hours)
    d = PriceAction(bar_hours=bar_hours).decide(_snap([c], bar_hours))
    assert "SPY" in d and d["SPY"].weight == pytest.approx(0.25)
    r = d["SPY"].reason
    assert r["setup"] == "breakout" and r["candle"] == "bullish"
    assert 104.0 < r["level"] < 110.0 * (1 - 0.002)
    assert r["volume_ratio"] == pytest.approx(3.0)
    assert r["stop"] < r["close"] and 0.0 < d["SPY"].conviction <= 1.0


def test_breakdown_on_volume_goes_short():
    c = _breakout_series("GLD", 24, direction=-1)
    d = PriceAction(bar_hours=24).decide(_snap([c], 24))
    assert "GLD" in d and d["GLD"].weight == pytest.approx(-0.25)
    r = d["GLD"].reason
    assert r["setup"] == "breakdown" and r["candle"] == "bearish"
    assert 90.0 * (1 + 0.002) < r["level"] < 96.0
    assert r["stop"] > r["close"]


def test_breakout_without_volume_or_with_bearish_candle_is_ignored():
    quiet = _breakout_series("SPY", 24)
    quiet.loc[quiet.index[-1], "volume"] = FLAT_VOL
    assert PriceAction(bar_hours=24).decide(_snap([quiet], 24)) == {}
    bearish = _breakout_series("SPY", 24)
    bearish.loc[bearish.index[-1], "open"] = 111.0  # closes 110 below its open
    bearish.loc[bearish.index[-1], "high"] = 111.5
    assert PriceAction(bar_hours=24).decide(_snap([bearish], 24)) == {}


# ------------------------------------------------------------ fib pullbacks


def _fib_series(sym: str) -> pd.DataFrame:
    """Range around 100, impulse 100 -> 150, pullback to ~127 (50 %) finished by a hammer."""
    idx = _index(122, 24)
    base = _range(80, 40, level=100.0, amp=2.0)
    impulse = np.linspace(100.0, 150.0, 21)[1:]  # 20 bars up
    pullback = np.linspace(150.0, 126.0, 21)[1:]  # 20 bars down (the 150 top becomes a confirmed pivot)
    close = np.concatenate([base, impulse, pullback])
    f = _frame(sym, idx, close)
    hammer = pd.DataFrame(
        [
            {
                "symbol": sym,
                "ts": idx[len(close)],
                "open": 126.0,
                "high": 127.3,
                "low": 120.0,
                "close": 127.0,
                "volume": FLAT_VOL,
            }
        ]
    )
    return pd.concat([f, hammer], ignore_index=True)


def test_fib_pullback_with_hammer_goes_long():
    c = _fib_series("ETH")
    d = PriceAction(bar_hours=24).decide(_snap([c], 24))
    assert "ETH" in d and d["ETH"].weight > 0
    r = d["ETH"].reason
    assert r["setup"] == "fib_pullback_long" and r["candle"] == "hammer"
    assert r["zone_low"] <= r["close"] <= r["zone_high"]
    assert 0.382 <= r["retrace"] <= 0.618
    assert r["volume_ratio"] is None
    assert d["ETH"].conviction > 0.5, "a ~50 % retrace sits near the middle of the zone"


def test_fib_pullback_without_reversal_candle_is_ignored():
    c = _fib_series("ETH")
    c.loc[c.index[-1], "low"] = 125.5  # no lower wick: plain small candle, not a hammer
    assert PriceAction(bar_hours=24).decide(_snap([c], 24)) == {}


# -------------------------------------------------------------------- stops


def test_close_below_stop_flattens_a_held_long():
    stop_hit = {"open": 109.0, "high": 109.5, "low": 84.5, "close": 85.0, "volume": FLAT_VOL}
    c = _breakout_series("SPY", 24, extra_rows=[stop_hit])
    comp = PriceAction(bar_hours=24)
    entry_ts = c["ts"].iloc[-2]
    d1 = comp.decide(_snap([c], 24, ts=entry_ts))
    assert d1["SPY"].weight > 0
    d2 = comp.decide(_snap([c], 24))
    assert "SPY" not in d2


def test_held_long_survives_a_quiet_day_and_updates_its_stop():
    quiet = {"open": 110.0, "high": 110.8, "low": 109.2, "close": 110.3, "volume": FLAT_VOL}
    c = _breakout_series("SPY", 24, extra_rows=[quiet])
    comp = PriceAction(bar_hours=24)
    d1 = comp.decide(_snap([c], 24, ts=c["ts"].iloc[-2]))
    d2 = comp.decide(_snap([c], 24))
    assert d2["SPY"].weight == pytest.approx(d1["SPY"].weight)
    assert d2["SPY"].reason["setup"] == "breakout"
    assert d2["SPY"].reason["close"] == pytest.approx(110.3)


# ------------------------------------------------------------ caps and gross


def test_k_cap_and_gross_never_exceed_limits():
    syms = [f"S{i}" for i in range(6)]
    frames = [_breakout_series(s, 24) for s in syms]
    d = PriceAction(bar_hours=24).decide(_snap(frames, 24))
    assert len(d) == 4
    assert sum(abs(t.weight) for t in d.values()) <= 1.0 + 1e-12
    d_fat = PriceAction({"max_weight": 0.35, "k": 6}, bar_hours=24).decide(_snap(frames, 24))
    assert len(d_fat) == 6
    assert sum(abs(t.weight) for t in d_fat.values()) == pytest.approx(1.0)


def test_warmup_honoured_and_days_based():
    hourly, daily = PriceAction(bar_hours=1), PriceAction(bar_hours=24)
    assert daily.warmup_bars() == 60 + 10 + 1
    assert hourly.warmup_bars() == 24 * 70 + 1
    c = _breakout_series("SPY", 24)
    early = _snap([c], 24, ts=c["ts"].iloc[daily.warmup_bars() - 2])
    assert PriceAction(bar_hours=24).decide(early) == {}


# ---------------------------------------------------------------- purity


def test_no_lookahead_and_deterministic():
    stop_hit = {"open": 109.0, "high": 109.5, "low": 84.5, "close": 85.0, "volume": FLAT_VOL}
    c = _breakout_series("SPY", 24, extra_rows=[stop_hit])
    ts = c["ts"].iloc[-2]
    truncated = _snap([c[c["ts"] <= ts]], 24)
    full_view = _snap([c], 24).at(ts)
    d_trunc = PriceAction(bar_hours=24).decide(truncated)
    d_full = PriceAction(bar_hours=24).decide(full_view)
    assert d_trunc == d_full and d_trunc, "decision should not be trivially empty"
    comp = PriceAction(bar_hours=24)
    assert comp.decide(truncated) == comp.decide(truncated)
    assert PriceAction(bar_hours=24).decide(truncated) == PriceAction(bar_hours=24).decide(truncated)


def test_hourly_universe_is_registered_and_runs_on_random_walk(snapshot):
    from arena.competitors.base import REGISTRY

    assert REGISTRY["price_action"] is PriceAction
    d = PriceAction().decide(snapshot)
    assert all(abs(t.weight) <= 0.25 + 1e-12 for t in d.values())
    assert sum(abs(t.weight) for t in d.values()) <= 1.0 + 1e-12


# --------------------------------------------------------------- presets


def test_presets_applied_by_style_but_explicit_values_win():
    assert presets("swing") == {"pivot_days": 10, "level_lookback_days": 60, "stop_days": 20}
    assert presets("position") == {"pivot_days": 30, "level_lookback_days": 180, "stop_days": 60}
    pos = PriceAction({"style": "position"})
    assert pos.params["pivot_days"] == 30 and pos.params["level_lookback_days"] == 180 and pos.params["stop_days"] == 60
    assert pos.warmup_bars() == 24 * 210 + 1
    custom = PriceAction({"style": "position", "pivot_days": 15})
    assert custom.params["pivot_days"] == 15 and custom.params["level_lookback_days"] == 180
    assert PriceAction().params["pivot_days"] == 10 and not PriceAction()._weekly


def test_position_style_redecides_on_mondays_only():
    calls = []

    class Counting(PriceAction):
        def compute(self, snap):
            calls.append(snap.ts)
            return {"BTC": Target(0.25)}

    comp = Counting({"style": "position"})
    assert comp._weekly

    def at(s):
        return Snapshot(pd.Timestamp(s, tz="UTC"), ["BTC"], {})

    assert comp.decide(at("2025-03-05T00:00:00")) == {"BTC": Target(0.25)}  # Wednesday: first call decides
    assert comp.decide(at("2025-03-06T00:00:00")) == {"BTC": Target(0.25)}  # Thursday: hold
    assert len(calls) == 1
    comp.decide(at("2025-03-10T00:00:00"))  # Monday 00:00: re-decide
    assert len(calls) == 2
    comp.decide(at("2025-03-10T04:00:00"))  # Monday 04:00: hold
    assert len(calls) == 2


# --------------------------------------------------------------- explain


def test_explain_sentences_for_breakout_and_fib():
    t = Target(
        0.25,
        reason={
            "setup": "breakout",
            "level": 512.3,
            "close": 520.0,
            "volume_ratio": 1.6,
            "candle": "bullish",
            "stop": 498.1,
        },
    )
    s = explain_target("price_action", "SPY", t)
    assert s == (
        "Achat SPY (25 % du capital) : cassure de la résistance 512.30 avec un volume 1.6× la normale, "
        "chandelier haussier ; stop sous 498.10."
    )
    t2 = Target(
        0.25,
        reason={
            "setup": "fib_pullback_long",
            "level": 2900.0,
            "close": 3190.0,
            "volume_ratio": None,
            "candle": "hammer",
            "stop": 3050.0,
            "retrace": 0.5,
            "zone_low": 3120.0,
            "zone_high": 3260.0,
        },
    )
    s2 = explain_target("price_action", "ETH", t2)
    assert "repli à 50 %" in s2 and "zone 3 120–3 260" in s2 and "marteau" in s2 and "stop sous 3 050" in s2
    t3 = Target(
        -0.25,
        reason={
            "setup": "breakdown",
            "level": 95.0,
            "close": 90.0,
            "volume_ratio": 3.0,
            "candle": "bearish",
            "stop": 105.0,
        },
    )
    s3 = explain_target("price_action", "GLD", t3)
    assert s3.startswith("Vente GLD") and "cassure du support 95.00" in s3 and "stop au-dessus de 105.00" in s3

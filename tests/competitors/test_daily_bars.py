"""Competitors on a daily-bar universe (classic markets) behave like on hourly bars."""

import numpy as np
import pandas as pd

from arena.competitors.regime import regime_label
from arena.competitors.trend_ts import TrendTS
from arena.competitors.xs_momentum import XSMomentum
from arena.core.snapshot import Snapshot


def _daily(symbols, days=900, seed=0, drift=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-02", periods=days, freq="1D", tz="UTC")
    frames = []
    for i, sym in enumerate(symbols):
        r = rng.normal((drift or {}).get(sym, 0.0), 0.01, days)
        close = 100.0 * (1 + i) * np.exp(np.cumsum(r))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "ts": idx,
                    "open": close,
                    "high": close * 1.005,
                    "low": close * 0.995,
                    "close": close,
                    "volume": 1.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_days_helper_and_warmups_scale_with_bar():
    hourly, daily = TrendTS(bar_hours=1), TrendTS(bar_hours=24)
    assert hourly.days(30) == 720 and daily.days(30) == 30
    assert daily.warmup_bars() == 200 + 1  # slow EMA dominates once days are cheap
    assert XSMomentum(bar_hours=24).warmup_bars() == 33


def test_trend_follows_daily_drift_and_snapshot_refuses_finer_tf():
    c = _daily(["SPY", "GLD", "BTC"], drift={"SPY": 0.002, "GLD": -0.002})
    snap = Snapshot.from_long(c["ts"].max(), ["SPY", "GLD", "BTC"], c, bar_hours=24)
    assert snap.bars_per_day == 1 and snap.bars_per_year == 365
    d = TrendTS(bar_hours=24).decide(snap)
    assert d["SPY"].weight > 0 and d["GLD"].weight < 0
    assert regime_label(snap, "BTC") in ("bull_calm", "bull_vol", "bear", "range")
    assert not snap.candles("SPY", "1d").empty
    try:
        snap.candles("SPY", "4h")
    except ValueError:
        pass
    else:
        raise AssertionError("4h bars must not exist on a daily universe")

"""Currency families on daily bars, and the market fit that decides who may run where."""

from __future__ import annotations

import numpy as np
import pandas as pd

from arena.competitors import REGISTRY
from arena.competitors.base import ALL_MARKETS
from arena.competitors.fx import FxBreakout, FxTsmom
from arena.core.snapshot import Snapshot

PAIRS = ["EURUSD=X", "USDJPY=X", "AUDUSD=X"]


def _daily(drift: dict[str, float], days: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=days, freq="1D", tz="UTC")
    frames = []
    for sym in PAIRS:
        r = rng.normal(drift.get(sym, 0.0), 0.005, days)  # ~8 % annualised, a real currency
        close = 1.0 * np.exp(np.cumsum(r))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "ts": idx,
                    "open": close,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "close": close,
                    "volume": 0.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _snap(candles: pd.DataFrame) -> Snapshot:
    return Snapshot.from_long(candles["ts"].max(), PAIRS, candles, None, bar_hours=24)


def test_tsmom_votes_and_sizes_to_ten_percent_vol():
    snap = _snap(_daily({"EURUSD=X": 0.0006, "USDJPY=X": -0.0006}))
    comp = FxTsmom(bar_hours=24)
    d = comp.decide(snap)
    assert d["EURUSD=X"].weight > 0 and d["USDJPY=X"].weight < 0
    assert (
        d["USDJPY=X"].reason["votes"] == 3
        and abs(d["USDJPY=X"].weight) > abs(d.get("AUDUSD=X", d["USDJPY=X"]).weight) - 1e-9
    )
    # a 0.5 %/day currency is ~8 % annualised: the 10 % target sizes it above one third of NAV, capped at max_weight
    assert abs(d["EURUSD=X"].weight) <= 0.34 + 1e-9 and d["EURUSD=X"].reason["votes"] >= 2


def test_breakout_enters_on_the_channel_and_holds_inside_it():
    c = _daily({"EURUSD=X": 0.0008})
    eur = c["symbol"] == "EURUSD=X"
    last = c.index[eur][-1]
    c.loc[last, ["close", "high"]] = c.loc[eur, "high"].iloc[:-1].max() * 1.01  # today closes above the channel
    snap = _snap(c)
    comp = FxBreakout(bar_hours=24)
    d = comp.decide(snap)
    assert d["EURUSD=X"].weight > 0  # a steady climb closes above its 55-day high
    # next day inside the channel: the leg is kept, not re-decided away
    comp2 = FxBreakout(bar_hours=24)
    comp2.restore_state(comp.state())
    later = c.copy()
    later.loc[later["symbol"] == "EURUSD=X", "close"] *= 0.999
    d2 = comp2.compute(_snap(later))
    assert "EURUSD=X" in d2 and d2["EURUSD=X"].weight > 0


def test_families_declare_their_markets():
    assert REGISTRY["fx_tsmom"].markets == {"fx"} and REGISTRY["fx_breakout"].markets == {"fx"}
    assert REGISTRY["carry"].markets == {"crypto"} and REGISTRY["news"].markets == {"crypto"}
    assert REGISTRY["trend_ts"].markets == ALL_MARKETS  # a generic rule may run anywhere (and be pruned by evidence)

import numpy as np
import pandas as pd
import pytest

from arena.competitors.regime import Regime, regime_label
from arena.core.snapshot import Snapshot
from tests.conftest import SYMBOLS, make_funding

BARS = 24 * 260


def _candles(drift: float, sigma_early: float, sigma_late: float, seed: int = 0) -> pd.DataFrame:
    """BTC with a vol change 60 days before the end; ETH/SOL are plain noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01T01:00:00Z", periods=BARS, freq="1h")
    frames = []
    for sym in SYMBOLS:
        if sym == "BTC":
            sig = np.where(np.arange(BARS) < BARS - 24 * 60, sigma_early, sigma_late)
            r = rng.normal(drift, sig)
        else:
            r = rng.normal(0, 0.01, BARS)
        close = 100 * np.exp(np.cumsum(r))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "ts": idx,
                    "open": close,
                    "high": close * 1.001,
                    "low": close * 0.999,
                    "close": close,
                    "volume": 1.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _snap(c, funding=None):
    return Snapshot.from_long(c["ts"].max(), SYMBOLS, c, funding)


@pytest.mark.parametrize(
    "drift,early,late,label",
    [
        (0.001, 0.02, 0.005, "bull_calm"),
        (0.001, 0.005, 0.02, "bull_vol"),
        (-0.001, 0.005, 0.02, "bear"),
        (-0.001, 0.02, 0.005, "range"),
    ],
)
def test_labels_on_constructed_series(drift, early, late, label):
    assert regime_label(_snap(_candles(drift, early, late))) == label


def test_unknown_when_history_short():
    c = _candles(0.001, 0.02, 0.005)
    snap = _snap(c).at(c["ts"].iloc[1000])
    assert regime_label(snap) == "unknown"
    assert Regime().decide(snap) == {}


def test_decisions_per_label():
    bull = Regime().decide(_snap(_candles(0.001, 0.02, 0.005)))
    assert {s: t.weight for s, t in bull.items()} == {"BTC": 0.25, "ETH": 0.25}
    assert bull["BTC"].reason["regime"] == "bull_calm"

    vol = Regime().decide(_snap(_candles(0.001, 0.005, 0.02)))
    assert {s: t.weight for s, t in vol.items()} == {"BTC": 0.25}

    assert Regime().decide(_snap(_candles(-0.001, 0.005, 0.02))) == {}

    c = _candles(-0.001, 0.02, 0.005)
    rng = Regime().decide(_snap(c, make_funding(candles=c, per_symbol={"BTC": 0.0, "ETH": 0.0003, "SOL": 0.0002})))
    assert set(rng) == {"ETH", "SOL"}
    assert all(t.kind == "carry" and t.weight == 0.25 and t.reason["regime"] == "range" for t in rng.values())

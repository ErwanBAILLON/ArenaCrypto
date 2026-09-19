import pandas as pd

from arena.competitors.trend_ts import TrendTS
from arena.core.snapshot import Snapshot
from tests.conftest import SYMBOLS, make_candles


def _snap(drift, seed=0):
    c = make_candles(bars=24 * 200, seed=seed, drift=drift)
    return Snapshot.from_long(c["ts"].max(), SYMBOLS, c), c


def test_strong_uptrend_goes_long():
    snap, _ = _snap({"BTC": 0.001})
    d = TrendTS().decide(snap)
    assert "BTC" in d and d["BTC"].weight > 0
    assert d["BTC"].weight <= 0.5 and d["BTC"].conviction > 0


def test_strong_downtrend_goes_short():
    snap, _ = _snap({"ETH": -0.001})
    d = TrendTS().decide(snap)
    assert "ETH" in d and d["ETH"].weight < 0


def _positioned_fraction(drift, seeds=range(12)) -> float:
    n = 0
    for seed in seeds:
        snap, _ = _snap(drift, seed=seed)
        n += len(TrendTS().decide(snap))
    return n / (len(seeds) * len(SYMBOLS))


def test_noise_is_positioned_far_less_often_than_a_real_trend():
    # A random walk has persistent runs, so the filter is not silent on noise;
    # it must however fire much less than on a planted drift.
    flat = _positioned_fraction({})
    trend = _positioned_fraction({s: 0.001 for s in SYMBOLS})
    assert trend > 0.8
    assert flat < trend - 0.25


def test_gross_never_exceeds_one_and_warmup_honoured():
    snap, c = _snap({"BTC": 0.002, "ETH": 0.002, "SOL": 0.002})
    d = TrendTS().decide(snap)
    assert len(d) == 3
    assert sum(abs(t.weight) for t in d.values()) <= 1.0 + 1e-12
    early = snap.at(c["ts"].iloc[TrendTS().warmup_bars() - 2])
    assert TrendTS().decide(early) == {}


def test_vol_targeting_reduces_size_when_vol_high():
    calm, _ = _snap({"BTC": 0.001})
    d_calm = TrendTS({"target_vol": 0.20}).decide(calm)
    d_small = TrendTS({"target_vol": 0.05}).decide(calm)
    assert d_small["BTC"].weight < d_calm["BTC"].weight

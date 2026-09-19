import pandas as pd
import pytest

from arena.competitors.nulls import BenchBtcHold, BenchCarryEqual, NullCash, NullRandom
from arena.core.snapshot import Snapshot
from tests.conftest import SYMBOLS


def test_null_cash_is_empty(snapshot):
    assert NullCash().decide(snapshot) == {}


def test_null_random_deterministic_per_seed_and_ts(snapshot):
    a, b = NullRandom(seed=1), NullRandom(seed=1)
    assert a.decide(snapshot) == b.decide(snapshot)
    ts_series = [snapshot.ts - pd.Timedelta(hours=h) for h in range(40)]
    d1 = [NullRandom(seed=1).decide(snapshot.at(t)) for t in ts_series]
    d2 = [NullRandom(seed=2).decide(snapshot.at(t)) for t in ts_series]
    assert d1 != d2
    assert any(d for d in d1)  # trades sometimes
    assert all(sum(abs(t.weight) for t in d.values()) <= 1.0 for d in d1)
    assert all(abs(t.weight) <= 0.3 for d in d1 for t in d.values())


def test_null_random_skips_symbols_without_data(candles):
    ts = candles["ts"].max()
    snap = Snapshot.from_long(ts, SYMBOLS + ["XYZ"], candles)
    for h in range(30):
        d = NullRandom(seed=0).decide(snap.at(ts - pd.Timedelta(hours=h)))
        assert "XYZ" not in d


def test_btc_hold(snapshot):
    d = BenchBtcHold().decide(snapshot)
    assert set(d) == {"BTC"} and d["BTC"].weight == 1.0 and d["BTC"].conviction == 1.0


def test_carry_equal_sums_to_at_most_one(snapshot, candles):
    d = BenchCarryEqual().decide(snapshot)
    assert set(d) == set(SYMBOLS)
    assert sum(t.weight for t in d.values()) == pytest.approx(1.0)
    assert all(t.kind == "carry" for t in d.values())
    assert BenchCarryEqual().decide(Snapshot.from_long(snapshot.ts, SYMBOLS, candles)) == {}

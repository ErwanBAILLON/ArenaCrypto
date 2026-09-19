"""Decisions depend only on data at or before the decision timestamp."""

import pytest

from arena.competitors.carry import Carry
from arena.competitors.regime import Regime
from arena.competitors.trend_ts import TrendTS
from arena.competitors.xs_momentum import XSMomentum
from arena.core.snapshot import Snapshot
from tests.conftest import SYMBOLS, make_candles, make_funding

FACTORIES = [
    lambda: Carry(),
    lambda: TrendTS(),
    lambda: XSMomentum({"k": 1}),
    lambda: Regime(),
]


@pytest.fixture(scope="module")
def history():
    c = make_candles(bars=24 * 320, drift={"BTC": 0.0005, "ETH": -0.0005})
    f = make_funding(candles=c, per_symbol={"BTC": 0.0003, "ETH": 0.00005, "SOL": 0.0002})
    return c, f


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda f: f().family)
def test_future_rows_do_not_change_decision(history, factory):
    c, f = history
    ts = c["ts"].iloc[24 * 280]
    truncated = Snapshot.from_long(ts, SYMBOLS, c[c["ts"] <= ts], f[f["ts"] <= ts])
    full_view = Snapshot.from_long(c["ts"].max(), SYMBOLS, c, f).at(ts)
    d_trunc = factory().decide(truncated)
    d_full = factory().decide(full_view)
    assert d_trunc == d_full
    assert d_trunc, "decision should not be trivially empty"


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda f: f().family)
def test_same_snapshot_same_decision(history, factory):
    c, f = history
    snap = Snapshot.from_long(c["ts"].max(), SYMBOLS, c, f)
    comp = factory()
    assert comp.decide(snap) == comp.decide(snap)
    assert factory().decide(snap) == factory().decide(snap)

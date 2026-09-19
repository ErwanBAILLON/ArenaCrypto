import pandas as pd

from arena.competitors.xs_momentum import XSMomentum
from arena.core.snapshot import Snapshot
from tests.conftest import SYMBOLS, make_candles


def test_xs_momentum_state_roundtrip():
    c = make_candles(bars=24 * 60, drift={"BTC": 0.001, "SOL": -0.001})
    snap = Snapshot.from_long(c["ts"].max(), SYMBOLS, c)
    a = XSMomentum({"k": 1, "lookback_days": 20})
    first = a.decide(snap)
    st = a.state()
    b = XSMomentum({"k": 1, "lookback_days": 20})
    b.restore_state(st)
    later = snap.at(c["ts"].max())  # same bar: must not re-rank, must reproduce
    assert b.decide(later) == first
    assert b.state() == st

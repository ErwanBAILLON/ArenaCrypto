import pandas as pd

from arena.competitors.xs_momentum import XSMomentum
from arena.core.snapshot import Snapshot

SYMS = ["A", "B", "C", "D", "E"]


def _snap(drift, syms=SYMS, bars=24 * 60):
    from tests.conftest import make_candles

    c = make_candles(symbols=syms, bars=bars, drift=drift)
    return Snapshot.from_long(c["ts"].max(), syms, c), c


def test_ranking_picks_planted_best_and_worst():
    snap, _ = _snap({"A": 0.003, "E": -0.003})
    d = XSMomentum({"k": 1}).decide(snap)
    assert d["A"].weight == 0.5 and d["E"].weight == -0.5
    assert set(d) == {"A", "E"}
    assert d["A"].conviction == 1.0 and d["E"].conviction == 1.0


def test_hysteresis_keeps_held_symbol_within_band():
    snap, _ = _snap({"A": 0.003, "B": 0.0015, "E": -0.003})
    kept = XSMomentum({"k": 1, "band": 1, "long_only": True})
    kept._prev = {"B": 1}  # B is rank 2, within k+band
    assert set(kept.decide(snap)) == {"B"}
    strict = XSMomentum({"k": 1, "band": 0, "long_only": True})
    strict._prev = {"B": 1}
    assert set(strict.decide(snap)) == {"A"}


def test_no_rerank_between_rebalances():
    snap, c = _snap({"A": 0.003, "E": -0.003})
    comp = XSMomentum({"k": 1})
    first = comp.decide(snap.at(c["ts"].iloc[-200]))
    later = comp.decide(snap.at(c["ts"].iloc[-100]))
    assert set(first) == set(later) == {"A", "E"}
    assert comp._last_rebalance_ts == c["ts"].iloc[-200]
    comp.decide(snap)  # 199h later: due for rebalance
    assert comp._last_rebalance_ts == snap.ts


def test_long_only_has_no_shorts_and_full_weight():
    snap, _ = _snap({"A": 0.003, "E": -0.003})
    d = XSMomentum({"k": 2, "long_only": True}).decide(snap)
    assert all(t.weight > 0 for t in d.values())
    assert sum(t.weight for t in d.values()) == 1.0
    assert "A" in d and "E" not in d


def test_insufficient_history_is_flat():
    snap, _ = _snap({}, bars=24 * 10)
    assert XSMomentum().decide(snap) == {}

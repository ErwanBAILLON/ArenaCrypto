"""The majors trend filter: long what rose, flat what fell, two names, vol-targeted."""

from arena.competitors.majors_tsmom import MajorsTsmom
from arena.core.snapshot import Snapshot
from tests.conftest import SYMBOLS, make_candles


def _snap(drift, seed=0):
    c = make_candles(bars=24 * 60, seed=seed, drift=drift)
    return Snapshot.from_long(c["ts"].max(), SYMBOLS, c)


def test_long_the_riser_flat_the_faller_and_never_the_rest():
    d = MajorsTsmom().decide(_snap({"BTC": 0.001, "ETH": -0.001, "SOL": 0.002}))
    assert d["BTC"].weight > 0 and "ETH" not in d and "SOL" not in d


def test_sizes_to_the_vol_target_shared_across_the_basket():
    d = MajorsTsmom().decide(_snap({"BTC": 0.001, "ETH": 0.001}))
    assert set(d) == {"BTC", "ETH"}
    assert all(0 < t.weight <= 0.5 for t in d.values()) and sum(t.weight for t in d.values()) <= 1.0 + 1e-9


def test_accepts_full_perp_names():
    c = make_candles(symbols=["BTCUSDT", "ETHUSDT"], bars=24 * 60, drift={"BTCUSDT": 0.001})
    snap = Snapshot.from_long(c["ts"].max(), ["BTCUSDT", "ETHUSDT"], c)
    assert "BTCUSDT" in MajorsTsmom().decide(snap)

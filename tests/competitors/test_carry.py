import pytest

from arena.competitors.carry import Carry, rank_funding
from arena.core.snapshot import Snapshot
from tests.conftest import SYMBOLS, make_candles, make_funding


@pytest.fixture(scope="module")
def candles():
    return make_candles(bars=24 * 20)


def _snap(candles, funding, hl=None):
    return Snapshot.from_long(candles["ts"].max(), SYMBOLS, candles, funding, hl_funding=hl)


def test_positive_funding_on_eth_only(candles):
    snap = _snap(candles, make_funding(candles=candles, rate=0.0, per_symbol={"ETH": 0.0003}))
    d = Carry().decide(snap)
    assert set(d) == {"ETH"}
    t = d["ETH"]
    assert t.kind == "carry"
    assert t.weight == pytest.approx(min(0.34, 1 / 3))
    assert t.conviction == pytest.approx(1.0)
    assert t.reason["mean_funding_8h"] == pytest.approx(0.0003) and t.reason["rank"] == 0


def test_below_min_rate_is_flat(candles):
    snap = _snap(candles, make_funding(candles=candles, rate=0.00001))
    assert Carry().decide(snap) == {}


def test_top_k_and_ordering(candles):
    snap = _snap(candles, make_funding(candles=candles, per_symbol={"BTC": 0.0002, "ETH": 0.0004, "SOL": 0.0003}))
    d = Carry({"k": 2}).decide(snap)
    assert set(d) == {"ETH", "SOL"}
    assert d["ETH"].reason["rank"] == 0 and d["SOL"].reason["rank"] == 1
    assert d["ETH"].weight == pytest.approx(0.34)
    assert sum(t.weight for t in d.values()) <= 1.0


def test_hl_funding_blended(candles):
    fund = make_funding(candles=candles, rate=0.0001)
    plain = dict(rank_funding(_snap(candles, fund), 3))
    blended = dict(rank_funding(_snap(candles, fund, hl={"BTC": 0.0001}), 3))
    assert plain["BTC"] == pytest.approx(0.0001)
    assert blended["BTC"] == pytest.approx((0.0001 + 0.0008) / 2)
    assert blended["ETH"] == pytest.approx(0.0001)


def test_rank_funding_respects_lookback(candles):
    fund = make_funding(candles=candles, rate=0.0001)
    old = fund["ts"] < fund["ts"].max() - __import__("pandas").Timedelta(days=3)
    fund.loc[old, "rate"] = 0.01  # huge but outside the window
    assert dict(rank_funding(_snap(candles, fund), 3))["BTC"] == pytest.approx(0.0001)

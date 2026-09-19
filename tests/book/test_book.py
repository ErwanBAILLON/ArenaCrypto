import pytest

from arena.book.book import Book, FeeModel
from arena.core.types import Target

TS = "2024-01-01T01:00:00Z"
FEES = FeeModel(perp_taker=0.0005, slippage=0.0002, spot_taker=0.0010)


def test_open_long_charges_fee_only():
    b = Book(nav=10_000, fees=FEES)
    row = b.step(TS, {"BTC": 100.0}, {"BTC": 100.0}, {}, {"BTC": Target(0.5)})
    assert row.turnover == pytest.approx(0.5)
    assert row.fees == pytest.approx(0.5 * 0.0007)
    assert row.ret == pytest.approx(-0.5 * 0.0007)
    assert row.gross == pytest.approx(0.5)


def test_long_earns_price_move():
    b = Book(nav=10_000, fees=FEES)
    b.step(TS, {"BTC": 100.0}, {"BTC": 100.0}, {}, {"BTC": Target(0.5)})
    row = b.step(TS, {"BTC": 110.0}, {"BTC": 100.0}, {}, {"BTC": Target(0.5)})
    assert row.ret == pytest.approx(0.05)
    assert row.turnover == 0.0 and row.fees == 0.0


def test_short_pays_price_move_and_receives_funding():
    b = Book(nav=10_000, fees=FEES)
    b.step(TS, {"BTC": 100.0}, {"BTC": 100.0}, {}, {"BTC": Target(-0.4)})
    row = b.step(TS, {"BTC": 105.0}, {"BTC": 100.0}, {"BTC": 0.0001}, {"BTC": Target(-0.4)})
    assert row.funding_pnl == pytest.approx(0.4 * 0.0001)
    assert row.ret == pytest.approx(-0.4 * 0.05 + 0.4 * 0.0001)


def test_long_pays_funding():
    b = Book(nav=10_000, fees=FEES)
    b.step(TS, {"BTC": 100.0}, {"BTC": 100.0}, {}, {"BTC": Target(1.0)})
    row = b.step(TS, {"BTC": 100.0}, {"BTC": 100.0}, {"BTC": 0.0002}, {"BTC": Target(1.0)})
    assert row.funding_pnl == pytest.approx(-0.0002)


def test_carry_has_no_price_pnl_and_earns_funding():
    b = Book(nav=10_000, fees=FEES)
    open_row = b.step(TS, {"ETH": 100.0}, {"ETH": 100.0}, {}, {"ETH": Target(0.5, kind="carry")})
    assert open_row.fees == pytest.approx(0.5 * (0.0005 + 0.0002 + 0.0010))
    row = b.step(TS, {"ETH": 150.0}, {"ETH": 100.0}, {"ETH": 0.0003}, {"ETH": Target(0.5, kind="carry")})
    assert row.ret == pytest.approx(0.5 * 0.0003)


def test_gross_capped_to_one():
    b = Book(nav=10_000, fees=FEES)
    row = b.step(TS, {"A": 1.0, "B": 1.0}, {"A": 1.0, "B": 1.0}, {}, {"A": Target(0.8), "B": Target(-0.8)})
    assert row.gross == pytest.approx(1.0)
    assert b.positions["A"][1] == pytest.approx(0.5)
    assert b.positions["B"][1] == pytest.approx(-0.5)


def test_nav_compounds():
    b = Book(nav=10_000, fees=FeeModel(0, 0, 0))
    b.step(TS, {"A": 1.0}, {"A": 1.0}, {}, {"A": Target(1.0)})
    b.step(TS, {"A": 1.1}, {"A": 1.0}, {}, {"A": Target(1.0)})
    b.step(TS, {"A": 1.21}, {"A": 1.1}, {}, {"A": Target(0.0)})
    assert b.nav == pytest.approx(12_100.0)
    assert b.positions == {}


def test_restore_state():
    b = Book.restore(9_500.0, {"BTC": ("perp", 0.3)}, FEES)
    row = b.step(TS, {"BTC": 100.0}, {"BTC": 100.0}, {}, {"BTC": Target(0.3)})
    assert row.turnover == 0.0 and row.nav == pytest.approx(9_500.0)

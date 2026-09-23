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


# --------------------------------------------------------------------------- per-symbol impact


def _liq(adv, vol=0.10):
    from arena.core.costs import SymbolLiquidity

    return SymbolLiquidity(adv_usd=adv, daily_vol=vol)


def _impact_fees(**kw):
    from arena.book.book import FeeModel
    from arena.core.costs import ImpactModel

    return FeeModel(perp_taker=0.0005, slippage=0.0002, spot_taker=0.001, impact=ImpactModel(**kw))


def test_flat_model_is_untouched_when_no_impact_is_configured():
    """Opt-in by construction: no past verdict may be silently revised."""
    from arena.book.book import FeeModel

    f = FeeModel()
    assert f.cost("perp") == f.perp_cost
    assert f.cost("carry") == f.carry_cost
    assert f.cost("perp", 0.5, _liq(1_000.0)) == f.perp_cost  # size and liquidity ignored


def test_impact_makes_a_thin_symbol_cost_more_than_a_liquid_one():
    from arena.book.book import Book

    fees = _impact_fees(capacity_nav=1_000_000.0)
    prices, prev = {"A": 100.0, "B": 100.0}, {"A": 100.0, "B": 100.0}
    thin = Book(nav=10_000.0, fees=fees).step(
        TS, prices, prev, {}, {"A": Target(weight=0.5)}, liquidity={"A": _liq(2_000_000.0)}
    )
    deep = Book(nav=10_000.0, fees=fees).step(
        TS, prices, prev, {}, {"A": Target(weight=0.5)}, liquidity={"A": _liq(500_000_000.0)}
    )
    assert thin.fees > 5 * deep.fees
    assert thin.turnover == deep.turnover  # same trade, different price of trading it


def test_carry_pays_impact_on_both_legs():
    from arena.book.book import FeeModel
    from arena.core.costs import ImpactModel

    f = FeeModel(perp_taker=0.0005, spot_taker=0.001, impact=ImpactModel(half_spread=0.0, k=1.0))
    liq = _liq(10_000_000.0)
    slip = f.impact.slippage(0.1, liq)
    assert f.cost("perp", 0.1, liq) == pytest.approx(0.0005 + slip)
    assert f.cost("carry", 0.1, liq) == pytest.approx(0.0005 + 0.001 + 2 * slip)


def test_a_symbol_with_no_liquidity_data_is_charged_as_unknown():
    from arena.core.costs import UNKNOWN_LIQUIDITY_SLIPPAGE

    f = _impact_fees()
    assert f.cost("perp", 0.1, None) == pytest.approx(0.0005 + UNKNOWN_LIQUIDITY_SLIPPAGE)


def test_capacity_changes_the_verdict_on_the_same_book():
    """The same weights, the same symbol: only the money behind them differs."""
    from arena.book.book import Book

    prices, prev = {"A": 100.0}, {"A": 100.0}
    liquidity = {"A": _liq(3_000_000.0)}
    small = Book(nav=10_000.0, fees=_impact_fees(capacity_nav=10_000.0)).step(
        TS, prices, prev, {}, {"A": Target(weight=0.4)}, liquidity=liquidity
    )
    mid = Book(nav=10_000.0, fees=_impact_fees(capacity_nav=1_000_000.0)).step(
        TS, prices, prev, {}, {"A": Target(weight=0.4)}, liquidity=liquidity
    )
    large = Book(nav=10_000.0, fees=_impact_fees(capacity_nav=5_000_000.0)).step(
        TS, prices, prev, {}, {"A": Target(weight=0.4)}, liquidity=liquidity
    )
    # only the impact term scales; taker and half-spread are fixed, so compare the impact alone
    fixed = 0.0005 + 0.0002
    impact_small = small.fees / 0.4 - fixed
    impact_mid = mid.fees / 0.4 - fixed
    assert impact_mid == pytest.approx(10.0 * impact_small, rel=1e-6)  # sqrt(100) between 10k and 1M
    # 40 % of a 5M book is 2M into a 3M-ADV perp: two thirds of a day's volume, so the
    # cap fires. A backtest that keeps filling there is writing fiction.
    assert large.fees / 0.4 == pytest.approx(0.0005 + _impact_fees().impact.cap)

"""Point-in-time membership: the survivorship defence, tested as such."""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import pytest

from arena.core.membership import MembershipRule, exits, rebalance_dates, select

START = "2024-01-01"
BARS = 24 * 120


def _seed(name: str) -> int:
    """Stable across processes: PYTHONHASHSEED randomises hash() and made this file flaky."""
    return zlib.crc32(name.encode()) % 100_000


def _panel(spec: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide (closes, volumes) from {symbol: {price, units, first_bar, last_bar}}."""
    idx = pd.date_range(START, periods=BARS, freq="1h", tz="UTC")
    closes, volumes = {}, {}
    for sym, s in spec.items():
        rng = np.random.default_rng(_seed(sym))
        price = pd.Series(s.get("price", 100.0) * np.exp(np.cumsum(rng.normal(0, 0.005, BARS))), index=idx)
        units = pd.Series(float(s.get("units", 1_000.0)), index=idx)
        alive = pd.Series(True, index=idx)
        if s.get("first_bar") is not None:
            alive &= idx >= pd.Timestamp(s["first_bar"], tz="UTC")
        if s.get("last_bar") is not None:
            alive &= idx <= pd.Timestamp(s["last_bar"], tz="UTC")
        closes[sym] = price.where(alive)
        volumes[sym] = units.where(alive)
    return pd.DataFrame(closes), pd.DataFrame(volumes)


LOOSE = MembershipRule(top_n=10, min_adv_usd=0.0)


class TestPointInTime:
    def test_future_bars_cannot_influence_membership(self):
        closes, volumes = _panel({"A": {"units": 1_000}, "B": {"units": 2_000}})
        ts = closes.index[24 * 60]
        full = select(closes, volumes, ts, LOOSE)
        truncated = select(closes.loc[:ts], volumes.loc[:ts], ts, LOOSE)
        assert [m.symbol for m in full] == [m.symbol for m in truncated]
        assert full[0].adv_usd == pytest.approx(truncated[0].adv_usd)

    def test_a_symbol_not_yet_listed_is_not_a_member(self):
        closes, volumes = _panel({"OLD": {}, "NEW": {"first_bar": "2024-03-01", "units": 99_999}})
        early = select(closes, volumes, pd.Timestamp("2024-02-01", tz="UTC"), LOOSE)
        assert [m.symbol for m in early] == ["OLD"]

    def test_a_delisted_symbol_stops_being_a_member(self):
        """TOMOUSDT and SRMUSDT both died in May 2024, inside the arena's window."""
        closes, volumes = _panel({"ALIVE": {}, "DEAD": {"last_bar": "2024-02-15", "units": 99_999}})
        before = select(closes, volumes, pd.Timestamp("2024-02-10", tz="UTC"), LOOSE)
        after = select(closes, volumes, pd.Timestamp("2024-03-10", tz="UTC"), LOOSE)
        assert "DEAD" in [m.symbol for m in before]
        assert "DEAD" not in [m.symbol for m in after]

    def test_a_symbol_that_went_quiet_is_dropped_even_with_history(self):
        closes, volumes = _panel({"A": {}, "GHOST": {"last_bar": "2024-02-20", "units": 99_999}})
        # two days after its last bar, still well inside the 30-day window
        ts = pd.Timestamp("2024-02-22", tz="UTC")
        assert "GHOST" not in [m.symbol for m in select(closes, volumes, ts, LOOSE)]


class TestRanking:
    def test_ranked_by_dollar_volume_not_by_units(self):
        closes, volumes = _panel(
            {"CHEAP": {"price": 1.0, "units": 1_000_000}, "DEAR": {"price": 10_000.0, "units": 1_000}}
        )
        ranked = select(closes, volumes, closes.index[-1], LOOSE)
        assert [m.symbol for m in ranked] == ["DEAR", "CHEAP"]  # 10M vs 1M dollars
        assert ranked[0].rank == 0 and ranked[1].rank == 1

    def test_the_floor_excludes_outright(self):
        closes, volumes = _panel({"BIG": {"units": 100_000}, "SMALL": {"units": 1}})
        rule = MembershipRule(top_n=10, min_adv_usd=1_000_000.0)
        assert [m.symbol for m in select(closes, volumes, closes.index[-1], rule)] == ["BIG"]

    def test_top_n_truncates(self):
        closes, volumes = _panel({f"S{i}": {"units": 10.0**i} for i in range(8)})
        picked = select(closes, volumes, closes.index[-1], MembershipRule(top_n=3, min_adv_usd=0.0))
        assert [m.symbol for m in picked] == ["S7", "S6", "S5"]

    def test_members_carry_their_liquidity_for_the_cost_model(self):
        closes, volumes = _panel({"A": {"units": 10_000}})
        m = select(closes, volumes, closes.index[-1], LOOSE)[0]
        liq = m.liquidity
        assert liq.usable and liq.adv_usd == m.adv_usd and liq.daily_vol == m.daily_vol

    def test_ties_break_deterministically(self):
        closes, volumes = _panel({"B": {"price": 100.0, "units": 1_000}, "A": {"price": 100.0, "units": 1_000}})
        first = [m.symbol for m in select(closes, volumes, closes.index[-1], LOOSE)]
        second = [m.symbol for m in select(closes, volumes, closes.index[-1], LOOSE)]
        assert first == second


class TestSchedule:
    def test_weekly_mondays(self):
        dates = rebalance_dates(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-31", tz="UTC"))
        assert all(d.weekday() == 0 and d.hour == 0 for d in dates)
        assert len(dates) == 5

    def test_naive_inputs_are_read_as_utc(self):
        import datetime as dt

        assert rebalance_dates(dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 15))[0].tzinfo is not None


class TestExits:
    def test_a_departure_is_reported_so_it_can_be_closed(self):
        assert exits(["A", "B", "C"], ["A", "C"]) == ["B"]

    def test_arrivals_are_not_exits(self):
        assert exits(["A"], ["A", "B"]) == []

    def test_it_accepts_members_or_names(self):
        closes, volumes = _panel({"A": {}, "B": {}})
        members = select(closes, volumes, closes.index[-1], LOOSE)
        assert exits(members, ["A"]) == ["B"]

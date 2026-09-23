"""Assembling the wide universe: the logic, not the downloads."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from arena.core.membership import Member, MembershipRule
from arena.runner import wide_backfill as wb

START, END = dt.datetime(2024, 1, 1), dt.datetime(2024, 3, 31)


def _frame(days=120, price=100.0, units=1_000.0, first=0, last=None, seed=0):
    idx = pd.date_range(START, periods=days, freq="1D", tz="UTC")
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, 0.01, days)))
    frame = pd.DataFrame({"ts": idx, "close": close, "volume": units})
    return frame.iloc[first : (last if last is not None else days)]


class TestToWide:
    def test_it_aligns_symbols_on_one_index(self):
        closes, volumes = wb.to_wide({"A": _frame(), "B": _frame(first=30)})
        assert list(closes.columns) == ["A", "B"]
        assert closes["B"].isna().sum() == 30  # B had not listed yet
        assert volumes.shape == closes.shape

    def test_no_frames_gives_no_panel(self):
        closes, volumes = wb.to_wide({})
        assert closes.empty and volumes.empty


class TestMembership:
    def test_it_produces_one_entry_per_weekly_date(self):
        closes, volumes = wb.to_wide({f"S{i}": _frame(units=1_000 * (i + 1), seed=i) for i in range(5)})
        members = wb.build_membership(closes, volumes, START, END, MembershipRule(top_n=3, min_adv_usd=0.0))
        assert 10 < len(members) < 15  # ~13 Mondays in the window
        assert all(len(v) == 3 for v in members.values())
        assert all(ts.weekday() == 0 for ts in members)

    def test_it_ranks_by_dollar_volume(self):
        closes, volumes = wb.to_wide({"BIG": _frame(units=1e6), "SMALL": _frame(units=1.0)})
        members = wb.build_membership(closes, volumes, START, END, MembershipRule(top_n=1, min_adv_usd=0.0))
        assert {m.symbol for v in members.values() for m in v} == {"BIG"}


class TestWideBuild:
    def _build(self, schedule):
        return wb.WideBuild(
            members_by_date={
                pd.Timestamp(ts, tz="UTC"): [Member(s, i, 1e7, 0.05) for i, s in enumerate(names)]
                for ts, names in schedule.items()
            },
            probed=len(schedule),
            candidates=99,
        )

    def test_symbols_is_everyone_who_was_ever_a_member(self):
        build = self._build({"2024-01-01": ["A", "B"], "2024-01-08": ["B", "C"]})
        assert build.symbols == ["A", "B", "C"]

    def test_churn_measures_what_left(self):
        """One of two replaced each week is 50 % churn -- the number that kills a fixed list."""
        build = self._build({"2024-01-01": ["A", "B"], "2024-01-08": ["B", "C"]})
        assert build.churn == pytest.approx(0.5)

    def test_a_stable_universe_has_no_churn(self):
        build = self._build({"2024-01-01": ["A", "B"], "2024-01-08": ["A", "B"]})
        assert build.churn == 0.0

    def test_a_single_date_cannot_churn(self):
        assert self._build({"2024-01-01": ["A"]}).churn == 0.0

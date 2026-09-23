"""Crowded trend: the trend signal, minus the legs the crowd has already taken."""

from __future__ import annotations

import pytest

from arena.competitors.crowded_trend import CrowdedTrend
from arena.competitors.trend_ts import TrendTS
from arena.core.snapshot import Snapshot
from tests.conftest import make_candles, make_funding

SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]
UP, DOWN = 0.0006, -0.0006


@pytest.fixture(scope="module")
def trending():
    """BTC and ETH trend up, XRP and DOGE trend down; the rest drift."""
    return make_candles(symbols=SYMS, bars=24 * 320, drift={"BTC": UP, "ETH": UP, "XRP": DOWN, "DOGE": DOWN}, seed=4)


def _snap(candles, per_symbol):
    funding = make_funding(symbols=SYMS, candles=candles, rate=0.0001, per_symbol=per_symbol)
    return Snapshot.from_long(candles["ts"].max(), SYMS, candles, funding)


def _same_params() -> dict:
    """The trend half of CrowdedTrend, for a like-for-like comparison with trend_ts."""
    return {k: v for k, v in CrowdedTrend.default_params.items() if k in TrendTS.default_params}


class TestFilter:
    def test_without_a_funding_cross_section_it_is_the_trend_signal(self, trending):
        """The filter is additive: no funding data, no filtering, the same book as trend_ts.

        Only the `reason` dicts differ (this family reports the crowding it saw);
        the positions must match to the last basis point, otherwise the
        three-way comparison with trend_ts and funding_skew measures two things.
        """
        flat = _snap(trending, {})  # dispersion below the floor -> no z-scores at all
        mine = {s: t.weight for s, t in CrowdedTrend().decide(flat).items()}
        theirs = {s: t.weight for s, t in TrendTS(_same_params()).decide(flat).items()}
        assert mine == pytest.approx(theirs)
        assert mine

    def test_a_crowded_long_is_dropped(self, trending):
        crowded = _snap(trending, {"BTC": 0.0015, "DOGE": -0.0010})
        plain = TrendTS(_same_params()).decide(crowded)
        filtered = CrowdedTrend().decide(crowded)
        assert "BTC" in plain and plain["BTC"].weight > 0
        assert "BTC" not in filtered

    def test_a_crowded_short_is_dropped(self, trending):
        # DOGE falls and its funding is far below the pack: shorts there are already being paid
        crowded = _snap(trending, {"DOGE": -0.0015, "BTC": 0.0010})
        plain = TrendTS(_same_params()).decide(crowded)
        assert plain.get("DOGE") is not None and plain["DOGE"].weight < 0
        assert "DOGE" not in CrowdedTrend().decide(crowded)

    def test_an_uncrowded_trend_survives(self, trending):
        crowded = _snap(trending, {"BTC": 0.0015, "DOGE": -0.0010})
        filtered = CrowdedTrend().decide(crowded)
        assert "ETH" in filtered and filtered["ETH"].weight > 0
        assert filtered["ETH"].reason["funding_z"] is not None

    def test_a_permissive_threshold_keeps_everything(self, trending):
        crowded = _snap(trending, {"BTC": 0.0015, "DOGE": -0.0010})
        wide = CrowdedTrend({"max_long_z": 99.0, "min_short_z": -99.0}).decide(crowded)
        assert set(wide) == set(TrendTS(_same_params()).decide(crowded))

    def test_a_strict_threshold_can_empty_the_book(self, trending):
        crowded = _snap(trending, {"BTC": 0.0015, "DOGE": -0.0010})
        assert CrowdedTrend({"max_long_z": -99.0, "min_short_z": 99.0}).decide(crowded) == {}


class TestContract:
    def test_future_rows_do_not_change_the_decision(self, trending):
        ts = trending["ts"].iloc[24 * 280]
        funding = make_funding(symbols=SYMS, candles=trending, rate=0.0001, per_symbol={"BTC": 0.0015})
        truncated = Snapshot.from_long(ts, SYMS, trending[trending["ts"] <= ts], funding[funding["ts"] <= ts])
        full_view = Snapshot.from_long(trending["ts"].max(), SYMS, trending, funding).at(ts)
        d = CrowdedTrend().decide(truncated)
        assert d and d == CrowdedTrend().decide(full_view)

    def test_same_snapshot_same_decision(self, trending):
        snap = _snap(trending, {"BTC": 0.0015})
        comp = CrowdedTrend()
        assert comp.decide(snap) == comp.decide(snap)

    def test_gross_never_exceeds_one(self, trending):
        d = CrowdedTrend({"max_weight": 0.9, "max_long_z": 99.0, "min_short_z": -99.0}).decide(_snap(trending, {}))
        assert sum(abs(t.weight) for t in d.values()) <= 1.0 + 1e-9

    def test_it_is_registered_and_warms_up_on_the_longest_window(self):
        from arena.competitors.base import REGISTRY

        assert REGISTRY["crowded_trend"] is CrowdedTrend
        assert CrowdedTrend().warmup_bars() == 24 * 90 + 1  # lb_long_days dominates

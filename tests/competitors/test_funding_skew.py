"""Funding skew: cross-sectional crowding, not the funding level."""

from __future__ import annotations

import pandas as pd
import pytest

from arena.competitors.features import funding_zscores
from arena.competitors.funding_skew import FundingSkew
from arena.core.snapshot import Snapshot
from tests.conftest import make_candles, make_funding

SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]


@pytest.fixture(scope="module")
def candles():
    return make_candles(symbols=SYMS, bars=24 * 30)


def _snap(candles, per_symbol, rate=0.0001, ts=None):
    funding = make_funding(symbols=SYMS, candles=candles, rate=rate, per_symbol=per_symbol)
    return Snapshot.from_long(ts or candles["ts"].max(), SYMS, candles, funding)


def _crowded(candles, ts=None):
    """BTC pays far above the pack, DOGE far below: one crowded long, one crowded short."""
    return _snap(candles, {"BTC": 0.0010, "DOGE": -0.0008}, ts=ts)


class TestZScores:
    def test_a_flat_cross_section_says_nothing(self, candles):
        assert funding_zscores(_snap(candles, {}), 7) == {}

    def test_a_universe_wide_funding_regime_is_not_crowding(self, candles):
        """Every perp paying 0.05 % is a bull market, not a crowded symbol."""
        z = funding_zscores(_snap(candles, {}, rate=0.0005), 7)
        assert z == {}  # no dispersion, no signal -- the level is deliberately invisible here

    def test_the_outlier_is_the_one_with_the_z_score(self, candles):
        z = funding_zscores(_crowded(candles), 7)
        assert z["BTC"] > 1.0 and z["DOGE"] < -1.0
        assert abs(z["ETH"]) < 1.0

    def test_too_few_symbols_to_speak_of_a_cross_section(self, candles):
        three = Snapshot.from_long(
            candles["ts"].max(),
            SYMS[:3],
            candles[candles["symbol"].isin(SYMS[:3])],
            make_funding(symbols=SYMS[:3], candles=candles, per_symbol={"BTC": 0.001}),
        )
        assert funding_zscores(three, 7) == {}


class TestDecisions:
    def test_it_shorts_the_crowded_longs_and_buys_the_crowded_shorts(self, candles):
        d = FundingSkew().decide(_crowded(candles))
        assert d["BTC"].weight < 0 and d["BTC"].reason["crowd"] == "long"
        assert d["DOGE"].weight > 0 and d["DOGE"].reason["crowd"] == "short"

    def test_it_stands_aside_when_nobody_is_crowded(self, candles):
        assert FundingSkew().decide(_snap(candles, {})) == {}

    def test_gross_is_balanced_between_the_two_sides(self, candles):
        d = FundingSkew().decide(_crowded(candles))
        longs = sum(t.weight for t in d.values() if t.weight > 0)
        shorts = -sum(t.weight for t in d.values() if t.weight < 0)
        assert longs == pytest.approx(shorts)

    def test_one_sided_crowding_is_refused_when_market_neutral(self, candles):
        """A long-only version of this family would be a directional bet in disguise."""
        one_sided = _snap(candles, {"BTC": 0.0010})
        assert FundingSkew().decide(one_sided) == {}
        assert FundingSkew({"market_neutral": False}).decide(one_sided)

    def test_a_higher_bar_takes_fewer_legs(self, candles):
        snap = _crowded(candles)
        assert len(FundingSkew({"min_z": 0.3}).decide(snap)) >= len(FundingSkew({"min_z": 1.4}).decide(snap))

    def test_conviction_grows_with_the_z_score(self, candles):
        d = FundingSkew().decide(_crowded(candles))
        assert d["BTC"].conviction > 0.4
        assert all(0.0 <= t.conviction <= 1.0 for t in d.values())

    def test_gross_never_exceeds_one(self, candles):
        d = FundingSkew({"k": 5, "max_weight": 0.5, "min_z": 0.0}).decide(_crowded(candles))
        assert sum(abs(t.weight) for t in d.values()) <= 1.0 + 1e-9


class TestContract:
    def test_future_rows_do_not_change_the_decision(self, candles):
        ts = candles["ts"].iloc[24 * 20]
        funding = make_funding(symbols=SYMS, candles=candles, per_symbol={"BTC": 0.0010, "DOGE": -0.0008})
        truncated = Snapshot.from_long(ts, SYMS, candles[candles["ts"] <= ts], funding[funding["ts"] <= ts])
        full_view = Snapshot.from_long(candles["ts"].max(), SYMS, candles, funding).at(ts)
        d = FundingSkew().decide(truncated)
        assert d and d == FundingSkew().decide(full_view)

    def test_the_lookback_window_is_respected(self, candles):
        funding = make_funding(symbols=SYMS, candles=candles, rate=0.0001)
        old = funding["ts"] < funding["ts"].max() - pd.Timedelta(days=14)
        funding.loc[old & (funding["symbol"] == "BTC"), "rate"] = 0.05  # enormous, and out of the window
        snap = Snapshot.from_long(candles["ts"].max(), SYMS, candles, funding)
        assert funding_zscores(snap, 7) == {}

    def test_it_is_registered_under_its_family_name(self):
        from arena.competitors.base import REGISTRY

        assert REGISTRY["funding_skew"] is FundingSkew
        assert FundingSkew().warmup_bars() == 24 * 7 + 1


def test_a_dispersion_of_floating_point_dust_is_not_crowding(candles):
    """Identical funding everywhere gives sd ~1e-20, and dividing by it invents a signal."""
    flat = _snap(candles, {})
    assert funding_zscores(flat, 7, min_dispersion=0.0) != {}  # what the naive guard allowed through
    assert funding_zscores(flat, 7) == {}
    assert FundingSkew().decide(flat) == {}

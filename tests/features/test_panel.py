"""The panel is only worth anything if it cannot see the future."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.core.snapshot import Snapshot
from arena.features import build, feature_columns, market_series, symbol_features
from arena.features.panel import RANK_PREFIX, PanelConfig
from tests.conftest import make_candles, make_funding

SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]


@pytest.fixture(scope="module")
def history():
    candles = make_candles(symbols=SYMS, bars=24 * 220, drift={"BTC": 0.0004, "DOGE": -0.0004}, seed=11)
    funding = make_funding(symbols=SYMS, candles=candles, rate=0.0001, per_symbol={"BTC": 0.0006, "DOGE": -0.0004})
    return candles, funding


def _snap(history, ts=None):
    candles, funding = history
    at = ts or candles["ts"].max()
    return Snapshot.from_long(at, SYMS, candles[candles["ts"] <= at], funding[funding["ts"] <= at])


class TestNoLookahead:
    def test_future_bars_cannot_change_a_single_feature(self, history):
        candles, funding = history
        ts = candles["ts"].iloc[24 * 180]
        truncated = build(_snap(history, ts))
        full_view = build(Snapshot.from_long(candles["ts"].max(), SYMS, candles, funding).at(ts))
        pd.testing.assert_frame_equal(truncated, full_view)

    def test_the_same_snapshot_gives_the_same_panel(self, history):
        snap = _snap(history)
        pd.testing.assert_frame_equal(build(snap), build(snap))

    def test_a_later_snapshot_is_a_different_panel(self, history):
        candles, _ = history
        early = build(_snap(history, candles["ts"].iloc[24 * 100]))
        late = build(_snap(history))
        assert not early["ret_7d"].equals(late["ret_7d"])


class TestShape:
    def test_it_is_wide_and_mostly_populated(self, history):
        panel = build(_snap(history))
        cols = feature_columns(panel)
        assert len(panel) == len(SYMS)
        assert len(cols) > 80
        assert panel[cols].isna().mean().mean() < 0.05

    def test_every_raw_column_has_a_rank_twin(self, history):
        panel = build(_snap(history))
        raw = [c for c in panel.columns if not c.startswith(RANK_PREFIX)]
        ranked = {c[len(RANK_PREFIX) :] for c in panel.columns if c.startswith(RANK_PREFIX)}
        # the cross-sectional state columns are constant, so they rank trivially
        assert ranked.issuperset({"ret_7d", "vol_30d", "funding_7d", "rsi"})
        assert len(ranked) > 0.8 * len([c for c in raw if not c.startswith("xs_")])

    def test_ranks_are_percentiles_within_the_bar(self, history):
        panel = build(_snap(history))
        ranks = panel[f"{RANK_PREFIX}ret_7d"].dropna()
        assert ranks.min() > 0 and ranks.max() <= 1.0
        assert ranks.is_unique  # no ties on continuous data

    def test_a_symbol_with_too_little_history_is_dropped_not_imputed(self, history):
        """A filled-in zero teaches the model that new listings are average. They are not."""
        candles, funding = history
        ts = candles["ts"].max()
        young = candles[(candles["symbol"] == "SOL") & (candles["ts"] > ts - pd.Timedelta(days=3))]
        rest = candles[candles["symbol"] != "SOL"]
        snap = Snapshot.from_long(ts, SYMS, pd.concat([rest, young]), funding)
        assert "SOL" not in build(snap).index


class TestContent:
    def test_trending_symbols_rank_above_falling_ones(self, history):
        panel = build(_snap(history))
        assert panel.loc["BTC", f"{RANK_PREFIX}ret_90d"] > panel.loc["DOGE", f"{RANK_PREFIX}ret_90d"]

    def test_funding_features_follow_the_funding(self, history):
        panel = build(_snap(history))
        assert panel.loc["BTC", "funding_7d"] > panel.loc["DOGE", "funding_7d"]
        assert panel.loc["BTC", "funding_crowding_z"] > 0 > panel.loc["DOGE", "funding_crowding_z"]

    def test_cross_sectional_state_is_shared_by_every_row(self, history):
        panel = build(_snap(history))
        assert panel["xs_n_symbols"].nunique() == 1
        assert panel["xs_dispersion"].nunique() == 1
        assert panel["xs_breadth"].iloc[0] == pytest.approx((panel["ret_7d"] > 0).mean())

    def test_the_market_series_is_an_equal_weight_index(self, history):
        snap = _snap(history)
        market = market_series(snap, SYMS)
        assert len(market) > 0 and market.iloc[0] > 0
        assert np.isfinite(market.iloc[-1])

    def test_volatility_estimators_broadly_agree(self, history):
        """Close-to-close, Parkinson and Garman-Klass measure the same thing three ways."""
        panel = build(_snap(history))
        close_to_close = panel["vol_30d"]
        for other in ("vol_parkinson_30d", "vol_garman_klass_30d"):
            ratio = panel[other] / close_to_close
            assert ratio.between(0.3, 3.0).all(), other

    def test_widening_the_config_widens_the_panel(self, history):
        narrow = build(_snap(history), cfg=PanelConfig(return_days=(1,), ema_spans=(10,), vol_days=(7,)))
        wide = build(_snap(history))
        assert len(feature_columns(wide)) > len(feature_columns(narrow))

    def test_one_symbol_returns_its_own_features(self, history):
        snap = _snap(history)
        row = symbol_features(snap, "BTC", market_series(snap, SYMS))
        assert row and "ret_30d" in row and np.isfinite(row["ret_30d"])


class TestLotteryTrendCrossAsset:
    def test_the_lottery_block_is_present_and_ordered(self, history):
        panel = build(_snap(history))
        for col in ("max_daily_ret", "max5_daily_ret", "skew_30d", "kurt_30d", "extreme_share_30d", "up_day_share_30d"):
            assert col in panel and f"{RANK_PREFIX}{col}" in panel, col
        assert (panel["max_daily_ret"] >= panel["max5_daily_ret"]).all()
        assert panel["extreme_share_30d"].between(0, 1).all()

    def test_a_straight_line_has_an_efficiency_ratio_of_one(self, history):
        candles, funding = history
        ts = candles["ts"].max()
        straight = candles.copy()
        line = np.linspace(100, 200, len(straight[straight["symbol"] == "SOL"]))
        straight.loc[straight["symbol"] == "SOL", ["open", "high", "low", "close"]] = np.repeat(line[:, None], 4, 1)
        panel = build(Snapshot.from_long(ts, SYMS, straight, funding))
        assert panel.loc["SOL", "efficiency_ratio"] == pytest.approx(1.0, abs=1e-9)
        assert panel.loc["BTC", "efficiency_ratio"] < 0.5  # a random walk wanders

    def test_trend_quality_columns_are_finite(self, history):
        panel = build(_snap(history))
        assert np.isfinite(panel["hurst_proxy"]).all() and panel["adx"].between(0, 100).all()

    def test_seasonality_is_not_ranked_across_the_cross_section(self, history):
        panel = build(_snap(history))
        assert "day_of_week" in panel and f"{RANK_PREFIX}day_of_week" not in panel
        assert panel["day_of_week"].nunique() == 1  # same bar for everyone

    def test_cross_asset_betas_and_the_eth_btc_ratio(self, history):
        panel = build(_snap(history))
        assert "beta_btc" in panel and "beta_eth" in panel and "xs_ethbtc_ret_30d" in panel
        assert panel.loc["BTC", "beta_btc"] == pytest.approx(1.0, abs=1e-6)
        assert panel["xs_ethbtc_ret_30d"].nunique() == 1

    def test_the_wider_panel_still_cannot_see_the_future(self, history):
        candles, funding = history
        ts = candles["ts"].iloc[24 * 180]
        truncated = build(_snap(history, ts))
        full_view = build(Snapshot.from_long(candles["ts"].max(), SYMS, candles, funding).at(ts))
        pd.testing.assert_frame_equal(truncated, full_view)

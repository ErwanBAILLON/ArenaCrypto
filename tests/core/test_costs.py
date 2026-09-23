"""Square-root impact, and the capacity question it exists to answer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.core.costs import (
    UNKNOWN_LIQUIDITY_SLIPPAGE,
    ImpactModel,
    SymbolLiquidity,
    liquidity_from_bars,
)

LIQUID = SymbolLiquidity(adv_usd=500_000_000.0, daily_vol=0.03)  # BTC-like
THIN = SymbolLiquidity(adv_usd=2_000_000.0, daily_vol=0.12)  # a rank-200 perp


class TestImpact:
    def test_nothing_traded_costs_nothing(self):
        assert ImpactModel().slippage(0.0, LIQUID) == 0.0

    def test_a_liquid_symbol_is_near_the_spread(self):
        cost = ImpactModel().slippage(0.05, LIQUID)
        assert 0.0002 < cost < 0.001  # a few basis points above the half-spread

    def test_a_thin_symbol_costs_an_order_of_magnitude_more(self):
        assert ImpactModel().slippage(0.05, THIN) > 10 * ImpactModel().slippage(0.05, LIQUID)

    def test_impact_is_concave_in_size(self):
        """Four times the size costs twice the impact, not four times: that is the square root."""
        m = ImpactModel(half_spread=0.0, k=1.0)
        small, big = m.slippage(0.01, THIN), m.slippage(0.04, THIN)
        assert big == pytest.approx(2.0 * small, rel=1e-9)

    def test_it_is_capped(self):
        assert ImpactModel(cap=0.05).slippage(1.0, SymbolLiquidity(adv_usd=1_000.0, daily_vol=2.0)) == 0.05

    def test_unknown_liquidity_is_expensive_not_free(self):
        """Missing data must never be the cheapest symbol in the cross-section."""
        assert ImpactModel().slippage(0.05, None) == UNKNOWN_LIQUIDITY_SLIPPAGE
        assert ImpactModel().slippage(0.05, SymbolLiquidity(adv_usd=0.0, daily_vol=0.1)) == UNKNOWN_LIQUIDITY_SLIPPAGE
        assert ImpactModel().slippage(0.05, SymbolLiquidity(adv_usd=1e6, daily_vol=float("nan"))) == (
            UNKNOWN_LIQUIDITY_SLIPPAGE
        )

    def test_sign_of_the_trade_does_not_matter(self):
        m = ImpactModel()
        assert m.slippage(-0.05, THIN) == m.slippage(0.05, THIN)


class TestCapacity:
    def test_pocket_change_hides_the_cost(self):
        """The reason capacity_nav exists: 5 % of a 10k book is 2.5 % of a 2M-ADV perp at 1M."""
        pocket = ImpactModel(capacity_nav=10_000.0).slippage(0.05, THIN)
        real = ImpactModel(capacity_nav=1_000_000.0).slippage(0.05, THIN)
        assert pocket < 0.0025  # ~21 bps: an annoyance
        assert real > 0.015  # ~190 bps: a different strategy entirely
        assert real / pocket > 5

    def test_capacity_scales_as_a_square_root(self):
        m = ImpactModel(half_spread=0.0)
        assert m.at_capacity(4e6).slippage(0.05, THIN) == pytest.approx(
            2.0 * m.at_capacity(1e6).slippage(0.05, THIN), rel=1e-9
        )

    def test_k_can_be_swept_without_rebuilding_the_model(self):
        m = ImpactModel(half_spread=0.0, k=1.0)
        assert m.with_k(2.0).slippage(0.05, THIN) == pytest.approx(2.0 * m.slippage(0.05, THIN), rel=1e-9)


class TestLiquidityFromBars:
    def _bars(self, n=24 * 40, price=100.0, vol_units=1_000.0, sigma=0.01, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        closes = pd.Series(price * np.exp(np.cumsum(rng.normal(0, sigma, n))), index=idx)
        volumes = pd.Series(np.full(n, vol_units), index=idx)
        return closes, volumes

    def test_adv_is_a_daily_figure_not_a_window_total(self):
        closes, volumes = self._bars(sigma=0.0)
        liq = liquidity_from_bars(closes, volumes)
        assert liq is not None
        assert liq.adv_usd == pytest.approx(24 * 100.0 * 1_000.0, rel=0.02)

    def test_daily_vol_is_annualised_from_the_bar_size(self):
        closes, volumes = self._bars(sigma=0.01)
        liq = liquidity_from_bars(closes, volumes)
        assert liq.daily_vol == pytest.approx(0.01 * np.sqrt(24), rel=0.25)

    def test_a_window_too_short_says_so_rather_than_guessing(self):
        closes, volumes = self._bars(n=10)
        assert liquidity_from_bars(closes, volumes) is None

    def test_it_only_reads_the_trailing_window(self):
        closes, volumes = self._bars(n=24 * 200, sigma=0.0)
        volumes.iloc[: 24 * 150] = 1e9  # ancient volume, far outside 30 days
        liq = liquidity_from_bars(closes, volumes, window_days=30)
        assert liq.adv_usd == pytest.approx(24 * 100.0 * 1_000.0, rel=0.05)

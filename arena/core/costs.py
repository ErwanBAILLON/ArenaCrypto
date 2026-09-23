"""Per-symbol execution cost: the square-root impact law, priced at a stated capacity.

A flat slippage number is conservative on BTC and fiction on the 200th perp by
volume. Widening the universe without widening the cost model is how a backtest
manufactures alpha out of thin liquidity, so the two have to move together.

The model is the square-root law, empirically confirmed on BTC/USD futures and
[remarkably universal](https://bouchaud.substack.com/p/the-square-root-law-of-market-impact)
across equities, futures, FX and crypto::

    slippage = half_spread + k · σ_daily · (notional / ADV)^δ

with ``δ ≈ 0.5``. ``k`` is **an assumption, not a fit** -- the arena has no
order-book data -- so it is a swept parameter with a pessimistic default, and a
strategy whose edge survives only at optimistic ``k`` has not shown an edge.

``capacity_nav`` is the point of the module. The arena books 10 000 € per
competitor; at that size the impact term is negligible on every symbol, which
silently flatters illiquid names and answers the wrong question. Costs are
therefore computed as if the same weights were traded at ``capacity_nav``, so
the backtest reports what the rule would cost with real money rather than with
pocket change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

DEFAULT_CAPACITY_NAV = 1_000_000.0
UNKNOWN_LIQUIDITY_SLIPPAGE = 0.01  # 100 bps: a symbol we cannot size is one we should not trade


@dataclass(frozen=True)
class SymbolLiquidity:
    """Point-in-time execution context for one symbol."""

    adv_usd: float  # trailing average daily dollar volume
    daily_vol: float  # realised daily volatility, as a fraction

    @property
    def usable(self) -> bool:
        return bool(
            np.isfinite(self.adv_usd) and self.adv_usd > 0 and np.isfinite(self.daily_vol) and self.daily_vol > 0
        )


@dataclass(frozen=True)
class ImpactModel:
    """Square-root impact, plus a fixed half-spread, as a fraction of notional traded.

    ``cap`` bounds a single fill: past it the position is not executable at any
    sensible price, and a backtest that keeps filling there is writing fiction.
    Unknown liquidity is charged ``UNKNOWN_LIQUIDITY_SLIPPAGE`` rather than zero,
    so missing data can never look cheap.
    """

    half_spread: float = 0.0002
    k: float = 1.0
    delta: float = 0.5
    cap: float = 0.05
    capacity_nav: float = DEFAULT_CAPACITY_NAV

    def slippage(self, weight_delta: float, liq: SymbolLiquidity | None) -> float:
        """Cost fraction for trading ``|weight_delta|`` of the book at ``capacity_nav``."""
        size = abs(float(weight_delta))
        if size <= 0.0:
            return 0.0
        if liq is None or not liq.usable:
            return min(UNKNOWN_LIQUIDITY_SLIPPAGE, self.cap)
        participation = (size * self.capacity_nav) / liq.adv_usd
        impact = self.k * liq.daily_vol * participation**self.delta
        return float(min(self.half_spread + impact, self.cap))

    def at_capacity(self, capacity_nav: float) -> ImpactModel:
        """Same model priced at a different deployment size (used to sweep capacity)."""
        return replace(self, capacity_nav=float(capacity_nav))

    def with_k(self, k: float) -> ImpactModel:
        """Same model at a different impact coefficient (used to sweep the assumption)."""
        return replace(self, k=float(k))


def liquidity_from_bars(closes, volumes, bars_per_day: int = 24, window_days: int = 30) -> SymbolLiquidity | None:
    """``SymbolLiquidity`` from the trailing window of one symbol's bars.

    ``adv_usd`` is the mean daily dollar volume (close × volume summed per day),
    ``daily_vol`` the realised standard deviation of daily log returns. Returns
    None when the window is too short to say anything, which the impact model
    then charges as unknown rather than free.
    """
    import pandas as pd

    close = pd.Series(closes).dropna()
    volume = pd.Series(volumes).reindex(close.index).fillna(0.0)
    window = window_days * bars_per_day
    if len(close) < max(2 * bars_per_day, bars_per_day + 2):
        return None
    close, volume = close.tail(window), volume.tail(window)
    dollar = (close * volume).sum()
    days = max(len(close) / bars_per_day, 1e-9)
    adv = float(dollar / days)
    per_day = np.log(close).diff().dropna()
    if per_day.empty:
        return None
    daily_vol = float(per_day.std(ddof=1) * np.sqrt(bars_per_day))
    if not np.isfinite(adv) or not np.isfinite(daily_vol):
        return None
    return SymbolLiquidity(adv_usd=adv, daily_vol=daily_vol)

"""Virtual book: marks positions to market, accrues funding, charges fees.

The same class scores backtests and the live arena; there is exactly one
accounting code path.

Return for one step::

    ret = sum_perp w_prev * (p/p_prev - 1)   # signed; carry has no price PnL
        + funding_pnl                        # short perp receives +rate*|w|, long pays; carry receives +rate*w
        - fees                               # on turnover after the move
    nav = nav_prev * (1 + ret)

Targets whose gross exposure exceeds 1 are scaled down proportionally.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from arena.core.costs import ImpactModel, SymbolLiquidity
from arena.core.types import BookRow, Decision, Kind


@dataclass(frozen=True)
class FeeModel:
    """Taker fees, plus slippage either flat or from a per-symbol impact model.

    ``impact`` is opt-in and defaults to None, in which case every number this
    book produces is byte-identical to before it existed -- no past verdict is
    silently revised. With it, slippage is the square-root law evaluated at the
    model's stated capacity, and a carry position pays it on **both** legs,
    because both of them cross a book.
    """

    perp_taker: float = 0.0005
    slippage: float = 0.0002
    spot_taker: float = 0.0010
    impact: ImpactModel | None = None

    @property
    def perp_cost(self) -> float:
        return self.perp_taker + self.slippage

    @property
    def carry_cost(self) -> float:
        # both legs move: perp leg + spot leg
        return self.perp_taker + self.slippage + self.spot_taker

    def cost(self, kind: Kind, weight_delta: float = 0.0, liq: SymbolLiquidity | None = None) -> float:
        """Cost fraction of one unit of turnover in ``kind``, for this symbol and size."""
        if self.impact is None:
            return self.carry_cost if kind == "carry" else self.perp_cost
        slip = self.impact.slippage(weight_delta, liq)
        if kind == "carry":
            return self.perp_taker + self.spot_taker + 2.0 * slip
        return self.perp_taker + slip


@dataclass
class Book:
    nav: float = 10_000.0
    fees: FeeModel = field(default_factory=FeeModel)
    positions: dict[str, tuple[Kind, float]] = field(default_factory=dict)

    @classmethod
    def restore(cls, nav: float, positions: dict[str, tuple[Kind, float]], fees: FeeModel | None = None) -> Book:
        return cls(nav=nav, fees=fees or FeeModel(), positions=dict(positions))

    @staticmethod
    def cap_gross(targets: Decision) -> dict[str, tuple[Kind, float]]:
        wanted = {s: (t.kind, float(t.weight)) for s, t in targets.items() if t.weight != 0.0}
        gross = sum(abs(w) for _, w in wanted.values())
        if gross > 1.0:
            wanted = {s: (k, w / gross) for s, (k, w) in wanted.items()}
        return wanted

    def step(
        self,
        ts: datetime,
        prices: dict[str, float],
        prev_prices: dict[str, float],
        funding: dict[str, float],
        targets: Decision,
        liquidity: dict[str, SymbolLiquidity] | None = None,
        fills: Iterable[tuple[str, str, float, float]] = (),
    ) -> BookRow:
        """Advance the book one bar.

        ``fills`` are legs closed *inside* the bar by the live watcher, as
        ``(symbol, kind, weight_before, fill_price)``. Their positions are
        already gone from ``self.positions`` (the watcher re-stated the book),
        so this step earns them ``w × (fill / prev_close − 1)`` and charges the
        closing turnover, as if the tick had sold them itself at that price.
        """
        # 1. mark to market with positions held over (prev_ts, ts]
        price_ret = 0.0
        funding_pnl = 0.0
        turnover = 0.0
        fees = 0.0
        for sym, kind, w, fill_price in fills:
            pp = prev_prices.get(sym)
            if pp and kind == "perp":
                price_ret += w * (fill_price / pp - 1.0)
            turnover += abs(w)
            fees += abs(w) * self.fees.cost(kind, abs(w), (liquidity or {}).get(sym))
        for sym, (kind, w) in self.positions.items():
            rate = float(funding.get(sym, 0.0) or 0.0)
            if kind == "perp":
                p, pp = prices.get(sym), prev_prices.get(sym)
                if p is not None and pp:
                    price_ret += w * (p / pp - 1.0)
                funding_pnl += -w * rate  # long pays when rate>0, short receives
            else:  # carry: long spot / short perp, delta neutral
                funding_pnl += w * rate

        # 2. rebalance to new targets, pay fees on turnover
        new = self.cap_gross(targets)
        symbols = set(self.positions) | set(new)
        for sym in symbols:
            k_old, w_old = self.positions.get(sym, ("perp", 0.0))
            k_new, w_new = new.get(sym, (k_old, 0.0))
            liq = (liquidity or {}).get(sym)
            if k_old != k_new and w_old != 0.0:
                # kind change: close old fully, open new fully
                d_close, d_open = abs(w_old), abs(w_new)
                turnover += d_close + d_open
                fees += d_close * self.fees.cost(k_old, d_close, liq) + d_open * self.fees.cost(k_new, d_open, liq)
            else:
                d = abs(w_new - w_old)
                turnover += d
                fees += d * self.fees.cost(k_new, d, liq)

        ret = price_ret + funding_pnl - fees
        self.nav *= 1.0 + ret
        self.positions = new
        gross = sum(abs(w) for _, w in new.values())
        return BookRow(ts=ts, nav=self.nav, ret=ret, gross=gross, turnover=turnover, fees=fees, funding_pnl=funding_pnl)

    def _cost(self, kind: Kind) -> float:
        """Flat cost of one unit of turnover; kept for callers that do not size trades."""
        return self.fees.cost(kind)

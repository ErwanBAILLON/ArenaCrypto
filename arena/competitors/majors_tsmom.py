"""Time-series momentum on the majors alone, weekly, long or flat.

The night of 2026-09-28 put every weekly rule through the fast harness on five
years of point-in-time data with the arena's costs. One thing stood above its
own null: on BTC and ETH only, being long when the 30-day return is positive and
flat otherwise, sized to a 20 % volatility target, kept the return of always
being long (median quarterly Sharpe 0.43 against 0.40) with half the drawdown
(14 % against 28 %). That is the classic finding on time-series momentum
(Moskowitz, Ooi & Pedersen 2012): the filter does not add return, it removes
the worst of the beta. Longer lookbacks (60, 90 days) and wider baskets (5, 10
names) were worse than doing nothing; shorting on a negative return lost money.

So the rule is deliberately small: two names, one lookback, one side, one
decision a week. It runs as a challenger and the judge decides, forward, whether
what survived five years of backtest survives the next quarters live.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from arena.competitors.base import HoldingCompetitor, cap_gross, register
from arena.competitors.features import last_realised_vol, pct_return
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target

VOL_FLOOR = 0.05


@register
class MajorsTsmom(HoldingCompetitor):
    family = "majors_tsmom"
    markets: ClassVar[frozenset[str]] = frozenset({"crypto"})
    rebalance_weekday: ClassVar[int | None] = 0  # Mondays: the cadence the harness tested
    default_params: ClassVar[dict[str, Any]] = {
        "symbols": ["BTC", "ETH"],
        "lookback_days": 30,
        "target_vol": 0.20,
        "vol_window_days": 30,
        "max_weight": 0.5,
    }

    def warmup_bars(self) -> int:
        p = self.params
        return max(self.days(p["lookback_days"]), self.days(p["vol_window_days"])) + 1

    def _members(self, snap: Snapshot) -> list[str]:
        """The configured names as the universe spells them (``BTC`` or ``BTCUSDT`` alike)."""
        wanted = [str(s) for s in self.params["symbols"]]
        out = []
        for sym in snap.symbols:
            base = sym[:-4] if sym.endswith("USDT") else sym
            if sym in wanted or base in wanted:
                out.append(sym)
        return out

    def compute(self, snap: Snapshot) -> Decision:
        p = self.params
        names = self._members(snap)
        out: Decision = {}
        for sym in names:
            c = snap.candles(sym)
            if len(c) < self.warmup_bars():
                continue
            close = c["close"].dropna()
            r = pct_return(close, self.days(p["lookback_days"]))
            if not np.isfinite(r) or r <= 0:
                continue
            vol = last_realised_vol(close, self.days(p["vol_window_days"]), self.bars_per_year)
            if not np.isfinite(vol):
                continue
            size = min(float(p["max_weight"]), float(p["target_vol"]) / max(vol, VOL_FLOOR) / len(names))
            out[sym] = Target(
                weight=size,
                conviction=float(np.clip(r / 0.20, 0.1, 1.0)),
                reason={"ret": round(float(r), 4), "vol": round(float(vol), 4)},
            )
        return cap_gross(out)

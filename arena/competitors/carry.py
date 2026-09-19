"""Funding carry.

Perp funding is paid by leveraged longs to shorts when the crowd is long;
holding spot against a short perp collects it delta-neutrally. We rank symbols
by recent mean funding and carry the top-k above a threshold that covers fees.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from arena.competitors.base import Competitor, cap_gross, register
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target

HL_HOURS_PER_BINANCE_INTERVAL = 8  # Hyperliquid funding is hourly, Binance every 8h


def rank_funding(snap: Snapshot, lookback_days: int) -> list[tuple[str, float]]:
    """Symbols sorted by mean 8h funding over ``lookback_days``, best first.

    Hyperliquid's current hourly rate, when known, is scaled to an 8h rate and
    averaged in as a second venue reading. Symbols without funding are omitted.
    """
    since = snap.ts - timedelta(days=lookback_days)
    ranked: list[tuple[str, float]] = []
    for sym in snap.symbols:
        f = snap.funding(sym)
        f = f[f.index > since]
        if f.empty:
            continue
        mean = float(f.mean())
        hl = snap.hl_funding(sym)
        if hl is not None:
            mean = (mean + float(hl) * HL_HOURS_PER_BINANCE_INTERVAL) / 2.0
        ranked.append((sym, mean))
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked


@register
class Carry(Competitor):
    family = "carry"
    default_params = {"lookback_days": 3, "k": 3, "min_rate": 0.0001, "max_weight": 0.34}

    def warmup_bars(self) -> int:
        return int(self.params["lookback_days"]) * 24

    def decide(self, snap: Snapshot) -> Decision:
        k = int(self.params["k"])
        min_rate = float(self.params["min_rate"])
        ranked = rank_funding(snap, int(self.params["lookback_days"]))
        picks = [(s, m) for s, m in ranked if m >= min_rate][:k]
        if not picks:
            return {}
        w = min(float(self.params["max_weight"]), 1.0 / k)
        out: Decision = {}
        for i, (sym, mean) in enumerate(picks):
            conviction = 1.0 if min_rate <= 0 else float(np.clip(mean / (3 * min_rate), 0.0, 1.0))
            out[sym] = Target(weight=w, conviction=conviction, kind="carry",
                              reason={"mean_funding_8h": mean, "rank": i})
        return cap_gross(out)

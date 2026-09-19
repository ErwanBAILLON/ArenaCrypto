"""Slow narrative sentiment (news-only competitor).

Headlines move prices in minutes, which we cannot exploit; what survives is the
slow drift of a narrative over days. We go long the assets whose 7-day
sentiment is positive and rising, sized by how much is being written about
them. The competitor is standalone so its marginal value can be measured
against the null models, and it is never gated by backtest (its scores only
exist forward, see the design §8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arena.competitors.base import Competitor, cap_gross, register
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target

STALE_HOURS = 2  # news features older than this are treated as missing
RISING_LAG_BARS = 24  # compare 7d sentiment with the value one day earlier
DENSITY_FULL = 10  # articles per 7 days that count as full size


@register
class News(Competitor):
    family = "news"
    default_params = {"min_sent_7d": 0.10, "k": 3, "max_weight": 0.30, "min_n_7d": 3}

    def warmup_bars(self) -> int:
        return RISING_LAG_BARS + 1

    def decide(self, snap: Snapshot) -> Decision:
        p = self.params
        scored: list[tuple[str, float, float]] = []
        for sym in snap.symbols:
            f = snap.news(sym)
            if f.empty or (snap.ts - f.index[-1]) > pd.Timedelta(hours=STALE_HOURS):
                continue
            if len(f) <= RISING_LAG_BARS:
                continue
            now, before = f.iloc[-1], f.iloc[-1 - RISING_LAG_BARS]
            if now["n_7d"] < p["min_n_7d"]:
                continue
            if now["sent_7d"] > p["min_sent_7d"] and now["sent_7d"] > before["sent_7d"]:
                scored.append((sym, float(now["sent_7d"]), float(now["n_7d"])))
        scored.sort(key=lambda x: (-x[1], x[0]))
        k = int(p["k"])
        out: Decision = {}
        for sym, sent, n in scored[:k]:
            density = min(1.0, n / DENSITY_FULL)
            weight = min(float(p["max_weight"]), 1.0 / k) * density
            out[sym] = Target(
                weight=weight,
                conviction=float(np.clip(sent, 0.0, 1.0)),
                reason={"sent_7d": round(sent, 3), "n_7d": int(n)},
            )
        return cap_gross(out)

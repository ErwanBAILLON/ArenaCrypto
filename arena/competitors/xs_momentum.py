"""Cross-sectional momentum (12-1 style, compressed to crypto horizons).

Recent relative winners keep outperforming relative losers for weeks; we rank
by 30d return skipping the last 2d (short-term reversal), long the top k and
short the bottom k, rebalancing weekly with a rank-hysteresis band to cut
turnover.

State: ``_prev`` (signs held) and ``_last_rebalance_ts`` exist only for the
hysteresis. A fresh instance fed bars in chronological order reproduces the
same decisions, so backtests and live remain identical.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from arena.competitors.base import Competitor, cap_gross, register
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target


@register
class XSMomentum(Competitor):
    family = "xs_momentum"
    default_params = {
        "lookback_days": 30, "skip_days": 2, "k": 3, "band": 1, "rebalance_hours": 168, "long_only": False,
    }

    def __init__(self, params: dict | None = None, seed: int = 0):
        super().__init__(params, seed)
        self._prev: dict[str, int] = {}
        self._last_rebalance_ts: pd.Timestamp | None = None

    def state(self) -> dict:
        return {
            "prev": dict(self._prev),
            "last_rebalance_ts": self._last_rebalance_ts.isoformat() if self._last_rebalance_ts is not None else None,
        }

    def restore_state(self, state: dict) -> None:
        self._prev = {str(k): int(v) for k, v in (state.get("prev") or {}).items()}
        ts = state.get("last_rebalance_ts")
        self._last_rebalance_ts = pd.Timestamp(ts) if ts else None

    def warmup_bars(self) -> int:
        return (int(self.params["lookback_days"]) + int(self.params["skip_days"])) * 24 + 1

    def _scores(self, snap: Snapshot) -> pd.Series:
        closes = snap.closes()
        if len(closes) < self.warmup_bars():
            return pd.Series(dtype=float)
        skip = int(self.params["skip_days"]) * 24
        span = (int(self.params["lookback_days"]) + int(self.params["skip_days"])) * 24
        score = closes.iloc[-1 - skip] / closes.iloc[-1 - span] - 1.0
        return score.dropna()

    def _select(self, ranked: list[str], held: set[str], k: int, band: int) -> list[str]:
        kept = [s for s in ranked[: k + band] if s in held]
        for s in ranked:
            if len(kept) >= k:
                break
            if s not in kept:
                kept.append(s)
        return kept[:k]

    def decide(self, snap: Snapshot) -> Decision:
        p = self.params
        k, band = int(p["k"]), int(p["band"])
        score = self._scores(snap)
        if score.empty:
            return {}
        due = (
            self._last_rebalance_ts is None
            or snap.ts - self._last_rebalance_ts >= timedelta(hours=int(p["rebalance_hours"]))
        )
        desc = list(score.sort_values(ascending=False).index)
        if due:
            longs = self._select(desc, {s for s, d in self._prev.items() if d > 0}, k, band)
            shorts = [] if p["long_only"] else self._select(
                desc[::-1], {s for s, d in self._prev.items() if d < 0}, k, band)
            shorts = [s for s in shorts if s not in longs]
            self._prev = {**{s: 1 for s in longs}, **{s: -1 for s in shorts}}
            self._last_rebalance_ts = snap.ts
        longs = [s for s, d in self._prev.items() if d > 0 and s in score.index]
        shorts = [s for s, d in self._prev.items() if d < 0 and s in score.index]
        w = (1.0 / k) if p["long_only"] else (0.5 / k)
        n = max(len(desc) - 1, 1)
        rank = {s: i for i, s in enumerate(desc)}
        out: Decision = {}
        for s in longs:
            out[s] = Target(weight=w, conviction=1.0 - rank[s] / n, reason={"rank": rank[s], "score": float(score[s])})
        for s in shorts:
            out[s] = Target(weight=-w, conviction=rank[s] / n, reason={"rank": rank[s], "score": float(score[s])})
        return cap_gross(out)

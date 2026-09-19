"""Meta-labeling (López de Prado, *Advances in Financial Machine Learning*, ch. 3).

Base rules (trend_ts, xs_momentum) decide *where* and *which side*; a small
gradient-boosting model decides *whether to take the signal and how much*,
from the context it was emitted in (volatility, regime, funding, open
interest, news, calendar). The learning target is well defined (did this
signal pay after fees?), which keeps the model small and honest. Without a
trained model the competitor is a pass-through of its bases, so the model's
marginal value is measurable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from arena.competitors.base import REGISTRY, Competitor, cap_gross, register
from arena.competitors.features import pct_return, realised_vol
from arena.competitors.regime import regime_label
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target

REGIMES = ["bull_calm", "bull_vol", "bear", "range"]
FEATURE_COLUMNS = [
    "base_weight",
    "n_agree",
    "vol_30d",
    "r_7d",
    "funding_3d",
    "oi_change_24h",
    "sent_24h",
    "sent_7d",
    "n_7d",
    "hour",
    "macro_today",
    *[f"regime_{r}" for r in REGIMES],
]
DEFAULT_BASES = [{"family": "trend_ts", "params": {}}, {"family": "xs_momentum", "params": {}}]


def build_features(snap: Snapshot, symbol: str, base_weight: float, n_agree: int, regime: str) -> dict[str, float]:
    """Context features for one base signal on ``symbol`` at ``snap.ts``."""
    close = snap.candles(symbol, "1h")["close"]
    vol = realised_vol(close, 24 * 30)
    fund = snap.funding(symbol)
    fund_3d = fund[fund.index > snap.ts - pd.Timedelta(days=3)]
    oi = snap.open_interest(symbol)
    oi_change = float("nan")
    if len(oi) > 24 and oi.iloc[-25] > 0:
        oi_change = float(oi.iloc[-1] / oi.iloc[-25] - 1.0)
    news = snap.news(symbol)
    last_news = news.iloc[-1] if not news.empty else None
    feats: dict[str, float] = {
        "base_weight": float(base_weight),
        "n_agree": float(n_agree),
        "vol_30d": float(vol.iloc[-1]) if len(vol) else float("nan"),
        "r_7d": pct_return(close, 24 * 7),
        "funding_3d": float(fund_3d.mean()) if len(fund_3d) else 0.0,
        "oi_change_24h": oi_change,
        "sent_24h": float(last_news["sent_24h"]) if last_news is not None else 0.0,
        "sent_7d": float(last_news["sent_7d"]) if last_news is not None else 0.0,
        "n_7d": float(last_news["n_7d"]) if last_news is not None else 0.0,
        "hour": float(snap.ts.hour),
        "macro_today": float(snap.macro_today()),
    }
    for r in REGIMES:
        feats[f"regime_{r}"] = 1.0 if regime == r else 0.0
    return feats


def combine_bases(decisions: list[Decision]) -> dict[str, tuple[float, int]]:
    """Average base weights per symbol; return (mean_weight, number of bases positioned)."""
    acc: dict[str, list[float]] = {}
    for d in decisions:
        for sym, t in d.items():
            if t.kind == "perp" and t.weight != 0.0:
                acc.setdefault(sym, []).append(float(t.weight))
    n = max(1, len(decisions))
    return {sym: (sum(ws) / n, len(ws)) for sym, ws in acc.items()}


@register
class MetaLabel(Competitor):
    family = "meta_label"
    default_params: dict[str, Any] = {"bases": DEFAULT_BASES, "threshold": 0.55, "model_str": None}

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0):
        super().__init__(params, seed)
        self.bases: list[Competitor] = [
            REGISTRY[b["family"]](b.get("params") or {}, seed=seed) for b in self.params["bases"]
        ]
        self._booster = None
        if self.params.get("model_str"):
            import lightgbm as lgb

            self._booster = lgb.Booster(model_str=self.params["model_str"])

    def warmup_bars(self) -> int:
        return max([b.warmup_bars() for b in self.bases] + [24 * 30 + 1])

    def base_signals(self, snap: Snapshot) -> tuple[dict[str, tuple[float, int]], str]:
        decisions = [b.decide(snap) for b in self.bases]
        return combine_bases(decisions), regime_label(snap)

    def decide(self, snap: Snapshot) -> Decision:
        combined, regime = self.base_signals(snap)
        if not combined:
            return {}
        out: Decision = {}
        if self._booster is None:
            for sym, (w, n) in combined.items():
                out[sym] = Target(w, conviction=n / len(self.bases), reason={"mode": "pass_through", "n_agree": n})
            return cap_gross(out)
        rows = [build_features(snap, sym, w, n, regime) for sym, (w, n) in combined.items()]
        X = pd.DataFrame(rows)[FEATURE_COLUMNS].to_numpy(dtype=float)
        probs = np.asarray(self._booster.predict(X), dtype=float).reshape(-1)
        thr = float(self.params["threshold"])
        for (sym, (w, n)), p in zip(combined.items(), probs, strict=True):
            if p > thr:
                out[sym] = Target(
                    w * float(p),
                    conviction=float(p),
                    reason={"p_win": round(float(p), 3), "regime": regime, "n_agree": n},
                )
        return cap_gross(out)

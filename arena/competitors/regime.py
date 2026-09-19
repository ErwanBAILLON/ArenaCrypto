"""Regime conditioning, rule-based v1.

Trend rules pay in calm bull markets, carry pays in ranges, nothing pays in a
volatile bear: rather than one rule for all weather, classify BTC's state
(trend sign x realised-vol tercile) and map each state to a fixed allocation.
``regime_label`` is also consumed by other families as a context feature.
"""

from __future__ import annotations

from typing import Any

from arena.competitors.base import Competitor, cap_gross, register
from arena.competitors.carry import rank_funding
from arena.competitors.features import ema, realised_vol
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target

DEFAULT_PARAMS: dict[str, Any] = {
    "vol_window": 24 * 30, "vol_lookback_days": 180, "trend_fast": 50, "trend_slow": 200,
    "bull_weight": 0.5, "range_carry_weight": 0.5,
}
LABELS = ("bull_calm", "bull_vol", "bear", "range", "unknown")
CARRY_LOOKBACK_DAYS = 3


def warmup_bars(params: dict[str, Any]) -> int:
    return max(int(params["trend_slow"]), int(params["vol_window"]) + int(params["vol_lookback_days"]) * 24) + 1


def regime_label(snap: Snapshot, symbol: str = "BTC", params: dict[str, Any] | None = None) -> str:
    """Classify the market state from ``symbol``'s 1h closes at ``snap.ts``."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = snap.candles(symbol, "1h")["close"]
    if len(close) < warmup_bars(p):
        return "unknown"
    trend_up = float(ema(close, int(p["trend_fast"])).iloc[-1]) > float(ema(close, int(p["trend_slow"])).iloc[-1])
    vol = realised_vol(close, int(p["vol_window"])).dropna()
    recent = vol.iloc[-int(p["vol_lookback_days"]) * 24:]
    lo, hi = recent.quantile(1 / 3), recent.quantile(2 / 3)
    v = float(vol.iloc[-1])
    high_vol = v > hi
    if trend_up:
        return "bull_vol" if high_vol else "bull_calm"
    return "bear" if high_vol else "range"


@register
class Regime(Competitor):
    family = "regime"
    default_params = DEFAULT_PARAMS

    def warmup_bars(self) -> int:
        return warmup_bars(self.params)

    def decide(self, snap: Snapshot) -> Decision:
        p = self.params
        label = regime_label(snap, "BTC", p)
        reason = {"regime": label}
        out: Decision = {}
        if label == "bull_calm":
            w = float(p["bull_weight"]) / 2
            for sym in ("BTC", "ETH"):
                if sym in snap.symbols:
                    out[sym] = Target(weight=w, conviction=0.7, reason=reason)
        elif label == "bull_vol":
            out["BTC"] = Target(weight=float(p["bull_weight"]) / 2, conviction=0.5, reason=reason)
        elif label == "range":
            w = float(p["range_carry_weight"]) / 2
            top = [(s, m) for s, m in rank_funding(snap, CARRY_LOOKBACK_DAYS) if m > 0][:2]
            for i, (sym, mean) in enumerate(top):
                out[sym] = Target(weight=w, conviction=0.5, kind="carry",
                                  reason={**reason, "mean_funding_8h": mean, "rank": i})
        return cap_gross(out)

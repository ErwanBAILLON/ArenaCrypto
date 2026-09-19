"""Time-series momentum (Moskowitz, Ooi, Pedersen 2012).

An asset that has been going up over 1-3 months tends to keep going up for a
while; we require EMA alignment and positive 30d and 90d returns to agree, size
each leg to a target annualised volatility and exit on an ATR stop.
"""

from __future__ import annotations

import numpy as np

from arena.competitors.base import Competitor, cap_gross, register
from arena.competitors.features import atr, ema, pct_return, realised_vol
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target

VOL_FLOOR = 0.05  # avoid infinite sizing on a dead-flat series
CONVICTION_SCALE = 0.20  # |30d return| that counts as full conviction


@register
class TrendTS(Competitor):
    family = "trend_ts"
    default_params = {
        "fast": 50, "slow": 200, "lb_short_days": 30, "lb_long_days": 90,
        "target_vol": 0.20, "vol_window": 24 * 30, "max_weight": 0.5, "atr_stop_mult": 3.0,
    }

    def warmup_bars(self) -> int:
        p = self.params
        return max(int(p["slow"]), int(p["lb_long_days"]) * 24, int(p["vol_window"])) + 1

    def decide(self, snap: Snapshot) -> Decision:
        p = self.params
        out: Decision = {}
        for sym in snap.symbols:
            c = snap.candles(sym, "1h")
            if len(c) < self.warmup_bars():
                continue
            close = c["close"]
            e_fast = float(ema(close, int(p["fast"])).iloc[-1])
            e_slow = float(ema(close, int(p["slow"])).iloc[-1])
            r_short = pct_return(close, int(p["lb_short_days"]) * 24)
            r_long = pct_return(close, int(p["lb_long_days"]) * 24)
            if e_fast > e_slow and r_short > 0 and r_long > 0:
                direction = 1
            elif e_fast < e_slow and r_short < 0 and r_long < 0:
                direction = -1
            else:
                continue
            last = float(close.iloc[-1])
            stop_band = float(p["atr_stop_mult"]) * float(atr(c).iloc[-1])
            if direction == 1 and last < e_fast - stop_band:
                continue
            if direction == -1 and last > e_fast + stop_band:
                continue
            vol = float(realised_vol(close, int(p["vol_window"])).iloc[-1])
            size = min(float(p["max_weight"]), float(p["target_vol"]) / max(vol, VOL_FLOOR))
            conviction = float(np.clip(abs(r_short) / CONVICTION_SCALE, 0.0, 1.0))
            out[sym] = Target(
                weight=direction * size, conviction=conviction,
                reason={"r_short": r_short, "r_long": r_long, "vol": vol, "ema_fast": e_fast, "ema_slow": e_slow},
            )
        return cap_gross(out)

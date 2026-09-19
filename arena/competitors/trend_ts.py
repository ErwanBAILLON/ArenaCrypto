"""Time-series momentum (Moskowitz, Ooi, Pedersen 2012).

An asset that has been going up over 1-3 months tends to keep going up for a
while; we require EMA alignment and positive 30d and 90d returns to agree, size
each leg to a target annualised volatility and exit on an ATR stop.
"""

from __future__ import annotations

import numpy as np

from arena.competitors.base import HoldingCompetitor, cap_gross, register
from arena.competitors.features import last_atr, last_ema, last_realised_vol, pct_return
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target

VOL_FLOOR = 0.05  # avoid infinite sizing on a dead-flat series
CONVICTION_SCALE = 0.20  # |30d return| that counts as full conviction


@register
class TrendTS(HoldingCompetitor):
    family = "trend_ts"
    default_params = {
        "fast": 50,
        "slow": 200,
        "lb_short_days": 30,
        "lb_long_days": 90,
        "target_vol": 0.20,
        "vol_window_days": 30,
        "max_weight": 0.5,
        "atr_stop_mult": 3.0,
    }

    def warmup_bars(self) -> int:
        p = self.params
        return max(int(p["slow"]), self.days(p["lb_long_days"]), self.days(p["vol_window_days"])) + 1

    def compute(self, snap: Snapshot) -> Decision:
        p = self.params
        out: Decision = {}
        for sym in snap.symbols:
            c = snap.candles(sym, "1h")
            if len(c) < self.warmup_bars():
                continue
            close = c["close"]
            e_fast = last_ema(close, int(p["fast"]))
            e_slow = last_ema(close, int(p["slow"]))
            r_short = pct_return(close, self.days(p["lb_short_days"]))
            r_long = pct_return(close, self.days(p["lb_long_days"]))
            if e_fast > e_slow and r_short > 0 and r_long > 0:
                direction = 1
            elif e_fast < e_slow and r_short < 0 and r_long < 0:
                direction = -1
            else:
                continue
            last = float(close.iloc[-1])
            stop_band = float(p["atr_stop_mult"]) * last_atr(c)
            if direction == 1 and last < e_fast - stop_band:
                continue
            if direction == -1 and last > e_fast + stop_band:
                continue
            vol = last_realised_vol(close, self.days(p["vol_window_days"]), self.bars_per_year)
            size = min(float(p["max_weight"]), float(p["target_vol"]) / max(vol, VOL_FLOOR))
            conviction = float(np.clip(abs(r_short) / CONVICTION_SCALE, 0.0, 1.0))
            out[sym] = Target(
                weight=direction * size,
                conviction=conviction,
                reason={"r_short": r_short, "r_long": r_long, "vol": vol, "ema_fast": e_fast, "ema_slow": e_slow},
            )
        return cap_gross(out)

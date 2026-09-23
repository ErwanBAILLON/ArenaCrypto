"""Time-series momentum, filtered by who is paying to hold it.

The hypothesis, stated so it can lose: crypto trend following underperforms its
equity-futures ancestor because the trades are levered. A trend the whole market
is levered into unwinds through liquidations, not through a gentle loss of
momentum, which is exactly the kind of drawdown a stop cannot protect against.
A trend nobody is paying a premium to hold has no such exit queue behind it.

So this family takes the same signal as ``trend_ts`` -- EMA alignment plus
agreement between the 30d and 90d returns, sized to a target volatility -- and
then throws away every leg that the funding cross-section says is crowded in
the direction of the trade:

* a **long** is kept only when the symbol's funding z-score is below
  ``max_long_z``: going long something whose longs already pay a premium is
  joining the queue;
* a **short** is kept only when the z-score is above ``min_short_z``: shorting
  something whose shorts are already paid is the same mistake mirrored.

Against ``trend_ts`` and ``funding_skew``, this is a three-way test of one
question: is funding a *signal* (trade against the crowd), a *filter* (trade
with the trend, away from the crowd), or *noise*? The three families share the
judge, the fee model and the book, so whichever wins, the answer is readable.

``trend_ts`` is deliberately not imported. Sharing code between two competitors
would mean a change to one silently re-parameterises the other, and the arena's
whole premise is that a running model is never modified in place.
"""

from __future__ import annotations

import numpy as np

from arena.competitors.base import HoldingCompetitor, cap_gross, register
from arena.competitors.features import funding_zscores, last_atr, last_ema, last_realised_vol, pct_return
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target

VOL_FLOOR = 0.05
CONVICTION_SCALE = 0.20


@register
class CrowdedTrend(HoldingCompetitor):
    family = "crowded_trend"

    default_params = {
        "fast": 50,
        "slow": 200,
        "lb_short_days": 30,
        "lb_long_days": 90,
        "target_vol": 0.20,
        "vol_window_days": 30,
        "max_weight": 0.5,
        "atr_stop_mult": 3.0,
        "funding_lookback_days": 7,
        "max_long_z": 0.5,  # a long is dropped above this: the crowd is already there
        "min_short_z": -0.5,  # a short is dropped below this
    }

    def warmup_bars(self) -> int:
        p = self.params
        return (
            max(
                int(p["slow"]),
                self.days(p["lb_long_days"]),
                self.days(p["vol_window_days"]),
                self.days(p["funding_lookback_days"]),
            )
            + 1
        )

    def _direction(self, snap: Snapshot, sym: str) -> tuple[int, dict] | None:
        """The ``trend_ts`` signal for one symbol, or None when it does not fire."""
        p = self.params
        c = snap.candles(sym, "1h")
        if len(c) < self.warmup_bars():
            return None
        close = c["close"]
        e_fast, e_slow = last_ema(close, int(p["fast"])), last_ema(close, int(p["slow"]))
        r_short = pct_return(close, self.days(p["lb_short_days"]))
        r_long = pct_return(close, self.days(p["lb_long_days"]))
        if e_fast > e_slow and r_short > 0 and r_long > 0:
            direction = 1
        elif e_fast < e_slow and r_short < 0 and r_long < 0:
            direction = -1
        else:
            return None
        last = float(close.iloc[-1])
        stop_band = float(p["atr_stop_mult"]) * last_atr(c)
        if direction == 1 and last < e_fast - stop_band:
            return None
        if direction == -1 and last > e_fast + stop_band:
            return None
        vol = last_realised_vol(close, self.days(p["vol_window_days"]), self.bars_per_year)
        return direction, {"r_short": r_short, "r_long": r_long, "vol": vol}

    def compute(self, snap: Snapshot) -> Decision:
        p = self.params
        z = funding_zscores(snap, int(p["funding_lookback_days"]))
        out: Decision = {}
        for sym in snap.symbols:
            found = self._direction(snap, sym)
            if found is None:
                continue
            direction, info = found
            crowding = z.get(sym)
            if crowding is not None:
                if direction > 0 and crowding > float(p["max_long_z"]):
                    continue  # rising, and everyone is paying to be long it
                if direction < 0 and crowding < float(p["min_short_z"]):
                    continue  # falling, and everyone is paid to be short it
            size = min(float(p["max_weight"]), float(p["target_vol"]) / max(info["vol"], VOL_FLOOR))
            out[sym] = Target(
                weight=direction * size,
                conviction=float(np.clip(abs(info["r_short"]) / CONVICTION_SCALE, 0.0, 1.0)),
                reason={**info, "funding_z": crowding},
            )
        return cap_gross(out)

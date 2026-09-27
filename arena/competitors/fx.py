"""Families built for currencies, on daily bars.

Currencies are not crypto with lower volatility. They mean-revert at long
horizons (purchasing-power parity), trend at medium ones, carry funds the
whole market, and their volatility sits around 7-10 % a year against 60-90 %
for a perp -- so a rule sized for crypto is either asleep or levered ten times
on FX. Two families that the literature actually supports on currency pairs:

* :class:`FxTsmom` -- time-series momentum on several horizons at once
  (Moskowitz, Ooi & Pedersen 2012 on FX futures; Menkhoff et al. 2012 on
  spot). The sign of the 1-, 3- and 12-month return vote; a pair is traded
  only when the votes agree, sized to a 10 % volatility target.
* :class:`FxBreakout` -- the Donchian channel (the "turtle" rule): long a
  close above the 55-day high, short a close below the 55-day low, out when
  the close crosses the 20-day channel the other way. It is the same premium
  as momentum bought with a different entry, and it is here as a control on
  the first: two rules for one premium tell the arena whether the premium
  or the rule is doing the work.

Both hold between weekly re-decisions and are sized in fractions of NAV; the
book's fee model prices spot FX at a basis point or two, which is what a
retail broker charges on majors.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from arena.competitors.base import HoldingCompetitor, cap_gross, register
from arena.competitors.features import last_realised_vol, pct_return
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target

VOL_FLOOR = 0.02  # a currency at 2 % annualised is a peg, not a trade


def _vol_size(close, days: int, bars_per_year: int, target_vol: float, max_weight: float) -> tuple[float, float]:
    vol = last_realised_vol(close, days, bars_per_year)
    if not np.isfinite(vol):
        return 0.0, float("nan")
    return min(float(max_weight), float(target_vol) / max(vol, VOL_FLOOR)), float(vol)


@register
class FxTsmom(HoldingCompetitor):
    family = "fx_tsmom"
    markets: ClassVar[frozenset[str]] = frozenset({"fx"})
    rebalance_weekday: ClassVar[int | None] = 0  # Mondays: a weekly cadence on daily bars
    default_params: ClassVar[dict[str, Any]] = {
        "lookbacks_days": [21, 63, 252],
        "min_agreement": 2,  # votes out of len(lookbacks) that must share a sign
        "target_vol": 0.10,
        "vol_window_days": 60,
        "max_weight": 0.34,
    }

    def warmup_bars(self) -> int:
        p = self.params
        return max(self.days(max(p["lookbacks_days"])), self.days(p["vol_window_days"])) + 1

    def compute(self, snap: Snapshot) -> Decision:
        p = self.params
        out: Decision = {}
        for sym in snap.symbols:
            c = snap.candles(sym)
            if len(c) < self.warmup_bars():
                continue
            close = c["close"].dropna()
            votes = [np.sign(pct_return(close, self.days(lb))) for lb in p["lookbacks_days"]]
            votes = [v for v in votes if np.isfinite(v) and v != 0]
            if not votes:
                continue
            score = float(np.mean(votes))
            agree = max(votes.count(1.0), votes.count(-1.0))
            if agree < int(p["min_agreement"]):
                continue
            size, vol = _vol_size(
                close, self.days(p["vol_window_days"]), self.bars_per_year, p["target_vol"], p["max_weight"]
            )
            if size <= 0:
                continue
            out[sym] = Target(
                weight=float(np.sign(score)) * size * abs(score),
                conviction=abs(score),
                reason={"votes": agree, "score": round(score, 3), "vol": round(vol, 4)},
            )
        return cap_gross(out)


@register
class FxBreakout(HoldingCompetitor):
    family = "fx_breakout"
    markets: ClassVar[frozenset[str]] = frozenset({"fx"})
    default_params: ClassVar[dict[str, Any]] = {
        "entry_days": 55,
        "exit_days": 20,
        "target_vol": 0.10,
        "vol_window_days": 60,
        "max_weight": 0.34,
    }

    def warmup_bars(self) -> int:
        p = self.params
        return max(self.days(p["entry_days"]), self.days(p["vol_window_days"])) + 2

    def compute(self, snap: Snapshot) -> Decision:
        p = self.params
        out: Decision = {}
        n_in, n_out = self.days(p["entry_days"]), self.days(p["exit_days"])
        for sym in snap.symbols:
            c = snap.candles(sym)
            if len(c) < self.warmup_bars():
                continue
            close = float(c["close"].iloc[-1])
            hi_in, lo_in = float(c["high"].iloc[-n_in - 1 : -1].max()), float(c["low"].iloc[-n_in - 1 : -1].min())
            hi_out, lo_out = float(c["high"].iloc[-n_out - 1 : -1].max()), float(c["low"].iloc[-n_out - 1 : -1].min())
            held = self._held.get(sym)
            direction = 0
            if close > hi_in:
                direction = 1
            elif close < lo_in:
                direction = -1
            elif held is not None:
                # inside the entry channel: keep the leg until the exit channel is crossed against it
                d = 1 if held.weight > 0 else -1
                direction = d if (d == 1 and close > lo_out) or (d == -1 and close < hi_out) else 0
            if direction == 0:
                continue
            size, vol = _vol_size(
                c["close"], self.days(p["vol_window_days"]), self.bars_per_year, p["target_vol"], p["max_weight"]
            )
            if size <= 0:
                continue
            width = (hi_in - lo_in) / close if close else 0.0
            out[sym] = Target(
                weight=direction * size,
                conviction=float(np.clip(1.0 - width / 0.10, 0.2, 1.0)),
                reason={"channel_high": round(hi_in, 5), "channel_low": round(lo_in, 5), "vol": round(vol, 4)},
            )
        return cap_gross(out)

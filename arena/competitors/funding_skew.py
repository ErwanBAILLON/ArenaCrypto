"""Funding skew: a directional bet against the crowded side of the leverage trade.

Perp funding is the price of leverage. When one symbol's funding sits far above
the rest of the universe, leveraged longs are concentrated there and are paying
every eight hours to stay. That is not a yield opportunity -- ``carry`` already
harvests the level, delta-neutrally -- it is a positioning signal: the crowded
side is the side a liquidation cascade clears first.

So this family shorts the perps whose funding z-score is highest and buys those
whose z-score is lowest (crowded shorts paying longs), unhedged, and holds.

Three choices that make this a testable claim rather than a hunch:

* **Cross-sectional z-score, not the level.** In a bull market every perp pays
  positive funding; ranking on the level would rank beta and call it crowding.
* **Market neutral by construction.** Long and short gross are equalised, so
  what is being tested is the funding *spread*, not a directional view wearing
  a funding costume. A family that can win by being long crypto in a bull run
  has not demonstrated anything the buy-and-hold benchmark does not.
* **A dispersion floor read from the data.** When the cross-section is flat
  there is no crowding to trade and the family stands aside. The project has
  already been bitten by the opposite mistake -- a carry threshold set from
  memory, so the rule never entered -- so ``min_z`` gates on the z-score, which
  is scale-free, rather than on a funding rate in absolute terms.

Known weakness, stated up front: crowding and momentum are correlated. A perp
with extreme positive funding is usually one that has been rising. Shorting it
is partly a short-term reversal bet, and the entry gate cannot tell the two
apart. ``crowded_trend`` is the other half of the experiment -- same signal,
used as a filter instead of a signal -- and the pair is informative whichever
way it goes.
"""

from __future__ import annotations

import numpy as np

from arena.competitors.base import HoldingCompetitor, cap_gross, register
from arena.competitors.features import funding_zscores
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target

CONVICTION_Z = 2.0  # |z| that counts as full conviction


@register
class FundingSkew(HoldingCompetitor):
    family = "funding_skew"
    rebalance_weekday = 0  # Mondays: funding ranks swap daily and each swap pays two legs of fees

    default_params = {
        "lookback_days": 7,
        "k": 3,  # legs per side
        "min_z": 0.75,  # below this the cross-section is too flat to call anyone crowded
        "max_weight": 0.15,
        "market_neutral": True,
    }

    def warmup_bars(self) -> int:
        return self.days(self.params["lookback_days"]) + 1

    def compute(self, snap: Snapshot) -> Decision:
        p = self.params
        z = funding_zscores(snap, int(p["lookback_days"]))
        if not z:
            return {}
        k, min_z = int(p["k"]), float(p["min_z"])
        ranked = sorted(z.items(), key=lambda kv: (-kv[1], kv[0]))
        crowded_longs = [(s, v) for s, v in ranked if v >= min_z][:k]  # they pay -> we short
        crowded_shorts = [(s, v) for s, v in reversed(ranked) if v <= -min_z][:k]  # they receive -> we buy
        if p["market_neutral"] and (not crowded_longs or not crowded_shorts):
            return {}  # one-sided crowding is a directional view, which is not what this family claims
        if not crowded_longs and not crowded_shorts:
            return {}
        out: Decision = {}
        for side, legs in ((-1, crowded_longs), (+1, crowded_shorts)):
            if not legs:
                continue
            weight = min(float(p["max_weight"]), 0.5 / len(legs))
            for rank, (sym, value) in enumerate(legs):
                out[sym] = Target(
                    weight=side * weight,
                    conviction=float(np.clip(abs(value) / CONVICTION_Z, 0.0, 1.0)),
                    reason={"funding_z": value, "rank": rank, "crowd": "long" if side < 0 else "short"},
                )
        return cap_gross(out)

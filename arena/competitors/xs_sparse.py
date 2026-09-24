"""Cross-sectional composite of a handful of ranks, combined without fitting.

This is Nagel's control, made executable. [Nagel
(2025)](https://voices.uchicago.edu/stefannagel/files/2025/07/Complexity_2.pdf)
argues that the out-of-sample gain reported for massively over-parameterised
return models largely reduces to volatility-timed momentum -- three parameters,
not three thousand. If he is right about crypto panels too, this family should
match ``xs_complex`` while consuming almost no trials, and the deflated Sharpe
will then rightly prefer it.

So it is built to be the cheapest defensible version of the same idea: take a
few economically motivated cross-sectional ranks, average them with equal
weight, and trade the extremes dollar-neutral. Nothing is estimated from the
data. There is no training step, no artefact and no search space beyond the
choice of which ranks to use and how many names to hold.

The signals, each oriented so that a higher score means "expected to outperform
the universe":

* ``rank_ret_30d_skip_7d`` -- cross-sectional momentum, skipping the last week
  to sidestep short-horizon reversal.
* ``rank_ret_7d`` **negated** -- one-week reversal, which is strong in
  retail-driven markets.
* ``rank_funding_crowding_z`` **negated** -- the crowding measure ``funding_skew``
  already trades: paying far above the cross-section marks the crowded side.
* ``rank_vol_30d`` **negated** -- low-volatility and lottery-demand effects both
  say the most volatile names underperform per unit of risk.
* ``rank_donchian_position`` -- where the price sits in its own recent range.

Vol scaling happens in the sizing, not the score, which is precisely the
mechanism Nagel says does the work.
"""

from __future__ import annotations

import pandas as pd

from arena.competitors.base import register
from arena.competitors.ladder import LadderHoldingCompetitor
from arena.core.snapshot import Snapshot
from arena.core.types import Decision
from arena.features import build as build_panel

# column -> sign. Negative means "a high rank predicts underperformance".
# Pre-registered on 2026-09-24 after the five-signal rule's per-signal ICs were read:
# low volatility carried the result (t = 5.2 out of sample) and three of the other
# four were negative. Kept as a *named preset*, not as the new default, because
# selecting it on the same held-out year it was read from would be fitting to the
# test set. Its verdict belongs to data that did not exist when this was written.
TWO_SIGNALS: dict[str, float] = {"rank_vol_30d": -1.0, "rank_donchian_position": 1.0}

DEFAULT_SIGNALS: dict[str, float] = {
    "rank_ret_30d_skip_7d": 1.0,
    "rank_ret_7d": -1.0,
    "rank_funding_crowding_z": -1.0,
    "rank_vol_30d": -1.0,
    "rank_donchian_position": 1.0,
}


@register
class XsSparse(LadderHoldingCompetitor):
    family = "xs_sparse"
    rebalance_weekday = 0  # Mondays; the ladder can still close any day

    default_params = {
        "k": 8,  # names per side
        "max_weight": 0.08,
        "target_vol": 0.20,
        "min_symbols": 10,
        "signals": None,  # None -> DEFAULT_SIGNALS
        "stop": 0.08,
        "roi_steps": None,  # None -> the module's default ladder
        "hedge": "picks",  # picks | index | anchor
        "vol_mode": "names",  # names | portfolio
        "max_gross": 1.0,
        "max_leverage": 1.0,
        "rebalance_every_weeks": 1,
    }

    def warmup_bars(self) -> int:
        return 95 * self.bars_per_day  # the longest panel lookback plus margin

    def _signals(self) -> dict[str, float]:
        configured = self.params.get("signals")
        return {str(k): float(v) for k, v in configured.items()} if configured else dict(DEFAULT_SIGNALS)

    def score(self, panel: pd.DataFrame) -> pd.Series:
        """Equal-weight average of the signed, centred ranks. No estimation anywhere."""
        signals = self._signals()
        present = [c for c in signals if c in panel.columns]
        if not present:
            return pd.Series(dtype=float)
        centred = panel[present].sub(0.5)  # percentile ranks: 0.5 is the middle of the cross-section
        signed = centred.mul(pd.Series({c: signals[c] for c in present}), axis=1)
        return signed.mean(axis=1, skipna=True).dropna()

    def select(self, snap: Snapshot) -> Decision:
        panel = build_panel(snap)
        if len(panel) < int(self.params["min_symbols"]):
            return {}
        scores = self.score(panel)
        if scores.empty:
            return {}
        return self.size_book(snap, panel, scores, lambda sym: {"score": round(float(scores[sym]), 4)})

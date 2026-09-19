"""Null models and benchmarks, permanently in the arena.

They define the floor: a strategy that cannot beat cash, a seeded coin flip,
plain BTC or equal-weight carry has no business being called a signal.
"""

from __future__ import annotations

import numpy as np

from arena.competitors.base import Competitor, cap_gross, register
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target


@register
class NullCash(Competitor):
    family = "null_cash"

    def decide(self, snap: Snapshot) -> Decision:
        return {}


@register
class NullRandom(Competitor):
    """Coin-flip positions, deterministic per (seed, bar) so runs are replayable."""

    family = "null_random"
    default_params = {"scale": 0.3, "p_trade": 0.3}

    def decide(self, snap: Snapshot) -> Decision:
        rng = np.random.default_rng(self.seed + int(snap.ts.timestamp()) // 3600)
        out: Decision = {}
        for sym in snap.symbols:
            if not snap.has(sym, self.warmup_bars()):
                continue
            trade = rng.uniform() < self.params["p_trade"]
            w = rng.uniform(-1.0, 1.0) * self.params["scale"]
            if trade:
                out[sym] = Target(weight=float(w), conviction=0.5, reason={"null": "random"})
        return cap_gross(out)


@register
class BenchBtcHold(Competitor):
    family = "bench_btc_hold"

    def decide(self, snap: Snapshot) -> Decision:
        return {"BTC": Target(weight=1.0, conviction=1.0, reason={"bench": "btc_hold"})}


@register
class BenchCarryEqual(Competitor):
    """Equal-weight carry on every symbol that has funding data."""

    family = "bench_carry_equal"

    def decide(self, snap: Snapshot) -> Decision:
        syms = [s for s in snap.symbols if not snap.funding(s).empty]
        if not syms:
            return {}
        w = 1.0 / len(syms)
        return {s: Target(weight=w, conviction=0.5, kind="carry", reason={"bench": "carry_equal"}) for s in syms}

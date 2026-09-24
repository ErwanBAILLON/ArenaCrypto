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
    """Coin-flip positions redrawn once per ``hold_hours`` (default a week).

    Deterministic per (seed, week bucket) so runs are replayable. Weekly
    holding keeps turnover realistic: an hourly coin flip pays ~700 % of NAV a
    year in fees and would make any competitor look good against it.
    """

    family = "null_random"
    default_params = {"scale": 0.3, "p_trade": 0.3, "hold_hours": 168}

    def decide(self, snap: Snapshot) -> Decision:
        bucket = int(snap.ts.timestamp()) // (3600 * int(self.params["hold_hours"]))
        rng = np.random.default_rng(self.seed * 1_000_003 + bucket)
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
class BenchHold(Competitor):
    """Buy the universe's reference asset (param ``symbol``) and never move: the classic-markets benchmark."""

    family = "bench_hold"
    default_params = {"symbol": None}

    def decide(self, snap: Snapshot) -> Decision:
        sym = self.params.get("symbol") or snap.reference
        return {sym: Target(weight=1.0, conviction=1.0, reason={"bench": "hold"})}


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


@register
class NullNeutral(Competitor):
    """A random *dollar-neutral* book at the cross-sectional families' own cadence and size.

    ``null_random`` is not neutral and redraws its whole book weekly; with a
    realistic impact model it lost 86-96 % in a year, almost all of it fees.
    Beating it then proves only that a competitor trades less than a coin flip.
    This null holds ``k`` random longs against ``k`` random shorts, re-drawn
    every ``hold_weeks``, at the same per-name cap the families use -- so the
    comparison isolates *selection*, which is the only thing a cross-sectional
    rule claims.
    """

    family = "null_neutral"
    default_params = {"k": 8, "max_weight": 0.08, "hold_weeks": 1, "gross": 0.2}

    def decide(self, snap: Snapshot) -> Decision:
        bucket = int(snap.ts.timestamp()) // (3600 * 24 * 7 * int(self.params["hold_weeks"]))
        rng = np.random.default_rng(self.seed * 7_919 + bucket)
        names = [s for s in snap.symbols if snap.has(s, self.warmup_bars())]
        k = min(int(self.params["k"]), len(names) // 2)
        if k < 1:
            return {}
        picks = list(rng.permutation(names))
        weight = min(float(self.params["max_weight"]), float(self.params["gross"]) / (2 * k))
        out: Decision = {}
        for sym in picks[:k]:
            out[sym] = Target(weight=weight, conviction=0.5, reason={"null": "neutral"})
        for sym in picks[k : 2 * k]:
            out[sym] = Target(weight=-weight, conviction=0.5, reason={"null": "neutral"})
        return cap_gross(out)

"""Competitor contract and family registry.

A competitor is a pure function of ``(params, Snapshot) -> Decision``: same
input, same output, no database, no clock. That purity is what lets the judge
count trials and replay any decision from stored data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, ClassVar

from arena.core.snapshot import Snapshot
from arena.core.types import CompetitorSpec, Decision


class Competitor(ABC):
    family: ClassVar[str]
    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0):
        self.params: dict[str, Any] = {**self.default_params, **(params or {})}
        self.seed = seed

    def warmup_bars(self) -> int:
        """Number of 1h bars required before the first decision."""
        return 1

    @abstractmethod
    def decide(self, snap: Snapshot) -> Decision: ...

    def state(self) -> dict[str, Any]:
        """Serialisable state to carry across processes (hysteresis only). Default: none."""
        return {}

    def restore_state(self, state: dict[str, Any]) -> None:
        """Inverse of :meth:`state`. Default: nothing to restore."""


REGISTRY: dict[str, type[Competitor]] = {}


def register(cls: type[Competitor]) -> type[Competitor]:
    REGISTRY[cls.family] = cls
    return cls


def build(spec: CompetitorSpec) -> Competitor:
    cls = REGISTRY[spec.family]
    return cls(spec.params, seed=spec.params.get("seed", 0))


def cap_gross(decision: Decision) -> Decision:
    """Scale weights so that sum(|w|) <= 1, preserving every other Target field."""
    gross = sum(abs(t.weight) for t in decision.values())
    if gross <= 1.0:
        return decision
    return {sym: replace(t, weight=t.weight / gross) for sym, t in decision.items()}


class HoldingCompetitor(Competitor):
    """Competitor that re-decides once a day and holds in between.

    Hourly re-decision is what turned sound rules into fee machines (trend_ts
    paid 52 % of NAV per year in fees on real data). Subclasses implement
    :meth:`compute`; ``decide`` calls it only at ``rebalance_hour`` UTC (or on
    the first call) and otherwise returns the held decision. Small weight
    changes below ``band`` are ignored to avoid dust trades. The held decision
    is part of :meth:`state` so a fresh process replays identically.
    """

    rebalance_hour: ClassVar[int] = 0
    band: ClassVar[float] = 0.10

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0):
        super().__init__(params, seed)
        self._held: Decision = {}
        self._held_ts: str | None = None

    def compute(self, snap: Snapshot) -> Decision:  # pragma: no cover - abstract by convention
        raise NotImplementedError

    def decide(self, snap: Snapshot) -> Decision:
        if self._held_ts is not None and snap.ts.hour != self.rebalance_hour:
            return dict(self._held)
        if self._held_ts is not None and self._held_ts == snap.ts.isoformat():
            return dict(self._held)
        fresh = self.compute(snap)
        merged: Decision = {}
        for sym in set(fresh) | set(self._held):
            new, old = fresh.get(sym), self._held.get(sym)
            if new is None:
                continue  # exit: symbol dropped
            if old is not None and old.kind == new.kind and abs(new.weight - old.weight) < self.band:
                merged[sym] = replace(old, conviction=new.conviction, reason=new.reason)
            else:
                merged[sym] = new
        self._held, self._held_ts = merged, snap.ts.isoformat()
        return dict(merged)

    def state(self) -> dict[str, Any]:
        return {
            "held_ts": self._held_ts,
            "held": {s: {"weight": t.weight, "conviction": t.conviction, "kind": t.kind, "reason": t.reason}
                     for s, t in self._held.items()},
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        from arena.core.types import Target

        self._held_ts = state.get("held_ts")
        self._held = {s: Target(weight=float(v["weight"]), conviction=float(v.get("conviction", 0.5)),
                                kind=v.get("kind", "perp"), reason=dict(v.get("reason", {})))
                      for s, v in (state.get("held") or {}).items()}

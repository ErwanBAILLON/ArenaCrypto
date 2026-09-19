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

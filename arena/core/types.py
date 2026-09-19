"""Shared value types. Every module imports these; none redefines them."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

Kind = Literal["perp", "carry"]
Role = Literal["null", "benchmark", "competitor"]
Status = Literal["candidate", "challenger", "champion", "retired"]


@dataclass(frozen=True)
class Target:
    """Desired exposure for one symbol.

    weight: fraction of NAV in [-1, 1]. Negative = short perp.
            For kind="carry", weight > 0 is the size of the delta-neutral
            long-spot / short-perp position that earns funding.
    conviction: [0, 1], used only for reporting and allocator tie-breaks.
    """

    weight: float
    conviction: float = 0.5
    kind: Kind = "perp"
    reason: dict[str, Any] = field(default_factory=dict)


Decision = dict[str, Target]  # symbol -> Target ; missing symbol == flat


@dataclass(frozen=True)
class CompetitorSpec:
    id: int | None
    name: str
    family: str
    version: int
    params: dict[str, Any]
    role: Role = "competitor"
    status: Status = "candidate"
    parent_id: int | None = None
    rationale: str = ""


@dataclass
class BookRow:
    ts: datetime
    nav: float
    ret: float
    gross: float
    turnover: float
    fees: float
    funding_pnl: float


@dataclass
class Verdict:
    admitted: bool
    metrics: dict[str, float]
    failed: list[str]


@dataclass(frozen=True)
class Article:
    source: str
    url: str
    title: str
    summary: str
    published_at: datetime
    fetched_at: datetime
    id: int | None = None


@dataclass(frozen=True)
class ArticleScore:
    article_id: int
    asset: str  # universe symbol or "MARKET"
    sentiment: float  # [-1, 1]
    event_type: str
    intensity: float  # [0, 1]
    scorer_version: int


@dataclass(frozen=True)
class Alert:
    kind: str
    payload: dict[str, Any]
    competitor_id: int | None = None
    symbol: str | None = None

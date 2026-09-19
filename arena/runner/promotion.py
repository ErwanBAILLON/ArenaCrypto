"""Champion / challenger promotion rules (design §10).

A challenger replaces the champion of its family when, over their common
window, it has run long enough, its Sharpe beats the champion's and it sits
above the null models' 95th percentile. Nothing is deleted: the old champion
is retired and keeps its books.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from arena.core.types import Alert, CompetitorSpec
from arena.judge.metrics import sharpe

MIN_DAYS = 42
MIN_DECISIONS = 100
MAX_CHALLENGERS_PER_FAMILY = 3
NULL_Q = 0.95


@dataclass(frozen=True)
class Candidate:
    spec: CompetitorSpec
    first_ts: datetime
    decisions: int
    returns: pd.Series  # since first_ts


def null_threshold(null_returns: pd.DataFrame, since: datetime) -> float:
    """95th percentile of null competitors' Sharpe over ``[since, now]``."""
    window = null_returns[null_returns.index >= pd.Timestamp(since)]
    vals = [sharpe(window[c].dropna()) for c in window.columns if window[c].notna().sum() > 24]
    return float(np.quantile(vals, NULL_Q)) if vals else 0.0


def ready(c: Candidate, now: datetime) -> bool:
    age = pd.Timestamp(now) - pd.Timestamp(c.first_ts)
    return age >= timedelta(days=MIN_DAYS) or c.decisions >= MIN_DECISIONS


def should_promote(challenger: Candidate, champion: Candidate | None, null_returns: pd.DataFrame, now: datetime) -> tuple[bool, dict]:
    """Return (promote?, evidence). ``champion`` is None when the family has no champion yet."""
    if not ready(challenger, now):
        return False, {"reason": "not_ready"}
    since = challenger.first_ts
    ch_sr = sharpe(challenger.returns[challenger.returns.index >= pd.Timestamp(since)].dropna())
    thr = null_threshold(null_returns, since)
    evidence = {"challenger_sharpe": ch_sr, "null95": thr}
    if ch_sr <= thr:
        return False, {**evidence, "reason": "below_null"}
    if champion is not None:
        champ_sr = sharpe(champion.returns[champion.returns.index >= pd.Timestamp(since)].dropna())
        evidence["champion_sharpe"] = champ_sr
        if ch_sr <= champ_sr:
            return False, {**evidence, "reason": "below_champion"}
    return True, evidence


def promotion_alert(challenger: CompetitorSpec, champion: CompetitorSpec | None, evidence: dict) -> Alert:
    old = champion.name if champion else "none"
    return Alert(kind="promotion", competitor_id=challenger.id,
                 payload={"detail": f"{challenger.name} promoted to champion of {challenger.family} (was {old})", **evidence})


def surplus_challengers(challengers: list[CompetitorSpec]) -> list[CompetitorSpec]:
    """Oldest challengers beyond the per-family budget (to be retired)."""
    by_family: dict[str, list[CompetitorSpec]] = {}
    for c in challengers:
        by_family.setdefault(c.family, []).append(c)
    out: list[CompetitorSpec] = []
    for fam, lst in by_family.items():
        lst = sorted(lst, key=lambda s: s.id or 0)
        out.extend(lst[: max(0, len(lst) - MAX_CHALLENGERS_PER_FAMILY)])
    return out

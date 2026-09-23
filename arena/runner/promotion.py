"""Champion / challenger promotion rules (design §10).

A challenger replaces the champion of its family only when the evidence says
so, not when a point estimate happens to be higher. Four barriers, in order:

1. **Gate.** A competitor the entry gate refused is never promoted. It keeps
   its forward book — watching a refused rule lose is the point of the arena —
   but it cannot become the champion of a family just because that family has
   no champion yet. Families that cannot be backtested at all (``news``, whose
   scores only exist forward) are exempt by name, not by accident.
2. **Length.** ``MIN_DAYS`` or ``MIN_DECISIONS`` in the arena.
3. **Luck.** ``P(true Sharpe > null 95th percentile) >= PROMOTION_CONFIDENCE``,
   computed on the autocorrelation-adjusted sample size. The threshold itself
   needs ``MIN_NULL_SAMPLES`` null series covering the window, otherwise the
   arena says so instead of promoting against a quantile of five numbers.
4. **The incumbent.** ``P(true Sharpe of the difference > 0) >=
   PROMOTION_CONFIDENCE`` on the paired series ``challenger - champion``,
   which is the test that survives the two books sharing market moves.

Nothing is deleted: the old champion is retired and keeps its books.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from arena.core.types import Alert, CompetitorSpec
from arena.judge.metrics import PPY_HOURLY, probabilistic_sharpe, sharpe, track_record_verdict

MIN_DAYS = 42
MIN_DECISIONS = 100
MAX_CHALLENGERS_PER_FAMILY = 3
NULL_Q = 0.95
MIN_NULL_SAMPLES = 20
MIN_NULL_COVERAGE = 0.8
PROMOTION_CONFIDENCE = 0.95
FORWARD_ONLY_FAMILIES = frozenset({"news"})


@dataclass(frozen=True)
class Candidate:
    spec: CompetitorSpec
    first_ts: datetime
    decisions: int
    returns: pd.Series  # since first_ts


def gate_cleared(spec: CompetitorSpec) -> bool:
    """True when this competitor is allowed to become a champion at all."""
    return bool(spec.gate_admitted) or spec.family in FORWARD_ONLY_FAMILIES


def null_sharpes(null_returns: pd.DataFrame, since: datetime, ppy: int = PPY_HOURLY) -> list[float]:
    """Annualised Sharpe of every null series that actually covers ``[since, now]``.

    A null model registered last week cannot speak about a challenger that has
    been running for six: only columns present for at least ``MIN_NULL_COVERAGE``
    of the window are counted, so the threshold is never a mix of horizons.
    """
    if null_returns is None or null_returns.empty:
        return []
    window = null_returns[null_returns.index >= pd.Timestamp(since)]
    if window.empty:
        return []
    need = max(24, int(MIN_NULL_COVERAGE * len(window)))
    return [sharpe(window[c].dropna(), ppy) for c in window.columns if window[c].notna().sum() >= need]


def null_threshold(null_returns: pd.DataFrame, since: datetime, ppy: int = PPY_HOURLY) -> float:
    """``NULL_Q`` quantile of the null models' Sharpe over ``[since, now]`` (0 when none)."""
    vals = null_sharpes(null_returns, since, ppy)
    return float(np.quantile(vals, NULL_Q)) if vals else 0.0


def ready(c: Candidate, now: datetime) -> bool:
    age = pd.Timestamp(now) - pd.Timestamp(c.first_ts)
    return age >= timedelta(days=MIN_DAYS) or c.decisions >= MIN_DECISIONS


def _paired(challenger: pd.Series, champion: pd.Series) -> pd.Series:
    """``challenger - champion`` on the bars both booked, which is the only fair comparison."""
    joined = pd.concat([challenger.rename("a"), champion.rename("b")], axis=1, join="inner").dropna()
    return joined["a"] - joined["b"] if not joined.empty else pd.Series(dtype=float)


def should_promote(
    challenger: Candidate,
    champion: Candidate | None,
    null_returns: pd.DataFrame,
    now: datetime,
    ppy: int = PPY_HOURLY,
) -> tuple[bool, dict]:
    """Return (promote?, evidence). ``champion`` is None when the family has no champion yet."""
    if not gate_cleared(challenger.spec):
        return False, {"reason": "never_admitted"}
    if not ready(challenger, now):
        return False, {"reason": "not_ready"}
    since = challenger.first_ts
    ch = challenger.returns[challenger.returns.index >= pd.Timestamp(since)].dropna()
    ch_sr = sharpe(ch, ppy)
    nulls = null_sharpes(null_returns, since, ppy)
    thr = float(np.quantile(nulls, NULL_Q)) if nulls else 0.0
    record = track_record_verdict(ch, thr, PROMOTION_CONFIDENCE, ppy)
    evidence = {
        "challenger_sharpe": ch_sr,
        "null95": thr,
        "null_samples": len(nulls),
        "psr_vs_null": record["psr"],
        "effective_bars": record["effective"],
        "bars_missing": record["missing"],
    }
    if len(nulls) < MIN_NULL_SAMPLES:
        return False, {**evidence, "reason": "null_underpowered"}
    if record["psr"] < PROMOTION_CONFIDENCE:
        return False, {**evidence, "reason": "below_null"}
    if champion is not None:
        champ_sr = sharpe(champion.returns[champion.returns.index >= pd.Timestamp(since)].dropna(), ppy)
        diff = _paired(ch, champion.returns)
        psr_vs_champ = probabilistic_sharpe(diff, 0.0, ppy) if len(diff) >= 3 else 0.5
        evidence["champion_sharpe"] = champ_sr
        evidence["psr_vs_champion"] = psr_vs_champ
        evidence["paired_bars"] = int(len(diff))
        if psr_vs_champ < PROMOTION_CONFIDENCE:
            return False, {**evidence, "reason": "below_champion"}
    return True, evidence


def promotion_alert(challenger: CompetitorSpec, champion: CompetitorSpec | None, evidence: dict) -> Alert:
    old = champion.name if champion else "none"
    return Alert(
        kind="promotion",
        competitor_id=challenger.id,
        payload={"detail": f"{challenger.name} promoted to champion of {challenger.family} (was {old})", **evidence},
    )


def surplus_challengers(challengers: list[CompetitorSpec]) -> list[CompetitorSpec]:
    """Oldest challengers beyond the per-family budget (to be retired)."""
    by_family: dict[str, list[CompetitorSpec]] = {}
    for c in challengers:
        by_family.setdefault(c.family, []).append(c)
    out: list[CompetitorSpec] = []
    for lst in by_family.values():
        lst = sorted(lst, key=lambda s: s.id or 0)
        out.extend(lst[: max(0, len(lst) - MAX_CHALLENGERS_PER_FAMILY)])
    return out

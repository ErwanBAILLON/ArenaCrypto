"""Arena-wide multiple testing: what the per-family gate cannot see.

The deflated Sharpe deflates a candidate by the number of trials **its own
family** has recorded. That is the right correction for "did I overfit this
rule?" and the wrong one for the question that actually killed the six
freqtrade bots: *103 strategies were selected on the same in-sample data, and
the best one looked wonderful*. Run enough families, search each of them every
Sunday, and something clears a 10 % gate by construction.

This module asks the arena-level question. Over every gate decision ever
recorded, in one universe:

* how many tests were run, and how many false discoveries the nominal level
  therefore *predicts* even if no family has any edge;
* which admissions survive a Benjamini-Hochberg correction at ``q``, which
  controls the expected share of false positives among the admitted rather
  than the error rate of each test taken alone;
* what each admitted candidate's deflated Sharpe becomes when it pays for the
  whole arena's search instead of only its family's.

It changes no verdict. It is a number to put next to the leaderboard, so that
"three families admitted something this quarter" can be read against "we ran
four hundred tests to get there".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psycopg

from arena.judge.gate import DEFAULT_GATE, GateConfig
from arena.judge.metrics import deflated_sharpe

GATE_KINDS = ("walkforward",)


@dataclass(frozen=True)
class AuditRow:
    trial_id: int
    family: str
    competitor: str | None
    verdict: str | None
    sharpe: float | None
    bootstrap_p: float | None
    dsr_family: float | None
    dsr_arena: float | None
    survives_bh: bool


@dataclass(frozen=True)
class Audit:
    universe: str
    tests: int
    admitted: int
    search_evaluations: int
    expected_false_positives: float
    bh_q: float
    bh_threshold: float | None
    survivors: int
    rows: list[AuditRow] = field(default_factory=list)

    @property
    def headline(self) -> str:
        return (
            f"{self.universe}: {self.tests} gate decisions ({self.search_evaluations} search evaluations behind "
            f"them), {self.admitted} admitted, {self.survivors} still admitted after Benjamini-Hochberg at "
            f"q={self.bh_q:.2f}; chance alone predicts about {self.expected_false_positives:.1f} admissions."
        )


def benjamini_hochberg(p_values: list[float], q: float = 0.10) -> tuple[float | None, list[bool]]:
    """``(threshold, rejected)`` under Benjamini-Hochberg at false discovery rate ``q``.

    The largest ``k`` with ``p_(k) <= k/m * q`` sets the threshold; every
    p-value at or below it is rejected. ``threshold`` is None when nothing
    survives, in which case no discovery in the batch is defensible.
    """
    m = len(p_values)
    if m == 0:
        return None, []
    order = np.argsort(np.asarray(p_values, dtype=float))
    ranked = np.asarray(p_values, dtype=float)[order]
    limits = (np.arange(1, m + 1) / m) * q
    passing = np.nonzero(ranked <= limits)[0]
    if passing.size == 0:
        return None, [False] * m
    threshold = float(ranked[passing.max()])
    return threshold, [float(p) <= threshold for p in p_values]


def run(conn: psycopg.Connection, universe: str = "crypto", q: float = 0.10, cfg: GateConfig = DEFAULT_GATE) -> Audit:
    """Audit every gate decision recorded for ``universe``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT t.id, t.family, t.verdict, t.metrics, c.name AS competitor"
            " FROM trials t LEFT JOIN competitors c ON c.id = t.competitor_id"
            " WHERE t.universe = %s AND t.kind = ANY(%s) AND t.verdict IN ('admitted', 'rejected')"
            " ORDER BY t.id",
            (universe, list(GATE_KINDS)),
        )
        trials = cur.fetchall()
        cur.execute("SELECT count(*) AS n FROM trials WHERE universe = %s", (universe,))
        total_trials = int(cur.fetchone()["n"])
        cur.execute(
            "SELECT count(*) AS n FROM trials WHERE universe = %s AND kind = 'optimize'",
            (universe,),
        )
        searches = int(cur.fetchone()["n"])

    p_values = [_p_of(t["metrics"]) for t in trials]
    threshold, rejected = benjamini_hochberg(p_values, q)
    rows: list[AuditRow] = []
    for trial, survives in zip(trials, rejected or [False] * len(trials), strict=False):
        m = dict(trial["metrics"] or {})
        rows.append(
            AuditRow(
                trial_id=int(trial["id"]),
                family=trial["family"],
                competitor=trial["competitor"],
                verdict=trial["verdict"],
                sharpe=_f(m.get("sharpe")),
                bootstrap_p=_f(m.get("bootstrap_p")),
                dsr_family=_f(m.get("dsr")),
                dsr_arena=_arena_dsr(m, total_trials),
                survives_bh=bool(survives) and trial["verdict"] == "admitted",
            )
        )
    admitted = sum(1 for t in trials if t["verdict"] == "admitted")
    return Audit(
        universe=universe,
        tests=len(trials),
        admitted=admitted,
        search_evaluations=searches,
        expected_false_positives=len(trials) * cfg.max_bootstrap_p,
        bh_q=q,
        bh_threshold=threshold,
        survivors=sum(1 for r in rows if r.survives_bh),
        rows=rows,
    )


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _p_of(metrics: dict[str, Any] | None) -> float:
    """The gate's own p-value, or 1.0 when the trial predates it being recorded."""
    p = _f((metrics or {}).get("bootstrap_p"))
    return 1.0 if p is None else min(max(p, 0.0), 1.0)


def _arena_dsr(metrics: dict[str, Any], n_trials: int) -> float | None:
    """The deflated Sharpe recomputed against every trial in the arena, not just the family's."""
    sr = _f(metrics.get("sr_period"))
    obs = _f(metrics.get("observations"))
    if sr is None or obs is None or obs < 2:
        return None
    return deflated_sharpe(
        sr, max(n_trials, 1), int(obs), _f(metrics.get("skew")) or 0.0, _f(metrics.get("kurtosis")) or 3.0
    )

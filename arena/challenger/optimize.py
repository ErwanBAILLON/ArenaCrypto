"""Weekly parameter search for rule families → at most one new challenger per family.

Objective = mean out-of-fold Sharpe of an anchored walk-forward. Every
evaluation is written to ``trials`` before the gate sees the best candidate, so
the deflated Sharpe pays for the whole search, not just for the winner.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import optuna
import psycopg

from arena.book.book import FeeModel
from arena.competitors.base import REGISTRY
from arena.core.types import Alert, CompetitorSpec
from arena.judge.admission import admit
from arena.judge.backtest import HistoryFrames
from arena.judge.metrics import sharpe
from arena.judge.walkforward import run_walkforward
from arena.store import books as bstore
from arena.store import registry

from .spaces import SPACES

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective_factory(conn, family: str, history: HistoryFrames, symbols: list[str], start, end, fees: FeeModel):
    space = SPACES[family]

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        folds = run_walkforward(lambda: REGISTRY[family](params), history, symbols, start, end, fees)
        srs = [sharpe(res.returns) for _, res in folds]
        score = float(np.mean(srs)) if srs else -10.0
        registry.add_trial(conn, family, "optimize", params, {"fold_sharpes": srs, "mean_sharpe": score}, None)
        conn.commit()
        return score

    return objective


def next_version(conn, family: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT coalesce(max(version), 0) AS v FROM competitors WHERE family = %s", (family,))
        return int(cur.fetchone()["v"]) + 1


def run(conn: psycopg.Connection, family: str, history: HistoryFrames, symbols: list[str], start: datetime, end: datetime,
        fees: FeeModel, null_thr: float, n_trials: int = 40, seed: int = 0) -> CompetitorSpec | None:
    """Search, gate the best candidate, insert it as challenger. Returns the spec or None."""
    if family not in SPACES:
        raise ValueError(f"no search space for {family}")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective_factory(conn, family, history, symbols, start, end, fees), n_trials=n_trials)
    best = dict(study.best_params)
    champions = registry.list_competitors(conn, statuses=["champion"])
    champion = next((c for c in champions if c.family == family), None)
    if champion is not None and champion.params == best:
        return None
    adm = admit(conn, family, best, history, symbols, start, end, fees, null_thr, notes="optimize best")
    if not adm.verdict.admitted:
        bstore.add_alert(conn, Alert(kind="rejected", payload={"detail": f"{family} optimize best rejected: {', '.join(adm.verdict.failed)}"}))
        conn.commit()
        return None
    version = next_version(conn, family)
    spec = CompetitorSpec(None, f"{family}_v{version}", family, version, best, status="challenger",
                          parent_id=champion.id if champion else None,
                          rationale=f"optuna best of {n_trials} (mean OOF Sharpe {study.best_value:.2f}); trial {adm.trial_id}")
    cid = registry.insert_competitor(conn, spec)
    conn.commit()
    return CompetitorSpec(**{**spec.__dict__, "id": cid})

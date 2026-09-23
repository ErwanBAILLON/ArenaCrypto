"""Weekly parameter search for rule families → at most one new challenger per family.

Objective = mean out-of-fold Sharpe of an anchored walk-forward. Every
evaluation is written to ``trials`` before the gate sees the best candidate, so
the deflated Sharpe pays for the whole search, not just for the winner, and the
per-bar returns of every evaluation are kept in memory so the search can also
be scored as a whole: ``metrics.pbo_cscv`` answers "does picking the best of
these generalise?", which the deflated Sharpe does not ask.

Everything here is parameterised by the arena it runs in. A search on daily
bars that builds hourly competitors and annualises by 8760 produces confident
nonsense; ``bar_hours`` and ``universe`` are threaded through the objective,
the gate, the trial counter and the inserted competitor.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg

from arena.book.book import FeeModel
from arena.competitors.base import REGISTRY
from arena.core.types import Alert, CompetitorSpec
from arena.judge.admission import admit
from arena.judge.backtest import HistoryFrames
from arena.judge.metrics import pbo_cscv, sharpe
from arena.judge.walkforward import run_walkforward
from arena.store import books as bstore
from arena.store import registry

from .spaces import SPACES

log = logging.getLogger(__name__)
MIN_CONFIGS_FOR_PBO = 8


def objective_factory(
    conn,
    family: str,
    history: HistoryFrames,
    symbols: list[str],
    start,
    end,
    fees: FeeModel,
    bar_hours: int = 1,
    universe: str = "crypto",
    collected: list[pd.Series] | None = None,
):
    """Optuna objective: mean out-of-fold Sharpe, one recorded trial per evaluation.

    ``collected`` receives the concatenated out-of-fold return series of each
    evaluation, in trial order, for the PBO computation at the end of the run.
    """
    space = SPACES[family]
    ppy = 8760 // bar_hours

    def objective(trial) -> float:
        params = space(trial)
        folds = run_walkforward(
            lambda: REGISTRY[family](params, bar_hours=bar_hours),
            history,
            symbols,
            start,
            end,
            fees,
            bar_hours=bar_hours,
        )
        srs = [sharpe(res.returns, ppy) for _, res in folds]
        score = float(np.mean(srs)) if srs else -10.0
        if collected is not None:
            series = [res.returns for _, res in folds if len(res.returns)]
            collected.append(pd.concat(series).sort_index() if series else pd.Series(dtype=float))
        registry.add_trial(
            conn,
            family,
            "optimize",
            params,
            {"fold_sharpes": srs, "mean_sharpe": score},
            "scored",
            notes=f"optuna trial {trial.number}",
            universe=universe,
            finished=True,
        )
        conn.commit()
        return score

    return objective


def next_version(conn, family: str, universe: str = "crypto") -> int:
    """One version counter per family **and** arena; the two never share a name."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT coalesce(max(version), 0) AS v FROM competitors WHERE family = %s AND universe = %s",
            (family, universe),
        )
        return int(cur.fetchone()["v"]) + 1


def search_pbo(collected: list[pd.Series]) -> dict:
    """PBO over the evaluations of one search, aligned on their common bars.

    Walk-forward folds share their test windows across evaluations, so the
    series align; an evaluation that produced nothing is dropped. Fewer than
    ``MIN_CONFIGS_FOR_PBO`` usable configurations means no opinion (``pbo`` 1.0,
    which fails the criterion — a search too small to be validated is a search
    whose winner has not been validated).
    """
    usable = [s for s in collected if len(s) > 0]
    if len(usable) < MIN_CONFIGS_FOR_PBO:
        return {"pbo": 1.0, "n_splits": 0, "n_configs": len(usable), "logits": []}
    matrix = pd.concat(usable, axis=1, join="inner").dropna()
    if matrix.empty or matrix.shape[1] < MIN_CONFIGS_FOR_PBO:
        return {"pbo": 1.0, "n_splits": 0, "n_configs": int(matrix.shape[1]), "logits": []}
    out = pbo_cscv(matrix.to_numpy())
    out.pop("logits", None)  # the distribution is large and the summary is what the gate reads
    return out


def run(
    conn: psycopg.Connection,
    family: str,
    history: HistoryFrames,
    symbols: list[str],
    start: datetime,
    end: datetime,
    fees: FeeModel,
    null_thr: float,
    n_trials: int = 40,
    seed: int = 0,
    bar_hours: int = 1,
    universe: str = "crypto",
) -> CompetitorSpec | None:
    """Search, score the search, gate the best candidate, insert it as challenger."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if family not in SPACES:
        raise ValueError(f"no search space for {family}")
    registry.abandon_stale_trials(conn)
    conn.commit()
    collected: list[pd.Series] = []
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        objective_factory(conn, family, history, symbols, start, end, fees, bar_hours, universe, collected),
        n_trials=n_trials,
    )
    best = dict(study.best_params)
    champions = registry.list_competitors(conn, statuses=["champion"], universe=universe)
    champion = next((c for c in champions if c.family == family), None)
    if champion is not None and champion.params == best:
        return None
    pbo = search_pbo(collected)
    adm = admit(
        conn,
        family,
        best,
        history,
        symbols,
        start,
        end,
        fees,
        null_thr,
        notes=f"optimize best ({universe})",
        bar_hours=bar_hours,
        universe=universe,
        pbo=pbo["pbo"],
    )
    if not adm.verdict.admitted:
        bstore.add_alert(
            conn,
            Alert(
                kind="rejected",
                payload={
                    "detail": (
                        f"{family} optimize best rejected: {', '.join(adm.verdict.failed)}"
                        f" (pbo {pbo['pbo']:.2f} over {pbo['n_configs']} configs)"
                    )
                },
            ),
        )
        conn.commit()
        return None
    version = next_version(conn, family, universe)
    base = f"{family}_v{version}"
    spec = CompetitorSpec(
        None,
        base if universe == "crypto" else f"{base}_{universe}",
        family,
        version,
        best,
        status="challenger",
        parent_id=champion.id if champion else None,
        universe=universe,
        gate_admitted=True,
        rationale=(
            f"optuna best of {n_trials} (mean OOF Sharpe {study.best_value:.2f}, "
            f"pbo {pbo['pbo']:.2f}); trial {adm.trial_id}"
        ),
    )
    cid = registry.insert_competitor(conn, spec)
    conn.commit()
    return CompetitorSpec(**{**spec.__dict__, "id": cid})

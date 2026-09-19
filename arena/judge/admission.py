"""Entry gate as a single call: walk-forward + null distribution + trial record.

Used by ``bootstrap`` (founding families) and by the challenger loop (every
optimisation candidate). Every call adds a row to ``trials`` so the deflated
Sharpe of the next candidate accounts for it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from arena.book.book import FeeModel
from arena.competitors.base import REGISTRY
from arena.core.types import Verdict
from arena.judge.backtest import HistoryFrames
from arena.judge.gate import DEFAULT_GATE, GateConfig, evaluate, null_sharpe_threshold, run_null_distribution
from arena.judge.walkforward import run_walkforward
from arena.store import registry

N_NULL = 50


@dataclass
class Admission:
    verdict: Verdict
    trial_id: int
    fold_sharpes: list[float]


def null_threshold(
    history: HistoryFrames, symbols: list[str], start: datetime, end: datetime, fees: FeeModel, n: int = N_NULL
) -> float:
    """95th percentile Sharpe of seeded null_random runs over ``[start, end]``."""
    return null_sharpe_threshold(null_sharpes(history, symbols, start, end, fees, n))


def null_sharpes(
    history: HistoryFrames, symbols: list[str], start: datetime, end: datetime, fees: FeeModel, n: int = N_NULL
) -> list[float]:
    from arena.judge.metrics import sharpe

    make_null = lambda seed: REGISTRY["null_random"]({"seed": seed}, seed=seed)  # noqa: E731
    return [sharpe(r.returns) for r in run_null_distribution(make_null, history, symbols, start, end, fees, n=n)]


def cached_null_threshold(
    conn: psycopg.Connection,
    history: HistoryFrames,
    symbols: list[str],
    start: datetime,
    end: datetime,
    fees: FeeModel,
    n: int = N_NULL,
    q: float = 0.95,
) -> float:
    """Null threshold reused within the same ISO week (the distribution barely moves day to day).

    The per-seed Sharpes are stored as a ``trials`` row (family ``null_random``,
    kind ``backtest``) so the weekly challenger and any ``arena judge`` call
    share one computation instead of 50 full backtests each.
    """
    import numpy as np

    week = pd.Timestamp(end).strftime("%G-W%V")
    key = {"start": pd.Timestamp(start).isoformat(), "week": week, "n": n, "symbols": sorted(symbols)}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT metrics FROM trials WHERE family = 'null_random' AND kind = 'backtest' AND params = %s "
            "ORDER BY id DESC LIMIT 1",
            (Jsonb(key),),
        )
        row = cur.fetchone()
    if row and row["metrics"].get("sharpes"):
        return float(np.quantile(row["metrics"]["sharpes"], q))
    sharpes = null_sharpes(history, symbols, start, end, fees, n)
    registry.add_trial(
        conn,
        "null_random",
        "backtest",
        key,
        {"sharpes": sharpes, "q95": float(np.quantile(sharpes, q))},
        "admitted",
        notes="null distribution cache",
    )
    conn.commit()
    return float(np.quantile(sharpes, q))


def admit(
    conn: psycopg.Connection,
    family: str,
    params: dict[str, Any],
    history: HistoryFrames,
    symbols: list[str],
    start: datetime,
    end: datetime,
    fees: FeeModel,
    null_thr: float,
    make_competitor: Callable[[], Any] | None = None,
    cfg: GateConfig = DEFAULT_GATE,
    notes: str = "",
) -> Admission:
    """Walk-forward ``family`` with ``params`` and record the trial. Never raises on rejection."""
    n_trials = registry.count_trials(conn, family) + 1
    trial_id = registry.add_trial(conn, family, "walkforward", params, {}, None, notes=notes)
    conn.commit()
    make = make_competitor or (lambda: REGISTRY[family](params))
    folds = run_walkforward(make, history, symbols, start, end, fees)
    verdict = evaluate(folds, null_thr, n_trials, cfg)
    from arena.judge.metrics import sharpe

    fold_sharpes = [sharpe(res.returns) for _, res in folds]
    metrics = {**verdict.metrics, "fold_sharpes": fold_sharpes, "failed": verdict.failed}
    registry.finish_trial(conn, trial_id, metrics, "admitted" if verdict.admitted else "rejected")
    conn.commit()
    return Admission(verdict=verdict, trial_id=trial_id, fold_sharpes=fold_sharpes)

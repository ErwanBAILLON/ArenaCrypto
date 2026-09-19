"""Entry gate (spec §8): turn walk-forward results into an ``admitted`` / ``rejected`` verdict.

A competitor is admitted only if **all** criteria hold:

* ``folds_positive``: net return > 0 on at least 2/3 of the test folds;
* ``sharpe_above_null``: annualised Sharpe above the 95th percentile of the
  null (seeded random) distribution on the same period;
* ``dsr``: Deflated Sharpe Ratio > 0.90 given the number of trials for the family;
* ``bootstrap_p``: stationary block bootstrap p-value of ``Sharpe <= 0`` below 0.10;
* ``max_drawdown``: below 30 %;
* ``min_decisions``: at least 30 bars with a non-flat target.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from arena.book.book import FeeModel
from arena.core.types import Verdict
from arena.judge import metrics as m
from arena.judge.backtest import BacktestResult, HistoryFrames, run
from arena.judge.walkforward import Fold


@dataclass(frozen=True)
class GateConfig:
    min_folds_positive_frac: float = 2 / 3
    min_dsr: float = 0.90
    max_bootstrap_p: float = 0.10
    max_drawdown: float = 0.30
    min_decisions: int = 30
    null_quantile: float = 0.95


DEFAULT_GATE = GateConfig()


def null_sharpe_threshold(null_results: list[BacktestResult], q: float = 0.95) -> float:
    """Empirical ``q`` quantile of annualised Sharpe over the null runs (0 if none)."""
    if not null_results:
        return 0.0
    return float(np.quantile([m.sharpe(res.returns) for res in null_results], q))


_NULL_JOB: dict[str, Any] = {}


def _null_worker(seed: int) -> BacktestResult:
    j = _NULL_JOB
    return run(j["make_null"](seed), j["history"], j["symbols"], j["start"], j["end"], j["fees"])


def run_null_distribution(
    make_null: Callable[[int], Any],
    history: HistoryFrames,
    symbols: list[str],
    start: datetime | str,
    end: datetime | str,
    fees: FeeModel,
    n: int = 200,
    workers: int | None = None,
) -> list[BacktestResult]:
    """Backtest ``make_null(seed)`` for ``seed in range(n)`` on the same period.

    Runs are independent, so they are spread over ``workers`` forked processes
    (default: ``ARENA_WORKERS`` env, else 1). Fork shares the history frames
    copy-on-write; nothing is pickled but the seed and the result.
    """
    workers = workers or int(os.environ.get("ARENA_WORKERS", "1"))
    if workers <= 1 or n <= 1:
        return [run(make_null(seed), history, symbols, start, end, fees) for seed in range(n)]
    _NULL_JOB.update(make_null=make_null, history=history, symbols=symbols, start=start, end=end, fees=fees)
    try:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            return list(pool.map(_null_worker, range(n)))
    finally:
        _NULL_JOB.clear()


def evaluate(
    fold_results: list[tuple[Fold, BacktestResult]],
    null_threshold: float,
    n_trials: int,
    cfg: GateConfig = DEFAULT_GATE,
) -> Verdict:
    """Concatenate the test-window returns of every fold and apply the §8 criteria.

    The DSR is computed on the per-period Sharpe ``mean/std`` of the
    concatenated series with its sample skew/kurtosis and ``T = len(r)``.
    """
    series = [res.returns for _, res in fold_results if len(res.returns)]
    r = pd.concat(series).sort_index() if series else pd.Series(dtype=float)
    T = int(len(r))
    sr_period = float(r.mean() / r.std(ddof=1)) if T > 1 and r.std(ddof=1) > 0 else 0.0
    skew, kurt = m.skew_kurt(r)
    fold_returns = [m.total_return(res.returns) for _, res in fold_results]
    folds_positive_frac = float(np.mean([x > 0 for x in fold_returns])) if fold_returns else 0.0

    metrics: dict[str, float] = {
        "sharpe": m.sharpe(r),
        "sortino": m.sortino(r),
        "max_drawdown": m.max_drawdown(r),
        "profit_factor": m.profit_factor(r),
        "total_return": m.total_return(r),
        "folds_positive_frac": folds_positive_frac,
        "dsr": m.deflated_sharpe(sr_period, n_trials, T, skew, kurt),
        "bootstrap_p": m.block_bootstrap_p(r),
        "decisions": float(sum(res.decisions for _, res in fold_results)),
        "turnover": float(sum(res.turnover for _, res in fold_results)),
        "null_threshold": float(null_threshold),
        "n_trials": float(n_trials),
    }
    checks = {
        "folds_positive": metrics["folds_positive_frac"] >= cfg.min_folds_positive_frac,
        "sharpe_above_null": metrics["sharpe"] > null_threshold,
        "dsr": metrics["dsr"] > cfg.min_dsr,
        "bootstrap_p": metrics["bootstrap_p"] < cfg.max_bootstrap_p,
        "max_drawdown": metrics["max_drawdown"] < cfg.max_drawdown,
        "min_decisions": metrics["decisions"] >= cfg.min_decisions,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return Verdict(admitted=not failed, metrics=metrics, failed=failed)

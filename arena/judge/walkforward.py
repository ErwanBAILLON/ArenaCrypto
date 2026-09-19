"""Anchored walk-forward: expanding train window, contiguous test folds.

Parameters are fixed inside a fold; a fresh competitor instance is built per
fold so that no state (fitted models, caches) leaks between folds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from arena.book.book import FeeModel
from arena.judge.backtest import BacktestResult, HistoryFrames, run

MIN_LAST_FOLD_DAYS = 30
BAR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class Fold:
    """Train on ``[train_start, train_end)``, test on ``[test_start, test_end)``; ``train_end == test_start``."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _utc(ts: datetime | str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def folds(start: datetime | str, end: datetime | str, test_days: int = 90, min_train_days: int = 180) -> list[Fold]:
    """Anchored folds over ``[start, end)``.

    ``train_start`` is always ``start``. The first test window opens after
    ``min_train_days``; test windows are contiguous, non-overlapping and
    ``test_days`` long. The trailing window may be shorter, but is dropped when
    under ``MIN_LAST_FOLD_DAYS`` (30) days.
    """
    s, e = _utc(start), _utc(end)
    out: list[Fold] = []
    test_start = s + pd.Timedelta(days=min_train_days)
    while test_start < e:
        test_end = min(test_start + pd.Timedelta(days=test_days), e)
        if test_end - test_start < pd.Timedelta(days=MIN_LAST_FOLD_DAYS):
            break
        out.append(Fold(train_start=s, train_end=test_start, test_start=test_start, test_end=test_end))
        test_start = test_end
    return out


def run_walkforward(
    make_competitor: Callable[[], Any],
    history: HistoryFrames,
    symbols: list[str],
    start: datetime | str,
    end: datetime | str,
    fees: FeeModel,
    test_days: int = 90,
    min_train_days: int = 180,
    bar_hours: int = 1,
) -> list[tuple[Fold, BacktestResult]]:
    """Backtest a fresh ``make_competitor()`` instance on each fold's test window.

    The Snapshot always holds the full history, so warm-up uses bars before
    ``test_start``; decisions are recorded only inside ``[test_start, test_end)``.
    """
    results: list[tuple[Fold, BacktestResult]] = []
    for fold in folds(start, end, test_days=test_days, min_train_days=min_train_days):
        competitor = make_competitor()
        res = run(competitor, history, symbols, fold.test_start, fold.test_end - BAR, fees, bar_hours=bar_hours)
        results.append((fold, res))
    return results

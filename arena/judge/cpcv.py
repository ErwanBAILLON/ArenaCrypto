"""Purged, embargoed, combinatorial cross-validation for overlapping labels.

Barrier labels span time: a label stamped Monday is only resolved on Thursday.
Ordinary k-fold then trains on Monday's label while testing on Wednesday's bar,
and Wednesday's bar is part of what decides Monday's label. The leak is not
subtle -- it is the single most common way a financial model reports an accuracy
it does not have.

Two corrections (López de Prado, *Advances in Financial Machine Learning*, 2018,
ch. 7):

* **purging** drops every training label whose lifetime overlaps the test block;
* **embargo** drops a further margin of training labels immediately after the
  test block, because serial correlation leaks forward even without overlap.

And one extension (ch. 12): instead of k train/test splits, take ``n_groups``
contiguous blocks and test on every combination of ``n_test`` of them. That
yields ``C(n, k)`` splits and, more usefully, many distinct *backtest paths*
through the sample, so a strategy can be scored on a distribution of outcomes
rather than on one. That distribution is what ``metrics.pbo_cscv`` consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

NO_EMBARGO = pd.Timedelta(0)


@dataclass(frozen=True)
class Split:
    """Positional indices into the event table."""

    train: np.ndarray
    test: np.ndarray
    test_groups: tuple[int, ...]

    @property
    def sizes(self) -> tuple[int, int]:
        return int(self.train.size), int(self.test.size)


def _ns(values) -> np.ndarray:
    """Integer nanoseconds, whatever resolution the index happens to carry.

    ``DatetimeIndex.asi8`` returns the index's own unit, which pandas 2 may set
    to microseconds, while ``Timedelta.value`` is always nanoseconds. Mixing the
    two turned a four-hour embargo into a 198-day one and silently emptied the
    training set of every split touching the first block.
    """
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def _bounds(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Label lifetimes as integer nanoseconds, for cheap interval arithmetic."""
    start = _ns(events["ts"])
    end = _ns(events["exit_ts"])
    return start, np.maximum(end, start)


def purge_blocks(events: pd.DataFrame, blocks: list[np.ndarray], embargo: pd.Timedelta = NO_EMBARGO) -> np.ndarray:
    """Training rows surviving purging and embargo against several test blocks.

    Each block is purged against **its own** time interval. Taking the hull of
    all test rows instead would, for a combinatorial split whose blocks sit at
    opposite ends of the sample, span the entire history and purge everything --
    which silently empties a third of the splits and over-purges the rest.
    """
    start, end = _bounds(events)
    n = len(events)
    blocks = [b for b in blocks if b.size]
    if not blocks:
        return np.arange(n)
    embargo_ns = int(embargo.value) if embargo is not None else 0
    keep = np.ones(n, dtype=bool)
    for block in blocks:
        lo, hi = int(start[block].min()), int(end[block].max())
        keep &= ~((start <= hi) & (end >= lo))  # lifetime overlaps this block
        keep &= ~((start > hi) & (start <= hi + embargo_ns))  # begins in its embargo
        keep[block] = False
    return np.flatnonzero(keep)


def purge(events: pd.DataFrame, test_rows: np.ndarray, embargo: pd.Timedelta = NO_EMBARGO) -> np.ndarray:
    """Training rows surviving purging and embargo against one contiguous test block."""
    return purge_blocks(events, [np.asarray(test_rows)], embargo)


def group_rows(events: pd.DataFrame, n_groups: int) -> list[np.ndarray]:
    """Contiguous, chronologically ordered blocks of event rows of near-equal size."""
    order = np.argsort(_ns(events["ts"]), kind="stable")
    return [g for g in np.array_split(order, n_groups) if g.size]


def cpcv_splits(
    events: pd.DataFrame,
    n_groups: int = 6,
    n_test: int = 2,
    embargo_frac: float = 0.01,
) -> list[Split]:
    """Every combination of ``n_test`` blocks out of ``n_groups``, purged and embargoed.

    ``embargo_frac`` is a share of the sample's total time span. Splits whose
    training set is empty after purging are dropped rather than returned as
    degenerate.
    """
    if events.empty or n_test <= 0 or n_test >= n_groups:
        return []
    groups = group_rows(events, n_groups)
    if len(groups) < n_groups:
        return []
    span = pd.DatetimeIndex(events["ts"]).max() - pd.DatetimeIndex(events["ts"]).min()
    embargo = pd.Timedelta(span * float(embargo_frac))
    out: list[Split] = []
    for combo in combinations(range(len(groups)), n_test):
        blocks = [groups[i] for i in combo]
        test = np.sort(np.concatenate(blocks))
        train = purge_blocks(events, blocks, embargo)
        if train.size == 0:
            continue
        out.append(Split(train=train, test=test, test_groups=combo))
    return out


def n_paths(n_groups: int, n_test: int) -> int:
    """Distinct backtest paths the combinatorial scheme produces.

    Each block is held out in ``C(n-1, k-1)`` of the ``C(n, k)`` splits, so the
    splits reassemble into that many complete walks through the sample -- which
    is the point: one path is an anecdote, a hundred is a distribution.
    """
    from math import comb

    if n_test <= 0 or n_test >= n_groups:
        return 0
    return comb(n_groups - 1, n_test - 1)


def leakage_report(events: pd.DataFrame, splits: list[Split]) -> dict:
    """Assert-style summary: how much overlap survived, which must be none.

    Cheap enough to run in anger. If ``max_overlap_ns`` is anything but zero the
    purge is broken and every number downstream is decoration.
    """
    start, end = _bounds(events)
    worst = 0
    for split in splits:
        if split.train.size == 0 or split.test.size == 0:
            continue
        # per contiguous run of test rows, exactly as the purge saw them
        runs = np.split(split.test, np.flatnonzero(np.diff(split.test) > 1) + 1)
        for run in runs:
            t_lo, t_hi = int(start[run].min()), int(end[run].max())
            overlap = np.minimum(end[split.train], t_hi) - np.maximum(start[split.train], t_lo)
            worst = max(worst, int(overlap.max()) if overlap.size else 0)
    return {
        "splits": len(splits),
        "max_overlap_ns": max(worst, 0),
        "mean_train": float(np.mean([s.train.size for s in splits])) if splits else 0.0,
        "mean_test": float(np.mean([s.test.size for s in splits])) if splits else 0.0,
    }

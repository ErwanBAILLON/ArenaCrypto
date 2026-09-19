"""Allocator: an *opinion* on how capital would be split between champions.

weight_i ∝ max(0, Sharpe_60d_i), EMA-smoothed so that the split does not chase
last week's winner; whatever is not allocated is cash. It is published to
Telegram and never turned into orders.
"""

from __future__ import annotations

import pandas as pd

from arena.judge.metrics import sharpe

MIN_BARS = 24 * 7  # a week of returns before a competitor gets any weight
ALPHA_PER_TICK = 0.1 / 24  # ~10 % of the way to the raw target per day


def raw_weights(returns: pd.DataFrame) -> dict[int, float]:
    """Positive-Sharpe-proportional weights over the columns of ``returns`` (competitor ids)."""
    scores: dict[int, float] = {}
    for cid in returns.columns:
        r = returns[cid].dropna()
        if len(r) < MIN_BARS:
            continue
        scores[int(cid)] = max(0.0, sharpe(r))
    total = sum(scores.values())
    if total <= 0:
        return {}
    return {cid: s / total for cid, s in scores.items() if s > 0}


def compute(returns: pd.DataFrame, prev: dict[int, float], alpha: float = ALPHA_PER_TICK) -> dict[int, float]:
    """EMA of the raw weights toward ``prev``; competitors absent from both are dropped."""
    raw = raw_weights(returns)
    ids = set(raw) | set(prev)
    out = {cid: (1 - alpha) * prev.get(cid, 0.0) + alpha * raw.get(cid, 0.0) for cid in ids}
    return {cid: w for cid, w in out.items() if w > 1e-4}

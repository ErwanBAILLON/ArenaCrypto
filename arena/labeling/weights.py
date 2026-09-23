"""Sample weights: how much each label is allowed to say.

Two corrections, both from López de Prado, both aimed at the same illusion --
that a dataset of N labels contains N independent opinions.

**Uniqueness.** Barrier labels overlap: a trade opened on Monday and one opened
on Tuesday can be resolved by the same Thursday move. Counting them as two
observations lets one market event vote twice. Each label is weighted by the
average, over its own lifetime, of one divided by how many labels were live at
the same moment (*Advances in Financial Machine Learning*, 2018, §4).

**Trend strength.** Not every labelled episode is equally informative: a barrier
touched during a clean trend says more than one touched in noise. Trend scanning
(*Machine Learning for Asset Managers*, 2020) regresses log price on time over a
range of horizons and keeps the largest |t|, which becomes a second weight.
Recent work is blunt that the value lies in filtering on it rather than in
labelling alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def concurrency(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Number of labels live at each bar of ``index``.

    ``events`` needs ``ts`` (entry) and ``exit_ts`` columns. A bar touched by no
    label counts zero.
    """
    counts = pd.Series(0, index=index, dtype="int64")
    if events.empty:
        return counts
    starts = index.searchsorted(pd.DatetimeIndex(events["ts"]), side="left")
    ends = index.searchsorted(pd.DatetimeIndex(events["exit_ts"]), side="right")
    delta = np.zeros(len(index) + 1, dtype="int64")
    for s, e in zip(starts, ends, strict=True):
        if e > s:
            delta[s] += 1
            delta[e] -= 1
    counts.iloc[:] = np.cumsum(delta[:-1])
    return counts


def average_uniqueness(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Per label, the mean of ``1 / concurrency`` over the bars it spans.

    1.0 means the label had the market to itself; 0.1 means ten labels were
    resolving on the same bars and each deserves a tenth of a vote.
    """
    if events.empty:
        return pd.Series(dtype=float)
    live = concurrency(events, index).to_numpy()
    starts = index.searchsorted(pd.DatetimeIndex(events["ts"]), side="left")
    ends = index.searchsorted(pd.DatetimeIndex(events["exit_ts"]), side="right")
    out = np.ones(len(events), dtype=float)
    for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
        if e <= s:
            continue
        span = live[s:e]
        span = np.where(span > 0, span, 1)
        out[i] = float(np.mean(1.0 / span))
    return pd.Series(out, index=events.index, name="uniqueness")


def trend_tstat(closes: pd.Series, min_bars: int = 6, max_bars: int = 48) -> tuple[float, int]:
    """``(t-statistic, horizon)`` of the strongest linear trend in log price starting now.

    Fits ``log p_t = a + b·t`` over every horizon in ``[min_bars, max_bars]``
    forward from the first point and keeps the one whose slope has the largest
    absolute t. The sign carries the direction, the magnitude the conviction.
    Returns ``(0.0, 0)`` when there is not enough data to fit anything.
    """
    y_all = np.log(pd.Series(closes, dtype=float).dropna().to_numpy())
    if y_all.size < min_bars:
        return 0.0, 0
    best_t, best_h = 0.0, 0
    for h in range(min_bars, min(max_bars, y_all.size) + 1):
        y = y_all[:h]
        x = np.arange(h, dtype=float)
        x_centred = x - x.mean()
        denom = float(x_centred @ x_centred)
        if denom <= 0:
            continue
        slope = float(x_centred @ (y - y.mean())) / denom
        resid = y - (y.mean() + slope * x_centred)
        dof = h - 2
        if dof <= 0:
            continue
        se = np.sqrt(float(resid @ resid) / dof / denom)
        if se <= 0 or not np.isfinite(se):
            continue
        t = slope / se
        if abs(t) > abs(best_t):
            best_t, best_h = float(t), h
    return best_t, best_h


def trend_tstats(closes: pd.Series, events: pd.DatetimeIndex, min_bars: int = 6, max_bars: int = 48) -> pd.Series:
    """``trend_tstat`` evaluated forward from each event timestamp.

    Look-ahead by construction -- it reads the future of each event -- which is
    why it may only ever be a *weight*, never a feature.
    """
    series = pd.Series(closes, dtype=float)
    out = {}
    for event in events:
        start = series.index.searchsorted(pd.Timestamp(event), side="right")
        out[pd.Timestamp(event)] = trend_tstat(series.iloc[start : start + max_bars], min_bars, max_bars)[0]
    return pd.Series(out, name="trend_t")


def sample_weights(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    trend_t: pd.Series | None = None,
    trend_cap: float = 4.0,
) -> pd.Series:
    """Uniqueness, optionally scaled by clipped trend strength, normalised to mean 1.

    Normalising keeps the effective sample size interpretable: the weights
    redistribute attention, they do not inflate it.
    """
    weights = average_uniqueness(events, index)
    if weights.empty:
        return weights
    if trend_t is not None:
        strength = pd.Series(
            [abs(float(trend_t.get(pd.Timestamp(t), 0.0))) for t in events["ts"]], index=events.index
        ).clip(upper=trend_cap)
        weights = weights * (strength / trend_cap).clip(lower=0.05)
    total = float(weights.sum())
    if total <= 0:
        return pd.Series(1.0, index=events.index, name="weight")
    return (weights * (len(weights) / total)).rename("weight")

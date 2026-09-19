"""Random-window robustness (spec §8, step 7): does the model hold in every season?

``n`` windows of 2 to 6 months are drawn uniformly at random inside the
history. A **fresh** competitor is backtested on each one (the Snapshot holds
the full history, so warm-up uses the bars before the window). Each window is
labelled by the BTC return over it (``bull`` > +10 %, ``bear`` < -10 %, else
``range``) and the win rate (total return > 0) is reported per regime.

Windows are independent, so they are spread over forked workers exactly like
``gate.run_null_distribution``.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from arena.book.book import FeeModel
from arena.judge import metrics as m
from arena.judge.backtest import REFERENCE_SYMBOL, HistoryFrames, run

REGIMES = ("bull", "bear", "range")
BULL_THRESHOLD = 0.10
BEAR_THRESHOLD = -0.10
BAR = pd.Timedelta(hours=1)
Window = tuple[pd.Timestamp, pd.Timestamp]


def _utc(ts: datetime | str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def label_window(btc_closes: pd.Series, start: datetime | str, end: datetime | str) -> str:
    """``bull`` / ``bear`` / ``range`` from the BTC close-to-close return over ``[start, end]``.

    Fewer than two closes in the window means no measurable move: ``range``.
    """
    s, e = _utc(start), _utc(end)
    idx = pd.DatetimeIndex(btc_closes.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    vals = btc_closes.to_numpy(dtype=float)[(idx >= s) & (idx <= e)]
    vals = vals[~np.isnan(vals)]
    if vals.size < 2 or vals[0] <= 0:
        return "range"
    ret = vals[-1] / vals[0] - 1.0
    if ret > BULL_THRESHOLD:
        return "bull"
    if ret < BEAR_THRESHOLD:
        return "bear"
    return "range"


def sample_windows(
    start: datetime | str,
    end: datetime | str,
    n: int = 200,
    min_days: int = 60,
    max_days: int = 180,
    seed: int = 0,
) -> list[Window]:
    """``n`` windows ``(w_start, w_end)`` with ``w_end - w_start`` in ``[min_days, max_days]`` days.

    Length (whole days) and offset (whole hours) are uniform; every window lies
    inside ``[start, end]``. Deterministic for a given ``seed``. ``max_days`` is
    clamped to the span; a span shorter than ``min_days`` raises ``ValueError``.
    """
    s, e = _utc(start), _utc(end)
    span_hours = int((e - s) / BAR)
    if span_hours < min_days * 24:
        raise ValueError(f"span {span_hours / 24:.1f} days is shorter than min_days={min_days}")
    max_days = min(max_days, span_hours // 24)
    rng = np.random.default_rng(seed)
    out: list[Window] = []
    for _ in range(n):
        length_h = int(rng.integers(min_days, max_days + 1)) * 24
        offset_h = int(rng.integers(0, span_hours - length_h + 1))
        w_start = s + offset_h * BAR
        out.append((w_start, w_start + length_h * BAR))
    return out


def _reference_closes(history: HistoryFrames, symbols: list[str]) -> pd.Series:
    c = history.candles
    present = set(c["symbol"].unique())
    ref = REFERENCE_SYMBOL if REFERENCE_SYMBOL in present else next((s for s in symbols if s in present), None)
    if ref is None:
        return pd.Series(dtype=float)
    ser = c.loc[c["symbol"] == ref, ["ts", "close"]].set_index("ts")["close"].sort_index()
    idx = pd.DatetimeIndex(ser.index)
    ser.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return ser.astype(float)


_ROB_JOB: dict[str, Any] = {}


def _window_worker(window: Window) -> tuple[float, float]:
    j = _ROB_JOB
    w_start, w_end = window
    res = run(
        j["make_competitor"](), j["history"], j["symbols"], w_start, w_end - BAR, j["fees"], bar_hours=j["bar_hours"]
    )
    return m.sharpe(res.returns, 8760 // j["bar_hours"]), m.total_return(res.returns)


def _run_windows(windows: list[Window], workers: int) -> list[tuple[float, float]]:
    if workers <= 1 or len(windows) <= 1:
        return [_window_worker(w) for w in windows]
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(_window_worker, windows))


def _median(xs: list[float]) -> float | None:
    return float(np.median(xs)) if xs else None


def summarise(results: list[tuple[float, float]], labels: list[str], min_days: int, max_days: int) -> dict[str, Any]:
    """Aggregate per-window ``(sharpe, total_return)`` and regime labels into the robustness dict (plain JSON types)."""
    rets = [float(r) for _, r in results]
    wins = [r > 0 for r in rets]
    by_regime: dict[str, float | None] = {}
    n_by_regime: dict[str, int] = {}
    for reg in REGIMES:
        sel = [w for w, lab in zip(wins, labels, strict=True) if lab == reg]
        n_by_regime[reg] = len(sel)
        by_regime[reg] = float(np.mean(sel)) if sel else None
    return {
        "n_windows": len(results),
        "min_days": int(min_days),
        "max_days": int(max_days),
        "win_rate_overall": float(np.mean(wins)) if wins else 0.0,
        "win_rate_by_regime": by_regime,
        "n_by_regime": n_by_regime,
        "median_sharpe": _median([float(s) for s, _ in results]),
        "median_return": _median(rets),
        "worst_return": float(min(rets)) if rets else None,
    }


def run_robustness(
    make_competitor: Callable[[], Any],
    history: HistoryFrames,
    symbols: list[str],
    start: datetime | str,
    end: datetime | str,
    fees: FeeModel,
    n: int = 200,
    seed: int = 0,
    workers: int | None = None,
    bar_hours: int = 1,
    min_days: int = 60,
    max_days: int = 180,
) -> dict[str, Any]:
    """Backtest a fresh ``make_competitor()`` on ``n`` random windows of ``[start, end]``.

    Returns ``{"n_windows", "win_rate_overall", "win_rate_by_regime", "n_by_regime",
    "median_sharpe", "median_return", "worst_return", "min_days", "max_days"}``;
    a regime with no window has ``None`` as win rate. Workers default to the
    ``ARENA_WORKERS`` env (else 1); fork shares the history copy-on-write.
    """
    workers = workers or int(os.environ.get("ARENA_WORKERS", "1"))
    windows = sample_windows(start, end, n=n, min_days=min_days, max_days=max_days, seed=seed)
    closes = _reference_closes(history, symbols)
    labels = [label_window(closes, ws, we) for ws, we in windows]
    _ROB_JOB.update(make_competitor=make_competitor, history=history, symbols=symbols, fees=fees, bar_hours=bar_hours)
    try:
        results = _run_windows(windows, workers)
    finally:
        _ROB_JOB.clear()
    return summarise(results, labels, min_days, max_days)

"""Drift checks: deterministic rules that raise alerts, never actions."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from arena.core.types import Alert, CompetitorSpec
from arena.judge.metrics import sharpe, total_return

LIVE_WINDOW_DAYS = 30
SHARPE_FLOOR = -1.0
RETURN_FLOOR = -0.10
SILENT_DAYS = 7


def stale_data(last_bar: datetime | None, now: datetime, max_lag_bars: int = 2) -> Alert | None:
    if last_bar is None:
        return Alert(kind="stale", payload={"detail": "no candles at all"})
    lag = (pd.Timestamp(now) - pd.Timestamp(last_bar)) / pd.Timedelta(hours=1)
    if lag > max_lag_bars:
        return Alert(kind="stale", payload={"detail": f"last closed bar {last_bar:%Y-%m-%d %H:%M} UTC, lag {lag:.1f}h"})
    return None


def champion_bleeding(spec: CompetitorSpec, returns: pd.Series, bars_per_day: int = 24) -> Alert | None:
    """A champion whose live 30d Sharpe and return are both clearly negative.

    ``bars_per_day`` makes the window a month in both arenas: 720 hourly bars
    for crypto, 30 daily bars for classic markets.
    """
    window = LIVE_WINDOW_DAYS * bars_per_day
    r = returns.dropna().tail(window)
    if len(r) < window // 2:
        return None
    s, tr = sharpe(r, 365 * bars_per_day), total_return(r)
    if s < SHARPE_FLOOR and tr < RETURN_FLOOR:
        return Alert(
            kind="drift", competitor_id=spec.id, payload={"detail": f"{spec.name}: 30d Sharpe {s:.2f}, return {tr:.1%}"}
        )
    return None


def silent_competitor(spec: CompetitorSpec, last_decision_ts: datetime | None, now: datetime) -> Alert | None:
    if spec.role != "competitor":
        return None
    if last_decision_ts is None or pd.Timestamp(now) - pd.Timestamp(last_decision_ts) > timedelta(days=SILENT_DAYS):
        return Alert(
            kind="drift",
            competitor_id=spec.id,
            payload={"detail": f"{spec.name}: no non-zero target for {SILENT_DAYS}+ days"},
        )
    return None


def broken_book(spec: CompetitorSpec, nav: float | None) -> Alert | None:
    if nav is None or not np.isfinite(nav) or nav <= 0:
        return Alert(kind="error", competitor_id=spec.id, payload={"detail": f"{spec.name}: nav is {nav}"})
    return None

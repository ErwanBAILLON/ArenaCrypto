"""Read-only queries behind the dashboard.

Everything the pages show is computed here from the store; the routes only
render. Statistics reuse the same functions as the Telegram digest so both
views agree to the last digit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import psycopg

from arena.core.types import CompetitorSpec
from arena.judge.metrics import max_drawdown, sharpe, total_return
from arena.runner import promotion
from arena.runner.digest import WINDOW
from arena.store import books as bstore
from arena.store import registry

STALE_AFTER = timedelta(hours=3)
CHART_DAYS = 90
MIN_BARS_FOR_SHARPE = 24
TRIAL_METRIC_KEYS = ("sharpe", "dsr", "bootstrap_p", "max_drawdown", "fold_sharpes")


@dataclass(frozen=True)
class Leaderboard:
    """Everything the front page needs, in one object."""

    rows: list[dict[str, Any]]
    allocation: list[tuple[str, float]]
    null95: float | None
    last_bar: datetime | None
    stale: bool
    chart_ids: list[int] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """JSON-serialisable view (timestamps as ISO 8601)."""
        return {
            "rows": self.rows,
            "allocation": [{"name": n, "weight": w} for n, w in self.allocation],
            "null95": self.null95,
            "last_bar": self.last_bar.isoformat() if self.last_bar else None,
            "stale": self.stale,
        }


@dataclass(frozen=True)
class CompetitorDetail:
    spec: CompetitorSpec
    parent_name: str | None
    first_ts: datetime | None
    current_ts: datetime | None
    current_targets: list[dict[str, Any]]
    recent_targets: list[dict[str, Any]]
    trials: list[dict[str, Any]]
    model_metrics: dict[str, Any] | None


def last_bar(conn: psycopg.Connection) -> datetime | None:
    """Timestamp of the most recent book row across all competitors, or None."""
    with conn.cursor() as cur:
        cur.execute("SELECT max(ts) AS ts FROM books")
        return cur.fetchone()["ts"]


def is_stale(last: datetime | None, now: datetime) -> bool:
    """True when no book row is younger than ``STALE_AFTER``."""
    return last is None or (now - last) > STALE_AFTER


def leaderboard(conn: psycopg.Connection, now: datetime) -> Leaderboard:
    """Same rows and ordering as the daily digest, plus ids and staleness."""
    specs = registry.list_competitors(conn, statuses=["champion", "challenger"])
    ids = [s.id for s in specs]
    rets = bstore.read_returns(conn, ids, now - WINDOW, now) if ids else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for s in specs:
        r = rets[s.id].dropna() if s.id in rets.columns else pd.Series(dtype=float)
        last = bstore.last_book_row(conn, s.id)
        rows.append(
            {
                "id": s.id,
                "name": s.name,
                "family": s.family,
                "version": s.version,
                "status": s.status,
                "role": s.role,
                "sharpe_30d": sharpe(r) if len(r) > MIN_BARS_FOR_SHARPE else 0.0,
                "ret_30d": total_return(r) if len(r) else 0.0,
                "mdd_30d": max_drawdown(r) if len(r) else 0.0,
                "nav": last.nav if last else 0.0,
                "bars_30d": int(len(r)),
            }
        )
    rows.sort(key=lambda x: (x["role"] != "competitor", -x["sharpe_30d"]))
    names = {s.id: s.name for s in specs}
    alloc = [
        (names.get(cid, str(cid)), w) for cid, w in sorted(bstore.last_allocations(conn).items(), key=lambda kv: -kv[1])
    ]
    nulls = [s.id for s in specs if s.family == "null_random" and s.id in rets.columns]
    null95 = promotion.null_threshold(rets[nulls], now - WINDOW) if nulls else None
    chart_ids = [s.id for s in specs if s.role == "benchmark" or (s.role == "competitor" and s.status == "champion")]
    last = last_bar(conn)
    return Leaderboard(rows, alloc, null95, last, is_stale(last, now), chart_ids)


def nav_series(conn: psycopg.Connection, ids: list[int], start: datetime, end: datetime) -> dict[str, pd.Series]:
    """``name -> NAV / NAV[0]`` for each competitor with at least one book row in the window."""
    names = {s.id: s.name for s in registry.list_competitors(conn)}
    out: dict[str, pd.Series] = {}
    for cid in ids:
        nav = bstore.read_nav(conn, cid, start, end)
        if nav.empty or nav.iloc[0] == 0:
            continue
        out[names.get(cid, str(cid))] = nav / nav.iloc[0]
    return out


def _competitor_by_id(conn: psycopg.Connection, competitor_id: int) -> CompetitorSpec | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, family, version, params, role, status, parent_id, rationale"
            " FROM competitors WHERE id = %s",
            (competitor_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return CompetitorSpec(
        id=int(row["id"]),
        name=row["name"],
        family=row["family"],
        version=int(row["version"]),
        params=dict(row["params"]),
        role=row["role"],
        status=row["status"],
        parent_id=row["parent_id"],
        rationale=row["rationale"],
    )


def _trial_row(r: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(r["metrics"])
    return {
        "id": int(r["id"]),
        "competitor_id": r["competitor_id"],
        "family": r["family"],
        "kind": r["kind"],
        "verdict": r["verdict"],
        "started_at": r["started_at"],
        "finished_at": r["finished_at"],
        "notes": r["notes"],
        "metrics": metrics,
        "highlights": {k: metrics.get(k) for k in TRIAL_METRIC_KEYS},
    }


def competitor_detail(conn: psycopg.Connection, competitor_id: int) -> CompetitorDetail | None:
    """Spec, targets, family trials and latest model metrics; None when the id is unknown."""
    spec = _competitor_by_id(conn, competitor_id)
    if spec is None:
        return None
    parent = _competitor_by_id(conn, spec.parent_id) if spec.parent_id is not None else None
    with conn.cursor() as cur:
        cur.execute("SELECT min(ts) AS first FROM books WHERE competitor_id = %s", (competitor_id,))
        first_ts = cur.fetchone()["first"]
        cur.execute(
            "SELECT ts, symbol, weight, kind, conviction, reason FROM targets"
            " WHERE competitor_id = %s AND ts = (SELECT max(ts) FROM targets WHERE competitor_id = %s)"
            " ORDER BY abs(weight) DESC, symbol",
            (competitor_id, competitor_id),
        )
        current = [dict(r) for r in cur.fetchall()]
        cur.execute(
            "SELECT ts, symbol, weight, kind, conviction, reason FROM targets"
            " WHERE competitor_id = %s ORDER BY ts DESC, symbol LIMIT 50",
            (competitor_id,),
        )
        recent = [dict(r) for r in cur.fetchall()]
        cur.execute(
            "SELECT id, competitor_id, family, kind, verdict, started_at, finished_at, notes, metrics"
            " FROM trials WHERE family = %s ORDER BY started_at DESC, id DESC LIMIT 50",
            (spec.family,),
        )
        trials_rows = [_trial_row(r) for r in cur.fetchall()]
        model_metrics = None
        if spec.family == "meta_label":
            cur.execute(
                "SELECT metrics FROM models WHERE competitor_id = %s ORDER BY trained_at DESC, id DESC LIMIT 1",
                (competitor_id,),
            )
            row = cur.fetchone()
            model_metrics = dict(row["metrics"]) if row else None
    return CompetitorDetail(
        spec=spec,
        parent_name=parent.name if parent else None,
        first_ts=first_ts,
        current_ts=current[0]["ts"] if current else None,
        current_targets=current,
        recent_targets=recent,
        trials=trials_rows,
        model_metrics=model_metrics,
    )


def alerts(conn: psycopg.Connection, limit: int = 200) -> list[dict[str, Any]]:
    """Most recent alerts with the competitor name resolved and a one-line detail."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT a.id, a.ts, a.kind, a.competitor_id, c.name AS competitor, a.symbol, a.payload, a.sent_at"
            " FROM alerts a LEFT JOIN competitors c ON c.id = a.competitor_id"
            " ORDER BY a.ts DESC, a.id DESC LIMIT %s",
            (limit,),
        )
        rows = []
        for r in cur.fetchall():
            payload = dict(r["payload"])
            rows.append({**r, "payload": payload, "detail": payload.get("detail") or _kv(payload)})
        return rows


def trials(conn: psycopg.Connection, limit: int = 200) -> list[dict[str, Any]]:
    """Most recent trials across all families with metric highlights extracted."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, competitor_id, family, kind, verdict, started_at, finished_at, notes, metrics"
            " FROM trials ORDER BY started_at DESC, id DESC LIMIT %s",
            (limit,),
        )
        return [_trial_row(r) for r in cur.fetchall()]


def _kv(d: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())

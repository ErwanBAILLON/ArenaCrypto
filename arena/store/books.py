"""Forward-run repository: targets, virtual books, allocations and alerts."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from arena.core.types import Alert, BookRow, Decision

_BOOK_COLS = ["ts", "nav", "ret", "gross", "turnover", "fees", "funding_pnl"]


def write_targets(conn: psycopg.Connection, competitor_id: int, ts: datetime, decision: Decision) -> None:
    """Persist the non-zero targets of a decision at ``ts`` (re-running overwrites)."""
    rows = [
        (competitor_id, ts, symbol, t.weight, t.conviction, t.kind, Jsonb(t.reason))
        for symbol, t in decision.items()
        if t.weight != 0.0
    ]
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO targets (competitor_id, ts, symbol, weight, conviction, kind, reason)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (competitor_id, ts, symbol) DO UPDATE SET"
            " weight = EXCLUDED.weight, conviction = EXCLUDED.conviction,"
            " kind = EXCLUDED.kind, reason = EXCLUDED.reason",
            rows,
        )


def last_targets(
    conn: psycopg.Connection, competitor_id: int
) -> tuple[datetime, dict[str, tuple[str, float]]] | None:
    """Latest target timestamp and ``symbol -> (kind, weight)`` for a competitor, or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, symbol, kind, weight FROM targets"
            " WHERE competitor_id = %s AND ts = (SELECT max(ts) FROM targets WHERE competitor_id = %s)",
            (competitor_id, competitor_id),
        )
        rows = cur.fetchall()
    if not rows:
        return None
    return rows[0]["ts"], {r["symbol"]: (r["kind"], float(r["weight"])) for r in rows}


def write_book_row(conn: psycopg.Connection, competitor_id: int, row: BookRow) -> None:
    """Persist one book step (re-running overwrites the same ``ts``)."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO books (competitor_id, ts, nav, ret, gross, turnover, fees, funding_pnl)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (competitor_id, ts) DO UPDATE SET"
            " nav = EXCLUDED.nav, ret = EXCLUDED.ret, gross = EXCLUDED.gross, turnover = EXCLUDED.turnover,"
            " fees = EXCLUDED.fees, funding_pnl = EXCLUDED.funding_pnl",
            (competitor_id, row.ts, row.nav, row.ret, row.gross, row.turnover, row.fees, row.funding_pnl),
        )


def last_book_row(conn: psycopg.Connection, competitor_id: int) -> BookRow | None:
    """Most recent book row for a competitor, or None."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_BOOK_COLS)} FROM books WHERE competitor_id = %s ORDER BY ts DESC LIMIT 1",
            (competitor_id,),
        )
        row = cur.fetchone()
    return BookRow(**row) if row else None


def read_returns(
    conn: psycopg.Connection, competitor_ids: Sequence[int], start: datetime, end: datetime
) -> pd.DataFrame:
    """Wide frame of hourly returns: index ``ts`` (UTC), one column per competitor id."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT competitor_id, ts, ret FROM books"
            " WHERE competitor_id = ANY(%s) AND ts BETWEEN %s AND %s ORDER BY ts",
            (list(competitor_ids), start, end),
        )
        df = pd.DataFrame(cur.fetchall(), columns=["competitor_id", "ts", "ret"])
    if df.empty:
        return pd.DataFrame(columns=list(competitor_ids), index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    wide = df.pivot(index="ts", columns="competitor_id", values="ret").astype(float)
    wide.columns.name = None
    return wide


def read_nav(conn: psycopg.Connection, competitor_id: int, start: datetime, end: datetime) -> pd.Series:
    """NAV series (UTC index) for a competitor between ``start`` and ``end``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, nav FROM books WHERE competitor_id = %s AND ts BETWEEN %s AND %s ORDER BY ts",
            (competitor_id, start, end),
        )
        rows = cur.fetchall()
    idx = pd.DatetimeIndex(pd.to_datetime([r["ts"] for r in rows], utc=True), name="ts")
    return pd.Series([float(r["nav"]) for r in rows], index=idx, name="nav", dtype=float)


def write_allocations(conn: psycopg.Connection, ts: datetime, weights: dict[int, float]) -> None:
    """Persist allocator weights per competitor at ``ts`` (re-running overwrites)."""
    if not weights:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO allocations (ts, competitor_id, weight) VALUES (%s, %s, %s)"
            " ON CONFLICT (ts, competitor_id) DO UPDATE SET weight = EXCLUDED.weight",
            [(ts, cid, w) for cid, w in weights.items()],
        )


def last_allocations(conn: psycopg.Connection) -> dict[int, float]:
    """``competitor_id -> weight`` at the latest allocation timestamp (empty if none)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT competitor_id, weight FROM allocations WHERE ts = (SELECT max(ts) FROM allocations)"
        )
        return {int(r["competitor_id"]): float(r["weight"]) for r in cur.fetchall()}


def add_alert(conn: psycopg.Connection, alert: Alert) -> int:
    """Queue an alert for Telegram delivery; returns its id."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO alerts (kind, competitor_id, symbol, payload) VALUES (%s, %s, %s, %s) RETURNING id",
            (alert.kind, alert.competitor_id, alert.symbol, Jsonb(alert.payload)),
        )
        return int(cur.fetchone()["id"])


def unsent_alerts(conn: psycopg.Connection) -> list[tuple[int, Alert, datetime]]:
    """Alerts not yet delivered, oldest first, as ``(id, alert, created_ts)``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, ts, kind, competitor_id, symbol, payload FROM alerts WHERE sent_at IS NULL ORDER BY ts, id"
        )
        return [
            (
                int(r["id"]),
                Alert(kind=r["kind"], payload=dict(r["payload"]), competitor_id=r["competitor_id"], symbol=r["symbol"]),
                r["ts"],
            )
            for r in cur.fetchall()
        ]


def mark_sent(conn: psycopg.Connection, ids: Sequence[int]) -> None:
    """Stamp ``sent_at = now()`` on the given alerts."""
    if not ids:
        return
    with conn.cursor() as cur:
        cur.execute("UPDATE alerts SET sent_at = now() WHERE id = ANY(%s)", (list(ids),))


def alerts_sent_today(conn: psycopg.Connection, now: datetime) -> int:
    """Number of alerts delivered on the UTC calendar day of ``now`` (rate-limit input)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) AS n FROM alerts"
            " WHERE sent_at IS NOT NULL AND date_trunc('day', sent_at AT TIME ZONE 'UTC')"
            "     = date_trunc('day', %s::timestamptz AT TIME ZONE 'UTC')",
            (now,),
        )
        return int(cur.fetchone()["n"])


def recent_alert_exists(
    conn: psycopg.Connection, kind: str, competitor_id: int | None, symbol: str | None, since: datetime
) -> bool:
    """True when an alert with the same kind/competitor/symbol was queued at or after ``since``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM alerts WHERE kind = %s AND competitor_id IS NOT DISTINCT FROM %s"
            " AND symbol IS NOT DISTINCT FROM %s AND ts >= %s LIMIT 1",
            (kind, competitor_id, symbol, since),
        )
        return cur.fetchone() is not None

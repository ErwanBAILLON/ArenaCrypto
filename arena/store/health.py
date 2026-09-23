"""Per-source ingestion health: when did we last ask, did it work, how fresh is the answer.

One row per ``(source, universe)``, written by ``arena.runner.ingest`` on every
attempt. Three timestamps that are deliberately not the same thing:

* ``last_fetch_at`` -- when the arena last asked the endpoint;
* ``last_ok_at`` -- when it last got a usable answer;
* ``last_data_ts`` -- the newest datum that answer carried.

Deriving any of them from ``max(fetched_at)`` of the data table is what made
the macro calendar look three days dead while it was being polled every thirty
minutes: a static weekly calendar upserted with ``DO NOTHING`` writes no row.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import psycopg

SOURCES = ("candles", "funding", "open_interest", "hl_snapshots", "articles", "macro_events")


def record_fetch(
    conn: psycopg.Connection,
    source: str,
    universe: str = "crypto",
    *,
    ok: bool = True,
    last_data_ts: datetime | None = None,
    rows_written: int = 0,
    detail: str = "",
    now: datetime | None = None,
) -> None:
    """Upsert the health row for one source. A failure keeps the previous ``last_ok_at``."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO feed_health (source, universe, last_fetch_at, last_ok_at, last_data_ts, rows_written,"
            " ok, detail) VALUES (%(src)s, %(u)s, coalesce(%(now)s, now()),"
            " CASE WHEN %(ok)s THEN coalesce(%(now)s, now()) END, %(data)s, %(rows)s, %(ok)s, %(detail)s)"
            " ON CONFLICT (source, universe) DO UPDATE SET"
            "   last_fetch_at = excluded.last_fetch_at,"
            "   last_ok_at    = coalesce(excluded.last_ok_at, feed_health.last_ok_at),"
            "   last_data_ts  = coalesce(excluded.last_data_ts, feed_health.last_data_ts),"
            "   rows_written  = excluded.rows_written,"
            "   ok            = excluded.ok,"
            "   detail        = excluded.detail",
            {
                "src": source,
                "u": universe,
                "now": now,
                "ok": bool(ok),
                "data": last_data_ts,
                "rows": int(rows_written),
                "detail": detail[:500],
            },
        )


def read_all(conn: psycopg.Connection, universe: str | None = None) -> list[dict[str, Any]]:
    """Every health row, ordered by source, optionally for one arena."""
    sql = "SELECT source, universe, last_fetch_at, last_ok_at, last_data_ts, rows_written, ok, detail FROM feed_health"
    params: list[Any] = []
    if universe is not None:
        sql += " WHERE universe = %s"
        params.append(universe)
    with conn.cursor() as cur:
        cur.execute(sql + " ORDER BY universe, source", params)
        return [dict(r) for r in cur.fetchall()]

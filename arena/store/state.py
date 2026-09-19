"""Competitor state persistence between ticks (see ``Competitor.state``)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


def save_state(conn: psycopg.Connection, competitor_id: int, ts: datetime, state: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO competitor_state (competitor_id, ts, state) VALUES (%s, %s, %s)"
            " ON CONFLICT (competitor_id) DO UPDATE SET ts = EXCLUDED.ts, state = EXCLUDED.state",
            (competitor_id, ts, Jsonb(state)),
        )


def load_state(conn: psycopg.Connection, competitor_id: int) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT state FROM competitor_state WHERE competitor_id = %s", (competitor_id,))
        row = cur.fetchone()
    return dict(row["state"]) if row else {}

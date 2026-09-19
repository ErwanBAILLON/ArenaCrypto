"""Postgres connection and migrations."""

from __future__ import annotations

from pathlib import Path

import psycopg
from psycopg.rows import dict_row

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def connect(database_url: str) -> psycopg.Connection:
    return psycopg.connect(database_url, row_factory=dict_row, autocommit=False, options="-c timezone=UTC")


def run_migrations(conn: psycopg.Connection) -> list[str]:
    """Apply every migrations/NNNN_*.sql not yet recorded. Returns applied names."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        cur.execute("SELECT name FROM schema_migrations")
        done = {r["name"] for r in cur.fetchall()}
        applied: list[str] = []
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue
            cur.execute(path.read_text())
            cur.execute("INSERT INTO schema_migrations(name) VALUES (%s)", (path.name,))
            applied.append(path.name)
    conn.commit()
    return applied

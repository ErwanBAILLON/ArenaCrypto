"""Competitor registry: competitors, trials and trained model artefacts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from arena.core.types import CompetitorSpec

_SPEC_COLS = "id, name, family, version, params, role, status, parent_id, rationale, universe, gate_admitted"


def _spec(row: dict[str, Any]) -> CompetitorSpec:
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
        universe=row.get("universe", "crypto"),
        gate_admitted=bool(row.get("gate_admitted", False)),
    )


def insert_competitor(conn: psycopg.Connection, spec: CompetitorSpec) -> int:
    """Insert a competitor (name must be unique); returns its id."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO competitors (name, family, version, parent_id, params, role, status, rationale, universe,"
            " gate_admitted) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (
                spec.name,
                spec.family,
                spec.version,
                spec.parent_id,
                Jsonb(spec.params),
                spec.role,
                spec.status,
                spec.rationale,
                spec.universe,
                spec.gate_admitted,
            ),
        )
        return int(cur.fetchone()["id"])


def get_competitor(conn: psycopg.Connection, name: str) -> CompetitorSpec | None:
    """Competitor by unique name, or None."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_SPEC_COLS} FROM competitors WHERE name = %s", (name,))
        row = cur.fetchone()
    return _spec(row) if row else None


def list_competitors(
    conn: psycopg.Connection,
    statuses: Sequence[str] | None = None,
    roles: Sequence[str] | None = None,
    universe: str | None = None,
) -> list[CompetitorSpec]:
    """Competitors ordered by id, optionally filtered by status and/or role."""
    clauses: list[str] = []
    params: list[Any] = []
    if statuses is not None:
        clauses.append("status = ANY(%s)")
        params.append(list(statuses))
    if roles is not None:
        clauses.append("role = ANY(%s)")
        params.append(list(roles))
    if universe is not None:
        clauses.append("universe = %s")
        params.append(universe)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_SPEC_COLS} FROM competitors{where} ORDER BY id", params)
        return [_spec(r) for r in cur.fetchall()]


def set_status(conn: psycopg.Connection, competitor_id: int, status: str) -> None:
    """Change a competitor's lifecycle status."""
    with conn.cursor() as cur:
        cur.execute("UPDATE competitors SET status = %s WHERE id = %s", (status, competitor_id))


def count_trials(conn: psycopg.Connection, family: str, universe: str | None = None) -> int:
    """Number of trials recorded for a family in one arena (input to the deflated Sharpe).

    Scoped by universe: a daily-bar search on classic markets must not deflate
    the Sharpe of an hourly-bar crypto candidate, and vice versa.
    """
    sql = "SELECT count(*) AS n FROM trials WHERE family = %s"
    params: list[Any] = [family]
    if universe is not None:
        sql += " AND universe = %s"
        params.append(universe)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return int(cur.fetchone()["n"])


def add_trial(
    conn: psycopg.Connection,
    family: str,
    kind: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    verdict: str | None,
    competitor_id: int | None = None,
    notes: str = "",
    universe: str = "crypto",
    finished: bool = False,
) -> int:
    """Record a trial (backtest / walkforward / optimize / retrain); returns its id.

    ``finished`` stamps ``finished_at`` immediately, for the one-shot trials
    (an optimisation evaluation) that nobody will come back to close. A trial
    left open and never closed shows up forever as "running" on the dashboard.
    """
    finished_at = "now()" if finished else "NULL"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO trials (competitor_id, family, kind, params, metrics, verdict, notes, universe,"
            f" finished_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, {finished_at}) RETURNING id",
            (competitor_id, family, kind, Jsonb(params), Jsonb(metrics), verdict, notes, universe),
        )
        return int(cur.fetchone()["id"])


def abandon_stale_trials(conn: psycopg.Connection, older_than_hours: int = 12) -> int:
    """Close trials left open by a crashed or killed run; returns how many.

    A walk-forward writes its row before it starts so the deflated Sharpe of
    the next candidate already pays for it. If the process dies in between,
    the row stays open forever. Called at the start of every judging run.
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE trials SET verdict = 'abandoned', finished_at = now()"
            " WHERE verdict IS NULL AND finished_at IS NULL"
            " AND started_at < now() - make_interval(hours => %s)",
            (int(older_than_hours),),
        )
        return int(cur.rowcount)


def set_gate_admitted(conn: psycopg.Connection, competitor_id: int, admitted: bool) -> None:
    """Record that a competitor has (or has not) cleared the entry gate."""
    with conn.cursor() as cur:
        cur.execute("UPDATE competitors SET gate_admitted = %s WHERE id = %s", (admitted, competitor_id))


def finish_trial(conn: psycopg.Connection, trial_id: int, metrics: dict[str, Any], verdict: str | None) -> None:
    """Close a trial with its final metrics and verdict."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE trials SET metrics = %s, verdict = %s, finished_at = now() WHERE id = %s",
            (Jsonb(metrics), verdict, trial_id),
        )


def save_model(conn: psycopg.Connection, competitor_id: int, artifact: bytes, metrics: dict[str, Any]) -> int:
    """Store a serialised model for a competitor; returns the model id."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO models (competitor_id, artifact, metrics) VALUES (%s, %s, %s) RETURNING id",
            (competitor_id, artifact, Jsonb(metrics)),
        )
        return int(cur.fetchone()["id"])


def latest_model(conn: psycopg.Connection, competitor_id: int) -> tuple[bytes, dict[str, Any]] | None:
    """Most recently trained (artifact, metrics) for a competitor, or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT artifact, metrics FROM models WHERE competitor_id = %s ORDER BY trained_at DESC, id DESC LIMIT 1",
            (competitor_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return bytes(row["artifact"]), dict(row["metrics"])

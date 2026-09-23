"""Stored point-in-time universe membership.

Membership is computed once per rebalance date and read back, never recomputed
on the fly: a backtest and the live tick must agree on who was in the universe
on a given Monday, and a rule that is re-evaluated later is a rule that can be
re-evaluated with hindsight.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import psycopg

from arena.core.membership import Member


def write_members(conn: psycopg.Connection, universe: str, ts: datetime, members: list[Member]) -> int:
    """Replace the membership stored for ``ts``; returns how many rows were written."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM universe_members WHERE universe = %s AND ts = %s", (universe, ts))
        if not members:
            return 0
        cur.executemany(
            "INSERT INTO universe_members (universe, ts, symbol, rank, adv_usd, daily_vol)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            [(universe, ts, m.symbol, m.rank, m.adv_usd, m.daily_vol) for m in members],
        )
        return len(members)


def members_at(conn: psycopg.Connection, universe: str, ts: datetime) -> list[Member]:
    """Membership in force at ``ts``: the most recent rebalance at or before it."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol, rank, adv_usd, daily_vol FROM universe_members"
            " WHERE universe = %s AND ts = ("
            "   SELECT max(ts) FROM universe_members WHERE universe = %s AND ts <= %s)"
            " ORDER BY rank",
            (universe, universe, ts),
        )
        return [Member(r["symbol"], int(r["rank"]), float(r["adv_usd"]), float(r["daily_vol"])) for r in cur.fetchall()]


def membership_frame(conn: psycopg.Connection, universe: str) -> pd.DataFrame:
    """Every stored membership row, for backtests that need the whole schedule at once."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, symbol, rank, adv_usd, daily_vol FROM universe_members WHERE universe = %s ORDER BY ts, rank",
            (universe,),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["ts", "symbol", "rank", "adv_usd", "daily_vol"])
    return pd.DataFrame(rows)


def rebalance_timestamps(conn: psycopg.Connection, universe: str) -> list[datetime]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT ts FROM universe_members WHERE universe = %s ORDER BY ts", (universe,))
        return [r["ts"] for r in cur.fetchall()]


def coverage(conn: psycopg.Connection, universe: str) -> dict[str, Any]:
    """Summary used to sanity-check a build: dates, distinct symbols, churn."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(DISTINCT ts) AS dates, count(DISTINCT symbol) AS symbols, count(*) AS rows,"
            " min(ts) AS first_ts, max(ts) AS last_ts FROM universe_members WHERE universe = %s",
            (universe,),
        )
        row = dict(cur.fetchone())
    row["avg_size"] = (row["rows"] / row["dates"]) if row["dates"] else 0.0
    return row

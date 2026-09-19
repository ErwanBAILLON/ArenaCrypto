"""Market data repository: candles, funding, open interest, Hyperliquid snapshots.

Every function takes an open psycopg connection (dict_row factory) and leaves
transaction control to the caller. Frames are long-format with a tz-aware UTC
``ts`` column; reads always return tz-aware UTC timestamps.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pandas as pd
import psycopg

TF = "1h"

CANDLE_COLS = ["symbol", "ts", "open", "high", "low", "close", "volume"]


def _rows(df: pd.DataFrame, cols: Sequence[str]) -> list[tuple]:
    """Materialise frame columns as native Python tuples (psycopg cannot adapt numpy scalars)."""
    if df.empty:
        return []
    out = df.loc[:, list(cols)].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True).dt.to_pydatetime()
    return [tuple(r) for r in out.itertuples(index=False, name=None)]


def _insert_ignore(conn: psycopg.Connection, sql: str, rows: list[tuple]) -> int:
    """Batch an ``INSERT ... ON CONFLICT DO NOTHING`` and return the number of rows actually inserted."""
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
        return cur.rowcount


def _frame(rows: list[dict], cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=cols)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def upsert_candles(conn: psycopg.Connection, exchange: str, df: pd.DataFrame) -> int:
    """Insert 1h candles (columns symbol, ts, open, high, low, close, volume); returns rows inserted."""
    rows = [(exchange, s, TF, ts, o, h, l, c, v) for s, ts, o, h, l, c, v in _rows(df, CANDLE_COLS)]
    return _insert_ignore(
        conn,
        "INSERT INTO candles (exchange, symbol, tf, ts, open, high, low, close, volume)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
        rows,
    )


def upsert_funding(conn: psycopg.Connection, exchange: str, df: pd.DataFrame) -> int:
    """Insert funding rates (columns symbol, ts, rate); returns rows inserted."""
    rows = [(exchange, s, ts, r) for s, ts, r in _rows(df, ["symbol", "ts", "rate"])]
    return _insert_ignore(
        conn,
        "INSERT INTO funding (exchange, symbol, ts, rate) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
        rows,
    )


def upsert_open_interest(conn: psycopg.Connection, exchange: str, df: pd.DataFrame) -> int:
    """Insert open interest (columns symbol, ts, oi); returns rows inserted."""
    rows = [(exchange, s, ts, oi) for s, ts, oi in _rows(df, ["symbol", "ts", "oi"])]
    return _insert_ignore(
        conn,
        "INSERT INTO open_interest (exchange, symbol, ts, oi) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
        rows,
    )


def upsert_hl_snapshot(conn: psycopg.Connection, ts: datetime, df: pd.DataFrame) -> int:
    """Insert one Hyperliquid snapshot (columns coin, funding, oi, mark) taken at ``ts``."""
    if df.empty:
        return 0
    rows = [
        (ts, str(coin), float(f), float(oi), float(mark))
        for coin, f, oi, mark in df.loc[:, ["coin", "funding", "oi", "mark"]].itertuples(index=False, name=None)
    ]
    return _insert_ignore(
        conn,
        "INSERT INTO hl_snapshots (ts, coin, funding, oi, mark) VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
        rows,
    )


def read_candles(
    conn: psycopg.Connection, exchange: str, symbols: Sequence[str], start: datetime, end: datetime
) -> pd.DataFrame:
    """1h candles for ``symbols`` with ``start <= ts <= end``, ordered by (symbol, ts)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol, ts, open, high, low, close, volume FROM candles"
            " WHERE exchange = %s AND tf = %s AND symbol = ANY(%s) AND ts BETWEEN %s AND %s"
            " ORDER BY symbol, ts",
            (exchange, TF, list(symbols), start, end),
        )
        return _frame(cur.fetchall(), CANDLE_COLS)


def read_funding(
    conn: psycopg.Connection, exchange: str, symbols: Sequence[str], start: datetime, end: datetime
) -> pd.DataFrame:
    """Funding rates for ``symbols`` with ``start <= ts <= end``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol, ts, rate FROM funding"
            " WHERE exchange = %s AND symbol = ANY(%s) AND ts BETWEEN %s AND %s ORDER BY symbol, ts",
            (exchange, list(symbols), start, end),
        )
        return _frame(cur.fetchall(), ["symbol", "ts", "rate"])


def read_open_interest(
    conn: psycopg.Connection, exchange: str, symbols: Sequence[str], start: datetime, end: datetime
) -> pd.DataFrame:
    """Open interest for ``symbols`` with ``start <= ts <= end``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol, ts, oi FROM open_interest"
            " WHERE exchange = %s AND symbol = ANY(%s) AND ts BETWEEN %s AND %s ORDER BY symbol, ts",
            (exchange, list(symbols), start, end),
        )
        return _frame(cur.fetchall(), ["symbol", "ts", "oi"])


def latest_hl_funding(conn: psycopg.Connection) -> dict[str, float]:
    """coin -> funding from the most recent Hyperliquid snapshot (empty if none)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT coin, funding FROM hl_snapshots"
            " WHERE ts = (SELECT max(ts) FROM hl_snapshots) AND funding IS NOT NULL"
        )
        return {r["coin"]: float(r["funding"]) for r in cur.fetchall()}


def last_candle_ts(conn: psycopg.Connection, exchange: str, symbol: str) -> datetime | None:
    """Timestamp of the newest stored 1h candle for ``symbol``, or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT max(ts) AS ts FROM candles WHERE exchange = %s AND symbol = %s AND tf = %s",
            (exchange, symbol, TF),
        )
        row = cur.fetchone()
    return row["ts"] if row and row["ts"] is not None else None

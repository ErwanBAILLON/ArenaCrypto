"""What the ETF lab reads: the watchlist, and one daily close series per line.

The lab overlays as many series as the person wants, each rebased to 100 at
its own chosen start -- the same ETF twice on two different timelines is a
first-class case, so a series is addressed by (symbol, start, end), never by
symbol alone. Everything else (rebasing, overlays, fees, points) happens in
the browser on these plain arrays.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import psycopg
import yaml

WATCHLIST = Path(__file__).resolve().parents[2] / "config" / "watchlist.yaml"
EXCHANGE = "yahoo"
TF = "1d"


def watchlist(path: Path | None = None) -> list[dict]:
    raw = yaml.safe_load((path or WATCHLIST).read_text()) or {}
    out = []
    for t in raw.get("tickers", []):
        out.append(
            {
                "symbol": str(t["symbol"]),
                "label": str(t.get("label") or t["symbol"]),
                "currency": str(t.get("currency") or "USD"),
                "ttf": bool(t.get("ttf", False)),
            }
        )
    return out


def symbols(conn: psycopg.Connection, path: Path | None = None) -> list[dict]:
    """The watchlist with what is actually stored for each line (first and last bar, count)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol, min(ts) AS first, max(ts) AS last, count(*) AS n FROM candles"
            " WHERE exchange = %s AND tf = %s GROUP BY symbol",
            (EXCHANGE, TF),
        )
        stored = {r["symbol"]: r for r in cur.fetchall()}
    out = []
    for t in watchlist(path):
        s = stored.get(t["symbol"])
        out.append(
            {
                **t,
                "first": s["first"].date().isoformat() if s else None,
                "last": s["last"].date().isoformat() if s else None,
                "bars": int(s["n"]) if s else 0,
            }
        )
    # stored lines the watchlist forgot are still chartable
    known = {t["symbol"] for t in out}
    for sym, s in sorted(stored.items()):
        if sym not in known:
            out.append(
                {
                    "symbol": sym,
                    "label": sym,
                    "currency": "USD",
                    "ttf": False,
                    "first": s["first"].date().isoformat(),
                    "last": s["last"].date().isoformat(),
                    "bars": int(s["n"]),
                }
            )
    return out


def series(conn: psycopg.Connection, symbol: str, start: datetime | None, end: datetime | None) -> dict:
    """Daily closes of ``symbol`` in ``[start, end]`` as ``{symbol, t: [unix days], close: [...]}``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, close FROM candles WHERE exchange = %s AND tf = %s AND symbol = %s"
            " AND (%s::timestamptz IS NULL OR ts >= %s) AND (%s::timestamptz IS NULL OR ts <= %s)"
            " AND close IS NOT NULL AND close = close ORDER BY ts",
            (EXCHANGE, TF, symbol, start, start, end, end),
        )
        rows = cur.fetchall()
    return {
        "symbol": symbol,
        "t": [int(r["ts"].timestamp()) // 86400 for r in rows],
        "close": [round(float(r["close"]), 6) for r in rows],
    }

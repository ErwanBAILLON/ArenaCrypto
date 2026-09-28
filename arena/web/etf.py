"""What the ETF lab reads: the watchlist, and one daily close series per line.

The lab overlays as many series as the person wants, each rebased to 100 at
its own chosen start -- the same ETF twice on two different timelines is a
first-class case, so a series is addressed by (symbol, start, end), never by
symbol alone. Everything else (rebasing, overlays, fees, points) happens in
the browser on these plain arrays.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import psycopg
import yaml


def _default_watchlist() -> Path:
    """``WATCHLIST_PATH`` if set, else the image's /app/config, else the repo checkout next to the package."""
    env = os.environ.get("WATCHLIST_PATH")
    if env:
        return Path(env)
    for candidate in (
        Path("/app/config/watchlist.yaml"),
        Path(__file__).resolve().parents[2] / "config" / "watchlist.yaml",
    ):
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parents[2] / "config" / "watchlist.yaml"


WATCHLIST = _default_watchlist()
EXCHANGE = "yahoo"
TF = "1d"
TFS = {"1d": 86400, "5m": 300}  # granularities the lab serves, and their bar length in seconds


def watchlist(path: Path | None = None) -> list[dict]:
    p = path or WATCHLIST
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text()) or {}
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
        cur.execute(
            "SELECT symbol, min(ts) AS first, max(ts) AS last, count(*) AS n FROM candles"
            " WHERE exchange = %s AND tf = '5m' GROUP BY symbol",
            (EXCHANGE,),
        )
        intra = {r["symbol"]: r for r in cur.fetchall()}
    out = []
    for t in watchlist(path):
        s = stored.get(t["symbol"])
        i = intra.get(t["symbol"])
        out.append(
            {
                **t,
                "first": s["first"].date().isoformat() if s else None,
                "last": s["last"].date().isoformat() if s else None,
                "bars": int(s["n"]) if s else 0,
                "first_5m": i["first"].isoformat() if i else None,
                "last_5m": i["last"].isoformat() if i else None,
                "bars_5m": int(i["n"]) if i else 0,
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


def series(conn: psycopg.Connection, symbol: str, start: datetime | None, end: datetime | None, tf: str = TF) -> dict:
    """Closes of ``symbol`` in ``[start, end]`` at granularity ``tf``.

    ``t`` is in whole bars of that granularity since the epoch (days for 1d,
    five-minute slots for 5m), so the browser rebases and aligns on integers.
    """
    if tf not in TFS:
        raise ValueError(f"unknown granularity {tf!r}")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, close FROM candles WHERE exchange = %s AND tf = %s AND symbol = %s"
            " AND (%s::timestamptz IS NULL OR ts >= %s) AND (%s::timestamptz IS NULL OR ts <= %s)"
            " AND close IS NOT NULL AND close = close ORDER BY ts",
            (EXCHANGE, tf, symbol, start, start, end, end),
        )
        rows = cur.fetchall()
    step = TFS[tf]
    return {
        "symbol": symbol,
        "tf": tf,
        "step": step,
        "t": [int(r["ts"].timestamp()) // step for r in rows],
        "close": [round(float(r["close"]), 6) for r in rows],
    }

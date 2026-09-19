"""Daily digest: leaderboard, allocator opinion, challengers, drift of the day."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import psycopg

from arena.judge.metrics import max_drawdown, sharpe, total_return
from arena.runner import promotion
from arena.store import books as bstore
from arena.store import registry
from arena.telegram import templates

WINDOW = timedelta(days=30)


def build(conn: psycopg.Connection, now: datetime) -> str:
    specs = [s for s in registry.list_competitors(conn, statuses=["champion", "challenger"])]
    ids = [s.id for s in specs]
    rets = bstore.read_returns(conn, ids, now - WINDOW, now) if ids else pd.DataFrame()
    rows = []
    for s in specs:
        r = rets[s.id].dropna() if s.id in rets.columns else pd.Series(dtype=float)
        last = bstore.last_book_row(conn, s.id)
        rows.append(
            {
                "name": s.name,
                "status": s.status,
                "role": s.role,
                "sharpe_30d": sharpe(r) if len(r) > 24 else 0.0,
                "ret_30d": total_return(r) if len(r) else 0.0,
                "mdd_30d": max_drawdown(r) if len(r) else 0.0,
                "nav": last.nav if last else 0.0,
            }
        )
    rows.sort(key=lambda x: (x["role"] != "competitor", -x["sharpe_30d"]))
    names = {s.id: s.name for s in specs}
    alloc = [
        (names.get(cid, str(cid)), w) for cid, w in sorted(bstore.last_allocations(conn).items(), key=lambda kv: -kv[1])
    ]
    nulls = [s.id for s in specs if s.family == "null_random"]
    null95 = None
    if nulls and not rets.empty:
        null95 = promotion.null_threshold(rets[[c for c in nulls if c in rets.columns]], now - WINDOW)
    challengers = []
    for s in specs:
        if s.status != "challenger" or s.role != "competitor":
            continue
        with conn.cursor() as cur:
            cur.execute("SELECT min(ts) AS first FROM books WHERE competitor_id = %s", (s.id,))
            first = cur.fetchone()["first"]
            cur.execute("SELECT count(DISTINCT ts) AS n FROM targets WHERE competitor_id = %s AND weight <> 0", (s.id,))
            n = int(cur.fetchone()["n"])
        days = (pd.Timestamp(now) - pd.Timestamp(first)).days if first else 0
        challengers.append(
            {
                "name": s.name,
                "days_in_arena": days,
                "decisions": n,
                "progress": f"{days}/{promotion.MIN_DAYS}d, {n}/{promotion.MIN_DECISIONS} dec",
            }
        )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT kind, payload FROM alerts WHERE ts >= %s AND kind IN ('drift','stale','error','rejected') "
            "ORDER BY ts",
            (now - timedelta(days=1),),
        )
        drift_lines = [f"{r['kind']}: {r['payload'].get('detail', '')}" for r in cur.fetchall()]
    return templates.daily_digest(pd.Timestamp(now).strftime("%Y-%m-%d"), rows, alloc, null95, challengers, drift_lines)

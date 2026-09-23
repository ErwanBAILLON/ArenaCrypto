"""Read-only queries behind the dashboard.

Everything the pages show is computed here from the store; the routes only
render. Statistics reuse the same functions as the Telegram digest so both
views agree to the last digit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from arena.core.types import CompetitorSpec
from arena.explain import families, verdicts
from arena.judge.metrics import max_drawdown, sharpe, sharpe_se, total_return, track_record_verdict
from arena.runner import drift, promotion
from arena.runner.digest import WINDOW
from arena.store import books as bstore
from arena.store import health as hstore
from arena.store import registry
from arena.web import fr

STALE_AFTER = timedelta(hours=3)
CHART_DAYS = 90
MIN_BARS_FOR_STATS = 24  # below this nothing is computed at all, not even a point estimate
PROMOTION_CONFIDENCE = promotion.PROMOTION_CONFIDENCE
TRIAL_METRIC_KEYS = ("sharpe", "dsr", "bootstrap_p", "max_drawdown", "pbo", "fold_sharpes")
NULL_RANDOM_FAMILY = "null_random"


def infer_ppy(index: pd.Index) -> int:
    """Periods per year from the spacing of a book series (8760 hourly, 365 daily).

    Read from the data rather than configured, so the two arenas cannot drift
    apart: annualising daily classic-market bars by 8760 inflates every Sharpe
    on the page by a factor of five.
    """
    idx = pd.DatetimeIndex(index)
    if len(idx) < 3:
        return int(promotion.PPY_HOURLY)
    gap_hours = float(pd.Series(idx).diff().dt.total_seconds().median() or 3600.0) / 3600.0
    if not np.isfinite(gap_hours) or gap_hours <= 0:
        return int(promotion.PPY_HOURLY)
    return max(1, int(round(365.0 * 24.0 / gap_hours)))


@dataclass(frozen=True)
class Leaderboard:
    """Everything the front page needs, in one object."""

    rows: list[dict[str, Any]]
    allocation: list[tuple[str, float]]
    null95: float | None
    last_bar: datetime | None
    stale: bool
    chart_ids: list[int] = field(default_factory=list)
    null95_by_universe: dict[str, float | None] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """JSON-serialisable view (timestamps as ISO 8601)."""
        return {
            "rows": self.rows,
            "allocation": [{"name": n, "weight": w} for n, w in self.allocation],
            "null95": self.null95,
            "null95_by_universe": self.null95_by_universe,
            "last_bar": self.last_bar.isoformat() if self.last_bar else None,
            "stale": self.stale,
        }


@dataclass(frozen=True)
class CompetitorDetail:
    spec: CompetitorSpec
    parent_name: str | None
    first_ts: datetime | None
    current_ts: datetime | None
    current_targets: list[dict[str, Any]]
    recent_targets: list[dict[str, Any]]
    trials: list[dict[str, Any]]
    model_metrics: dict[str, Any] | None


def last_bar(conn: psycopg.Connection) -> datetime | None:
    """Timestamp of the most recent book row across all competitors, or None."""
    with conn.cursor() as cur:
        cur.execute("SELECT max(ts) AS ts FROM books")
        return cur.fetchone()["ts"]


def is_stale(last: datetime | None, now: datetime) -> bool:
    """True when no book row is younger than ``STALE_AFTER``."""
    return last is None or (now - last) > STALE_AFTER


def leaderboard(conn: psycopg.Connection, now: datetime) -> Leaderboard:
    """The 30-day board, ranked on evidence rather than on a point estimate.

    Each row carries its Sharpe **and** the standard error of that Sharpe, plus
    ``psr`` = P(true Sharpe > the null models' 95th percentile) on the
    autocorrelation-adjusted sample size, and how many bars are still missing
    before that question can be answered. A four-day arena produces Sharpe
    values around 15 with a standard error around 9; ranking on the number
    alone put a coin flip in third place, which is why the board now sorts on
    ``psr`` and shows the interval next to every estimate.

    The thirty ``null_random`` competitors are collapsed into one row: they are
    a distribution, not thirty contestants.
    """
    specs = registry.list_competitors(conn, statuses=["champion", "challenger"])
    ids = [s.id for s in specs]
    rets = bstore.read_returns(conn, ids, now - WINDOW, now) if ids else pd.DataFrame()
    series = {s.id: (rets[s.id].dropna() if s.id in rets.columns else pd.Series(dtype=float)) for s in specs}

    null95_by_universe: dict[str, float | None] = {}
    for universe in {s.universe for s in specs}:
        cols = [s.id for s in specs if s.universe == universe and s.family == NULL_RANDOM_FAMILY and s.id in rets]
        ppy = infer_ppy(rets[cols[0]].dropna().index) if cols and len(series[cols[0]]) > 2 else promotion.PPY_HOURLY
        null95_by_universe[universe] = promotion.null_threshold(rets[cols], now - WINDOW, ppy) if cols else None

    rows: list[dict[str, Any]] = []
    for s in specs:
        if s.family == NULL_RANDOM_FAMILY:
            continue  # collapsed below
        r = series[s.id]
        last = bstore.last_book_row(conn, s.id)
        rows.append(
            {
                **_row_stats(r, null95_by_universe.get(s.universe)),
                "id": s.id,
                "name": s.name,
                "family": s.family,
                "version": s.version,
                "status": s.status,
                "role": s.role,
                "universe": s.universe,
                "nav": last.nav if last else 0.0,
                "count": 1,
            }
        )
    rows.extend(_null_rows(specs, series, null95_by_universe))
    # evidence first, then the point estimate; a row with no evidence yet falls back on its book
    rows.sort(
        key=lambda x: (
            x["universe"] != "crypto",
            x["role"] != "competitor",
            -(x["psr"] if x["psr"] is not None else -1.0),
            -(x["sharpe_30d"] if x["sharpe_30d"] is not None else -1e9),
        )
    )
    names = {s.id: s.name for s in specs}
    alloc = [
        (names.get(cid, str(cid)), w) for cid, w in sorted(bstore.last_allocations(conn).items(), key=lambda kv: -kv[1])
    ]
    last = last_bar(conn)
    chart_ids = [s.id for s in specs if s.role == "benchmark" or (s.role == "competitor" and s.status == "champion")]
    return Leaderboard(
        rows,
        alloc,
        null95_by_universe.get("crypto"),
        last,
        is_stale(last, now),
        chart_ids,
        null95_by_universe,
    )


def _row_stats(r: pd.Series, null95: float | None) -> dict[str, Any]:
    """Point estimate, its standard error, and what it would take to conclude."""
    if len(r) <= MIN_BARS_FOR_STATS:
        return {
            "sharpe_30d": None,
            "sharpe_se_30d": None,
            "psr": None,
            "ret_30d": total_return(r) if len(r) else 0.0,
            "mdd_30d": max_drawdown(r) if len(r) else 0.0,
            "bars_30d": int(len(r)),
            "bars_needed": None,
            "bars_missing": None,
            "days_missing": None,
            "conclusive": False,
        }
    ppy = infer_ppy(r.index)
    verdict = track_record_verdict(r, null95 or 0.0, PROMOTION_CONFIDENCE, ppy)
    missing = verdict["missing"]
    return {
        "sharpe_30d": verdict["sharpe"],
        "sharpe_se_30d": sharpe_se(r, ppy),
        "psr": verdict["psr"],
        "ret_30d": total_return(r),
        "mdd_30d": max_drawdown(r),
        "bars_30d": int(len(r)),
        "bars_needed": None if not np.isfinite(verdict["needed"]) else verdict["needed"],
        "bars_missing": None if not np.isfinite(missing) else missing,
        "days_missing": None if not np.isfinite(missing) else missing / max(1.0, ppy / 365.0),
        "conclusive": bool(verdict["conclusive"]),
    }


def _null_rows(specs, series, null95_by_universe) -> list[dict[str, Any]]:
    """One row per arena standing for its whole random-model distribution."""
    out: list[dict[str, Any]] = []
    for universe in sorted({s.universe for s in specs if s.family == NULL_RANDOM_FAMILY}):
        members = [s for s in specs if s.universe == universe and s.family == NULL_RANDOM_FAMILY]
        usable = [series[s.id] for s in members if len(series[s.id]) > MIN_BARS_FOR_STATS]
        if not usable:
            continue
        ppy = infer_ppy(usable[0].index)
        sharpes = [sharpe(r, ppy) for r in usable]
        out.append(
            {
                "id": members[0].id,
                "name": f"{len(members)} tirages",
                "family": NULL_RANDOM_FAMILY,
                "version": 1,
                "status": "champion",
                "role": "null",
                "universe": universe,
                "sharpe_30d": float(np.median(sharpes)),
                "sharpe_se_30d": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else None,
                "psr": None,
                "ret_30d": float(np.median([total_return(r) for r in usable])),
                "mdd_30d": float(np.median([max_drawdown(r) for r in usable])),
                "nav": 0.0,
                "bars_30d": int(max(len(r) for r in usable)),
                "bars_needed": None,
                "bars_missing": None,
                "days_missing": None,
                "conclusive": False,
                "count": len(members),
                "null95": null95_by_universe.get(universe),
            }
        )
    return out


def maturity(conn: psycopg.Connection, board: Leaderboard, now: datetime) -> dict[str, Any]:
    """How old the arena is, how much it has proven, and when it could first promote.

    The front page opens with this because everything under it is a number that
    looks like a result and is not one yet: on four days of hourly bars the
    standard error of a Sharpe is larger than the Sharpe. Stating the age of the
    experiment next to its output is the cheapest honesty available.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT min(b.ts) AS first FROM books b JOIN competitors c ON c.id = b.competitor_id"
            " WHERE c.role = 'competitor'"
        )
        first = cur.fetchone()["first"]
    competitors = [r for r in board.rows if r["role"] == "competitor"]
    eligible = None
    if first is not None:
        eligible = pd.Timestamp(first) + timedelta(days=promotion.MIN_DAYS)
        eligible = eligible.to_pydatetime()
    return {
        "days": (pd.Timestamp(now) - pd.Timestamp(first)).days if first is not None else 0,
        "first_book": first,
        "first_eligible": eligible if eligible is not None and eligible > now else None,
        "proven": sum(1 for r in competitors if r["conclusive"]),
        "competitors": len(competitors),
        "min_days": promotion.MIN_DAYS,
        "min_decisions": promotion.MIN_DECISIONS,
    }


def nav_series(conn: psycopg.Connection, ids: list[int], start: datetime, end: datetime) -> dict[str, pd.Series]:
    """``name -> NAV / NAV[0]`` for each competitor with at least one book row in the window."""
    names = {s.id: s.name for s in registry.list_competitors(conn)}
    out: dict[str, pd.Series] = {}
    for cid in ids:
        nav = bstore.read_nav(conn, cid, start, end)
        if nav.empty or nav.iloc[0] == 0:
            continue
        out[names.get(cid, str(cid))] = nav / nav.iloc[0]
    return out


def _competitor_by_id(conn: psycopg.Connection, competitor_id: int) -> CompetitorSpec | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, family, version, params, role, status, parent_id, rationale"
            " FROM competitors WHERE id = %s",
            (competitor_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
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
    )


def _trial_row(r: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(r["metrics"])
    return {
        "id": int(r["id"]),
        "competitor_id": r["competitor_id"],
        "family": r["family"],
        "kind": r["kind"],
        "verdict": r["verdict"],
        "started_at": r["started_at"],
        "finished_at": r["finished_at"],
        "notes": r["notes"],
        "metrics": metrics,
        "highlights": {k: metrics.get(k) for k in TRIAL_METRIC_KEYS},
    }


def competitor_detail(conn: psycopg.Connection, competitor_id: int) -> CompetitorDetail | None:
    """Spec, targets, family trials and latest model metrics; None when the id is unknown."""
    spec = _competitor_by_id(conn, competitor_id)
    if spec is None:
        return None
    parent = _competitor_by_id(conn, spec.parent_id) if spec.parent_id is not None else None
    with conn.cursor() as cur:
        cur.execute("SELECT min(ts) AS first FROM books WHERE competitor_id = %s", (competitor_id,))
        first_ts = cur.fetchone()["first"]
        cur.execute(
            "SELECT ts, symbol, weight, kind, conviction, reason FROM targets"
            " WHERE competitor_id = %s AND ts = (SELECT max(ts) FROM targets WHERE competitor_id = %s)"
            " ORDER BY abs(weight) DESC, symbol",
            (competitor_id, competitor_id),
        )
        current = [dict(r) for r in cur.fetchall()]
        cur.execute(
            "SELECT ts, symbol, weight, kind, conviction, reason FROM targets"
            " WHERE competitor_id = %s ORDER BY ts DESC, symbol LIMIT 50",
            (competitor_id,),
        )
        recent = [dict(r) for r in cur.fetchall()]
        cur.execute(
            "SELECT id, competitor_id, family, kind, verdict, started_at, finished_at, notes, metrics"
            " FROM trials WHERE family = %s ORDER BY started_at DESC, id DESC LIMIT 50",
            (spec.family,),
        )
        trials_rows = [_trial_row(r) for r in cur.fetchall()]
        model_metrics = None
        if spec.family == "meta_label":
            cur.execute(
                "SELECT metrics FROM models WHERE competitor_id = %s ORDER BY trained_at DESC, id DESC LIMIT 1",
                (competitor_id,),
            )
            row = cur.fetchone()
            model_metrics = dict(row["metrics"]) if row else None
    return CompetitorDetail(
        spec=spec,
        parent_name=parent.name if parent else None,
        first_ts=first_ts,
        current_ts=current[0]["ts"] if current else None,
        current_targets=current,
        recent_targets=recent,
        trials=trials_rows,
        model_metrics=model_metrics,
    )


def alerts(conn: psycopg.Connection, limit: int = 200) -> list[dict[str, Any]]:
    """Most recent alerts with the competitor name resolved and a one-line detail."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT a.id, a.ts, a.kind, a.competitor_id, c.name AS competitor, a.symbol, a.payload, a.sent_at"
            " FROM alerts a LEFT JOIN competitors c ON c.id = a.competitor_id"
            " ORDER BY a.ts DESC, a.id DESC LIMIT %s",
            (limit,),
        )
        rows = []
        for r in cur.fetchall():
            payload = dict(r["payload"])
            rows.append({**r, "payload": payload, "detail": payload.get("detail") or _kv(payload)})
        return rows


SEARCH_KIND = "optimize"


def trials(conn: psycopg.Connection, limit: int = 200, include_search: bool = False) -> list[dict[str, Any]]:
    """Most recent trials with metric highlights extracted.

    A weekly search writes one row per evaluation -- fifteen per family, five
    families, every Sunday -- and none of them is a verdict on anything. They
    are hidden unless asked for, so the page shows the gate decisions it is
    named after rather than the sampling that led to them.
    """
    where = "" if include_search else " WHERE kind <> %(search)s"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, competitor_id, family, kind, verdict, started_at, finished_at, notes, metrics"
            f" FROM trials{where} ORDER BY started_at DESC, id DESC LIMIT %(limit)s",
            {"limit": limit, "search": SEARCH_KIND},
        )
        return [_trial_row(r) for r in cur.fetchall()]


def search_trial_count(conn: psycopg.Connection) -> int:
    """How many parameter-search evaluations are hidden behind the filter."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM trials WHERE kind = %s", (SEARCH_KIND,))
        return int(cur.fetchone()["n"])


def _kv(d: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


# ---------------------------------------------------------------------------
# « En direct » : tick runs, positions, P&L in euros, data freshness, activity
# ---------------------------------------------------------------------------

NAV0 = 10_000.0
BENCH_BTC_FAMILY = "bench_btc_hold"
DIFF_LOOKBACK = timedelta(hours=24)
ACTIONS_LIMIT = 30
ACTIVITY_TICKS = 48
ACTIVITY_ALERTS = 20
_WINDOWS = (("today", None), ("7d", timedelta(days=7)), ("30d", timedelta(days=30)), ("all", None))


def last_tick(conn: psycopg.Connection) -> dict[str, Any] | None:
    """Most recent ``tick_runs`` row, or None when the runner never ran."""
    rows = ticks(conn, 1)
    return rows[0] if rows else None


def ticks(conn: psycopg.Connection, limit: int = ACTIVITY_TICKS) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, bar_ts, started_at, finished_at, booked, skipped, failed, changes, alerts, ingested, ok"
            " FROM tick_runs ORDER BY started_at DESC, id DESC LIMIT %s",
            (limit,),
        )
        return [{**r, "failed": list(r["failed"] or []), "ingested": dict(r["ingested"] or {})} for r in cur.fetchall()]


def _grouped_targets(rows: list[dict[str, Any]]) -> dict[datetime, dict[str, dict[str, Any]]]:
    """``ts -> symbol -> target row`` from flat target rows."""
    out: dict[datetime, dict[str, dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["ts"], {})[r["symbol"]] = dict(r)
    return out


def _targets_rows(conn: psycopg.Connection, cid: int, upto: datetime | None = None, limit: int | None = None) -> list:
    sql = "SELECT ts, symbol, weight, kind, conviction, reason FROM targets WHERE competitor_id = %s"
    params: list[Any] = [cid]
    if upto is not None:
        sql += " AND ts <= %s"
        params.append(upto)
    sql += " ORDER BY ts DESC, abs(weight) DESC, symbol"
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def latest_targets(conn: psycopg.Connection, cid: int, upto: datetime | None = None) -> tuple[datetime | None, dict]:
    """``(ts, symbol -> row)`` of the last decision at or before ``upto`` (empty when none)."""
    with conn.cursor() as cur:
        if upto is None:
            cur.execute("SELECT max(ts) AS ts FROM targets WHERE competitor_id = %s", (cid,))
        else:
            cur.execute("SELECT max(ts) AS ts FROM targets WHERE competitor_id = %s AND ts <= %s", (cid, upto))
        ts = cur.fetchone()["ts"]
        if ts is None:
            return None, {}
        cur.execute(
            "SELECT ts, symbol, weight, kind, conviction, reason FROM targets"
            " WHERE competitor_id = %s AND ts = %s ORDER BY abs(weight) DESC, symbol",
            (cid, ts),
        )
        return ts, {r["symbol"]: dict(r) for r in cur.fetchall()}


def _sign(w: float) -> int:
    return (w > 0) - (w < 0)


def first_ts_of_position(conn: psycopg.Connection, cid: int, symbol: str, weight: float, upto: datetime) -> datetime:
    """Earliest decision timestamp of the *contiguous* run where ``symbol`` was held with this sign, up to ``upto``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT d.ts, (t.symbol IS NOT NULL) AS held"
            " FROM (SELECT DISTINCT ts FROM targets WHERE competitor_id = %s AND ts <= %s) d"
            " LEFT JOIN targets t ON t.competitor_id = %s AND t.ts = d.ts AND t.symbol = %s AND sign(t.weight) = %s"
            " ORDER BY d.ts DESC",
            (cid, upto, cid, symbol, _sign(weight)),
        )
        since = upto
        for r in cur.fetchall():
            if not r["held"]:
                break
            since = r["ts"]
        return since


def positions_now(conn: psycopg.Connection, cid: int, family: str, now: datetime, nav: float | None) -> list[dict]:
    """Current positions with the French sentence, holding time and value in euros."""
    ts, rows = latest_targets(conn, cid)
    out: list[dict[str, Any]] = []
    for sym, r in rows.items():
        w = float(r["weight"])
        since = first_ts_of_position(conn, cid, sym, w, ts)
        out.append(
            {
                "symbol": sym,
                "weight": w,
                "kind": r["kind"],
                "conviction": float(r["conviction"]),
                "reason": dict(r["reason"] or {}),
                "ts": ts,
                "since": since,
                "held_for": now - since,
                "value_eur": abs(w) * float(nav) if nav else None,
                "sentence": fr.position_sentence(family, sym, w, r["kind"], r["reason"]),
            }
        )
    return out


def _diff(family: str, before: dict[str, dict], after: dict[str, dict], ts: datetime | None) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for sym, r in after.items():
        w_new = float(r["weight"])
        w_old = float(before[sym]["weight"]) if sym in before else 0.0
        if w_old == 0.0 or _sign(w_old) != _sign(w_new):
            change = "entrée"
        elif abs(w_new - w_old) >= 1e-9:
            change = "taille"
        else:
            continue
        lines.append(
            {
                "ts": ts,
                "symbol": sym,
                "change": change,
                "weight_before": w_old,
                "weight_after": w_new,
                "sentence": fr.position_sentence(family, sym, w_new, r["kind"], r["reason"]),
            }
        )
    for sym, r in before.items():
        if sym not in after:
            lines.append(
                {
                    "ts": ts,
                    "symbol": sym,
                    "change": "sortie",
                    "weight_before": float(r["weight"]),
                    "weight_after": 0.0,
                    "sentence": fr.exit_sentence(family, sym),
                }
            )
    return lines


def positions_diff_24h(conn: psycopg.Connection, cid: int, family: str, now: datetime) -> list[dict[str, Any]]:
    """Entries, exits and resizes between the last decision and the last one at least 24 h older."""
    ts_now, after = latest_targets(conn, cid, now)
    if ts_now is None:
        return []
    ts_old, before = latest_targets(conn, cid, now - DIFF_LOOKBACK)
    if ts_old == ts_now:
        return []
    return _diff(family, before, after, ts_now)


def target_changes(conn: psycopg.Connection, cid: int, family: str, limit: int = ACTIONS_LIMIT) -> list[dict]:
    """The last ``limit`` target changes (entries, exits, resizes) as dated French sentences, newest first."""
    grouped = _grouped_targets(_targets_rows(conn, cid, limit=4000))
    stamps = sorted(grouped, reverse=True)
    out: list[dict[str, Any]] = []
    for i, ts in enumerate(stamps):
        before = grouped[stamps[i + 1]] if i + 1 < len(stamps) else {}
        out.extend(_diff(family, before, grouped[ts], ts))
        if len(out) >= limit:
            break
    return out[:limit]


def _nav_rows(conn: psycopg.Connection, cid: int, upto: datetime) -> list[tuple[datetime, float]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, nav FROM books WHERE competitor_id = %s AND ts <= %s ORDER BY ts",
            (cid, upto),
        )
        return [(r["ts"], float(r["nav"])) for r in cur.fetchall()]


def pnl_windows(conn: psycopg.Connection, cid: int, now: datetime) -> dict[str, Any]:
    """P&L today / 7 d / 30 d / since start as ``{"frac", "eur"}`` on the virtual book, plus ``nav`` and ``nav_ts``.

    The base of each window is the last NAV at or before its start (or the first
    row when the book is younger than the window). Windows without data are None.
    """
    rows = _nav_rows(conn, cid, now)
    out: dict[str, Any] = {k: None for k, _ in _WINDOWS}
    if not rows:
        return {**out, "nav": None, "nav_ts": None, "first_ts": None}
    ts_last, nav_last = rows[-1]
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for key, span in _WINDOWS:
        if key == "all":
            base = rows[0][1]
        else:
            start = day_start if key == "today" else now - span
            earlier = [nav for ts, nav in rows if ts <= start]
            base = earlier[-1] if earlier else rows[0][1]
        if base <= 0:
            continue
        out[key] = {"frac": nav_last / base - 1.0, "eur": nav_last - base}
    return {**out, "nav": nav_last, "nav_ts": ts_last, "first_ts": rows[0][0]}


def benchmark_pnl(conn: psycopg.Connection, now: datetime) -> dict[str, Any] | None:
    """P&L windows of the « garder du BTC » benchmark book, or None when absent."""
    bench = [s for s in registry.list_competitors(conn) if s.family == BENCH_BTC_FAMILY and s.id is not None]
    if not bench:
        return None
    return pnl_windows(conn, bench[0].id, now)


def sources_freshness(conn: psycopg.Connection, now: datetime) -> list[dict[str, Any]]:
    """One row per data source, from ``feed_health``: what we asked, when, and what came back.

    Read from the ingestion log rather than from ``max(fetched_at)`` of each
    data table. A source that legitimately writes nothing -- the macro calendar
    of a week already stored -- is healthy, and used to show up as three days
    late.
    """
    rows = hstore.read_all(conn)
    known = {r["source"]: r for r in rows}
    out: list[dict[str, Any]] = []
    for key in hstore.SOURCES:
        r = known.get(key)
        last_ok = r["last_ok_at"] if r else None
        age = (now - last_ok) if last_ok else None
        limit = drift.feed_max_age(key)
        out.append(
            {
                "key": key,
                "label": fr.SOURCE_FR[key],
                "data_ts": r["last_data_ts"] if r else None,
                "fetched": r["last_fetch_at"] if r else None,
                "last_ok": last_ok,
                "age": age,
                "limit_hours": limit,
                "level": fr.freshness_level(age, limit),
                "ok": bool(r["ok"]) if r else False,
                "detail": (r["detail"] if r else "") or "",
                "count_24h": _articles_24h(conn, now) if key == "articles" else None,
            }
        )
    return out


def _articles_24h(conn: psycopg.Connection, now: datetime) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM articles WHERE fetched_at > %s", (now - timedelta(hours=24),))
        return int(cur.fetchone()["n"])


def recent_alerts_fr(conn: psycopg.Connection, limit: int = ACTIVITY_ALERTS) -> list[dict[str, Any]]:
    """Latest alerts with a French label and one-line sentence (signals rendered via ``explain_target``)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT a.id, a.ts, a.kind, a.competitor_id, c.name AS competitor, c.family, a.symbol, a.payload, a.sent_at"
            " FROM alerts a LEFT JOIN competitors c ON c.id = a.competitor_id"
            " ORDER BY a.ts DESC, a.id DESC LIMIT %s",
            (limit,),
        )
        rows = []
        for r in cur.fetchall():
            payload = dict(r["payload"] or {})
            rows.append(
                {
                    **r,
                    "payload": payload,
                    "kind_fr": fr.kind_fr(r["kind"]),
                    "sentence": fr.alert_sentence(r["kind"], r["family"], r["symbol"], payload),
                }
            )
        return rows


def activity(conn: psycopg.Connection, n_ticks: int = ACTIVITY_TICKS, n_alerts: int = ACTIVITY_ALERTS) -> dict:
    return {"ticks": ticks(conn, n_ticks), "alerts": recent_alerts_fr(conn, n_alerts)}


def champions_live(conn: psycopg.Connection, now: datetime) -> list[dict[str, Any]]:
    """« Qui parle en ce moment » : each champion model with positions, holding times and P&L in euros."""
    out: list[dict[str, Any]] = []
    for s in registry.list_competitors(conn, statuses=["champion"]):
        if s.role != "competitor" or s.id is None:
            continue
        pnl = pnl_windows(conn, s.id, now)
        out.append(
            {
                "spec": s,
                "card": families.card(s.family),
                "universe": s.universe,
                "pnl": pnl,
                "positions": positions_now(conn, s.id, s.family, now, pnl["nav"]),
            }
        )
    return out


def changes_24h(conn: psycopg.Connection, now: datetime) -> list[dict[str, Any]]:
    """« Ce qui a changé depuis 24 h » : one block per model with at least one entry, exit or resize."""
    out: list[dict[str, Any]] = []
    for s in registry.list_competitors(conn, statuses=["champion", "challenger"]):
        if s.role != "competitor" or s.id is None:
            continue
        lines = positions_diff_24h(conn, s.id, s.family, now)
        if lines:
            out.append({"spec": s, "card": families.card(s.family), "lines": lines})
    return out


@dataclass(frozen=True)
class LivePage:
    tick: dict[str, Any] | None
    last_bar: datetime | None
    stale: bool
    champions: list[dict[str, Any]]
    changes: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    ticks: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
    bench: dict[str, Any] | None

    @property
    def alarm(self) -> str | None:
        """French reason for the red banner, or None when everything is fine."""
        if self.tick is None:
            return "Aucun passage du système enregistré : il n'a encore jamais tourné."
        if self.tick["failed"]:
            names = ", ".join(self.tick["failed"])
            return f"Le dernier passage a échoué pour : {names}. Les autres modèles ont été évalués normalement."
        if self.stale:
            return "Aucune évaluation depuis plus de 3 heures : les données ou le système sont à l'arrêt."
        return None


def live_page(conn: psycopg.Connection, now: datetime) -> LivePage:
    last = last_bar(conn)
    act = activity(conn)
    return LivePage(
        tick=last_tick(conn),
        last_bar=last,
        stale=is_stale(last, now),
        champions=champions_live(conn, now),
        changes=changes_24h(conn, now),
        sources=sources_freshness(conn, now),
        ticks=act["ticks"],
        alerts=act["alerts"],
        bench=benchmark_pnl(conn, now),
    )


# ---------------------------------------------------------------------------
# « Fiche modèle » and « Les familles »
# ---------------------------------------------------------------------------


def gate_trial(conn: psycopg.Connection, spec: CompetitorSpec) -> dict[str, Any] | None:
    """Latest trial that judged this competitor (walk-forward first), else its family with the same params."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, competitor_id, family, kind, verdict, started_at, finished_at, notes, metrics"
            " FROM trials WHERE competitor_id = %s"
            " OR (competitor_id IS NULL AND family = %s AND params = %s::jsonb)"
            " ORDER BY (kind = 'walkforward') DESC, (competitor_id IS NOT NULL) DESC, started_at DESC, id DESC LIMIT 1",
            (spec.id, spec.family, Jsonb(spec.params)),
        )
        row = cur.fetchone()
    return _trial_row(row) if row else None


def challenger_progress(conn: psycopg.Connection, cid: int, now: datetime) -> dict[str, Any]:
    """Days in the arena and decisions taken, against the promotion thresholds."""
    with conn.cursor() as cur:
        cur.execute("SELECT min(ts) AS first FROM books WHERE competitor_id = %s", (cid,))
        first = cur.fetchone()["first"]
        cur.execute("SELECT count(DISTINCT ts) AS n FROM targets WHERE competitor_id = %s AND weight <> 0", (cid,))
        n = int(cur.fetchone()["n"])
    days = (pd.Timestamp(now) - pd.Timestamp(first)).days if first else 0
    return {"days": days, "decisions": n, "min_days": promotion.MIN_DAYS, "min_decisions": promotion.MIN_DECISIONS}


@dataclass(frozen=True)
class CompetitorPage:
    spec: CompetitorSpec
    card: families.FamilyCard
    parent_name: str | None
    params_words: list[str]
    status_sentence: str
    progress: dict[str, Any]
    gate: dict[str, Any] | None
    gate_lines: list[str]
    robustness: dict[str, Any] | None
    positions: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    pnl: dict[str, Any]
    bench: dict[str, Any] | None
    model_metrics: dict[str, Any] | None

    @property
    def first_ts(self) -> datetime | None:
        return self.pnl.get("first_ts")


def competitor_page(conn: psycopg.Connection, competitor_id: int, now: datetime) -> CompetitorPage | None:
    spec = _competitor_by_id(conn, competitor_id)
    if spec is None:
        return None
    parent = _competitor_by_id(conn, spec.parent_id) if spec.parent_id is not None else None
    progress = challenger_progress(conn, competitor_id, now)
    gate = gate_trial(conn, spec)
    metrics = gate["metrics"] if gate else {}
    robustness = metrics.get("robustness") if isinstance(metrics.get("robustness"), dict) else None
    pnl = pnl_windows(conn, competitor_id, now)
    model_metrics = None
    if spec.family == "meta_label":
        found = registry.latest_model(conn, competitor_id)
        model_metrics = dict(found[1]) if found else None
    return CompetitorPage(
        spec=spec,
        card=families.card(spec.family),
        parent_name=parent.name if parent else None,
        params_words=families.params_in_words(spec.family, spec.params),
        status_sentence=verdicts.explain_status(spec.status, spec.role, progress["days"], progress["decisions"]),
        progress=progress,
        gate=gate,
        gate_lines=verdicts.explain_gate(metrics, metrics.get("failed")) if gate else [],
        robustness=robustness,
        positions=positions_now(conn, competitor_id, spec.family, now, pnl["nav"]),
        actions=target_changes(conn, competitor_id, spec.family),
        pnl=pnl,
        bench=benchmark_pnl(conn, now) if spec.family != BENCH_BTC_FAMILY else None,
        model_metrics=model_metrics,
    )


def families_overview(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """Every documented family with the competitors that exist in it (all statuses)."""
    by_family: dict[str, list[CompetitorSpec]] = {}
    for s in registry.list_competitors(conn):
        by_family.setdefault(s.family, []).append(s)
    out = [{"card": c, "competitors": by_family.pop(k, [])} for k, c in families.CARDS.items()]
    for k, specs in by_family.items():  # families present in the store but not documented
        out.append({"card": families.card(k), "competitors": specs})
    return out

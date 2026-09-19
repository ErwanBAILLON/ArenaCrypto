"""Read-only queries behind the dashboard.

Everything the pages show is computed here from the store; the routes only
render. Statistics reuse the same functions as the Telegram digest so both
views agree to the last digit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from arena.core.types import CompetitorSpec
from arena.explain import families, verdicts
from arena.judge.metrics import max_drawdown, sharpe, total_return
from arena.runner import promotion
from arena.runner.digest import WINDOW
from arena.store import books as bstore
from arena.store import registry
from arena.web import fr

STALE_AFTER = timedelta(hours=3)
CHART_DAYS = 90
MIN_BARS_FOR_SHARPE = 24
TRIAL_METRIC_KEYS = ("sharpe", "dsr", "bootstrap_p", "max_drawdown", "fold_sharpes")


@dataclass(frozen=True)
class Leaderboard:
    """Everything the front page needs, in one object."""

    rows: list[dict[str, Any]]
    allocation: list[tuple[str, float]]
    null95: float | None
    last_bar: datetime | None
    stale: bool
    chart_ids: list[int] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """JSON-serialisable view (timestamps as ISO 8601)."""
        return {
            "rows": self.rows,
            "allocation": [{"name": n, "weight": w} for n, w in self.allocation],
            "null95": self.null95,
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
    """Same rows and ordering as the daily digest, plus ids and staleness."""
    specs = registry.list_competitors(conn, statuses=["champion", "challenger"])
    ids = [s.id for s in specs]
    rets = bstore.read_returns(conn, ids, now - WINDOW, now) if ids else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for s in specs:
        r = rets[s.id].dropna() if s.id in rets.columns else pd.Series(dtype=float)
        last = bstore.last_book_row(conn, s.id)
        rows.append(
            {
                "id": s.id,
                "name": s.name,
                "family": s.family,
                "version": s.version,
                "status": s.status,
                "role": s.role,
                "sharpe_30d": sharpe(r) if len(r) > MIN_BARS_FOR_SHARPE else 0.0,
                "ret_30d": total_return(r) if len(r) else 0.0,
                "mdd_30d": max_drawdown(r) if len(r) else 0.0,
                "nav": last.nav if last else 0.0,
                "bars_30d": int(len(r)),
            }
        )
    rows.sort(key=lambda x: (x["role"] != "competitor", -x["sharpe_30d"]))
    names = {s.id: s.name for s in specs}
    alloc = [
        (names.get(cid, str(cid)), w) for cid, w in sorted(bstore.last_allocations(conn).items(), key=lambda kv: -kv[1])
    ]
    nulls = [s.id for s in specs if s.family == "null_random" and s.id in rets.columns]
    null95 = promotion.null_threshold(rets[nulls], now - WINDOW) if nulls else None
    chart_ids = [s.id for s in specs if s.role == "benchmark" or (s.role == "competitor" and s.status == "champion")]
    last = last_bar(conn)
    return Leaderboard(rows, alloc, null95, last, is_stale(last, now), chart_ids)


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


def trials(conn: psycopg.Connection, limit: int = 200) -> list[dict[str, Any]]:
    """Most recent trials across all families with metric highlights extracted."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, competitor_id, family, kind, verdict, started_at, finished_at, notes, metrics"
            " FROM trials ORDER BY started_at DESC, id DESC LIMIT %s",
            (limit,),
        )
        return [_trial_row(r) for r in cur.fetchall()]


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
    """One row per data source: last data timestamp, last fetch, age level (ok / warn / bad)."""
    specs = [
        ("candles", "SELECT max(ts) AS data_ts, max(fetched_at) AS fetched FROM candles", None),
        ("funding", "SELECT max(ts) AS data_ts, max(fetched_at) AS fetched FROM funding", None),
        ("open_interest", "SELECT max(ts) AS data_ts, max(fetched_at) AS fetched FROM open_interest", None),
        ("hl_snapshots", "SELECT max(ts) AS data_ts, max(fetched_at) AS fetched FROM hl_snapshots", None),
        (
            "articles",
            "SELECT max(published_at) AS data_ts, max(fetched_at) AS fetched,"
            " count(*) FILTER (WHERE fetched_at > %s) AS n24 FROM articles",
            (now - timedelta(hours=24),),
        ),
        ("macro_events", "SELECT max(ts) AS data_ts, max(fetched_at) AS fetched FROM macro_events", None),
    ]
    out: list[dict[str, Any]] = []
    with conn.cursor() as cur:
        for key, sql, params in specs:
            cur.execute(sql, params)
            r = cur.fetchone()
            fetched = r["fetched"]
            age = (now - fetched) if fetched else None
            out.append(
                {
                    "key": key,
                    "label": fr.SOURCE_FR[key],
                    "data_ts": r["data_ts"],
                    "fetched": fetched,
                    "age": age,
                    "level": fr.freshness_level(age),
                    "count_24h": int(r["n24"]) if "n24" in r and r["n24"] is not None else None,
                }
            )
    return out


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

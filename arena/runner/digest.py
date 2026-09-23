"""Daily digest: who speaks, what changed since yesterday, the challengers, the benchmarks, health.

Builds a ``templates.DigestContext`` from the store and renders it in French.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psycopg

from arena.judge.metrics import PPY_HOURLY, sharpe, track_record_verdict
from arena.runner import promotion
from arena.store import books as bstore
from arena.store import registry
from arena.telegram import templates
from arena.telegram.templates import ChallengerView, ChampionView, DigestContext, PositionView

WINDOW = timedelta(days=30)
SIZE_STEP = 0.05  # a weight move below this is not a "change"
MIN_BARS = 24
CONFIDENCE = promotion.PROMOTION_CONFIDENCE


def _ppy(index) -> int:
    """Periods per year read from the bar spacing, so the two arenas are annualised apart."""
    idx = pd.DatetimeIndex(index)
    if len(idx) < 3:
        return PPY_HOURLY
    gap = float(pd.Series(idx).diff().dt.total_seconds().median() or 3600.0) / 3600.0
    return max(1, int(round(365.0 * 24.0 / gap))) if gap > 0 else PPY_HOURLY


def _evidence(r: pd.Series, null95: float | None) -> dict[str, float | None]:
    """``psr_vs_null`` and the days still missing before the arena may conclude."""
    if len(r) <= MIN_BARS:
        return {"psr_vs_null": None, "days_missing": None}
    ppy = _ppy(r.index)
    v = track_record_verdict(r, null95 or 0.0, CONFIDENCE, ppy)
    bars_per_day = max(1.0, ppy / 365.0)
    missing = v["missing"]
    return {
        "psr_vs_null": float(v["psr"]),
        "days_missing": None if not np.isfinite(missing) else float(missing) / bars_per_day,
    }


def _pnl(nav: pd.Series, now: datetime, days: int) -> float | None:
    """Euros gained over the last ``days`` days, as a fraction of NAV0 (what ``verdicts.money`` expects)."""
    if nav.empty:
        return None
    ts = pd.Timestamp(now)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    since = ts - timedelta(days=days)
    ref = nav[nav.index >= since]
    if ref.empty:
        return None
    return float(nav.iloc[-1] - ref.iloc[0]) / templates.NAV0


def _positions(conn: psycopg.Connection, competitor_id: int, at: datetime | None = None) -> list[PositionView]:
    """Full targets (weight, conviction, kind, reason) at the latest bar, or the latest bar at/before ``at``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol, weight, conviction, kind, reason FROM targets WHERE competitor_id = %s AND ts = ("
            "  SELECT max(ts) FROM targets WHERE competitor_id = %s AND (%s::timestamptz IS NULL OR ts <= %s))",
            (competitor_id, competitor_id, at, at),
        )
        return [
            PositionView(r["symbol"], float(r["weight"]), float(r["conviction"]), r["kind"], dict(r["reason"]))
            for r in cur.fetchall()
            if float(r["weight"]) != 0.0
        ]


def _changes(name: str, family: str, before: list[PositionView], after: list[PositionView]) -> list[str]:
    old = {p.symbol: p for p in before}
    new = {p.symbol: p for p in after}
    who = templates.display_name(family, templates.split_name(name)[1])
    out: list[str] = []
    for sym in sorted(set(old) | set(new), key=lambda s: -abs((new.get(s) or old[s]).weight)):
        if sym in new and sym not in old:
            out.append(f"{who} entre sur {sym} ({abs(new[sym].weight) * 100:.0f} % du capital).")
        elif sym in old and sym not in new:
            out.append(f"{who} sort de {sym}.")
        else:
            w0, w1 = old[sym].weight, new[sym].weight
            if (w0 > 0) != (w1 > 0):
                out.append(f"{who} retourne sa position sur {sym} ({w0 * 100:+.0f} % -> {w1 * 100:+.0f} %).")
            elif abs(w1 - w0) >= SIZE_STEP:
                verb = "renforce" if abs(w1) > abs(w0) else "allège"
                out.append(f"{who} {verb} {sym} : {abs(w0) * 100:.0f} % -> {abs(w1) * 100:.0f} % du capital.")
    return out


def _arena_progress(conn: psycopg.Connection, competitor_id: int, now: datetime) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT min(ts) AS first FROM books WHERE competitor_id = %s", (competitor_id,))
        first = cur.fetchone()["first"]
        cur.execute(
            "SELECT count(DISTINCT ts) AS n FROM targets WHERE competitor_id = %s AND weight <> 0", (competitor_id,)
        )
        n = int(cur.fetchone()["n"])
    days = (pd.Timestamp(now) - pd.Timestamp(first)).days if first else 0
    return days, n


def _last_tick(conn: psycopg.Connection) -> tuple[datetime | None, int | None]:
    with conn.cursor() as cur:
        cur.execute("SELECT finished_at, booked, skipped FROM tick_runs ORDER BY started_at DESC LIMIT 1")
        r = cur.fetchone()
    if not r:
        return None, None
    return r["finished_at"], int(r["booked"]) + int(r["skipped"])


def _drift_lines(conn: psycopg.Connection, now: datetime) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT kind, payload FROM alerts WHERE ts >= %s AND kind IN ('drift','stale','error','rejected') "
            "ORDER BY ts",
            (now - timedelta(days=1),),
        )
        return [templates.detail_fr(r["kind"], str(r["payload"].get("detail", ""))) for r in cur.fetchall()]


def build_context(conn: psycopg.Connection, now: datetime) -> DigestContext:
    specs = registry.list_competitors(conn, statuses=["champion", "challenger"])
    ids = [s.id for s in specs]
    rets = bstore.read_returns(conn, ids, now - WINDOW, now) if ids else pd.DataFrame()

    def _series(cid: int) -> pd.Series:
        return rets[cid].dropna() if cid in rets.columns else pd.Series(dtype=float)

    def sharpe_30d(cid: int) -> float | None:
        r = _series(cid)
        return sharpe(r, _ppy(r.index)) if len(r) > MIN_BARS else None

    btc_spec = next((s for s in specs if s.family == "bench_btc_hold"), None)
    btc_30d = _pnl(bstore.read_nav(conn, btc_spec.id, now - WINDOW, now), now, 30) if btc_spec else None

    champions: list[ChampionView] = []
    changes: list[str] = []
    champ_sharpe_by_family: dict[str, float | None] = {}
    for s in specs:
        if s.role != "competitor" or s.status != "champion":
            continue
        nav = bstore.read_nav(conn, s.id, now - WINDOW, now)
        champions.append(
            ChampionView(
                name=s.name,
                family=s.family,
                version=s.version,
                positions=_positions(conn, s.id),
                pnl_1d=_pnl(nav, now, 1),
                pnl_7d=_pnl(nav, now, 7),
                pnl_30d=_pnl(nav, now, 30),
                btc_30d=btc_30d,
            )
        )
        champ_sharpe_by_family[s.family] = sharpe_30d(s.id)
        changes += _changes(
            s.name, s.family, _positions(conn, s.id, now - timedelta(hours=24)), champions[-1].positions
        )

    nulls = [s.id for s in specs if s.family == "null_random" and s.id in rets.columns]
    null_ppy = _ppy(_series(nulls[0]).index) if nulls else PPY_HOURLY
    null95 = promotion.null_threshold(rets[nulls], now - WINDOW, null_ppy) if nulls else None

    challengers: list[ChallengerView] = []
    for s in specs:
        if s.role != "competitor" or s.status != "challenger":
            continue
        days, n = _arena_progress(conn, s.id, now)
        challengers.append(
            ChallengerView(
                name=s.name,
                family=s.family,
                version=s.version,
                days=days,
                decisions=n,
                days_required=promotion.MIN_DAYS,
                decisions_required=promotion.MIN_DECISIONS,
                pnl_30d=_pnl(bstore.read_nav(conn, s.id, now - WINDOW, now), now, 30),
                sharpe_30d=sharpe_30d(s.id),
                champion_sharpe_30d=champ_sharpe_by_family.get(s.family),
                **_evidence(_series(s.id), null95),
            )
        )

    last_tick, evaluated = _last_tick(conn)
    return DigestContext(
        date_str=pd.Timestamp(now).strftime("%d/%m/%Y"),
        champions=champions,
        changes=changes,
        challengers=challengers,
        btc_30d_eur=btc_30d * templates.NAV0 if btc_30d is not None else None,
        null95=null95,
        last_tick=last_tick,
        evaluated=evaluated,
        drift_lines=_drift_lines(conn, now),
    )


def build(conn: psycopg.Connection, now: datetime) -> str:
    return templates.daily_digest(build_context(conn, now))

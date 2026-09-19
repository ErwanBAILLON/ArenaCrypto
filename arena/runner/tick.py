"""The hourly tick: ingest → decide → book → allocate → drift → promote → alert.

Idempotent per bar: re-running for an already-booked bar does nothing for that
competitor. Every competitor is isolated: one failure is logged and alerted,
the others still run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import httpx
import pandas as pd
import psycopg

from arena.book.book import Book, FeeModel
from arena.competitors.base import build
from arena.competitors.regime import regime_label
from arena.core.types import Alert, CompetitorSpec, Decision
from arena.core.universe import Universe
from arena.runner import allocator, drift, promotion
from arena.runner.history import load_history, snapshot_from_history
from arena.runner.ingest import EXCHANGE, ingest_market
from arena.settings import Settings
from arena.store import books as bstore
from arena.store import candles as cstore
from arena.store import registry
from arena.store.state import load_state, save_state

log = logging.getLogger(__name__)

HISTORY_BARS = 24 * 260  # covers the largest warm-up (regime: ~5041 bars) with margin
SIGNAL_MIN_DELTA = 0.25
SIGNAL_COOLDOWN = timedelta(hours=4)
ALLOC_WINDOW = timedelta(days=60)
ACTIVE = ("champion", "challenger")


@dataclass
class TickReport:
    ts: datetime | None = None
    ingested: dict[str, int] = field(default_factory=dict)
    booked: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    alerts: int = 0


def decision_bar(now: datetime) -> pd.Timestamp:
    """Last closed 1h bar label (candles are labelled by close time)."""
    return pd.Timestamp(now).floor("1h")


def fees_of(universe: Universe) -> FeeModel:
    f = universe.fees
    return FeeModel(perp_taker=f.perp_taker, slippage=f.slippage, spot_taker=f.spot_taker)


def _with_model(conn: psycopg.Connection, spec: CompetitorSpec) -> CompetitorSpec:
    if spec.family != "meta_label" or spec.id is None:
        return spec
    found = registry.latest_model(conn, spec.id)
    if not found:
        return spec
    artifact, _ = found
    return CompetitorSpec(**{**spec.__dict__, "params": {**spec.params, "model_str": artifact.decode()}})


def _signal_alerts(
    conn,
    spec: CompetitorSpec,
    prev: dict[str, tuple[str, float]],
    decision: Decision,
    ts,
    regime: str,
    funding_8h: dict[str, float],
) -> list[Alert]:
    if spec.role != "competitor" or spec.status != "champion":
        return []
    out: list[Alert] = []
    for sym in set(prev) | set(decision):
        w_old = prev.get(sym, ("perp", 0.0))[1]
        t = decision.get(sym)
        w_new = t.weight if t else 0.0
        flipped = (w_old > 0 > w_new) or (w_old < 0 < w_new) or (w_old == 0) != (w_new == 0)
        if not flipped and abs(w_new - w_old) < SIGNAL_MIN_DELTA:
            continue
        if bstore.recent_alert_exists(conn, "signal", spec.id, sym, ts - SIGNAL_COOLDOWN):
            continue
        out.append(
            Alert(
                kind="signal",
                competitor_id=spec.id,
                symbol=sym,
                payload={
                    "weight": w_new,
                    "conviction": t.conviction if t else 0.0,
                    "kind": t.kind if t else "perp",
                    "regime": regime,
                    "funding_8h": funding_8h.get(sym),
                    "reason": t.reason if t else {"exit": True},
                },
            )
        )
    return out


def run(
    conn: psycopg.Connection,
    settings: Settings,
    universe: Universe,
    now: datetime,
    client: httpx.Client | None = None,
    ingest: bool = True,
) -> TickReport:
    rep = TickReport()
    if ingest and client is not None:
        rep.ingested = ingest_market(conn, client, universe, now)
    ts = decision_bar(now)
    last_bar = cstore.last_candle_ts(conn, EXCHANGE, "BTC")
    stale = drift.stale_data(last_bar, now)
    if stale:
        bstore.add_alert(conn, stale)
        conn.commit()
    if last_bar is None:
        return rep
    ts = min(ts, pd.Timestamp(last_bar))
    rep.ts = ts.to_pydatetime()

    history = load_history(conn, universe, ts - timedelta(hours=HISTORY_BARS), ts)
    if history.candles.empty:
        return rep
    snap = snapshot_from_history(history, ts, universe.symbols, cstore.latest_hl_funding(conn))
    closes = snap.closes()
    prices = {s: float(closes[s].iloc[-1]) for s in closes.columns if pd.notna(closes[s].iloc[-1])}
    regime = regime_label(snap)
    fund_8h = {s: float(snap.funding(s).iloc[-1]) for s in universe.symbols if len(snap.funding(s))}
    fees = fees_of(universe)

    specs = [s for s in registry.list_competitors(conn, statuses=list(ACTIVE))]
    for spec in specs:
        try:
            _run_one(conn, spec, snap, history, prices, closes, ts, regime, fund_8h, fees, universe.nav0, rep)
            conn.commit()
        except Exception as exc:  # isolate competitors
            conn.rollback()
            log.exception("competitor %s failed", spec.name)
            rep.failed.append(spec.name)
            bstore.add_alert(
                conn,
                Alert(
                    kind="error",
                    competitor_id=spec.id,
                    payload={"detail": f"{spec.name}: {type(exc).__name__}: {exc}"[:300]},
                ),
            )
            conn.commit()

    _allocate(conn, specs, ts)
    _drift_and_promote(conn, specs, ts, now)
    conn.commit()
    return rep


def _run_one(conn, spec, snap, history, prices, closes, ts, regime, fund_8h, fees, nav0, rep: TickReport) -> None:
    last_row = bstore.last_book_row(conn, spec.id)
    if last_row is not None and pd.Timestamp(last_row.ts) >= ts:
        rep.skipped.append(spec.name)
        return
    comp = build(_with_model(conn, spec))
    comp.restore_state(load_state(conn, spec.id))
    decision = comp.decide(snap)

    prev = bstore.last_targets(conn, spec.id)
    prev_positions = prev[1] if prev else {}
    if last_row is None:
        book = Book(nav=nav0, fees=fees)
        prev_prices, funding = dict(prices), {}
    else:
        book = Book.restore(last_row.nav, prev_positions, fees)
        prev_ts = pd.Timestamp(last_row.ts)
        prev_prices = {
            s: float(closes[s].loc[:prev_ts].iloc[-1]) for s in closes.columns if len(closes[s].loc[:prev_ts])
        }
        funding = {}
        if history.funding is not None and not history.funding.empty:
            f = history.funding[(history.funding["ts"] > prev_ts) & (history.funding["ts"] <= ts)]
            funding = f.groupby("symbol")["rate"].sum().to_dict()
    row = book.step(ts, prices, prev_prices, funding, decision)
    bstore.write_targets(conn, spec.id, ts, decision)
    bstore.write_book_row(conn, spec.id, row)
    save_state(conn, spec.id, ts, comp.state())
    for a in _signal_alerts(conn, spec, prev_positions, decision, ts, regime, fund_8h):
        bstore.add_alert(conn, a)
        rep.alerts += 1
    broken = drift.broken_book(spec, row.nav)
    if broken:
        bstore.add_alert(conn, broken)
    rep.booked.append(spec.name)


def _allocate(conn, specs: list[CompetitorSpec], ts) -> None:
    champs = [s.id for s in specs if s.role == "competitor" and s.status == "champion"]
    if not champs:
        return
    rets = bstore.read_returns(conn, champs, ts - ALLOC_WINDOW, ts)
    weights = allocator.compute(rets, bstore.last_allocations(conn))
    bstore.write_allocations(conn, ts, weights)


def _drift_and_promote(conn, specs: list[CompetitorSpec], ts, now) -> None:
    nulls = [s for s in specs if s.role == "null" and s.family == "null_random"]
    null_rets = (
        bstore.read_returns(conn, [s.id for s in nulls], ts - timedelta(days=365), ts) if nulls else pd.DataFrame()
    )
    by_family: dict[str, dict[str, list[CompetitorSpec]]] = {}
    for s in specs:
        if s.role == "competitor":
            by_family.setdefault(s.family, {"champion": [], "challenger": []})[s.status].append(s)
    for groups in by_family.values():
        champion = groups["champion"][0] if groups["champion"] else None
        champ_cand = _candidate(conn, champion, ts) if champion else None
        if champ_cand is not None:
            bleeding = drift.champion_bleeding(champion, champ_cand.returns)
            if bleeding and not bstore.recent_alert_exists(conn, "drift", champion.id, None, ts - timedelta(days=1)):
                bstore.add_alert(conn, bleeding)
        for ch in groups["challenger"]:
            cand = _candidate(conn, ch, ts)
            if cand is None:
                continue
            ok, evidence = promotion.should_promote(cand, champ_cand, null_rets, now)
            if ok:
                if champion is not None:
                    registry.set_status(conn, champion.id, "retired")
                registry.set_status(conn, ch.id, "champion")
                bstore.add_alert(
                    conn,
                    promotion.promotion_alert(
                        ch, champion, {k: round(v, 3) if isinstance(v, float) else v for k, v in evidence.items()}
                    ),
                )
                champion, champ_cand = ch, cand
        for s in promotion.surplus_challengers(groups["challenger"]):
            registry.set_status(conn, s.id, "retired")
    for s in specs:
        if s.role != "competitor":
            continue
        last_dec = _last_decision_ts(conn, s.id)
        first = _first_book_ts(conn, s.id)
        if first is not None and pd.Timestamp(now) - pd.Timestamp(first) > timedelta(days=drift.SILENT_DAYS):
            silent = drift.silent_competitor(s, last_dec, now)
            if silent and not bstore.recent_alert_exists(conn, "drift", s.id, None, ts - timedelta(days=7)):
                bstore.add_alert(conn, silent)


def _candidate(conn, spec: CompetitorSpec, ts) -> promotion.Candidate | None:
    first = _first_book_ts(conn, spec.id)
    if first is None:
        return None
    rets = bstore.read_returns(conn, [spec.id], first, ts)
    series = rets[spec.id] if spec.id in rets.columns else pd.Series(dtype=float)
    return promotion.Candidate(spec=spec, first_ts=first, decisions=_decision_count(conn, spec.id), returns=series)


def _first_book_ts(conn, competitor_id: int):
    with conn.cursor() as cur:
        cur.execute("SELECT min(ts) AS ts FROM books WHERE competitor_id = %s", (competitor_id,))
        r = cur.fetchone()
    return r["ts"] if r and r["ts"] else None


def _decision_count(conn, competitor_id: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(DISTINCT ts) AS n FROM targets WHERE competitor_id = %s AND weight <> 0", (competitor_id,)
        )
        return int(cur.fetchone()["n"])


def _last_decision_ts(conn, competitor_id: int):
    with conn.cursor() as cur:
        cur.execute("SELECT max(ts) AS ts FROM targets WHERE competitor_id = %s AND weight <> 0", (competitor_id,))
        r = cur.fetchone()
    return r["ts"] if r and r["ts"] else None

"""What the interactive dashboard reads: NAV series, trade events, and the live state.

The arena decides once an hour. Nothing here invents motion between ticks:
``/api/live`` tells the page when the last tick wrote and when the next one is
due, and the page animates *those* two facts every second -- an age and a
countdown -- while charts and positions change only when a tick has actually
changed them. Faking per-second movement on hourly data would be decoration
pretending to be information.

Trade events are reconstructed from ``targets`` bar to bar, so every "who bought
what, when" marker on a chart is a weight that really changed in the stored
book, at the price the book saw.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import psycopg

from arena.store import books as bstore
from arena.store import registry
from arena.web import queries

TICK_MINUTE = 5  # helm: schedules.tick "5 * * * *"
EVENT_EPS = 1e-6
SIZE_STEP = 0.05  # below this a weight move is dust, not an event
_WINDOWS = ("today", "7d", "30d", "all")  # the P&L bases the board can be re-marked against


def next_tick(now: datetime, minute: int = TICK_MINUTE) -> datetime:
    """The next hourly tick after ``now``."""
    base = now.replace(minute=minute, second=0, microsecond=0)
    return base if base > now else base + timedelta(hours=1)


@dataclass(frozen=True)
class Event:
    ts: datetime
    symbol: str
    kind: str  # entrée | sortie | renfort | allège | retournement
    before: float
    after: float
    price: float | None
    conviction: float
    reason: dict

    @property
    def side(self) -> str:
        w = self.after if self.after != 0 else self.before
        return "long" if w > 0 else "short"

    def to_json(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "t": int(self.ts.timestamp()),
            "symbol": self.symbol,
            "kind": self.kind,
            "side": self.side,
            "before": round(self.before, 4),
            "after": round(self.after, 4),
            "price": self.price,
            "conviction": round(self.conviction, 3),
            "reason": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.reason.items()},
        }


def _classify(before: float, after: float) -> str | None:
    if abs(before) < EVENT_EPS and abs(after) >= EVENT_EPS:
        return "entrée"
    if abs(before) >= EVENT_EPS and abs(after) < EVENT_EPS:
        return "sortie"
    if before * after < 0:
        return "retournement"
    if abs(after - before) < SIZE_STEP:
        return None
    return "renfort" if abs(after) > abs(before) else "allège"


def trade_events(conn: psycopg.Connection, competitor_id: int, start: datetime, end: datetime) -> list[Event]:
    """Every weight change that counts, with the close the book marked it at."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, symbol, weight, conviction, reason FROM targets"
            " WHERE competitor_id = %s AND ts BETWEEN %s AND %s ORDER BY ts, symbol",
            (competitor_id, start, end),
        )
        rows = cur.fetchall()
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    wide = frame.pivot_table(index="ts", columns="symbol", values="weight", aggfunc="last").fillna(0.0).sort_index()
    conviction = frame.pivot_table(index="ts", columns="symbol", values="conviction", aggfunc="last")
    reasons = {(pd.Timestamp(r["ts"]), r["symbol"]): dict(r["reason"] or {}) for r in rows}
    stamps = list(wide.index)
    symbols = sorted(set(frame["symbol"]))
    prices = _closes(conn, symbols, start, end)

    events: list[Event] = []
    # the book as it stood just before the window, so a position already open on
    # the first bar is not mis-read as a fresh entry
    prev = pd.Series(0.0, index=wide.columns)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol, weight FROM targets WHERE competitor_id = %s AND ts = ("
            "  SELECT max(ts) FROM targets WHERE competitor_id = %s AND ts < %s)",
            (competitor_id, competitor_id, start),
        )
        for r in cur.fetchall():
            if r["symbol"] in prev.index:
                prev[r["symbol"]] = float(r["weight"])
    for ts in stamps:
        cur_w = wide.loc[ts]
        for sym in wide.columns:
            kind = _classify(float(prev[sym]), float(cur_w[sym]))
            if kind is None:
                continue
            price = None
            if not prices.empty and sym in prices.columns:
                p = prices[sym].asof(ts)
                price = float(p) if p == p else None
            conv = conviction.loc[ts, sym] if sym in conviction.columns and ts in conviction.index else float("nan")
            events.append(
                Event(
                    ts=ts.to_pydatetime(),
                    symbol=str(sym),
                    kind=kind,
                    before=float(prev[sym]),
                    after=float(cur_w[sym]),
                    price=price,
                    conviction=float(conv) if conv == conv else 0.0,
                    reason=reasons.get((ts, sym), {}),
                )
            )
        prev = cur_w
    return events


def _closes(conn: psycopg.Connection, symbols: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, symbol, close FROM candles WHERE symbol = ANY(%s) AND ts BETWEEN %s AND %s ORDER BY ts",
            (symbols, start - timedelta(hours=2), end),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last").sort_index()


def nav_series(
    conn: psycopg.Connection, competitor_id: int, start: datetime, end: datetime, max_points: int = 2000
) -> dict:
    """``{t: [...unix s], nav: [...], ret: [...]}`` thinned to ``max_points`` for the wire."""
    nav = bstore.read_nav(conn, competitor_id, start, end)
    if nav.empty:
        return {"t": [], "nav": []}
    if len(nav) > max_points:
        step = -(-len(nav) // max_points)
        keep = nav.iloc[::step]
        if keep.index[-1] != nav.index[-1]:
            keep = pd.concat([keep, nav.iloc[[-1]]])
        nav = keep
    return {"t": [int(t.timestamp()) for t in nav.index], "nav": [round(float(v), 2) for v in nav.to_numpy()]}


def competitor_series(conn: psycopg.Connection, competitor_id: int, days: int, now: datetime) -> dict:
    """Everything one chart needs: NAV, the trade markers on it, and the symbols involved."""
    spec = registry.list_competitors(conn)
    spec = next((s for s in spec if s.id == competitor_id), None)
    if spec is None:
        return {}
    start = now - timedelta(days=days)
    nav = nav_series(conn, competitor_id, start, now)
    events = trade_events(conn, competitor_id, start, now)
    return {
        "id": competitor_id,
        "name": spec.name,
        "family": spec.family,
        "status": spec.status,
        "nav": nav,
        "events": [e.to_json() for e in events],
        "symbols": sorted({e.symbol for e in events}),
    }


def board_series(conn: psycopg.Connection, ids: list[int], days: int, now: datetime) -> dict:
    """Normalised NAV (start = 100) for several competitors on one time axis."""
    start = now - timedelta(days=days)
    names = {s.id: s for s in registry.list_competitors(conn)}
    out = []
    for cid in ids:
        nav = bstore.read_nav(conn, cid, start, now)
        if nav.empty or nav.iloc[0] == 0:
            continue
        norm = nav / float(nav.iloc[0]) * 100.0
        spec = names.get(cid)
        out.append(
            {
                "id": cid,
                "name": spec.name if spec else str(cid),
                "family": spec.family if spec else "",
                "role": spec.role if spec else "",
                "status": spec.status if spec else "",
                "t": [int(t.timestamp()) for t in norm.index],
                "v": [round(float(v), 3) for v in norm.to_numpy()],
            }
        )
    return {"series": out, "start": int(start.timestamp()), "end": int(now.timestamp())}


def live_state(conn: psycopg.Connection, now: datetime) -> dict:
    """The two facts the page may animate -- age of the last tick, time to the next --
    plus the positions and latest events, so a poll can tell whether anything changed."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bar_ts, finished_at, booked, changes, alerts, ok FROM tick_runs ORDER BY started_at DESC LIMIT 1"
        )
        tick = cur.fetchone()
        cur.execute("SELECT max(ts) AS ts FROM books")
        last_bar = cur.fetchone()["ts"]
        cur.execute(
            "SELECT greatest(coalesce(max(ts), 'epoch'),"
            " (SELECT coalesce(max(ts), 'epoch') FROM books)) AS ts FROM targets"
        )
        last_write = cur.fetchone()["ts"]
    active = [s for s in registry.list_competitors(conn, statuses=["champion", "challenger"]) if s.role == "competitor"]
    positions = []
    for s in active:
        held = bstore.last_targets(conn, s.id)
        if not held:
            continue
        ts, pos = held
        for sym, (kind, w) in sorted(pos.items(), key=lambda kv: -abs(kv[1][1])):
            if abs(w) < EVENT_EPS:
                continue
            positions.append(
                {
                    "competitor_id": s.id,
                    "name": s.name,
                    "family": s.family,
                    "status": s.status,
                    "symbol": sym,
                    "weight": round(w, 4),
                    "kind": kind,
                    "since": ts.isoformat(),
                }
            )
    recent = []
    since = now - timedelta(hours=48)
    for s in active:
        for e in trade_events(conn, s.id, since, now)[-6:]:
            recent.append({**e.to_json(), "competitor_id": s.id, "name": s.name, "family": s.family})
    recent.sort(key=lambda e: e["t"], reverse=True)
    nxt = next_tick(now)
    return {
        "now": now.isoformat(),
        "last_tick": tick["finished_at"].isoformat() if tick else None,
        "last_bar": last_bar.isoformat() if last_bar else None,
        "tick_ok": bool(tick["ok"]) if tick else None,
        "booked": int(tick["booked"]) if tick else 0,
        "changes": int(tick["changes"]) if tick else 0,
        "next_tick": nxt.isoformat(),
        "seconds_to_next": int((nxt - now).total_seconds()),
        "positions": positions,
        "recent_events": recent[:30],
        "version": int(last_write.timestamp()) if last_write else 0,  # changes when a tick or a live exit wrote
    }


def utc_now() -> datetime:
    return datetime.now(UTC)


def _exchanges(conn: psycopg.Connection, symbols: list[str]) -> dict[str, str]:
    if not symbols:
        return {}
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol, exchange FROM candles WHERE symbol = ANY(%s)", (symbols,))
        return {r["symbol"]: r["exchange"] for r in cur.fetchall()}


FUTURES_PRICES = "https://fapi.binance.com/fapi/v1/ticker/price"  # public, CORS-open, every pair in one call
SPOT_STREAM = "wss://data-stream.binance.vision/stream?streams="  # public market-data host, fallback only


def binance_pair(symbol: str) -> str:
    """The USDⓈ-M perpetual for a stored symbol (``BTC`` or ``BTCUSDT``)."""
    return symbol if symbol.endswith("USDT") else f"{symbol}USDT"


def book_state(conn: psycopg.Connection, now: datetime) -> dict:
    """Everything the live board needs to mark every agent to market in the browser.

    The tick writes each book once an hour at the bar's close. Between two
    ticks the positions are known and the prices move, so the page can value
    ``nav × (1 + Σ w·(p_live/p_ref − 1))`` on every price update -- an honest
    mark-to-market of the stored book, not a prediction of what the next tick
    will decide. Prices come from Binance's public futures REST (the perps the
    book actually trades; the futures websocket opens but streams nothing from
    some regions), with the spot market-data websocket as a fallback. Equity
    tickers have no public live feed and keep their last hourly close.
    """
    board = queries.leaderboard(conn, now)
    stats = {r["id"]: r for r in board.rows}
    specs = [
        s
        for s in registry.list_competitors(conn, statuses=["champion", "challenger"])
        if s.family != queries.NULL_RANDOM_FAMILY and s.role != "null"
    ]
    all_syms: set[str] = set()
    raw = []
    for s in specs:
        held = bstore.last_targets(conn, s.id)
        positions = dict(held[1]) if held else {}
        all_syms |= set(positions)
        raw.append((s, held[0] if held else None, positions))
    exchanges = _exchanges(conn, sorted(all_syms))
    agents = []
    for s, pos_ts, positions in raw:
        pnl = queries.pnl_windows(conn, s.id, now)
        if pnl["nav"] is None:
            continue
        ref_ts = pnl["nav_ts"]
        prices = _closes(conn, sorted(positions), ref_ts - timedelta(hours=3), ref_ts) if positions else pd.DataFrame()
        plist = []
        for sym, (kind, w) in sorted(positions.items(), key=lambda kv: -abs(kv[1][1])):
            ref = None
            if not prices.empty and sym in prices.columns:
                p = prices[sym].asof(pd.Timestamp(ref_ts))
                ref = float(p) if p == p else None
            since = queries.first_ts_of_position(conn, s.id, sym, w, pos_ts) if pos_ts else None
            plist.append(
                {
                    "symbol": sym,
                    "weight": round(w, 4),
                    "kind": kind,
                    "ref_price": ref,
                    "pair": binance_pair(sym) if exchanges.get(sym) == "binance" and ref else None,
                    "since": since.isoformat() if since else None,
                }
            )
        st = stats.get(s.id, {})
        agents.append(
            {
                "id": s.id,
                "name": s.name,
                "family": s.family,
                "status": s.status,
                "role": s.role,
                "universe": s.universe,
                "nav": round(float(pnl["nav"]), 2),
                "nav_ts": ref_ts.isoformat(),
                "base": {k: (round(float(pnl["nav"]) - pnl[k]["eur"], 2) if pnl[k] else None) for k in _WINDOWS},
                "sharpe_30d": st.get("sharpe_30d"),
                "sharpe_se_30d": st.get("sharpe_se_30d"),
                "psr": st.get("psr"),
                "mdd_30d": st.get("mdd_30d"),
                "bars_30d": st.get("bars_30d"),
                "positions": plist,
            }
        )
    pairs = sorted({p["pair"] for a in agents for p in a["positions"] if p["pair"]})
    with conn.cursor() as cur:
        cur.execute("SELECT max(ts) AS ts FROM books")
        last_bar = cur.fetchone()["ts"]
    return {
        "now": now.isoformat(),
        "agents": agents,
        "pairs": pairs,
        "rest": FUTURES_PRICES if pairs else None,
        "ws": SPOT_STREAM + "/".join(f"{p.lower()}@miniTicker" for p in pairs) if pairs else None,
        "next_tick": next_tick(now).isoformat(),
        "version": int(last_bar.timestamp()) if last_bar else 0,
    }

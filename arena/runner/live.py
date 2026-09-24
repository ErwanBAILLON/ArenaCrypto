"""The live watcher: exit rules executed between ticks, on streamed prices.

The tick decides once an hour on closed bars and that stays true: nothing here
opens a position or re-selects a book. What the tick could not do is honour a
stop or a take-profit at the moment the price crosses it -- it only saw the
close of the hour, so a stop touched at 14:20 and recovered by 14:59 did not
exist, and a stop touched at 14:20 that kept falling was paid at the 15:00
close. This process watches the perp prices every few seconds and, when a
rule fires, closes that leg *now*: it writes a ``fills`` row at the streamed
price, re-states the competitor's book without the leg, and tells the
competitor so it does not bring the position back at the next bar. The next
tick reads the fill and accounts ``w × (fill / prev_close − 1)`` plus the
closing costs, exactly as if it had sold at that price itself.

Two kinds of rules are honoured:

* the :class:`LadderHoldingCompetitor` families (``xs_sparse`` and friends)
  carry their own stop and ROI ladder, measured as excess return over the
  basket frozen at entry -- the same measure the tick uses, re-evaluated on
  live prices for the leg and for the basket;
* any competitor given ``live_stop`` (a fraction) and/or ``live_roi`` (ladder
  steps) in its params gets the same rules on the raw return of the leg since
  it was opened. That is a parameter change, hence a new challenger version,
  never a live edit of a running champion.

The deadline rung of a ladder (hold at most N bars) is left to the tick: it is
a function of bars, not of price, and the tick sees it first anyway.

Around each tick there is a quiet window: a fill written while the tick is
mid-flight would be stamped after the tick's own decision and read as the
latest book. Better to miss a stop by four minutes than to corrupt a book.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx
import psycopg

from arena.competitors.base import REGISTRY, build
from arena.competitors.ladder import LadderHoldingCompetitor
from arena.core.types import Alert, CompetitorSpec
from arena.core.universe import Universe
from arena.labeling.barriers import RoiLadder
from arena.runner import alerts
from arena.settings import Settings
from arena.store import books as bstore
from arena.store import registry
from arena.store.state import load_state, save_state

log = logging.getLogger(__name__)

FUTURES_PRICES = "https://fapi.binance.com/fapi/v1/ticker/price"
ACTIVE = ("champion", "challenger")
QUIET_BEFORE = timedelta(minutes=1)
QUIET_AFTER = timedelta(minutes=5)


@dataclass
class Watch:
    """One open leg and the rule that may close it."""

    competitor_id: int
    name: str
    family: str
    symbol: str
    pair: str  # exchange symbol on the price feed (BTCUSDT)
    kind: str
    weight: float
    entry_price: float
    bars_held: float  # bars completed at the last tick
    stop: float | None
    ladder: RoiLadder | None
    basket: dict[str, float] = field(default_factory=dict)  # pair -> price at entry (ladder families)
    positions: dict[str, tuple[str, float]] = field(default_factory=dict)  # the whole book, to re-state it
    bar_hours: int = 1

    @property
    def side(self) -> float:
        return 1.0 if self.weight > 0 else -1.0


def quiet(now: datetime, tick_minutes: tuple[int, ...]) -> bool:
    """True inside the window around each tick minute where the watcher must not write."""
    for m in tick_minutes:
        anchor = now.replace(minute=m % 60, second=0, microsecond=0)
        for a in (anchor, anchor - timedelta(hours=1), anchor + timedelta(hours=1)):
            if a - QUIET_BEFORE <= now < a + QUIET_AFTER:
                return True
    return False


def fetch_prices(client: httpx.Client) -> dict[str, float]:
    """Every USDⓈ-M perp's last price, one call."""
    r = client.get(FUTURES_PRICES)
    r.raise_for_status()
    return {row["symbol"]: float(row["price"]) for row in r.json()}


def _entry_close(conn: psycopg.Connection, exchange: str, symbol: str, ts: datetime) -> float | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT close FROM candles WHERE exchange = %s AND symbol = %s AND ts <= %s ORDER BY ts DESC LIMIT 1",
            (exchange, symbol, ts),
        )
        row = cur.fetchone()
    return float(row["close"]) if row else None


def _ladder_from(params: dict) -> RoiLadder | None:
    steps = params.get("live_roi")
    return RoiLadder(steps=tuple((int(b), float(p)) for b, p in steps)) if steps else None


def watchlist(conn: psycopg.Connection, universe: Universe, now: datetime) -> list[Watch]:
    """Every open leg of every active competitor of a Binance universe that carries an exit rule."""
    if universe.exchange != "binance":
        return []
    out: list[Watch] = []
    specs = [
        s
        for s in registry.list_competitors(conn, statuses=list(ACTIVE), universe=universe.name)
        if s.role == "competitor" and s.family in REGISTRY
    ]
    for spec in specs:
        held = bstore.last_targets(conn, spec.id)
        if not held:
            continue
        pos_ts, positions = held
        comp = build(spec, bar_hours=universe.bar_hours)
        state = load_state(conn, spec.id)
        comp.restore_state(state)
        if isinstance(comp, LadderHoldingCompetitor):
            entries, basis = state.get("entries") or {}, state.get("basis") or {}
            stop, ladder = comp._stop(), comp._ladder()
            for sym, (kind, w) in positions.items():
                e = entries.get(sym)
                if not e or not e.get("price"):
                    continue
                basket = {
                    universe.binance_symbol(s): float(p) for s, p in (basis.get(str(e.get("basis"))) or {}).items()
                }
                out.append(
                    Watch(
                        spec.id,
                        spec.name,
                        spec.family,
                        sym,
                        universe.binance_symbol(sym),
                        kind,
                        w,
                        float(e["price"]),
                        float(e.get("bars", 0.0)),
                        stop,
                        ladder,
                        basket,
                        dict(positions),
                        universe.bar_hours,
                    )
                )
            continue
        stop = spec.params.get("live_stop")
        ladder = _ladder_from(spec.params)
        if stop is None and ladder is None:
            continue
        for sym, (kind, w) in positions.items():
            since = bstore.entry_of_position(conn, spec.id, sym, w, pos_ts)
            price = _entry_close(conn, universe.exchange, sym, since)
            if not price:
                continue
            bars = (pos_ts - since).total_seconds() / 3600.0 / universe.bar_hours
            out.append(
                Watch(
                    spec.id,
                    spec.name,
                    spec.family,
                    sym,
                    universe.binance_symbol(sym),
                    kind,
                    w,
                    price,
                    bars,
                    abs(float(stop)) if stop is not None else None,
                    ladder,
                    {},
                    dict(positions),
                    universe.bar_hours,
                )
            )
    return out


def measure(w: Watch, prices: dict[str, float]) -> float | None:
    """The rule's measure at current prices: signed excess over the entry basket (or raw return)."""
    px = prices.get(w.pair)
    if not px or not w.entry_price:
        return None
    leg = px / w.entry_price - 1.0
    basket_ret = 0.0
    if w.basket:
        moves = [prices[p] / p0 - 1.0 for p, p0 in w.basket.items() if p0 and prices.get(p)]
        basket_ret = sum(moves) / len(moves) if moves else 0.0
    return w.side * (leg - basket_ret)


def verdict(w: Watch, prices: dict[str, float], now: datetime, last_bar: datetime | None) -> tuple[str, float] | None:
    """``("stop" | "roi", measure)`` when a rule fires, else None."""
    x = measure(w, prices)
    if x is None:
        return None
    if w.stop is not None and x <= -w.stop:
        return "stop", x
    if w.ladder is not None:
        elapsed = ((now - last_bar).total_seconds() / 3600.0 / w.bar_hours) if last_bar else 0.0
        target = w.ladder.target_at(int(w.bars_held + max(elapsed, 0.0)))
        if target > 0.0 and x >= target:
            return "roi", x
    return None


def execute(
    conn: psycopg.Connection,
    settings: Settings,
    spec: CompetitorSpec,
    w: Watch,
    price: float,
    reason: str,
    x: float,
    now: datetime,
    bar_hours: int = 1,
) -> int:
    """Close the leg now: fill, re-stated book, competitor told, alert queued. Returns the fill id."""
    comp = build(spec, bar_hours=bar_hours)
    comp.restore_state(load_state(conn, spec.id))
    comp.live_close(w.symbol, reason)
    save_state(conn, spec.id, now, comp.state())
    fill_id = bstore.write_fill(
        conn, spec.id, bstore.Fill(now, w.symbol, w.kind, w.weight, 0.0, price, reason, round(x, 6))
    )
    bstore.write_targets_snapshot(
        conn,
        spec.id,
        now,
        w.positions,
        w.symbol,
        {"live_exit": reason, "price": price, "excess": round(x, 5), "entry_price": w.entry_price},
    )
    side = "long" if w.weight > 0 else "short"
    bstore.add_alert(
        conn,
        Alert(
            kind="live_exit",
            competitor_id=spec.id,
            symbol=w.symbol,
            payload={
                "detail": f"{spec.name} closes {side} {w.symbol} ({abs(w.weight):.0%}) at {price:g}:"
                f" {reason}, {x:+.2%}",
                "reason": reason,
                "price": price,
                "excess": round(x, 5),
                "weight": w.weight,
            },
        ),
    )
    conn.commit()
    log.info("live exit %s %s %s @ %g (%s %+.2f%%)", spec.name, side, w.symbol, price, reason, x * 100)
    return fill_id


def _last_bar(conn: psycopg.Connection) -> datetime | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(ts) AS ts FROM books")
        return cur.fetchone()["ts"]


def run_once(
    conn: psycopg.Connection,
    settings: Settings,
    universes: list[Universe],
    prices: dict[str, float],
    now: datetime,
    watches: list[Watch] | None = None,
) -> list[int]:
    """Evaluate every watched leg once; returns the ids of the fills written."""
    last_bar = _last_bar(conn)
    specs = {s.id: s for s in registry.list_competitors(conn, statuses=list(ACTIVE))}
    fills: list[int] = []
    closed: set[int] = set()  # one fill per competitor per pass: its book must be re-read first
    for uni in universes:
        for w in watches if watches is not None else watchlist(conn, uni, now):
            if w.competitor_id in closed or w.competitor_id not in specs:
                continue
            v = verdict(w, prices, now, last_bar)
            if v is None:
                continue
            reason, x = v
            try:
                fills.append(
                    execute(conn, settings, specs[w.competitor_id], w, prices[w.pair], reason, x, now, w.bar_hours)
                )
                closed.add(w.competitor_id)
            except Exception:  # isolate competitors, as the tick does
                conn.rollback()
                log.exception("live exit failed for %s %s", w.name, w.symbol)
        if watches is not None:
            break
    if fills:
        names = {s.id: s.name for s in specs.values()}
        alerts.flush(conn, settings, now, names)
    return fills


def run_forever(
    conn: psycopg.Connection,
    settings: Settings,
    universes: list[Universe],
    client: httpx.Client,
    interval: float = 5.0,
    refresh: float = 60.0,
    tick_minutes: tuple[int, ...] = (5, 35),
) -> None:  # pragma: no cover - the loop; every piece is tested on its own
    watches: dict[str, list[Watch]] = {}
    stamp: tuple[datetime | None, float] = (None, 0.0)
    while True:
        now = datetime.now(UTC)
        try:
            if quiet(now, tick_minutes):
                time.sleep(interval)
                continue
            last_bar = _last_bar(conn)
            if last_bar != stamp[0] or time.monotonic() - stamp[1] > refresh:
                watches = {u.name: watchlist(conn, u, now) for u in universes}
                stamp = (last_bar, time.monotonic())
                log.info("watching %d legs", sum(len(v) for v in watches.values()))
            conn.commit()  # end the read transaction; a long-open one pins vacuum
            if any(watches.values()):
                prices = fetch_prices(client)
                for u in universes:
                    if run_once(conn, settings, [u], prices, now, watches.get(u.name, [])):
                        stamp = (None, 0.0)  # a book changed: re-read before the next pass
        except Exception:
            conn.rollback()
            log.exception("live pass failed")
        time.sleep(interval)

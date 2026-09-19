"""Load stored history into the frames the judge and the tick consume."""

from __future__ import annotations

from datetime import datetime, timedelta

import psycopg

from arena.core.snapshot import Snapshot
from arena.core.universe import Universe
from arena.judge.backtest import HistoryFrames
from arena.store import candles as cstore
from arena.store import news as nstore

EXCHANGE = "binance"
SCORER_VERSION = 1


def load_history(
    conn: psycopg.Connection, universe: Universe, start: datetime, end: datetime, with_news: bool = True
) -> HistoryFrames:
    syms = universe.symbols
    ex = universe.exchange
    candles = cstore.read_candles(conn, ex, syms, start, end, tf=universe.bar)
    has_funding = ex == "binance"  # classic markets have no perpetual funding nor open interest
    funding = cstore.read_funding(conn, ex, syms, start, end) if has_funding else None
    oi = cstore.read_open_interest(conn, ex, syms, start, end) if has_funding else None
    news = (
        nstore.news_features(conn, syms, max(start, end - timedelta(days=30)), end, SCORER_VERSION)
        if with_news
        else None
    )
    macro = nstore.macro_event_times(conn, start, end + timedelta(days=7))
    return HistoryFrames(candles=candles, funding=funding, open_interest=oi, news=news, macro_events=macro)


def snapshot_from_history(
    h: HistoryFrames, ts: datetime, symbols: list[str], hl_funding: dict[str, float] | None = None, bar_hours: int = 1
) -> Snapshot:
    return Snapshot.from_long(
        ts, symbols, h.candles, h.funding, h.open_interest, hl_funding, h.news, h.macro_events, bar_hours=bar_hours
    )

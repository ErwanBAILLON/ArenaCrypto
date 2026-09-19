"""Ingestion: public endpoints → store, idempotent, closed bars only."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd
import psycopg

from arena.core.universe import Universe
from arena.data import binance, hyperliquid, macro, rss, yahoo
from arena.nlp.scorer import VaderScorer
from arena.store import candles as cstore
from arena.store import news as nstore

log = logging.getLogger(__name__)
EXCHANGE = "binance"
FUNDING_LOOKBACK = timedelta(days=3)  # re-fetch window for idempotent funding refresh


def _ms(ts: datetime) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


def ingest_market(
    conn: psycopg.Connection, client: httpx.Client, universe: Universe, now: datetime, since: datetime | None = None
) -> dict[str, int]:
    """Market data of the universe's exchange for every symbol, from the last stored bar (or ``since``)."""
    if universe.exchange == "yahoo":
        return ingest_yahoo(conn, client, universe, now)
    counts = {"candles": 0, "funding": 0, "oi": 0}
    for sym in universe.symbols:
        bsym = universe.binance_symbol(sym)
        last = cstore.last_candle_ts(conn, EXCHANGE, sym)
        start = (last - timedelta(hours=2)) if last else (since or universe.history_start)
        try:
            k = binance.klines(client, bsym, _ms(start), now=now)
            if not k.empty:
                k["symbol"] = sym
                counts["candles"] += cstore.upsert_candles(conn, EXCHANGE, k)
            f_start = min(start, now - FUNDING_LOOKBACK)
            f = binance.funding(client, bsym, _ms(f_start))
            if not f.empty:
                f["symbol"] = sym
                counts["funding"] += cstore.upsert_funding(conn, EXCHANGE, f)
            oi = binance.open_interest_hist(client, bsym, limit=500 if last is None else 48)
            if not oi.empty:
                oi["symbol"] = sym
                counts["oi"] += cstore.upsert_open_interest(conn, EXCHANGE, oi)
            conn.commit()
        except Exception:  # one symbol must not stop the others
            conn.rollback()
            log.exception("ingest failed for %s", sym)
    try:
        ctx = hyperliquid.meta_and_asset_ctxs(client)
        ctx = ctx[ctx["coin"].isin([universe.hyperliquid_coin(s) for s in universe.symbols])]
        ts = pd.Timestamp(now).floor("1h")
        cstore.upsert_hl_snapshot(conn, ts, ctx)
        conn.commit()
    except Exception:
        conn.rollback()
        log.exception("hyperliquid snapshot failed")
    return counts


def ingest_yahoo(conn: psycopg.Connection, client: httpx.Client, universe: Universe, now: datetime) -> dict[str, int]:
    """Closed daily bars for every Yahoo ticker of a classic-markets universe (5y on first run)."""
    counts = {"candles": 0}
    for sym in universe.symbols:
        try:
            last = cstore.last_candle_ts(conn, "yahoo", sym, tf="1d")
            df = yahoo.daily(client, sym, range_="1mo" if last else "5y", now=now)
            if not df.empty:
                df["symbol"] = sym
                counts["candles"] += cstore.upsert_candles(conn, "yahoo", df, tf="1d")
            conn.commit()
        except Exception:
            conn.rollback()
            log.exception("yahoo ingest failed for %s", sym)
    return counts


def ingest_news(
    conn: psycopg.Connection, client: httpx.Client, now: datetime, scorer: VaderScorer | None = None
) -> dict[str, int]:
    """Fetch every RSS feed, store new articles point-in-time, score whatever is unscored."""
    scorer = scorer or VaderScorer()
    articles = rss.fetch_all(client, now)
    inserted = nstore.upsert_articles(conn, articles)
    pending = nstore.unscored_articles(conn, scorer.version)
    scores = [s for a in pending for s in scorer.score(a)]
    nstore.upsert_scores(conn, scores)
    conn.commit()
    return {"fetched": len(articles), "inserted": len(inserted), "scored": len(pending)}


def ingest_macro(conn: psycopg.Connection, client: httpx.Client, now: datetime) -> int:
    df = macro.calendar(client, now)
    n = nstore.upsert_macro_events(conn, df) if not df.empty else 0
    conn.commit()
    return int(n or 0)


def utc_now() -> datetime:
    return datetime.now(UTC)

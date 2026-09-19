"""News repository: articles, per-asset scores, macro calendar and hourly news features."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pandas as pd
import psycopg

from arena.core.types import Article, ArticleScore

MARKET = "MARKET"

FEATURE_COLS = ["symbol", "ts", "sent_24h", "sent_7d", "n_24h", "n_7d", "shock"]

SHOCK_INTENSITY = 0.8

_FEATURES_SQL = """
WITH grid AS (
    SELECT generate_series(%(start)s::timestamptz, %(end)s::timestamptz, interval '1 hour') AS ts
),
assets AS (
    SELECT unnest(%(assets)s::text[]) AS symbol
)
SELECT a.symbol,
       g.ts,
       coalesce(f.sent_24h, 0.0) AS sent_24h,
       coalesce(f.sent_7d, 0.0)  AS sent_7d,
       coalesce(f.n_24h, 0)      AS n_24h,
       coalesce(f.n_7d, 0)       AS n_7d,
       coalesce(f.shock, 0)      AS shock
  FROM grid g
 CROSS JOIN assets a
  LEFT JOIN LATERAL (
        SELECT avg(s.sentiment) FILTER (WHERE ar.published_at > g.ts - interval '24 hours') AS sent_24h,
               avg(s.sentiment)                                                             AS sent_7d,
               count(*)          FILTER (WHERE ar.published_at > g.ts - interval '24 hours') AS n_24h,
               count(*)                                                                     AS n_7d,
               max(CASE WHEN ar.published_at > g.ts - interval '24 hours'
                         AND s.intensity >= %(shock)s THEN 1 ELSE 0 END)                    AS shock
          FROM article_scores s
          JOIN articles ar ON ar.id = s.article_id
         WHERE s.asset = a.symbol
           AND s.scorer_version = %(version)s
           AND ar.published_at > g.ts - interval '7 days'
           AND ar.published_at <= g.ts
  ) f ON true
 ORDER BY a.symbol, g.ts
"""


def upsert_articles(conn: psycopg.Connection, articles: Sequence[Article]) -> list[int]:
    """Insert articles, skipping duplicates on (source, url, title); returns ids of new rows only."""
    ids: list[int] = []
    with conn.cursor() as cur:
        for a in articles:
            cur.execute(
                "INSERT INTO articles (source, url, title, summary, published_at, fetched_at)"
                " VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (source, url, title) DO NOTHING RETURNING id",
                (a.source, a.url, a.title, a.summary, a.published_at, a.fetched_at),
            )
            row = cur.fetchone()
            if row is not None:
                ids.append(int(row["id"]))
    return ids


def unscored_articles(conn: psycopg.Connection, scorer_version: int, limit: int = 500) -> list[Article]:
    """Oldest-first articles without any ``article_scores`` row for ``scorer_version``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, source, url, title, summary, published_at, fetched_at FROM articles a"
            " WHERE NOT EXISTS (SELECT 1 FROM article_scores s"
            "                   WHERE s.article_id = a.id AND s.scorer_version = %s)"
            " ORDER BY published_at, id LIMIT %s",
            (scorer_version, limit),
        )
        return [Article(**r) for r in cur.fetchall()]


def upsert_scores(conn: psycopg.Connection, scores: Sequence[ArticleScore]) -> None:
    """Insert or replace scores keyed on (article_id, asset, scorer_version)."""
    if not scores:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO article_scores (article_id, asset, sentiment, event_type, intensity, scorer_version)"
            " VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (article_id, asset, scorer_version) DO UPDATE SET"
            " sentiment = EXCLUDED.sentiment, event_type = EXCLUDED.event_type, intensity = EXCLUDED.intensity",
            [(s.article_id, s.asset, s.sentiment, s.event_type, s.intensity, s.scorer_version) for s in scores],
        )


def upsert_macro_events(conn: psycopg.Connection, df: pd.DataFrame) -> int:
    """Insert macro calendar rows (columns ts, currency, title, impact); returns rows inserted."""
    if df.empty:
        return 0
    ts = pd.to_datetime(df["ts"], utc=True).dt.to_pydatetime()
    rows = [
        (t, str(c), str(title), str(impact))
        for t, c, title, impact in zip(ts, df["currency"], df["title"], df["impact"])
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO macro_events (ts, currency, title, impact) VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (ts, currency, title) DO NOTHING",
            rows,
        )
        return cur.rowcount


def macro_event_times(conn: psycopg.Connection, start: datetime, end: datetime) -> pd.DatetimeIndex:
    """Distinct macro event timestamps within ``[start, end]`` as a tz-aware UTC index."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT ts FROM macro_events WHERE ts BETWEEN %s AND %s ORDER BY ts", (start, end)
        )
        times = [r["ts"] for r in cur.fetchall()]
    return pd.DatetimeIndex(pd.to_datetime(times, utc=True))


def news_features(
    conn: psycopg.Connection,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    scorer_version: int,
) -> pd.DataFrame:
    """Hourly news features per asset (plus ``MARKET``) on the grid ``start..end``.

    For each hour ``h``: ``sent_24h``/``sent_7d`` average sentiment of articles
    published in ``(h-24h, h]`` / ``(h-7d, h]``, ``n_*`` the matching counts and
    ``shock`` = 1 when an article of the last 24h has intensity >= 0.8. Hours
    without articles yield zeros.
    """
    assets = list(dict.fromkeys([*symbols, MARKET]))
    with conn.cursor() as cur:
        cur.execute(
            _FEATURES_SQL,
            {"start": start, "end": end, "assets": assets, "version": scorer_version, "shock": SHOCK_INTENSITY},
        )
        df = pd.DataFrame(cur.fetchall(), columns=FEATURE_COLS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["sent_24h"] = df["sent_24h"].astype(float)
    df["sent_7d"] = df["sent_7d"].astype(float)
    df["n_24h"] = df["n_24h"].astype(int)
    df["n_7d"] = df["n_7d"].astype(int)
    df["shock"] = df["shock"].astype(int)
    return df

"""Crypto news RSS feeds (HTTPS only) parsed into point-in-time Articles."""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone

import feedparser
import httpx

from arena.core.types import Article

log = logging.getLogger(__name__)

FEEDS: list[tuple[str, str]] = [
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("theblock", "https://www.theblock.co/rss.xml"),
    ("decrypt", "https://decrypt.co/feed"),
    ("bitcoinmagazine", "https://bitcoinmagazine.com/feed"),
    ("thedefiant", "https://thedefiant.io/api/feed"),
]

SUMMARY_MAX = 2000
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove tags, unescape entities and collapse whitespace."""
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", text))).strip()


def _published(entry: feedparser.FeedParserDict, fallback: datetime) -> datetime:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
    return fallback


def fetch_feed(client: httpx.Client, source: str, url: str, now: datetime) -> list[Article]:
    """Download one feed and convert its entries; entries without a link are skipped."""
    resp = client.get(url)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.text)
    articles: list[Article] = []
    for entry in parsed.entries:
        link = entry.get("link")
        if not link:
            continue
        summary = strip_html(entry.get("summary") or entry.get("description") or "")[:SUMMARY_MAX]
        articles.append(
            Article(
                source=source,
                url=link,
                title=strip_html(entry.get("title") or ""),
                summary=summary,
                published_at=_published(entry, now),
                fetched_at=now,
            )
        )
    return articles


def fetch_all(client: httpx.Client, now: datetime, feeds: list[tuple[str, str]] = FEEDS) -> list[Article]:
    """Fetch every feed; a failing feed is logged and skipped so one outage never blocks ingestion."""
    articles: list[Article] = []
    for source, url in feeds:
        try:
            articles.extend(fetch_feed(client, source, url, now))
        except Exception:  # noqa: BLE001 - per-feed isolation is the point
            log.warning("rss feed %s failed", source, exc_info=True)
    return articles

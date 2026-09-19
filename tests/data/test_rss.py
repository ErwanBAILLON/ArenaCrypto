"""RSS adapter: date fallback, HTML stripping, per-feed failure tolerance."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from arena.data import rss

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)

FEED_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Test</title>
<item>
  <title>Bitcoin &amp; ETF approved</title>
  <link>https://example.com/a</link>
  <description><![CDATA[<p>Big <b>news</b>&nbsp;today.</p><br/>More &lt;here&gt;.]]></description>
  <pubDate>Fri, 18 Sep 2026 08:30:00 +0200</pubDate>
</item>
<item>
  <title>No date here</title>
  <link>https://example.com/b</link>
  <description>%s</description>
</item>
<item>
  <title>No link, skipped</title>
  <description>x</description>
</item>
</channel></rss>
""" % ("y" * 2500)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_feed_parses_dates_and_strips_html():
    arts = rss.fetch_feed(_client(lambda r: httpx.Response(200, text=FEED_XML)), "test", "https://x/feed", NOW)
    assert len(arts) == 2
    a, b = arts
    assert a.source == "test" and a.url == "https://example.com/a"
    assert a.title == "Bitcoin & ETF approved"
    assert a.summary == "Big news today. More <here>."
    assert a.published_at == datetime(2026, 9, 18, 6, 30, tzinfo=timezone.utc)  # +0200 normalised to UTC
    assert a.fetched_at == NOW
    # missing date -> fallback to now ; long summary truncated
    assert b.published_at == NOW
    assert len(b.summary) == rss.SUMMARY_MAX


def test_fetch_all_tolerates_failing_feed(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        if "bad" in request.url.host:
            return httpx.Response(503)
        if "boom" in request.url.host:
            raise httpx.ConnectError("down", request=request)
        return httpx.Response(200, text=FEED_XML)

    feeds = [("good", "https://good/feed"), ("bad", "https://bad/feed"), ("boom", "https://boom/feed"),
             ("good2", "https://good2/feed")]
    with caplog.at_level("WARNING", logger="arena.data.rss"):
        arts = rss.fetch_all(_client(handler), NOW, feeds=feeds)
    assert sorted({a.source for a in arts}) == ["good", "good2"]
    assert len(arts) == 4
    failed = {r.getMessage() for r in caplog.records}
    assert failed == {"rss feed bad failed", "rss feed boom failed"}


def test_feeds_are_https_only():
    assert rss.FEEDS and all(url.startswith("https://") for _, url in rss.FEEDS)
    assert len({src for src, _ in rss.FEEDS}) == len(rss.FEEDS)

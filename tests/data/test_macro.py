"""Macro calendar adapter: High-impact filter and offset-aware timestamps."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from arena.data import macro

NOW = datetime(2026, 9, 19, tzinfo=UTC)
EVENTS = [
    {"title": "CPI m/m", "country": "USD", "date": "2026-09-22T08:30:00-04:00", "impact": "High", "forecast": "0.3%"},
    {"title": "Bank Holiday", "country": "JPY", "date": "2026-09-21T00:00:00-04:00", "impact": "Holiday"},
    {"title": "FOMC Statement", "country": "USD", "date": "2026-09-23T14:00:00-04:00", "impact": "High"},
    {"title": "Trade Balance", "country": "EUR", "date": "2026-09-21T05:00:00-04:00", "impact": "Low"},
    {"title": "Retail Sales", "country": "GBP", "date": "2026-09-24T02:00:00-04:00", "impact": "Medium"},
]


def test_calendar_keeps_high_only_and_converts_to_utc():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == macro.CALENDAR_URL
        return httpx.Response(200, json=EVENTS)

    df = macro.calendar(httpx.Client(transport=httpx.MockTransport(handler)), NOW)
    assert list(df.columns) == ["ts", "currency", "title", "impact"]
    assert df["title"].tolist() == ["CPI m/m", "FOMC Statement"]
    assert set(df["impact"]) == {"High"}
    assert str(df["ts"].dt.tz) == "UTC"
    assert df["ts"].iloc[0] == datetime(2026, 9, 22, 12, 30, tzinfo=UTC)


def test_calendar_empty_when_nothing_high():
    low = [e for e in EVENTS if e["impact"] != "High"]
    df = macro.calendar(httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=low))), NOW)
    assert df.empty and list(df.columns) == macro.COLUMNS
    assert str(df["ts"].dtype) == "datetime64[ns, UTC]"

"""HTTP helper: retry on transport errors / 5xx / 429 with backoff, raise after retries."""

from __future__ import annotations

import httpx
import pytest

from arena.data import http


@pytest.fixture
def sleeps(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(http, "_sleep", calls.append)
    return calls


def test_retries_then_succeeds(sleeps):
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectTimeout("slow", request=request)
        if len(attempts) == 2:
            return httpx.Response(429)
        return httpx.Response(200, json={"ok": True})

    out = http.get_json(httpx.Client(transport=httpx.MockTransport(handler)), "https://x/y", params={"a": 1})
    assert out == {"ok": True} and len(attempts) == 3
    assert sleeps == [0.5, 1.0]


def test_raises_after_exhausting_retries(sleeps):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        http.post_json(client, "https://x/y", json={"type": "x"})
    assert sleeps == [0.5, 1.0]


def test_client_error_not_retried(sleeps):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(httpx.HTTPStatusError):
        http.get_json(client, "https://x/y")
    assert sleeps == []

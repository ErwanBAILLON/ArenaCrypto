"""Shared HTTP client with retries. Every adapter goes through this module."""

from __future__ import annotations

import time
from typing import Any

import httpx

USER_AGENT = "arena/0.1"
BACKOFF_BASE_S = 0.5
RETRYABLE_STATUS = {429} | set(range(500, 600))

_sleep = time.sleep  # module-level so tests can neutralise the backoff


def make_client(timeout: float = 10.0) -> httpx.Client:
    """Return an httpx client identified as arena with a global timeout."""
    return httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": USER_AGENT})


def _request_json(client: httpx.Client, retries: int, **kwargs: Any) -> Any:
    """Send a request, retrying on transport errors, 5xx and 429 with exponential backoff.

    Backoff sleeps 0.5s, 1s, 2s… between attempts. The last error is re-raised
    once `retries` attempts are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.request(**kwargs)
        except httpx.TransportError as exc:
            last_exc = exc
        else:
            if resp.status_code not in RETRYABLE_STATUS:
                resp.raise_for_status()
                return resp.json()
            last_exc = httpx.HTTPStatusError(
                f"retryable status {resp.status_code}", request=resp.request, response=resp
            )
        if attempt < retries - 1:
            _sleep(BACKOFF_BASE_S * 2**attempt)
    assert last_exc is not None
    raise last_exc


def get_json(client: httpx.Client, url: str, params: dict[str, Any] | None = None, retries: int = 3) -> Any:
    """GET `url` and return the decoded JSON body."""
    return _request_json(client, retries, method="GET", url=url, params=params)


def post_json(client: httpx.Client, url: str, json: Any, retries: int = 3) -> Any:
    """POST a JSON payload to `url` and return the decoded JSON body."""
    return _request_json(client, retries, method="POST", url=url, json=json)

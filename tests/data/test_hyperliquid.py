"""Hyperliquid adapter: universe/ctxs pairing and funding history."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from arena.data import hyperliquid

META = {
    "universe": [{"name": "BTC", "szDecimals": 5}, {"name": "ETH", "szDecimals": 4}, {"name": "SOL", "szDecimals": 2}]
}
CTXS = [
    {"funding": "0.0000125", "openInterest": "1234.5", "markPx": "65000.5", "oraclePx": "65000"},
    {"funding": "-0.00002", "openInterest": "9876.0", "markPx": "3500.25", "oraclePx": "3500"},
    {"funding": "0.00005", "openInterest": "5.0", "markPx": "150.0", "oraclePx": "150"},
]


def test_meta_and_asset_ctxs_pairs_by_position():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST" and request.url == hyperliquid.INFO_URL
        assert json.loads(request.content) == {"type": "metaAndAssetCtxs"}
        return httpx.Response(200, json=[META, CTXS])

    df = hyperliquid.meta_and_asset_ctxs(httpx.Client(transport=httpx.MockTransport(handler)))
    assert list(df.columns) == ["coin", "funding", "oi", "mark"]
    assert df["coin"].tolist() == ["BTC", "ETH", "SOL"]
    eth = df.set_index("coin").loc["ETH"]
    assert eth["funding"] == pytest.approx(-0.00002)
    assert eth["oi"] == pytest.approx(9876.0)
    assert eth["mark"] == pytest.approx(3500.25)


def test_funding_history():
    t0 = 1_704_067_200_000
    rows = [
        {"coin": "BTC", "fundingRate": "0.0000125", "premium": "0.0", "time": t0 + 3_600_000},
        {"coin": "BTC", "fundingRate": "0.0000100", "premium": "0.0", "time": t0},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"type": "fundingHistory", "coin": "BTC", "startTime": t0}
        return httpx.Response(200, json=rows)

    df = hyperliquid.funding_history(httpx.Client(transport=httpx.MockTransport(handler)), "BTC", t0)
    assert list(df.columns) == ["ts", "rate"]
    assert df["ts"].iloc[0] == datetime(2024, 1, 1, tzinfo=UTC)
    assert str(df["ts"].dt.tz) == "UTC"
    assert df["rate"].tolist() == pytest.approx([0.00001, 0.0000125])


def test_funding_history_empty():
    df = hyperliquid.funding_history(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[]))), "BTC", 0
    )
    assert df.empty and str(df["ts"].dtype) == "datetime64[ns, UTC]"

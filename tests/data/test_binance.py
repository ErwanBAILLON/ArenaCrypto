"""Binance adapter: paging, closed-bar filter, tz-aware timestamps."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from arena.data import binance
from arena.data.http import make_client

H = 3_600_000
T0 = 1_704_067_200_000  # 2024-01-01T00:00:00Z


def _kline(open_ms: int, px: float = 100.0) -> list:
    return [open_ms, str(px), str(px + 1), str(px - 1), str(px + 0.5), "10.0", open_ms + H - 1, "0", 1, "0", "0", "0"]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": "arena/0.1"})


def test_klines_pages_and_filters_open_bar():
    calls: list[dict] = []
    first_page = [_kline(T0 + i * H, 100 + i) for i in range(binance.PAGE)]
    second_page = [_kline(T0 + (binance.PAGE + i) * H) for i in range(3)]

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        assert request.url.path == "/fapi/v1/klines"
        assert params["symbol"] == "BTCUSDT"
        page = first_page if int(params["startTime"]) == T0 else second_page
        return httpx.Response(200, json=page)

    # "now" sits inside the very last candle: it must be dropped as still forming.
    now = datetime.fromtimestamp((T0 + (binance.PAGE + 2) * H + 60_000) / 1000, tz=timezone.utc)
    df = binance.klines(_client(handler), "BTCUSDT", T0, now=now)

    assert len(calls) == 2
    assert int(calls[1]["startTime"]) == T0 + binance.PAGE * H
    assert len(df) == binance.PAGE + 2
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert str(df["ts"].dt.tz) == "UTC"
    # ts is close time rounded up: first candle opened 00:00 -> labelled 01:00
    assert df["ts"].iloc[0] == datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
    assert df["open"].iloc[1] == 101.0 and df["close"].dtype == "float64"


def test_klines_empty_response():
    df = binance.klines(_client(lambda r: httpx.Response(200, json=[])), "BTCUSDT", T0)
    assert df.empty and list(df.columns) == binance.KLINE_COLUMNS
    assert str(df["ts"].dtype) == "datetime64[ns, UTC]"


def test_funding_pages_until_short_page():
    calls = []
    page1 = [{"symbol": "ETHUSDT", "fundingTime": T0 + i * 8 * H, "fundingRate": "0.0001", "markPrice": "1"}
             for i in range(binance.PAGE)]
    page2 = [{"symbol": "ETHUSDT", "fundingTime": T0 + binance.PAGE * 8 * H, "fundingRate": "-0.0002", "markPrice": "1"}]

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        assert request.url.path == "/fapi/v1/fundingRate"
        return httpx.Response(200, json=page1 if len(calls) == 1 else page2)

    df = binance.funding(_client(handler), "ETHUSDT", T0)
    assert len(calls) == 2
    assert int(calls[1]["startTime"]) == page1[-1]["fundingTime"] + 1
    assert len(df) == binance.PAGE + 1
    assert df["rate"].iloc[-1] == pytest.approx(-0.0002)
    assert df["ts"].is_monotonic_increasing and str(df["ts"].dt.tz) == "UTC"


def test_open_interest_hist():
    body = [
        {"symbol": "BTCUSDT", "sumOpenInterest": "20403.63", "sumOpenInterestValue": "1e9", "timestamp": T0 + H},
        {"symbol": "BTCUSDT", "sumOpenInterest": "20500.00", "sumOpenInterestValue": "1e9", "timestamp": T0},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/futures/data/openInterestHist"
        assert request.url.params["period"] == "1h"
        return httpx.Response(200, json=body)

    df = binance.open_interest_hist(_client(handler), "BTCUSDT")
    assert list(df.columns) == ["ts", "oi"]
    assert df["oi"].tolist() == [20500.0, 20403.63]  # sorted by ts
    assert df["ts"].iloc[0] == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_make_client_user_agent():
    with make_client() as c:
        assert c.headers["User-Agent"] == "arena/0.1"
        assert c.timeout.read == 10.0

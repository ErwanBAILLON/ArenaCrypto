"""Binance USD-M futures public endpoints: 1h klines, funding rates, open interest history.

Functions take the exchange symbol (e.g. "BTCUSDT"); callers map from the
universe with `Universe.binance_symbol`. All timestamps returned are UTC
tz-aware pandas Timestamps.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd

from arena.data.http import get_json

BASE = "https://fapi.binance.com"
PAGE = 1000
INTERVAL_MS = {"1h": 3_600_000}

KLINE_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


def _utc(ms: pd.Series | list[int]) -> pd.Series:
    return pd.to_datetime(pd.Series(ms, dtype="int64"), unit="ms", utc=True)


def _empty(columns: list[str]) -> pd.DataFrame:
    """Typed empty frame: tz-aware `ts`, float64 for everything else."""
    return pd.DataFrame({
        c: pd.Series(dtype="datetime64[ns, UTC]" if c == "ts" else "float64") for c in columns
    })


def klines(
    client: httpx.Client,
    symbol: str,
    start_ms: int,
    end_ms: int | None = None,
    interval: str = "1h",
    now: datetime | None = None,
) -> pd.DataFrame:
    """Closed candles in [start_ms, end_ms] as DataFrame[ts, open, high, low, close, volume].

    `ts` is the bar close time rounded up to the interval (open_time + interval),
    so the candle labelled 13:00 covers 12:00–13:00. Candles whose close_time is
    not strictly before `now` (the still-forming bar) are dropped.
    """
    now_ms = int((now or datetime.now(timezone.utc)).timestamp() * 1000)
    step = INTERVAL_MS[interval]
    rows: list[list] = []
    cursor = start_ms
    while True:
        params: dict = {"symbol": symbol, "interval": interval, "startTime": cursor, "limit": PAGE}
        if end_ms is not None:
            params["endTime"] = end_ms
        page = get_json(client, f"{BASE}/fapi/v1/klines", params=params)
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE:
            break
        cursor = int(page[-1][0]) + step
        if end_ms is not None and cursor > end_ms:
            break
    if not rows:
        return _empty(KLINE_COLUMNS)
    raw = pd.DataFrame(rows).iloc[:, :7]
    raw.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    raw = raw[raw["close_time"].astype("int64") < now_ms]
    out = pd.DataFrame({"ts": _utc(raw["open_time"].astype("int64") + step)})
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = raw[col].astype("float64").to_numpy()
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def funding(client: httpx.Client, symbol: str, start_ms: int, end_ms: int | None = None) -> pd.DataFrame:
    """Realised 8h funding rates as DataFrame[ts, rate], paging 1000 per call."""
    rows: list[dict] = []
    cursor = start_ms
    while True:
        params: dict = {"symbol": symbol, "startTime": cursor, "limit": PAGE}
        if end_ms is not None:
            params["endTime"] = end_ms
        page = get_json(client, f"{BASE}/fapi/v1/fundingRate", params=params)
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE:
            break
        cursor = int(page[-1]["fundingTime"]) + 1
        if end_ms is not None and cursor > end_ms:
            break
    if not rows:
        return _empty(["ts", "rate"])
    out = pd.DataFrame({
        "ts": _utc([int(r["fundingTime"]) for r in rows]),
        "rate": [float(r["fundingRate"]) for r in rows],
    })
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def open_interest_hist(client: httpx.Client, symbol: str, period: str = "1h", limit: int = 500) -> pd.DataFrame:
    """Aggregated open interest (contracts) as DataFrame[ts, oi]. API keeps ~30 days only."""
    page = get_json(
        client,
        f"{BASE}/futures/data/openInterestHist",
        params={"symbol": symbol, "period": period, "limit": limit},
    )
    if not page:
        return _empty(["ts", "oi"])
    out = pd.DataFrame({
        "ts": _utc([int(r["timestamp"]) for r in page]),
        "oi": [float(r["sumOpenInterest"]) for r in page],
    })
    return out.sort_values("ts").reset_index(drop=True)

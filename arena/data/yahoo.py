"""Yahoo Finance public chart endpoint (no key): daily bars for classic markets.

``https://query1.finance.yahoo.com/v8/finance/chart/<symbol>`` returns up to
5 years of daily bars for equities, indices, futures and FX. Bars are labelled
by the session's *open* timestamp in the exchange time zone; we relabel each
daily bar to midnight UTC of the **next** calendar day so that, like our hourly
Binance candles, a bar is stamped by the time at which it is closed and known.
Only fully closed sessions are returned (the current day's partial bar is
dropped).
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pandas as pd

from arena.data.http import get_json

BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


def _empty() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=float) for c in COLUMNS})
    df["ts"] = pd.Series(dtype="datetime64[ns, UTC]")
    return df


def daily(client: httpx.Client, symbol: str, range_: str = "5y", now: datetime | None = None) -> pd.DataFrame:
    """Closed daily bars for ``symbol`` (Yahoo ticker such as ``SPY``, ``^GDAXI``, ``GC=F``, ``EURUSD=X``)."""
    now = now or datetime.now(UTC)
    payload = get_json(
        client, f"{BASE}/{symbol}", params={"interval": "1d", "range": range_, "includePrePost": "false"}
    )
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return _empty()
    res = result[0]
    stamps = res.get("timestamp") or []
    quote = (res.get("indicators") or {}).get("quote") or [{}]
    q = quote[0]
    if not stamps:
        return _empty()
    tz = (res.get("meta") or {}).get("exchangeTimezoneName") or "UTC"
    opened = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(tz)
    # session date in exchange tz -> bar known at next midnight UTC
    session_day = opened.normalize().tz_localize(None)
    ts = (pd.DatetimeIndex(session_day) + pd.Timedelta(days=1)).tz_localize("UTC").as_unit("ns")
    df = pd.DataFrame(
        {
            "ts": ts,
            "open": pd.to_numeric(pd.Series(q.get("open")), errors="coerce").to_numpy(),
            "high": pd.to_numeric(pd.Series(q.get("high")), errors="coerce").to_numpy(),
            "low": pd.to_numeric(pd.Series(q.get("low")), errors="coerce").to_numpy(),
            "close": pd.to_numeric(pd.Series(q.get("close")), errors="coerce").to_numpy(),
            "volume": pd.to_numeric(pd.Series(q.get("volume")), errors="coerce").fillna(0.0).to_numpy(),
        }
    )
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["ts"] <= pd.Timestamp(now)]  # drop the still-open session
    df = df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    return df[COLUMNS]

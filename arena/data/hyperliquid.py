"""Hyperliquid public info endpoint: current perp context and funding history.

Coins use the universe symbol directly ("BTC"). Timestamps are UTC tz-aware.
"""

from __future__ import annotations

import httpx
import pandas as pd

from arena.data.http import post_json

INFO_URL = "https://api.hyperliquid.xyz/info"


def meta_and_asset_ctxs(client: httpx.Client) -> pd.DataFrame:
    """Snapshot of every listed perp as DataFrame[coin, funding, oi, mark].

    The API returns `[meta, ctxs]`; `meta["universe"][i]["name"]` pairs with
    `ctxs[i]` by position.
    """
    meta, ctxs = post_json(client, INFO_URL, json={"type": "metaAndAssetCtxs"})
    names = [asset["name"] for asset in meta["universe"]]
    return pd.DataFrame(
        {
            "coin": names,
            "funding": [float(c["funding"]) for c in ctxs[: len(names)]],
            "oi": [float(c["openInterest"]) for c in ctxs[: len(names)]],
            "mark": [float(c["markPx"]) for c in ctxs[: len(names)]],
        }
    )


def funding_history(client: httpx.Client, coin: str, start_ms: int, end_ms: int | None = None) -> pd.DataFrame:
    """Hourly funding rates paid on `coin` since `start_ms` as DataFrame[ts, rate]."""
    payload: dict = {"type": "fundingHistory", "coin": coin, "startTime": start_ms}
    if end_ms is not None:
        payload["endTime"] = end_ms
    rows = post_json(client, INFO_URL, json=payload)
    if not rows:
        return pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns, UTC]"), "rate": pd.Series(dtype="float64")})
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(pd.Series([int(r["time"]) for r in rows], dtype="int64"), unit="ms", utc=True),
            "rate": [float(r["fundingRate"]) for r in rows],
        }
    )
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

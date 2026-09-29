"""OKX public data: the one venue that serves liquidations over plain REST.

Binance publishes forced orders only on its futures WebSocket, which is mute
from the cluster (the handshake succeeds, nothing ever arrives -- verified
2026-09-29 on ``!forceOrder@arr`` and ``btcusdt@aggTrade`` alike, while the
spot stream flows). OKX returns the recent filled liquidations of a perpetual
family with a GET, no key, so that is what the arena stores: not Binance's
liquidations, but the same cascades on the same coins at the same minutes,
which is what a positioning signal needs.

The endpoint returns the most recent orders (100 a page, paged with ``after``
on the last timestamp); a pull every hour on a quiet coin catches everything,
a cascade on BTC may exceed what a few pages hold. What is caught is stored
by (symbol, ts, side, price, qty), so pulling twice never double counts.
"""

from __future__ import annotations

import pandas as pd

from arena.data.http import get_json

BASE = "https://www.okx.com/api/v5/public"
COLUMNS = ["ts", "side", "price", "qty", "notional"]
MAX_PAGES = 5


def swap_family(binance_symbol: str) -> str:
    """``BTCUSDT`` -> ``BTC-USDT`` (the OKX ``uly`` of the USDT perpetual)."""
    for quote in ("USDT", "USDC", "BUSD"):
        if binance_symbol.endswith(quote):
            return f"{binance_symbol[: -len(quote)]}-{quote}"
    return f"{binance_symbol}-USDT"


def liquidations(
    client, binance_symbol: str, since: pd.Timestamp | None = None, pages: int = MAX_PAGES
) -> pd.DataFrame:
    """Filled liquidations of the coin's USDT perpetual, newest first from OKX, returned oldest first.

    DataFrame[ts, side, price, qty, notional]; ``side`` is the liquidated
    position ("long" or "short"), ``notional`` is price × contracts (OKX USDT
    swaps are quoted per contract; the coin-sized contract multiplier is left
    to the reader, the ratio long/short being what the signal uses).
    """
    uly = swap_family(binance_symbol)
    rows: list[dict] = []
    after: str | None = None
    for _ in range(max(1, pages)):
        params = {"instType": "SWAP", "uly": uly, "state": "filled", "limit": "100"}
        if after:
            params["after"] = after
        body = get_json(client, f"{BASE}/liquidation-orders", params=params)
        data = (body or {}).get("data") or []
        details = [d for item in data for d in item.get("details", [])]
        if not details:
            break
        for d in details:
            ts = pd.Timestamp(int(d["ts"]), unit="ms", tz="UTC")
            rows.append(
                {
                    "ts": ts,
                    "side": str(d.get("posSide") or ("long" if d.get("side") == "sell" else "short")),
                    "price": float(d["bkPx"]),
                    "qty": float(d["sz"]),
                    "notional": float(d["bkPx"]) * float(d["sz"]),
                }
            )
        oldest = min(int(d["ts"]) for d in details)
        if since is not None and pd.Timestamp(oldest, unit="ms", tz="UTC") <= since:
            break
        after = str(oldest)
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates(["ts", "side", "price", "qty"])
    if since is not None:
        out = out[out["ts"] > since]
    return out.sort_values("ts").reset_index(drop=True)

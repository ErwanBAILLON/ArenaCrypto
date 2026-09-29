"""OKX liquidations, offline."""

import pandas as pd

from arena.data import okx
from tests.data.test_positioning import _Client


def _page(rows):
    return {"code": "0", "data": [{"instFamily": "BTC-USDT", "details": rows}]}


def test_swap_family_maps_binance_names():
    assert okx.swap_family("BTCUSDT") == "BTC-USDT"
    assert okx.swap_family("1000PEPEUSDT") == "1000PEPE-USDT"


def test_liquidations_parse_dedupe_and_filter_since():
    t0 = 1_790_659_173_876
    rows = [
        {"bkPx": "83305.2", "sz": "0.01", "posSide": "short", "side": "buy", "ts": str(t0)},
        {"bkPx": "83305.2", "sz": "0.01", "posSide": "short", "side": "buy", "ts": str(t0)},  # repeated across pages
        {"bkPx": "83100.0", "sz": "2", "posSide": "long", "side": "sell", "ts": str(t0 - 3_600_000)},
    ]
    frame = okx.liquidations(_Client({"liquidation-orders": _page(rows)}), "BTCUSDT", pages=1)
    assert list(frame.columns) == okx.COLUMNS and len(frame) == 2
    assert frame["ts"].is_monotonic_increasing and str(frame["ts"].dt.tz) == "UTC"
    assert frame.iloc[0]["side"] == "long" and frame.iloc[0]["notional"] == 83100.0 * 2
    since = pd.Timestamp(t0 - 1_800_000, unit="ms", tz="UTC")
    assert len(okx.liquidations(_Client({"liquidation-orders": _page(rows)}), "BTCUSDT", since=since, pages=1)) == 1


def test_empty_body_gives_empty_frame():
    frame = okx.liquidations(_Client({"liquidation-orders": {"code": "0", "data": []}}), "ETHUSDT")
    assert frame.empty and list(frame.columns) == okx.COLUMNS

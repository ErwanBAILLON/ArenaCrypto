"""Positioning endpoints, offline."""

import pandas as pd

from arena.data import binance


class _Client:
    def __init__(self, pages):
        self.pages = pages

    def request(self, method, url, params=None, **kw):  # get_json goes through client.request
        class R:
            status_code = 200

            def __init__(self, body):
                self._b = body

            def raise_for_status(self):
                pass

            def json(self):
                return self._b

        for key, body in self.pages.items():
            if key in url:
                return R(body)
        return R([])


def test_positioning_merges_the_three_ratios_on_timestamp():
    ts = 1_790_251_200_000
    client = _Client(
        {
            "globalLongShortAccountRatio": [{"timestamp": ts, "longShortRatio": "1.25"}],
            "topLongShortPositionRatio": [{"timestamp": ts, "longShortRatio": "1.97"}],
            "takerlongshortRatio": [{"timestamp": ts, "buySellRatio": "1.11"}],
        }
    )
    frame = binance.positioning(client, "BTCUSDT")
    assert list(frame.columns) == ["ts", "global_ls_ratio", "top_ls_ratio", "taker_bs_ratio"]
    assert frame.iloc[0]["global_ls_ratio"] == 1.25 and frame.iloc[0]["taker_bs_ratio"] == 1.11
    assert frame["ts"].iloc[0] == pd.Timestamp(ts, unit="ms", tz="UTC")


def test_a_missing_endpoint_leaves_a_nan_column_not_a_crash():
    ts = 1_790_251_200_000
    client = _Client({"globalLongShortAccountRatio": [{"timestamp": ts, "longShortRatio": "1.25"}]})
    frame = binance.positioning(client, "BTCUSDT")
    assert len(frame) == 1 and frame["top_ls_ratio"].isna().all()


def test_positioning_rows_are_stored_once(conn):
    from arena.store.candles import upsert_positioning

    frame = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "ts": [pd.Timestamp("2026-09-24T12:00:00Z")],
            "global_ls_ratio": [1.2],
            "top_ls_ratio": [float("nan")],
            "taker_bs_ratio": [0.9],
        }
    )
    assert upsert_positioning(conn, "binance", frame) == 1
    assert upsert_positioning(conn, "binance", frame) == 0
    with conn.cursor() as cur:
        cur.execute("SELECT top_ls_ratio, taker_bs_ratio FROM positioning")
        row = cur.fetchone()
    assert row["top_ls_ratio"] is None and row["taker_bs_ratio"] == 0.9

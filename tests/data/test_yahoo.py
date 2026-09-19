from datetime import UTC, datetime

import httpx
import pandas as pd

from arena.data.yahoo import daily

# two NY sessions (open 13:30 UTC = 09:30 ET) + one still-open session today
_STAMPS = [1_726_752_600, 1_726_839_000, 1_758_288_600]  # 2024-09-19, 2024-09-20, 2025-09-19 09:30 ET


def _payload(stamps):
    n = len(stamps)
    return {
        "chart": {
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                    "timestamp": stamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0] * n,
                                "high": [101.0] * n,
                                "low": [99.0] * n,
                                "close": [100.5, 100.0, 102.0][:n],
                                "volume": [1000] * n,
                            }
                        ]
                    },
                }
            ]
        }
    }


def _client(payload):
    return httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload)))


def test_daily_relabels_to_next_midnight_utc_and_drops_open_session():
    now = datetime(2025, 9, 19, 15, 0, tzinfo=UTC)  # during the third session
    df = daily(_client(_payload(_STAMPS)), "SPY", now=now)
    assert list(df["ts"]) == [pd.Timestamp("2024-09-20", tz="UTC"), pd.Timestamp("2024-09-21", tz="UTC")]
    assert df["close"].tolist() == [100.5, 100.0]
    assert str(df["ts"].dtype) == "datetime64[ns, UTC]"


def test_daily_empty_and_nan_rows():
    assert daily(_client({"chart": {"result": []}}), "NOPE").empty
    p = _payload(_STAMPS[:2])
    p["chart"]["result"][0]["indicators"]["quote"][0]["close"][0] = None
    df = daily(_client(p), "SPY", now=datetime(2025, 1, 1, tzinfo=UTC))
    assert len(df) == 1

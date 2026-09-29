from datetime import UTC, datetime

import pandas as pd

from arena.store import candles as repo


def test_liquidations_upsert_is_idempotent_and_last_ts_reads_back(conn):
    ts = datetime(2026, 9, 29, 5, tzinfo=UTC)
    frame = pd.DataFrame(
        [
            {"symbol": "BTC", "ts": pd.Timestamp(ts), "side": "long", "price": 83100.0, "qty": 2.0, "notional": 166200.0},
            {"symbol": "BTC", "ts": pd.Timestamp(ts), "side": "short", "price": 83300.0, "qty": 1.0, "notional": 83300.0},
        ]
    )
    assert repo.last_liquidation_ts(conn, "okx", "BTC") is None
    assert repo.upsert_liquidations(conn, "okx", frame) == 2
    assert repo.upsert_liquidations(conn, "okx", frame) == 0
    assert repo.last_liquidation_ts(conn, "okx", "BTC") == ts
    assert repo.upsert_liquidations(conn, "okx", frame.iloc[0:0]) == 0

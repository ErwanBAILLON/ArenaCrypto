from datetime import UTC, datetime, timedelta

import pandas as pd

from arena.store import candles as repo
from tests.conftest import make_candles, make_funding

EX = "binance"
T0 = datetime(2024, 1, 1, 1, tzinfo=UTC)


def test_upsert_candles_idempotent_and_round_trip(conn):
    df = make_candles(symbols=["BTC", "ETH"], bars=48)
    assert repo.upsert_candles(conn, EX, df) == 96
    assert repo.upsert_candles(conn, EX, df) == 0

    out = repo.read_candles(conn, EX, ["BTC"], T0, T0 + timedelta(hours=47))
    assert list(out.columns) == ["symbol", "ts", "open", "high", "low", "close", "volume"]
    assert len(out) == 48
    assert str(out["ts"].dt.tz) == "UTC"
    src = df[df.symbol == "BTC"].sort_values("ts").reset_index(drop=True)
    pd.testing.assert_series_equal(out["close"], src["close"], check_names=False)
    assert out["ts"].iloc[0] == pd.Timestamp(T0)


def test_read_candles_window_and_unknown_symbol(conn):
    repo.upsert_candles(conn, EX, make_candles(symbols=["BTC"], bars=24))
    out = repo.read_candles(conn, EX, ["BTC"], T0 + timedelta(hours=5), T0 + timedelta(hours=9))
    assert len(out) == 5
    empty = repo.read_candles(conn, EX, ["ZZZ"], T0, T0 + timedelta(days=1))
    assert empty.empty and list(empty.columns) == ["symbol", "ts", "open", "high", "low", "close", "volume"]


def test_last_candle_ts(conn):
    assert repo.last_candle_ts(conn, EX, "BTC") is None
    repo.upsert_candles(conn, EX, make_candles(symbols=["BTC"], bars=10))
    last = repo.last_candle_ts(conn, EX, "BTC")
    assert last == T0 + timedelta(hours=9)
    assert last.tzinfo is not None


def test_funding_and_open_interest_round_trip(conn):
    c = make_candles(symbols=["BTC", "ETH"], bars=48)
    f = make_funding(symbols=["BTC", "ETH"], candles=c, per_symbol={"BTC": 0.0001, "ETH": -0.0002})
    assert repo.upsert_funding(conn, EX, f) == len(f)
    assert repo.upsert_funding(conn, EX, f) == 0
    out = repo.read_funding(conn, EX, ["ETH"], T0, T0 + timedelta(days=2))
    assert list(out.columns) == ["symbol", "ts", "rate"]
    assert (out["rate"] == -0.0002).all() and str(out["ts"].dt.tz) == "UTC"

    oi = pd.DataFrame({"symbol": ["BTC", "BTC"], "ts": [T0, T0 + timedelta(hours=1)], "oi": [1e9, 2e9]})
    assert repo.upsert_open_interest(conn, EX, oi) == 2
    got = repo.read_open_interest(conn, EX, ["BTC"], T0, T0 + timedelta(hours=1))
    assert got["oi"].tolist() == [1e9, 2e9]


def test_hl_snapshot_latest(conn):
    snap = pd.DataFrame({"coin": ["BTC", "ETH"], "funding": [1e-5, 2e-5], "oi": [1.0, 2.0], "mark": [50000.0, 3000.0]})
    assert repo.upsert_hl_snapshot(conn, T0, snap) == 2
    assert repo.upsert_hl_snapshot(conn, T0, snap) == 0
    later = snap.assign(funding=[3e-5, 4e-5])
    repo.upsert_hl_snapshot(conn, T0 + timedelta(hours=1), later)
    assert repo.latest_hl_funding(conn) == {"BTC": 3e-5, "ETH": 4e-5}


def test_empty_frames_are_noops(conn):
    empty = pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    assert repo.upsert_candles(conn, EX, empty) == 0
    assert repo.latest_hl_funding(conn) == {}

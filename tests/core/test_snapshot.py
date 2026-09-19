import numpy as np
import pandas as pd

from arena.core.snapshot import Snapshot, resample_candles
from tests.conftest import SYMBOLS, make_funding


def test_accessors_never_return_future(candles):
    ts = candles["ts"].iloc[1000]
    snap = Snapshot.from_long(ts, SYMBOLS, candles, make_funding(candles=candles))
    for tf in ("1h", "4h", "1d"):
        c = snap.candles("BTC", tf)
        assert not c.empty
        assert c.index.max() <= snap.ts
    assert snap.funding("BTC").index.max() <= snap.ts


def test_appending_future_rows_does_not_change_present_view(candles):
    ts = candles["ts"].iloc[2000]
    snap_a = Snapshot.from_long(ts, SYMBOLS, candles[candles["ts"] <= ts])
    snap_b = Snapshot.from_long(ts, SYMBOLS, candles)  # contains the future
    for tf in ("1h", "4h", "1d"):
        pd.testing.assert_frame_equal(snap_a.candles("ETH", tf), snap_b.candles("ETH", tf))
    assert snap_a.close("ETH") == snap_b.close("ETH")
    pd.testing.assert_frame_equal(snap_a.closes(), snap_b.closes())


def test_4h_bar_contains_correct_hours():
    idx = pd.date_range("2024-01-01T01:00:00Z", periods=12, freq="1h")
    c1h = pd.DataFrame(
        {"open": range(12), "high": range(12), "low": range(12), "close": range(12), "volume": 1.0},
        index=idx,
        dtype=float,
    )
    c1h.index.name = "ts"
    c4h = resample_candles(c1h, "4h")
    # bar labelled 04:00 has 1h candles 01:00,02:00,03:00,04:00 -> closes 0..3
    first = c4h.iloc[0]
    assert c4h.index[0] == pd.Timestamp("2024-01-01T04:00:00Z")
    assert first["open"] == 0 and first["close"] == 3 and first["high"] == 3 and first["volume"] == 4
    assert len(c4h) == 3


def test_partial_daily_bar_dropped():
    idx = pd.date_range("2024-01-01T01:00:00Z", periods=30, freq="1h")
    c1h = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    c1h.index.name = "ts"
    c1d = resample_candles(c1h, "1d")
    assert len(c1d) == 1
    assert c1d.index[0] == pd.Timestamp("2024-01-02T00:00:00Z")


def test_missing_symbol_returns_empty(snapshot):
    assert snapshot.candles("NOPE").empty
    assert snapshot.funding("NOPE").empty
    assert np.isnan(snapshot.close("NOPE"))
    assert snapshot.hl_funding("NOPE") is None


def test_macro_today_window():
    ev = pd.DatetimeIndex(["2024-03-01T14:00:00Z"])
    s = Snapshot("2024-03-01T05:00:00Z", ["BTC"], {}, macro_events=ev)
    assert s.macro_today()
    s2 = Snapshot("2024-02-28T00:00:00Z", ["BTC"], {}, macro_events=ev)
    assert not s2.macro_today()


def test_at_view_is_cheap_and_consistent(candles):
    snap = Snapshot.from_long(candles["ts"].max(), SYMBOLS, candles)
    earlier = snap.at(candles["ts"].iloc[500])
    assert earlier.candles("BTC").index.max() == candles["ts"].iloc[500]
    assert snap.candles("BTC").index.max() == candles["ts"].max()

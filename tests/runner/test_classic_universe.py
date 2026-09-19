"""A daily-bar Yahoo universe runs through the same tick, books and registry as the crypto one."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from arena.core.types import CompetitorSpec
from arena.core.universe import load_universe
from arena.runner.tick import run as run_tick
from arena.settings import Settings
from arena.store import books as bstore
from arena.store import candles as cstore
from arena.store import registry

SYMS = ["SPY", "GLD", "TLT"]


def _daily(days=800, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=days, freq="1D", tz="UTC")
    frames = []
    for i, sym in enumerate(SYMS):
        drift = {"SPY": 0.0008, "GLD": 0.0002, "TLT": -0.0005}[sym]
        close = 100.0 * (1 + i) * np.exp(np.cumsum(rng.normal(drift, 0.01, days)))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "ts": idx,
                    "open": close,
                    "high": close * 1.004,
                    "low": close * 0.996,
                    "close": close,
                    "volume": 1e6,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def classic(conn, tmp_path):
    c = _daily()
    cstore.upsert_candles(conn, "yahoo", c, tf="1d")
    for spec in [
        CompetitorSpec(
            None, "null_cash_classic", "null_cash", 1, {}, role="null", status="champion", universe="classic"
        ),
        CompetitorSpec(
            None,
            "bench_hold_classic",
            "bench_hold",
            1,
            {"symbol": "SPY"},
            role="benchmark",
            status="champion",
            universe="classic",
        ),
        CompetitorSpec(None, "trend_ts_v1_classic", "trend_ts", 1, {}, status="champion", universe="classic"),
        CompetitorSpec(None, "xs_momentum_v1_classic", "xs_momentum", 1, {}, status="challenger", universe="classic"),
        CompetitorSpec(None, "trend_ts_v1", "trend_ts", 1, {}, status="champion", universe="crypto"),  # other arena
    ]:
        registry.insert_competitor(conn, spec)
    conn.commit()
    path = tmp_path / "classic.yaml"
    path.write_text(
        "name: classic\nexchange: yahoo\nbar: 1d\nsymbols: [SPY, GLD, TLT]\nbinance_suffix: ''\n"
        "fees: {perp_taker: 0.0005, slippage: 0.0002, spot_taker: 0.001}\nnav0: 10000\nhistory_start: '2023-01-01T00:00:00Z'\n"
    )
    return conn, load_universe(path), c["ts"].max()


def test_daily_tick_books_only_its_universe(classic):
    conn, universe, last = classic
    assert universe.bar_hours == 24 and universe.reference == "SPY"
    settings = Settings("x", "", "", True, None)
    rep = run_tick(conn, settings, universe, (last + timedelta(minutes=20)).to_pydatetime(), client=None, ingest=False)
    assert rep.failed == []
    assert sorted(rep.booked) == [
        "bench_hold_classic",
        "null_cash_classic",
        "trend_ts_v1_classic",
        "xs_momentum_v1_classic",
    ]
    hold = registry.get_competitor(conn, "bench_hold_classic")
    assert bstore.last_targets(conn, hold.id)[1] == {"SPY": ("perp", 1.0)}
    crypto = registry.get_competitor(conn, "trend_ts_v1")
    assert bstore.last_book_row(conn, crypto.id) is None  # untouched by the classic tick
    trend = registry.get_competitor(conn, "trend_ts_v1_classic")
    positions = bstore.last_targets(conn, trend.id)[1]
    assert positions and positions.get("TLT", ("perp", -1.0))[1] <= 0  # planted downtrend on TLT is never bought
    assert registry.list_competitors(conn, universe="classic")[0].universe == "classic"


def test_explicit_exit_rows_are_written(conn):
    from arena.core.types import Target

    cid = registry.insert_competitor(conn, CompetitorSpec(None, "x", "null_cash", 1, {}, status="champion"))
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    bstore.write_targets(conn, cid, ts, {"BTC": Target(0.5)})
    bstore.write_targets(conn, cid, ts + timedelta(hours=1), {}, exits=["BTC"])
    last_ts, positions = bstore.last_targets(conn, cid)
    assert last_ts == ts + timedelta(hours=1) and positions == {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT weight, reason FROM targets WHERE competitor_id = %s AND ts = %s", (cid, ts + timedelta(hours=1))
        )
        row = cur.fetchone()
    assert row["weight"] == 0.0 and row["reason"] == {"exit": True}

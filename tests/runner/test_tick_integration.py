"""End-to-end tick on a local Postgres with synthetic data (no network)."""

from datetime import timedelta

import pytest

from arena.core.types import CompetitorSpec
from arena.core.universe import load_universe
from arena.runner.tick import run as run_tick
from arena.settings import Settings
from arena.store import books as bstore
from arena.store import candles as cstore
from arena.store import registry
from tests.conftest import make_candles, make_funding

SYMS = ["BTC", "ETH", "SOL"]


@pytest.fixture
def seeded(conn, tmp_path):
    c = make_candles(SYMS, bars=24 * 300, drift={"BTC": 0.0006, "SOL": -0.0006}, seed=7)
    cstore.upsert_candles(conn, "binance", c)
    cstore.upsert_funding(conn, "binance", make_funding(SYMS, candles=c, per_symbol={"ETH": 0.0004}))
    for spec in [
        CompetitorSpec(None, "null_cash", "null_cash", 1, {}, role="null", status="champion"),
        CompetitorSpec(None, "null_random_0", "null_random", 1, {"seed": 0}, role="null", status="champion"),
        CompetitorSpec(None, "bench_btc_hold", "bench_btc_hold", 1, {}, role="benchmark", status="champion"),
        CompetitorSpec(None, "carry_v1", "carry", 1, {}, status="champion"),
        CompetitorSpec(None, "trend_ts_v1", "trend_ts", 1, {}, status="champion"),
        CompetitorSpec(None, "xs_momentum_v1", "xs_momentum", 1, {"lookback_days": 20}, status="challenger"),
    ]:
        registry.insert_competitor(conn, spec)
    conn.commit()
    uni_path = tmp_path / "universe.yaml"
    uni_path.write_text(
        "symbols: [BTC, ETH, SOL]\nbinance_suffix: USDT\nfees: {perp_taker: 0.0005, slippage: 0.0002, spot_taker: 0.001}\nnav0: 10000\nhistory_start: '2024-01-01T00:00:00Z'\n"
    )
    return conn, load_universe(uni_path), c["ts"].max()


def test_tick_books_every_competitor_and_is_idempotent(seeded):
    conn, universe, last = seeded
    settings = Settings("x", "", "", True, None)
    now = (last + timedelta(minutes=5)).to_pydatetime()
    rep = run_tick(conn, settings, universe, now, client=None, ingest=False)
    assert rep.failed == []
    assert len(rep.booked) == 6
    for s in registry.list_competitors(conn):
        row = bstore.last_book_row(conn, s.id)
        assert row is not None and row.ts == last
    hold = registry.get_competitor(conn, "bench_btc_hold")
    assert bstore.last_targets(conn, hold.id)[1]["BTC"][1] == 1.0
    carry = registry.get_competitor(conn, "carry_v1")
    assert "ETH" in bstore.last_targets(conn, carry.id)[1]

    rep2 = run_tick(conn, settings, universe, now, client=None, ingest=False)
    assert rep2.booked == [] and len(rep2.skipped) == 6


def test_second_bar_marks_pnl_and_allocates(seeded):
    conn, universe, last = seeded
    settings = Settings("x", "", "", True, None)
    first_bar = last - timedelta(hours=1)
    run_tick(conn, settings, universe, (first_bar + timedelta(minutes=5)).to_pydatetime(), client=None, ingest=False)
    run_tick(conn, settings, universe, (last + timedelta(minutes=5)).to_pydatetime(), client=None, ingest=False)
    hold = registry.get_competitor(conn, "bench_btc_hold")
    rets = bstore.read_returns(conn, [hold.id], first_bar, last)
    assert len(rets) == 2
    btc = cstore.read_candles(conn, "binance", ["BTC"], first_bar, last).set_index("ts")["close"]
    funding = cstore.read_funding(conn, "binance", ["BTC"], first_bar + timedelta(seconds=1), last)["rate"].sum()
    expected = btc.iloc[-1] / btc.iloc[0] - 1 - funding  # long pays funding
    assert rets[hold.id].iloc[-1] == pytest.approx(expected, abs=1e-9)

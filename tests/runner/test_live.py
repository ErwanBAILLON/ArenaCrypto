"""The live watcher: rules on streamed prices, fills, and the tick that accounts them."""

from datetime import timedelta

import pytest

from arena.core.types import CompetitorSpec
from arena.core.universe import load_universe
from arena.labeling.barriers import RoiLadder
from arena.runner import live
from arena.runner.tick import run as run_tick
from arena.settings import Settings
from arena.store import books as bstore
from arena.store import candles as cstore
from arena.store import registry
from arena.store.state import load_state
from tests.conftest import make_candles, make_funding

SYMS = ["BTC", "ETH", "SOL"]
SETTINGS = Settings("x", "", "", True, None)


@pytest.fixture
def arena(conn, tmp_path):
    c = make_candles(SYMS, bars=24 * 300, drift={"BTC": 0.0006, "SOL": -0.0006}, seed=7)
    cstore.upsert_candles(conn, "binance", c)
    cstore.upsert_funding(conn, "binance", make_funding(SYMS, candles=c, per_symbol={"ETH": 0.0004}))
    # a hold-BTC benchmark has no rule; the trend follower gets a live stop and a take-profit
    registry.insert_competitor(
        conn, CompetitorSpec(None, "bench_btc_hold", "bench_btc_hold", 1, {}, role="benchmark", status="champion")
    )
    registry.insert_competitor(
        conn,
        CompetitorSpec(
            None,
            "trend_ts_v2",
            "trend_ts",
            2,
            {"live_stop": 0.05, "live_roi": [[0, 0.04], [48, 0.0]]},
            status="challenger",
        ),
    )
    conn.commit()
    p = tmp_path / "universe.yaml"
    p.write_text(
        "symbols: [BTC, ETH, SOL]\nbinance_suffix: USDT\nfees: {perp_taker: 0.0005, slippage: 0.0002, spot_taker: 0.001}\n"
        "nav0: 10000\nhistory_start: '2024-01-01T00:00:00Z'\n"
    )
    return conn, load_universe(p), c["ts"].max()


def _watch(**kw) -> live.Watch:
    base = dict(
        competitor_id=1,
        name="x",
        family="trend_ts",
        symbol="BTC",
        pair="BTCUSDT",
        kind="perp",
        weight=0.5,
        entry_price=100.0,
        bars_held=0.0,
        stop=0.05,
        ladder=None,
    )
    return live.Watch(**{**base, **kw})


class TestRules:
    def test_stop_fires_on_the_leg_side(self):
        assert live.verdict(_watch(), {"BTCUSDT": 94.9}, None, None) == ("stop", pytest.approx(-0.051))
        assert live.verdict(_watch(), {"BTCUSDT": 95.5}, None, None) is None
        # a short is stopped when the price rises
        assert live.verdict(_watch(weight=-0.5), {"BTCUSDT": 105.1}, None, None)[0] == "stop"
        assert live.verdict(_watch(weight=-0.5), {"BTCUSDT": 94.0}, None, None) is None

    def test_roi_uses_the_rung_for_the_bars_held(self):
        ladder = RoiLadder(steps=((0, 0.06), (24, 0.03), (48, 0.0)))
        w = _watch(stop=None, ladder=ladder, bars_held=10.0)
        assert live.verdict(w, {"BTCUSDT": 104.0}, None, None) is None  # 4 % < 6 % rung at 10 bars
        assert live.verdict(w, {"BTCUSDT": 106.5}, None, None)[0] == "roi"
        w = _watch(stop=None, ladder=ladder, bars_held=30.0)
        assert live.verdict(w, {"BTCUSDT": 104.0}, None, None)[0] == "roi"  # 3 % rung after 24 bars
        # the deadline rung (0 %) never fires here: bars are the tick's business
        w = _watch(stop=None, ladder=ladder, bars_held=60.0)
        assert live.verdict(w, {"BTCUSDT": 100.5}, None, None) is None

    def test_ladder_families_measure_excess_over_their_basket(self):
        w = _watch(basket={"BTCUSDT": 100.0, "ETHUSDT": 10.0, "SOLUSDT": 50.0})
        # leg −3 %, basket −7 % on average: the leg beat its basket, no stop
        assert live.verdict(w, {"BTCUSDT": 97.0, "ETHUSDT": 9.0, "SOLUSDT": 45.5}, None, None) is None
        # leg −3 %, basket (BTC −3 %, ETH +10 %, SOL +10 %) +5.7 %: excess −8.7 %, stop
        assert live.verdict(w, {"BTCUSDT": 97.0, "ETHUSDT": 11.0, "SOLUSDT": 55.0}, None, None)[0] == "stop"

    def test_missing_price_is_not_a_signal(self):
        assert live.verdict(_watch(), {"ETHUSDT": 1.0}, None, None) is None

    def test_quiet_window_brackets_each_tick_minute(self):
        from datetime import UTC, datetime

        t = lambda m, s=0: datetime(2026, 9, 24, 14, m, s, tzinfo=UTC)  # noqa: E731
        assert live.quiet(t(4, 30), (5, 35)) and live.quiet(t(5), (5, 35)) and live.quiet(t(9, 59), (5, 35))
        assert not live.quiet(t(10), (5, 35)) and not live.quiet(t(3, 59), (5, 35))
        assert live.quiet(t(36), (5, 35)) and not live.quiet(t(41), (5, 35))


class TestEndToEnd:
    def test_watchlist_fill_and_next_tick_accounting(self, arena):
        conn, universe, last = arena
        first_bar = last - timedelta(hours=1)
        run_tick(
            conn, SETTINGS, universe, (first_bar + timedelta(minutes=5)).to_pydatetime(), client=None, ingest=False
        )
        spec = registry.get_competitor(conn, "trend_ts_v2")
        held = bstore.last_targets(conn, spec.id)
        assert held and held[1], "the fixture needs the trend follower to hold something"
        sym, (kind, w) = next(iter(held[1].items()))

        now = (first_bar + timedelta(minutes=30)).to_pydatetime()
        watches = live.watchlist(conn, universe, now)
        assert {x.symbol for x in watches} == set(held[1])  # the benchmark has no rule and is not watched
        watch = next(x for x in watches if x.symbol == sym)
        assert watch.stop == 0.05 and watch.ladder is not None and watch.entry_price > 0
        assert watch.pair == f"{sym}USDT"

        # a price 6 % against the leg: the stop fires, the fill is written, the book is re-stated
        side = 1.0 if w > 0 else -1.0
        px = watch.entry_price * (1 - side * 0.06)
        prices = {x.pair: x.entry_price for x in watches}  # every other leg sits at its entry: no rule fires
        prices[watch.pair] = px
        fills = live.run_once(conn, SETTINGS, [universe], prices, now)
        assert len(fills) == 1
        after = bstore.last_targets(conn, spec.id)
        assert after is not None and after[0] == now and sym not in after[1]
        for other in held[1]:
            if other != sym:
                assert after[1][other] == held[1][other]  # surviving legs carried forward unchanged
        assert sym not in (load_state(conn, spec.id).get("held") or {})  # the competitor forgot the leg
        pending = bstore.unbooked_fills(conn, spec.id, last)
        assert len(pending) == 1 and pending[0][1].reason == "stop" and pending[0][1].price == pytest.approx(px)
        with conn.cursor() as cur:
            cur.execute("SELECT kind, symbol FROM alerts WHERE kind = 'live_exit'")
            assert cur.fetchone()["symbol"] == sym
        # nothing fires twice on the same leg
        assert live.run_once(conn, SETTINGS, [universe], prices, now + timedelta(seconds=5)) == []

        # the next tick accounts the fill from the previous close to the fill price, then books it
        run_tick(conn, SETTINGS, universe, (last + timedelta(minutes=5)).to_pydatetime(), client=None, ingest=False)
        assert bstore.unbooked_fills(conn, spec.id, last) == []
        rets = bstore.read_returns(conn, [spec.id], first_bar, last)[spec.id]
        closes = cstore.read_candles(conn, "binance", [sym], first_bar, last).set_index("ts")["close"]
        leg = w * (px / closes.iloc[0] - 1.0)
        # the closed leg contributed its move to the fill price (negative by construction) and is not held over
        assert rets.iloc[-1] < 0 and rets.iloc[-1] > leg - 0.05, rets.iloc[-1]
        # the competitor did not bring the leg back at the off-rebalance bar
        assert sym not in bstore.last_targets(conn, spec.id)[1]

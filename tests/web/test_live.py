"""The live layer: events are real weight changes, the clock is honest, the endpoints answer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from arena.core.types import CompetitorSpec
from arena.store import books as bstore
from arena.store import candles as cstore
from arena.store import registry
from arena.web import live

NOW = datetime(2024, 3, 1, 12, 20, tzinfo=UTC)
START = NOW - timedelta(hours=120)


def _settings(url):
    from arena.settings import Settings

    return Settings(database_url=url, telegram_bot_token="", telegram_chat_id="", dry_run=True, universe_path=None)


@pytest.fixture
def client(pg_url):
    from arena.web.app import create_app

    with TestClient(create_app(_settings(pg_url))) as c:
        yield c


def _target(conn, cid, hours, symbol, weight, conviction=0.5, reason=None):
    import json

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO targets (competitor_id, ts, symbol, weight, conviction, kind, reason) VALUES (%s,%s,%s,%s,%s,'perp',%s::jsonb)",
            (cid, START + timedelta(hours=hours), symbol, weight, conviction, json.dumps(reason or {})),
        )


@pytest.fixture
def seeded(conn):
    cid = registry.insert_competitor(
        conn, CompetitorSpec(None, "xs_live_v1", "xs_sparse", 1, {}, status="champion", gate_admitted=True)
    )
    idx = pd.date_range(START, periods=120, freq="1h", tz="UTC")
    for sym, step in (("BTC", 0.001), ("ETH", -0.0005)):
        close = 100.0 * np.cumprod(np.full(120, 1.0 + step))
        cstore.upsert_candles(
            conn,
            "binance",
            pd.DataFrame(
                {"symbol": sym, "ts": idx, "open": close, "high": close, "low": close, "close": close, "volume": 1e5}
            ),
        )
    nav = 10_000.0
    for i in range(120):
        nav *= 1.0001
        bstore.write_book_row(conn, cid, bstore.BookRow(START + timedelta(hours=i), nav, 0.0001, 0.2, 0.0, 0.0, 0.0))
    # BTC: enter at h10, resize at h30 (dust), reinforce at h50, exit at h80; ETH: short at h20, flip long at h60
    for h in range(10, 30):
        _target(conn, cid, h, "BTC", 0.10, reason={"score": 0.3})
    for h in range(30, 50):
        _target(conn, cid, h, "BTC", 0.11)
    for h in range(50, 80):
        _target(conn, cid, h, "BTC", 0.20)
    for h in range(20, 60):
        _target(conn, cid, h, "ETH", -0.10)
    for h in range(60, 100):
        _target(conn, cid, h, "ETH", 0.10)
    # the tick writes an explicit zero-weight row when a position leaves the book
    _target(conn, cid, 80, "BTC", 0.0)  # ETH stays held to the end
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO tick_runs (bar_ts, started_at, finished_at, booked, changes, alerts, ok) VALUES (%s,%s,%s,17,2,0,true)",
            (NOW - timedelta(minutes=15), NOW - timedelta(minutes=15), NOW - timedelta(minutes=14)),
        )
    conn.commit()
    return cid


class TestEvents:
    def test_every_kind_of_move_is_detected_once(self, conn, seeded):
        ev = live.trade_events(conn, seeded, START, NOW)
        kinds = [(e.symbol, e.kind) for e in ev]
        assert ("BTC", "entrée") in kinds and ("BTC", "renfort") in kinds and ("BTC", "sortie") in kinds
        assert ("ETH", "entrée") in kinds and ("ETH", "retournement") in kinds
        assert ("ETH", "sortie") not in kinds  # still held: no exit was written
        assert kinds.count(("BTC", "entrée")) == 1  # a held position is one entry, not sixty

    def test_dust_is_not_an_event(self, conn, seeded):
        ev = live.trade_events(conn, seeded, START, NOW)
        assert not any(
            e.symbol == "BTC" and e.kind in ("renfort", "allège") and abs(e.after - e.before) < 0.05 for e in ev
        )

    def test_events_carry_the_price_the_book_saw_and_the_reason(self, conn, seeded):
        entry = next(e for e in live.trade_events(conn, seeded, START, NOW) if e.symbol == "BTC" and e.kind == "entrée")
        assert entry.price is not None and entry.price > 100.0
        assert entry.reason == {"score": 0.3}
        assert entry.side == "long"

    def test_a_position_open_before_the_window_is_not_a_fresh_entry(self, conn, seeded):
        ev = live.trade_events(conn, seeded, START + timedelta(hours=40), NOW)
        assert not any(e.symbol == "BTC" and e.kind == "entrée" for e in ev)  # it was already held at h40
        assert any(e.symbol == "BTC" and e.kind == "renfort" for e in ev)

    def test_a_short_that_closes_is_a_short_side_exit(self):
        assert live._classify(-0.1, 0.0) == "sortie"
        assert live._classify(0.0, -0.1) == "entrée"
        assert live._classify(0.1, -0.1) == "retournement"
        assert live._classify(0.10, 0.12) is None


class TestClock:
    def test_next_tick_is_the_next_hour_at_minute_five(self):
        assert live.next_tick(datetime(2024, 1, 1, 12, 20, tzinfo=UTC)) == datetime(2024, 1, 1, 13, 5, tzinfo=UTC)
        assert live.next_tick(datetime(2024, 1, 1, 12, 2, tzinfo=UTC)) == datetime(2024, 1, 1, 12, 5, tzinfo=UTC)

    def test_live_state_reports_age_and_countdown_not_fake_motion(self, conn, seeded):
        state = live.live_state(conn, NOW)
        assert state["seconds_to_next"] == 45 * 60
        assert state["last_tick"] and state["tick_ok"] is True and state["booked"] == 17
        assert state["version"] == int((START + timedelta(hours=119)).timestamp())  # moves only when a bar is booked
        assert any(p["symbol"] == "ETH" and p["weight"] > 0 for p in state["positions"])


class TestEndpoints:
    def test_series_has_nav_events_and_symbols(self, client, seeded):
        body = client.get(f"/api/series/{seeded}?days=3650").json()
        assert body["name"] == "xs_live_v1" and len(body["nav"]["t"]) == 120
        assert {e["kind"] for e in body["events"]} == {"entrée", "sortie", "retournement", "renfort"}
        assert body["symbols"] == ["BTC", "ETH"]
        assert client.get("/api/series/999999").status_code == 404

    def test_board_normalises_to_one_hundred(self, client, seeded):
        body = client.get("/api/board?days=3650").json()
        s = next(x for x in body["series"] if x["name"] == "xs_live_v1")
        assert s["v"][0] == pytest.approx(100.0) and s["v"][-1] > 100.0

    def test_live_endpoint_and_static_assets(self, client, seeded):
        assert client.get("/api/live").json()["positions"]
        assert client.get("/static/arena.js").status_code == 200
        assert client.get("/static/uPlot.iife.min.js").status_code == 200

    def test_pages_carry_the_chart_mounts(self, client, seeded):
        assert 'data-chart="board"' in client.get("/").text
        page = client.get(f"/competitors/{seeded}").text
        assert 'data-chart="competitor"' in page and 'id="events-table"' in page


class TestBook:
    def test_book_marks_every_agent_with_reference_prices_and_streams(self, conn, seeded):
        state = live.book_state(conn, NOW)
        agent = next(a for a in state["agents"] if a["id"] == seeded)
        assert agent["nav"] > 0 and agent["base"]["all"] == pytest.approx(10_000.0 * 1.0001, rel=1e-3)
        eth = next(p for p in agent["positions"] if p["symbol"] == "ETH")
        assert eth["weight"] == 0.1 and eth["ref_price"] is not None
        assert eth["pair"] == "ETHUSDT"  # binance-sourced candles get a live perp price
        assert state["pairs"] == ["ETHUSDT"] and state["rest"].startswith("https://fapi.binance.com/")
        assert state["ws"] == "wss://data-stream.binance.vision/stream?streams=ethusdt@miniTicker"
        assert state["version"] > 0

    def test_binance_pair_names(self):
        assert live.binance_pair("BTC") == "BTCUSDT"
        assert live.binance_pair("BTCUSDT") == "BTCUSDT"

    def test_endpoints_and_pages(self, client, seeded):
        book = client.get("/api/book").json()
        assert [a["id"] for a in book["agents"]] == [seeded]
        home = client.get("/").text
        assert 'id="board"' in home and "/static/board.js" in home and 'data-chart="board"' in home
        assert client.get("/systeme").status_code == 200
        assert client.get("/static/board.js").status_code == 200

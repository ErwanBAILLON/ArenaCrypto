"""Dashboard routes against a seeded Postgres (fixture ``conn``)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

TestClient = pytest.importorskip("fastapi.testclient", reason="web extra not installed").TestClient

from arena.core.types import Alert, BookRow, CompetitorSpec, Target
from arena.settings import Settings
from arena.store import books as bstore
from arena.store import registry

NOW = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)


def _settings(pg_url: str) -> Settings:
    return Settings(database_url=pg_url, telegram_bot_token="", telegram_chat_id="", dry_run=True, universe_path=None)


def _book(conn, cid: int, start: datetime, bars: int, drift: float) -> None:
    nav = 10_000.0
    for i in range(bars):
        ret = drift + (0.001 if i % 2 else -0.0008)
        nav *= 1 + ret
        bstore.write_book_row(conn, cid, BookRow(start + timedelta(hours=i), nav, ret, 0.5, 0.0, 0.0, 0.0))


@pytest.fixture
def seeded(conn):
    """Two champions, a benchmark, a null, books for 40 bars ending 1h ago, targets, a trial, an alert."""
    start = NOW - timedelta(hours=40)
    trend = registry.insert_competitor(
        conn,
        CompetitorSpec(
            None, "trend_ts_v1", "trend_ts", 1, {"fast": 50, "slow": 200}, status="champion", rationale="founder"
        ),
    )
    meta = registry.insert_competitor(
        conn, CompetitorSpec(None, "meta_label_v1", "meta_label", 1, {}, status="challenger", parent_id=trend)
    )
    bench = registry.insert_competitor(
        conn, CompetitorSpec(None, "bench_btc_hold", "bench_btc_hold", 1, {}, role="benchmark", status="champion")
    )
    null = registry.insert_competitor(
        conn, CompetitorSpec(None, "null_random_0", "null_random", 1, {"seed": 0}, role="null", status="champion")
    )
    for cid, drift in ((trend, 0.0005), (meta, 0.0), (bench, 0.0002), (null, 0.0)):
        _book(conn, cid, start, 40, drift)
    bstore.write_targets(
        conn,
        trend,
        NOW - timedelta(hours=1),
        {"BTC": Target(0.4, 0.7, reason={"ema50": 1, "r30": 0.18}), "ETH": Target(-0.2, 0.55)},
    )
    bstore.write_allocations(conn, NOW - timedelta(hours=1), {trend: 0.6})
    registry.add_trial(
        conn,
        "trend_ts",
        "backtest",
        {"fast": 50},
        {"sharpe": 1.2, "dsr": 0.95, "bootstrap_p": 0.03, "max_drawdown": 0.12, "fold_sharpes": [1.0, 1.4]},
        "admitted",
        competitor_id=trend,
    )
    registry.save_model(conn, meta, b"\x00", {"auc": 0.61})
    bstore.add_alert(conn, Alert(kind="promotion", payload={"detail": "trend_ts_v1 promoted"}, competitor_id=trend))
    conn.commit()
    return {"trend": trend, "meta": meta, "bench": bench, "null": null}


@pytest.fixture
def client(pg_url):
    from arena.web.app import create_app

    with TestClient(create_app(_settings(pg_url))) as c:
        yield c


def test_index_lists_competitors_and_chart(client, seeded) -> None:
    r = client.get("/")
    assert r.status_code == 200
    for name in ("trend_ts_v1", "meta_label_v1", "bench_btc_hold", "null_random_0"):
        assert name in r.text
    assert 'data-series="trend_ts_v1"' in r.text
    assert 'data-series="bench_btc_hold"' in r.text
    assert 'data-series="null_random_0"' not in r.text
    assert "60%" in r.text
    assert "STALE" not in r.text


def test_competitor_pages(client, seeded) -> None:
    r = client.get(f"/competitors/{seeded['trend']}")
    assert r.status_code == 200
    assert "ema50=1" in r.text and "BTC" in r.text and "admitted" in r.text
    r = client.get(f"/competitors/{seeded['meta']}")
    assert r.status_code == 200
    assert "auc" in r.text and "trend_ts_v1" in r.text
    assert client.get("/competitors/999999").status_code == 404


def test_alerts_and_trials(client, seeded) -> None:
    r = client.get("/alerts")
    assert r.status_code == 200 and "trend_ts_v1 promoted" in r.text
    r = client.get("/trials")
    assert r.status_code == 200 and "backtest" in r.text and "1.00, 1.40" in r.text


def test_healthz_reflects_book_freshness(client, conn) -> None:
    cid = registry.insert_competitor(
        conn, CompetitorSpec(None, "old", "null_cash", 1, {}, role="null", status="champion")
    )
    _book(conn, cid, NOW - timedelta(days=5), 3, 0.0)
    conn.commit()
    r = client.get("/healthz")
    assert r.status_code == 503 and r.json()["ok"] is False
    bstore.write_book_row(conn, cid, BookRow(NOW - timedelta(minutes=30), 10_000.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    conn.commit()
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] is True and r.json()["last_bar"]


def test_leaderboard_json_shape(client, seeded) -> None:
    body = client.get("/api/leaderboard.json").json()
    assert set(body) == {"rows", "allocation", "null95", "last_bar", "stale"}
    assert body["stale"] is False and body["last_bar"]
    row = next(r for r in body["rows"] if r["name"] == "trend_ts_v1")
    assert {"id", "status", "role", "sharpe_30d", "ret_30d", "mdd_30d", "nav", "bars_30d"} <= set(row)
    assert body["allocation"] == [{"name": "trend_ts_v1", "weight": 0.6}]

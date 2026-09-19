"""Dashboard routes against a seeded Postgres (fixture ``conn``)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

TestClient = pytest.importorskip("fastapi.testclient", reason="web extra not installed").TestClient

from arena.core.types import Alert, BookRow, CompetitorSpec, Target
from arena.settings import Settings
from arena.store import books as bstore
from arena.store import candles as cstore
from arena.store import registry
from arena.web import fr, queries

NOW = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)


def _settings(pg_url: str) -> Settings:
    return Settings(database_url=pg_url, telegram_bot_token="", telegram_chat_id="", dry_run=True, universe_path=None)


def _book(conn, cid: int, start: datetime, bars: int, drift: float) -> None:
    nav = 10_000.0
    for i in range(bars):
        ret = drift + (0.001 if i % 2 else -0.0008)
        nav *= 1 + ret
        bstore.write_book_row(conn, cid, BookRow(start + timedelta(hours=i), nav, ret, 0.5, 0.0, 0.0, 0.0))


def _tick(conn, finished: datetime, booked: int, changes: int, alerts: int, failed: list[str]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO tick_runs (bar_ts, started_at, finished_at, booked, skipped, failed, changes, alerts,"
            " ingested, ok) VALUES (%s, %s, %s, %s, 0, %s, %s, %s, '{}'::jsonb, %s)",
            (
                finished.replace(minute=0),
                finished - timedelta(seconds=20),
                finished,
                booked,
                failed,
                changes,
                alerts,
                not failed,
            ),
        )


@pytest.fixture
def seeded(conn):
    """Trend and carry champions, a meta challenger, BTC benchmark, a null; books, targets at two
    timestamps (48 h and 1 h ago), tick runs, a trial with robustness, alerts and data sources."""
    start = NOW - timedelta(hours=72)
    trend = registry.insert_competitor(
        conn,
        CompetitorSpec(
            None, "trend_ts_v1", "trend_ts", 1, {"fast": 50, "slow": 200}, status="champion", rationale="founder"
        ),
    )
    carry = registry.insert_competitor(
        conn, CompetitorSpec(None, "carry_v1", "carry", 1, {"k": 5, "lookback_days": 14}, status="champion")
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
    for cid, drift in ((trend, 0.0005), (carry, 0.0002), (meta, 0.0), (bench, 0.0002), (null, 0.0)):
        _book(conn, cid, start, 72, drift)
    # trend: 48 h ago long BTC 0.4 + short ETH; 1 h ago long BTC 0.6 + long SOL (ETH exited, BTC resized, SOL entered)
    t_old, t_new = NOW - timedelta(hours=48), NOW - timedelta(hours=1)
    trend_reason = {"r_short": 0.18, "r_long": 0.35, "vol": 0.55}
    bstore.write_targets(
        conn,
        trend,
        t_old,
        {"BTC": Target(0.4, 0.7, reason=trend_reason), "ETH": Target(-0.2, 0.55, reason=trend_reason)},
    )
    bstore.write_targets(
        conn, trend, t_new, {"BTC": Target(0.6, 0.7, reason=trend_reason), "SOL": Target(0.2, 0.5, reason=trend_reason)}
    )
    carry_reason = {"mean_funding_8h": 0.0002, "rank": 0}
    for ts in (t_old, t_new):
        bstore.write_targets(conn, carry, ts, {"DOGE": Target(0.3, 0.6, kind="carry", reason=carry_reason)})
    bstore.write_allocations(conn, t_new, {trend: 0.6, carry: 0.4})
    registry.add_trial(
        conn,
        "trend_ts",
        "walkforward",
        {"fast": 50, "slow": 200},
        {
            "sharpe": 1.2,
            "null_threshold": 0.4,
            "dsr": 0.95,
            "n_trials": 7,
            "bootstrap_p": 0.03,
            "max_drawdown": 0.12,
            "folds_positive_frac": 1.0,
            "fold_sharpes": [1.0, 1.4],
            "decisions": 80,
            "total_return": 0.31,
            "failed": [],
            "robustness": {
                "win_rate_overall": 0.75,
                "win_rate_by_regime": {"bull": 0.9, "bear": 0.5, "range": 0.7},
                "n_windows": 8,
                "median_sharpe": 1.1,
            },
        },
        "admitted",
        competitor_id=trend,
    )
    registry.save_model(conn, meta, b"\x00", {"auc": 0.61})
    bstore.add_alert(conn, Alert(kind="promotion", payload={"detail": "trend_ts_v1 promoted"}, competitor_id=trend))
    bstore.add_alert(
        conn,
        Alert(
            kind="signal",
            competitor_id=carry,
            symbol="DOGE",
            payload={"weight": 0.3, "kind": "carry", "reason": carry_reason},
        ),
    )
    _tick(conn, NOW - timedelta(hours=2), 5, 0, 0, [])
    _tick(conn, NOW - timedelta(minutes=30), 5, 2, 2, [])
    # data sources: fresh candles, stale funding, articles
    idx = pd.date_range(NOW - timedelta(hours=3), periods=3, freq="1h", tz="UTC")
    candles = pd.DataFrame(
        {"symbol": "BTC", "ts": idx, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
    )
    cstore.upsert_candles(conn, "binance", candles)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO funding (exchange, symbol, ts, rate, fetched_at) VALUES ('binance', 'BTC', %s, 0.0001, %s)",
            (NOW - timedelta(hours=9), NOW - timedelta(hours=9)),
        )
        cur.execute(
            "INSERT INTO articles (source, url, title, published_at, fetched_at) VALUES ('x', 'u', 't', %s, %s)",
            (NOW - timedelta(hours=1), NOW - timedelta(hours=1)),
        )
    conn.commit()
    return {"trend": trend, "carry": carry, "meta": meta, "bench": bench, "null": null}


@pytest.fixture
def client(pg_url):
    from arena.web.app import create_app

    with TestClient(create_app(_settings(pg_url))) as c:
        yield c


def test_index_live_sections_in_french(client, seeded) -> None:
    r = client.get("/")
    assert r.status_code == 200
    for heading in ("En direct", "Qui parle", "Ce qui a changé", "Sur quoi l'arène se base", "Fil d'activité"):
        assert heading in r.text, heading
    assert "Classement des 30 derniers jours" in r.text and "10 000 € virtuels" in r.text
    # banner from the latest tick run
    assert "Dernier passage à" in r.text and "5 modèles évalués" in r.text and "2 ont changé de position" in r.text
    assert "Attention" not in r.text
    # champions speak in French, carry sentence mentions funding
    assert "Carry de funding" in r.text and "funding moyen 0.0200 % par 8 h" in r.text
    assert "Tendance (momentum temporel)" in r.text
    # 24 h diff: SOL entered, ETH exited, BTC resized
    assert "Sort de ETH" in r.text and "Achat SOL" in r.text and "taille" in r.text
    # data sources with freshness levels
    assert "Bougies Binance" in r.text and 'class="dot ok"' in r.text and 'class="dot bad"' in r.text
    # leaderboard and chart still there
    for name in ("trend_ts_v1", "meta_label_v1", "bench_btc_hold", "null_random_0"):
        assert name in r.text
    assert 'data-series="trend_ts_v1"' in r.text and 'data-series="bench_btc_hold"' in r.text
    assert 'data-series="null_random_0"' not in r.text
    assert "prétendant" in r.text and "repère" in r.text


def test_index_red_banner_on_failed_tick(client, seeded, conn) -> None:
    _tick(conn, NOW - timedelta(minutes=5), 4, 0, 1, ["carry_v1"])
    conn.commit()
    r = client.get("/")
    assert 'class="banner bad"' in r.text and "a échoué pour : carry_v1" in r.text


def test_competitor_page_sections(client, seeded) -> None:
    r = client.get(f"/competitors/{seeded['trend']}")
    assert r.status_code == 200
    for heading in (
        "Ce qu'il fait",
        "Sur quoi il se base",
        "Quand il décide",
        "Pourquoi il est champion",
        "Ce qu'il tient maintenant",
        "Ses dernières actions",
        "Résultats",
    ):
        assert heading in r.text, heading
    assert "il bat la chance" in r.text and "admis" in r.text
    assert "Tenue par type de marché" in r.text and "haussier" in r.text and "90 %" in r.text
    assert "Achat BTC (60 % du capital)" in r.text and "Sort de ETH" in r.text
    assert "Garder du BTC" in r.text and "Ne rien faire" in r.text
    r = client.get(f"/competitors/{seeded['meta']}")
    assert r.status_code == 200
    assert "auc" in r.text and "trend_ts_v1" in r.text and "Prétendant" in r.text and "Progression" in r.text
    r = client.get(f"/competitors/{seeded['bench']}")
    assert r.status_code == 200 and "Repère" in r.text
    assert client.get("/competitors/999999").status_code == 404


def test_modeles_lists_families(client, seeded) -> None:
    r = client.get("/modeles")
    assert r.status_code == 200
    assert "Les familles" in r.text and "Carry de funding" in r.text and "carry_v1" in r.text
    assert "Repère : garder du BTC" in r.text and "aucun pour l'instant" in r.text


def test_alerts_and_trials_in_french(client, seeded) -> None:
    r = client.get("/alerts")
    assert r.status_code == 200 and "trend_ts_v1 promoted" in r.text
    assert "Promotion" in r.text and "Nouveau signal" in r.text and "Carry sur DOGE" in r.text
    r = client.get("/trials")
    assert r.status_code == 200 and "test glissant" in r.text and "1.00, 1.40" in r.text and "admis" in r.text


def test_queries_pnl_and_positions(conn, seeded) -> None:
    pnl = queries.pnl_windows(conn, seeded["trend"], NOW)
    assert pnl["nav"] > 10_000 and pnl["all"]["eur"] > 400
    assert pnl["all"]["eur"] == pytest.approx(pnl["nav"] - pnl["nav"] / (1 + pnl["all"]["frac"]))
    assert pnl["today"] is not None and pnl["7d"]["frac"] == pytest.approx(pnl["all"]["frac"])
    pos = queries.positions_now(conn, seeded["carry"], "carry", NOW, pnl["nav"])
    assert [p["symbol"] for p in pos] == ["DOGE"] and pos[0]["since"] == NOW - timedelta(hours=48)
    assert "funding" in pos[0]["sentence"]
    pos = queries.positions_now(conn, seeded["trend"], "trend_ts", NOW, pnl["nav"])
    since = {p["symbol"]: p["since"] for p in pos}
    assert since["BTC"] == NOW - timedelta(hours=48) and since["SOL"] == NOW - timedelta(hours=1)
    diff = {d["symbol"]: d["change"] for d in queries.positions_diff_24h(conn, seeded["trend"], "trend_ts", NOW)}
    assert diff == {"BTC": "taille", "SOL": "entrée", "ETH": "sortie"}
    assert queries.positions_diff_24h(conn, seeded["carry"], "carry", NOW) == []
    levels = {s["key"]: s["level"] for s in queries.sources_freshness(conn, NOW)}
    assert levels["candles"] == "ok" and levels["funding"] == "bad" and levels["articles"] == "ok"
    assert levels["macro_events"] == "bad"
    assert fr.status_fr("competitor", "challenger") == "prétendant" and fr.kind_fr("stale") == "Données en retard"


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
    assert body["allocation"] == [{"name": "trend_ts_v1", "weight": 0.6}, {"name": "carry_v1", "weight": 0.4}]

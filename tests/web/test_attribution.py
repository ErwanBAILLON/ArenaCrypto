"""Attribution: derived from stored targets, so it works for every family."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from arena.core.types import CompetitorSpec
from arena.store import candles as cstore
from arena.store import registry
from arena.web import attribution

NOW = datetime(2024, 3, 1, tzinfo=UTC)
START = NOW - timedelta(hours=200)


def _seed_prices(conn, moves: dict[str, float], bars=200):
    """Each symbol compounds a fixed per-bar return, so episode maths is exact."""
    idx = pd.date_range(START, periods=bars, freq="1h", tz="UTC")
    frames = []
    for sym, step in moves.items():
        close = 100.0 * np.cumprod(np.full(bars, 1.0 + step))
        frames.append(
            pd.DataFrame(
                {"symbol": sym, "ts": idx, "open": close, "high": close, "low": close, "close": close, "volume": 1e5}
            )
        )
    cstore.upsert_candles(conn, "binance", pd.concat(frames, ignore_index=True))


def _hold(conn, cid, symbol, first, last, weight, conviction=0.5):
    idx = pd.date_range(START + timedelta(hours=first), START + timedelta(hours=last), freq="1h", tz="UTC")
    with conn.cursor() as cur:
        for ts in idx:
            cur.execute(
                "INSERT INTO targets (competitor_id, ts, symbol, weight, conviction, kind, reason)"
                " VALUES (%s, %s, %s, %s, %s, 'perp', '{}'::jsonb)",
                (cid, ts, symbol, weight, conviction),
            )


def _settings(url):
    from arena.settings import Settings

    return Settings(database_url=url, telegram_bot_token="", telegram_chat_id="", dry_run=True, universe_path=None)


@pytest.fixture
def client(pg_url):
    from starlette.testclient import TestClient

    from arena.web.app import create_app

    with TestClient(create_app(_settings(pg_url))) as c:
        yield c


@pytest.fixture
def seeded(conn):
    cid = registry.insert_competitor(
        conn, CompetitorSpec(None, "xs_test_v1", "xs_sparse", 1, {}, status="challenger", gate_admitted=True)
    )
    # the universe drifts up: WIN beats it, LOSE and FLAT do not
    _seed_prices(conn, {"WIN": 0.0020, "RISER": 0.0010, "LOSE": -0.0005, "FLAT": 0.0})
    conn.commit()
    return cid


class TestEpisodes:
    def test_a_continuous_hold_is_one_episode(self, conn, seeded):
        _hold(conn, seeded, "WIN", 10, 40, 0.5)
        conn.commit()
        eps = attribution.episodes(conn, seeded)
        assert len(eps) == 1
        assert eps[0].bars_held == 31 and eps[0].symbol == "WIN"

    def test_a_gap_splits_the_episodes(self, conn, seeded):
        _hold(conn, seeded, "WIN", 10, 20, 0.5)
        _hold(conn, seeded, "WIN", 60, 70, 0.5)
        conn.commit()
        assert len(attribution.episodes(conn, seeded)) == 2

    def test_returns_are_measured_against_the_market(self, conn, seeded):
        """WIN beats a rising universe, so its excess is smaller than its gross move."""
        _hold(conn, seeded, "WIN", 10, 40, 0.5)
        conn.commit()
        ep = attribution.episodes(conn, seeded)[0]
        assert ep.gross_return > ep.excess_return > 0
        assert ep.pnl == pytest.approx(ep.excess_return * 0.5)

    def test_a_short_on_a_faller_contributes_positively(self, conn, seeded):
        _hold(conn, seeded, "LOSE", 10, 40, -0.5)
        conn.commit()
        ep = attribution.episodes(conn, seeded)[0]
        assert ep.side == "short" and ep.excess_return < 0 and ep.pnl > 0 and ep.won

    def test_a_long_that_rose_less_than_the_market_is_a_loss(self, conn, seeded):
        """The point of excess returns: going up is not the same as winning."""
        _hold(conn, seeded, "FLAT", 10, 40, 0.5)
        conn.commit()
        ep = attribution.episodes(conn, seeded)[0]
        assert ep.gross_return == pytest.approx(0.0, abs=1e-9)
        assert ep.excess_return < 0 and ep.pnl < 0  # the universe rose and this did not

    def test_zero_weight_rows_are_not_positions(self, conn, seeded):
        _hold(conn, seeded, "WIN", 10, 40, 0.0)
        conn.commit()
        assert attribution.episodes(conn, seeded) == []

    def test_a_competitor_that_never_traded_has_no_episodes(self, conn, seeded):
        assert attribution.episodes(conn, seeded) == []


class TestDecomposition:
    @pytest.fixture
    def traded(self, conn, seeded):
        _hold(conn, seeded, "WIN", 10, 40, 0.5, conviction=0.9)
        _hold(conn, seeded, "LOSE", 10, 40, -0.4, conviction=0.8)
        _hold(conn, seeded, "FLAT", 50, 80, 0.3, conviction=0.2)
        conn.commit()
        return attribution.episodes(conn, seeded)

    def test_by_symbol_ranks_contributors(self, traded):
        rows = attribution.decompose(traded, "symbol")
        assert [r["bucket"] for r in rows][:2] == sorted(
            [r["bucket"] for r in rows][:2], key=lambda s: -dict((x["bucket"], x["pnl"]) for x in rows)[s]
        )
        assert rows[-1]["pnl"] < rows[0]["pnl"]

    def test_by_side_separates_long_from_short(self, traded):
        buckets = {r["bucket"] for r in attribution.decompose(traded, "side")}
        assert buckets == {"long", "short"}

    def test_every_row_carries_its_sample_size(self, traded):
        """A bucket of three trades is an anecdote however large its number."""
        for row in attribution.decompose(traded, "symbol"):
            assert row["trades"] >= 1 and 0.0 <= row["win_rate"] <= 1.0

    def test_an_unknown_dimension_returns_nothing(self, traded):
        assert attribution.decompose(traded, "not_a_column") == []

    def test_summary_flags_concentration(self, traded):
        s = attribution.summary(traded)
        assert s["trades"] == 3 and s["best_symbol"] and s["worst_symbol"]
        assert s["concentration"] == pytest.approx(1.0)  # three symbols is all of them

    def test_worst_is_sorted_by_damage(self, traded):
        worst = attribution.worst(traded, 2)
        assert worst[0].pnl <= worst[1].pnl


class TestCalibration:
    def test_a_model_whose_confidence_means_nothing_scores_badly(self, conn, seeded):
        """High conviction on the loser, low on the winner: the diagram must catch it."""
        _hold(conn, seeded, "WIN", 10, 40, 0.5, conviction=0.1)
        _hold(conn, seeded, "LOSE", 10, 40, 0.5, conviction=0.9)
        conn.commit()
        eps = attribution.episodes(conn, seeded)
        rows = attribution.reliability(eps)
        assert len(rows) == 2
        assert rows[0]["observed"] > rows[-1]["observed"]  # the confident one lost
        assert attribution.calibration_error(eps) > 0.5

    def test_a_well_calibrated_model_scores_well(self, conn, seeded):
        _hold(conn, seeded, "WIN", 10, 40, 0.5, conviction=0.95)
        _hold(conn, seeded, "LOSE", 10, 40, 0.5, conviction=0.05)
        conn.commit()
        assert attribution.calibration_error(attribution.episodes(conn, seeded)) < 0.15

    def test_one_conviction_level_cannot_be_calibrated(self, conn, seeded):
        _hold(conn, seeded, "WIN", 10, 40, 0.5, conviction=0.5)
        conn.commit()
        assert attribution.reliability(attribution.episodes(conn, seeded)) == []


class TestPage:
    def test_the_page_renders_for_a_competitor_that_traded(self, client, conn, seeded):
        _hold(conn, seeded, "WIN", 10, 40, 0.5, conviction=0.9)
        _hold(conn, seeded, "LOSE", 10, 40, -0.4, conviction=0.2)
        conn.commit()
        text = client.get(f"/competitors/{seeded}/attribution").text
        assert "Où" in text and "gagne et où il perd" in text
        assert "Est-ce qu'il sait ce qu'il ne sait pas" in text
        assert "Les douze pires positions" in text

    def test_an_unknown_competitor_is_a_404(self, client):
        assert client.get("/competitors/999999/attribution").status_code == 404

    def test_the_competitor_page_links_to_it(self, client, seeded):
        assert f"/competitors/{seeded}/attribution" in client.get(f"/competitors/{seeded}").text

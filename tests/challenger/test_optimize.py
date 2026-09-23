"""The weekly search must run in the arena it was pointed at.

Before this was threaded through, `arena-classic-challenger` built hourly
competitors on daily bars and annualised by 8760: eighteen folds of exactly
0.00 Sharpe every Sunday, rejected for `min_decisions`, every evaluation still
counted against the deflated Sharpe of the crypto arena.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from arena.book.book import FeeModel
from arena.challenger import optimize
from arena.competitors.base import REGISTRY, Competitor
from arena.core.types import CompetitorSpec, Target
from arena.judge.backtest import HistoryFrames
from arena.store import registry

FEES = FeeModel(perp_taker=0.0005, slippage=0.0002, spot_taker=0.001)
SEEN: list[int] = []


class _Spy(Competitor):
    """Records the bar size it was built with and always holds the first symbol."""

    family = "_spy"
    default_params = {"w": 0.5}

    def __init__(self, params=None, seed=0, bar_hours=1):
        super().__init__(params, seed, bar_hours)
        SEEN.append(self.bar_hours)

    def decide(self, snap):
        return {snap.symbols[0]: Target(weight=float(self.params["w"]), conviction=0.5)}


@pytest.fixture(autouse=True)
def _spy_space(monkeypatch):
    """Register the spy family for the test only: every real family owes the UI a card."""
    SEEN.clear()
    monkeypatch.setitem(REGISTRY, "_spy", _Spy)
    monkeypatch.setitem(optimize.SPACES, "_spy", lambda t: {"w": t.suggest_float("w", 0.1, 0.9)})


def _daily_history(days=600, seed=5) -> tuple[HistoryFrames, list[str], datetime, datetime]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=days, freq="1D", tz="UTC")
    rows = []
    for i, sym in enumerate(["SPY", "GLD"]):
        close = 100.0 * (1 + i) * np.exp(np.cumsum(rng.normal(0.0006, 0.01, days)))
        rows.append(
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
    candles = pd.concat(rows, ignore_index=True)
    history = HistoryFrames(candles=candles)
    return history, ["SPY", "GLD"], idx[0].to_pydatetime(), idx[-1].to_pydatetime()


class TestObjective:
    def test_it_builds_competitors_with_the_arena_bar_size(self, conn):
        history, syms, start, end = _daily_history()
        obj = optimize.objective_factory(
            conn, "_spy", history, syms, start, end, FEES, bar_hours=24, universe="classic"
        )
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        optuna.create_study(direction="maximize").optimize(obj, n_trials=2)
        assert SEEN and set(SEEN) == {24}

    def test_it_annualises_with_the_arena_bar_size(self, conn):
        """A daily Sharpe is sqrt(24) smaller than the same series read as hourly."""
        history, syms, start, end = _daily_history()
        scores = {}
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        for bar_hours in (1, 24):
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
            study.optimize(
                optimize.objective_factory(
                    conn, "_spy", history, syms, start, end, FEES, bar_hours=bar_hours, universe="classic"
                ),
                n_trials=1,
            )
            scores[bar_hours] = study.best_value
        assert scores[24] == pytest.approx(scores[1] / np.sqrt(24), rel=1e-6)

    def test_every_evaluation_is_recorded_closed_and_scoped(self, conn):
        history, syms, start, end = _daily_history()
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        optuna.create_study(direction="maximize").optimize(
            optimize.objective_factory(conn, "_spy", history, syms, start, end, FEES, bar_hours=24, universe="classic"),
            n_trials=3,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT verdict, finished_at, universe FROM trials WHERE family = '_spy'")
            rows = cur.fetchall()
        assert len(rows) == 3
        assert all(r["verdict"] == "scored" and r["finished_at"] is not None for r in rows)
        assert {r["universe"] for r in rows} == {"classic"}
        assert registry.count_trials(conn, "_spy", "crypto") == 0
        assert registry.count_trials(conn, "_spy", "classic") == 3


class TestNextVersion:
    def test_versions_are_counted_per_arena(self, conn):
        for universe in ("crypto", "classic"):
            registry.insert_competitor(
                conn,
                CompetitorSpec(
                    None, f"trend_ts_v7_{universe}", "trend_ts", 7, {}, status="challenger", universe=universe
                ),
            )
        conn.commit()
        assert optimize.next_version(conn, "trend_ts", "crypto") == 8
        assert optimize.next_version(conn, "trend_ts", "classic") == 8
        assert optimize.next_version(conn, "carry", "crypto") == 1


class TestSearchPbo:
    def _series(self, mu, n=800, seed=0):
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        return pd.Series(np.random.default_rng(seed).normal(mu, 0.01, n), index=idx)

    def test_a_search_too_small_to_validate_gets_no_confidence(self):
        assert optimize.search_pbo([self._series(0.0, seed=i) for i in range(3)])["pbo"] == 1.0

    def test_empty_evaluations_are_dropped(self):
        out = optimize.search_pbo([pd.Series(dtype=float)] * 20)
        assert out["pbo"] == 1.0 and out["n_configs"] == 0

    def test_a_noise_search_scores_badly(self):
        assert optimize.search_pbo([self._series(0.0, seed=i) for i in range(20)])["pbo"] > 0.3

    def test_a_search_containing_a_real_edge_scores_well(self):
        series = [self._series(0.0, seed=i) for i in range(20)]
        series[5] = series[5] + 0.003
        assert optimize.search_pbo(series)["pbo"] < 0.2

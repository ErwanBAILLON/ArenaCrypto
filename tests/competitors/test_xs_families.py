"""The two cross-sectional families, and the ROI ladder they share."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.competitors.base import REGISTRY
from arena.competitors.ladder import LadderHoldingCompetitor, neutral_book
from arena.competitors.xs_complex import XsComplex
from arena.competitors.xs_sparse import XsSparse
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target
from arena.features import build as build_panel
from arena.features import feature_columns
from arena.labeling.barriers import RoiLadder
from arena.models import rff
from tests.conftest import make_candles, make_funding

SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "NEAR"]


@pytest.fixture(scope="module")
def history():
    candles = make_candles(symbols=SYMS, bars=24 * 200, drift={"BTC": 0.0005, "DOGE": -0.0005, "ADA": 0.0003}, seed=7)
    funding = make_funding(symbols=SYMS, candles=candles, rate=0.0001, per_symbol={"BTC": 0.0008, "DOGE": -0.0005})
    return candles, funding


def _snap(history, ts=None):
    candles, funding = history
    at = ts or candles["ts"].max()
    return Snapshot.from_long(at, SYMS, candles[candles["ts"] <= at], funding[funding["ts"] <= at])


# --------------------------------------------------------------------------- neutral book


class TestNeutralBook:
    def test_it_is_dollar_neutral(self):
        scores = pd.Series({f"S{i}": float(i) for i in range(10)})
        weights = neutral_book(scores, k=3, max_weight=0.2)
        assert sum(weights.values()) == pytest.approx(0.0)

    def test_the_best_go_long_and_the_worst_go_short(self):
        scores = pd.Series({f"S{i}": float(i) for i in range(10)})
        weights = neutral_book(scores, k=2, max_weight=0.5)
        assert weights["S9"] > 0 and weights["S8"] > 0
        assert weights["S0"] < 0 and weights["S1"] < 0

    def test_a_symbol_is_never_both_long_and_short(self):
        scores = pd.Series({"A": 1.0, "B": 0.0, "C": -1.0})
        weights = neutral_book(scores, k=3, max_weight=1.0)
        assert all(w != 0 for w in weights.values())
        assert len(weights) == len(set(weights))

    def test_long_only_carries_market_exposure_on_purpose(self):
        scores = pd.Series({f"S{i}": float(i) for i in range(10)})
        weights = neutral_book(scores, k=3, max_weight=0.5, long_only=True)
        assert all(w > 0 for w in weights.values())

    def test_an_empty_score_gives_an_empty_book(self):
        assert neutral_book(pd.Series(dtype=float), k=3, max_weight=0.1) == {}


# --------------------------------------------------------------------------- the ladder


class _Fixed(LadderHoldingCompetitor):
    """Always wants the same long; only the ladder can close it."""

    family = "_fixed_ladder"
    ladder = RoiLadder(steps=((0, 0.02), (50, 0.0)))
    stop = 0.03
    cooldown_bars = 5
    default_params: dict = {}

    def compute(self, snap: Snapshot) -> Decision:
        return {"BTC": Target(weight=0.5, conviction=1.0)}


def _walk(comp, candles, funding, n_bars, start_index):
    """Feed consecutive snapshots and record the weight on BTC at each bar."""
    out = []
    stamps = sorted(candles["ts"].unique())[start_index : start_index + n_bars]
    for ts in stamps:
        snap = Snapshot.from_long(ts, SYMS, candles[candles["ts"] <= ts], funding[funding["ts"] <= ts])
        decision = comp.decide(snap)
        out.append(decision.get("BTC").weight if "BTC" in decision else 0.0)
    return out


class TestLadder:
    def test_a_winner_is_closed_on_its_target(self, history):
        candles, funding = history
        weights = _walk(_Fixed(), candles, funding, 80, 24 * 150)
        assert weights[0] != 0.0
        assert any(w == 0.0 for w in weights), "the ladder never closed a position"

    def test_the_cooldown_prevents_instant_re_entry(self, history):
        candles, funding = history
        comp = _Fixed()
        weights = _walk(comp, candles, funding, 80, 24 * 150)
        closed = [i for i, w in enumerate(weights) if w == 0.0]
        assert closed
        first = closed[0]
        assert all(weights[i] == 0.0 for i in range(first, min(first + 5, len(weights))))

    def test_state_round_trips_so_a_fresh_process_agrees(self, history):
        candles, funding = history
        comp = _Fixed()
        _walk(comp, candles, funding, 10, 24 * 150)
        clone = _Fixed()
        clone.restore_state(comp.state())
        assert clone.state() == comp.state()

    def test_exits_are_measured_against_the_market_not_the_price(self, history):
        """A position that rose less than the universe has not won."""
        candles, funding = history
        comp = _Fixed()
        snap = _snap(history, sorted(candles["ts"].unique())[24 * 150])
        comp.decide(snap)
        entry = comp.state()["entries"]["BTC"]
        assert "market" in entry and entry["market"] > 0

    def test_a_custom_ladder_comes_from_params(self):
        comp = _Fixed({"roi_steps": [[0, 0.5], [10, 0.0]]})
        assert comp._ladder().target_at(0) == 0.5
        assert comp._ladder().horizon == 10


# --------------------------------------------------------------------------- xs_sparse


class TestXsSparse:
    def test_it_is_registered_and_needs_no_training(self):
        assert REGISTRY["xs_sparse"] is XsSparse
        assert XsSparse().decide is not None

    def test_it_trades_and_stays_dollar_neutral(self, history):
        decision = XsSparse().decide(_snap(history))
        assert decision
        assert sum(t.weight for t in decision.values()) == pytest.approx(0.0, abs=1e-9)
        assert sum(abs(t.weight) for t in decision.values()) <= 1.0

    def test_the_score_is_an_unweighted_average_of_signed_ranks(self, history):
        panel = build_panel(_snap(history))
        comp = XsSparse()
        scores = comp.score(panel)
        manual = sum(sign * (panel[col] - 0.5) for col, sign in comp._signals().items() if col in panel) / len(
            comp._signals()
        )
        pd.testing.assert_series_equal(scores.sort_index(), manual.dropna().sort_index(), check_names=False)

    def test_flipping_a_signal_sign_flips_the_book(self, history):
        snap = _snap(history)
        base = XsSparse({"signals": {"rank_ret_30d_skip_7d": 1.0}}).decide(snap)
        flipped = XsSparse({"signals": {"rank_ret_30d_skip_7d": -1.0}}).decide(snap)
        longs_base = {s for s, t in base.items() if t.weight > 0}
        longs_flipped = {s for s, t in flipped.items() if t.weight > 0}
        assert longs_base and not (longs_base & longs_flipped)

    def test_too_few_symbols_means_no_opinion(self, history):
        assert XsSparse({"min_symbols": 99}).decide(_snap(history)) == {}

    def test_no_look_ahead(self, history):
        candles, funding = history
        ts = candles["ts"].iloc[24 * 160]
        truncated = _snap(history, ts)
        full_view = Snapshot.from_long(candles["ts"].max(), SYMS, candles, funding).at(ts)
        assert XsSparse().decide(truncated) == XsSparse().decide(full_view)


# --------------------------------------------------------------------------- xs_complex


def _train_toy_model(history):
    """A model fitted on the panel itself, enough to exercise the plumbing."""
    panel = build_panel(_snap(history))
    cols = feature_columns(panel)
    x = panel[cols].to_numpy(dtype=float)
    y = np.asarray(panel["ret_7d"].rank(pct=True) - 0.5, dtype=float)
    return rff.fit(x, y, cols, n_features=200, gamma=0.02, lam=1.0)[1.0]


class TestXsComplex:
    def test_without_an_artefact_it_stands_flat(self, history):
        comp = XsComplex()
        assert not comp.trained
        assert comp.decide(_snap(history)) == {}

    def test_with_an_artefact_it_trades(self, history):
        model = _train_toy_model(history)
        comp = XsComplex({"model_str": model.to_json()})
        assert comp.trained
        decision = comp.decide(_snap(history))
        assert decision and sum(t.weight for t in decision.values()) == pytest.approx(0.0, abs=1e-9)

    def test_the_score_is_demeaned_so_it_is_a_relative_call(self, history):
        model = _train_toy_model(history)
        comp = XsComplex({"model_str": model.to_json()})
        scores = comp.score(build_panel(_snap(history)))
        assert scores.mean() == pytest.approx(0.0, abs=1e-9)

    def test_the_reason_records_which_model_spoke(self, history):
        model = _train_toy_model(history)
        decision = XsComplex({"model_str": model.to_json()}).decide(_snap(history))
        reason = next(iter(decision.values())).reason
        assert reason["lambda"] == 1.0 and reason["features"] == 200

    def test_a_confidence_floor_can_silence_it(self, history):
        model = _train_toy_model(history)
        assert XsComplex({"model_str": model.to_json(), "min_abs_score": 1e9}).decide(_snap(history)) == {}

    def test_it_shares_every_mechanism_with_the_sparse_arm(self):
        """Only the predictor differs, so the comparison measures complexity alone."""
        for key in ("k", "max_weight", "target_vol", "min_symbols", "stop", "roi_steps"):
            assert XsComplex.default_params[key] == XsSparse.default_params[key]
        assert XsComplex.rebalance_weekday == XsSparse.rebalance_weekday

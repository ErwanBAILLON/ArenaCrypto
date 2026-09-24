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

    def test_exits_are_measured_against_a_basket_frozen_at_entry(self, history):
        """A position that rose less than the universe has not won, and the
        universe it is judged against is the one that existed when it opened."""
        candles, funding = history
        comp = _Fixed()
        snap = _snap(history, sorted(candles["ts"].unique())[24 * 150])
        comp.decide(snap)
        state = comp.state()
        key = state["entries"]["BTC"]["basis"]
        basket = state["basis"][key]
        assert set(basket) == set(SYMS) and all(v > 0 for v in basket.values())

    def test_the_basket_is_dropped_once_nothing_references_it(self, history):
        candles, funding = history
        comp = _Fixed()
        _walk(comp, candles, funding, 80, 24 * 150)
        live = {e["basis"] for e in comp.state()["entries"].values()}
        assert set(comp.state()["basis"]) == live

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


# --------------------------------------------------------------------------- hedge modes, vol, cadence


class TestHedgeModes:
    SCORES = pd.Series({f"S{i}": float(i) for i in range(12)} | {"BTCUSDT": 5.5, "ETHUSDT": 5.4})

    def test_index_hedge_shorts_everyone_else_a_little(self):
        w = neutral_book(self.SCORES, k=3, max_weight=0.2, hedge="index")
        longs = {s for s, x in w.items() if x > 0}
        shorts = {s for s, x in w.items() if x < 0}
        assert len(longs) == 3 and len(shorts) == len(self.SCORES) - 3
        assert sum(w.values()) == pytest.approx(0.0, abs=1e-12)

    def test_anchor_hedge_shorts_only_the_anchors(self):
        w = neutral_book(self.SCORES, k=3, max_weight=0.2, hedge="anchor", hedge_symbols=["BTCUSDT", "ETHUSDT"])
        assert {s for s, x in w.items() if x < 0} == {"BTCUSDT", "ETHUSDT"}
        assert sum(w.values()) == pytest.approx(0.0, abs=1e-12)

    def test_an_anchor_that_is_also_a_pick_is_not_shorted_against_itself(self):
        scores = pd.Series({"BTCUSDT": 9.0, "A": 1.0, "B": 2.0, "ETHUSDT": 0.5})
        w = neutral_book(scores, k=1, max_weight=0.5, hedge="anchor", hedge_symbols=["BTCUSDT", "ETHUSDT"])
        assert w["BTCUSDT"] > 0 and w["ETHUSDT"] < 0

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError):
            neutral_book(self.SCORES, k=2, max_weight=0.2, hedge="magic")


class TestPortfolioVol:
    def test_a_neutral_book_of_correlated_names_has_far_less_vol_than_its_names(self):
        from arena.competitors.ladder import portfolio_vol

        rng = np.random.default_rng(0)
        market = rng.normal(0, 0.01, 500)
        rets = pd.DataFrame({f"S{i}": market + rng.normal(0, 0.002, 500) for i in range(6)})
        weights = {"S0": 0.5, "S1": 0.5, "S2": -0.5, "S3": -0.5}
        book = portfolio_vol(weights, rets, 8760)
        single = float(rets["S0"].std() * np.sqrt(8760))
        assert book < 0.3 * single

    def test_portfolio_mode_levers_up_where_names_mode_does_not(self, history):
        """Sizing on the names' average vol left the book at a ninth of its risk budget."""
        snap = _snap(history)
        names = XsSparse({"vol_mode": "names", "max_leverage": 3.0}).decide(snap)
        port = XsSparse({"vol_mode": "portfolio", "max_leverage": 3.0}).decide(snap)
        assert sum(abs(t.weight) for t in port.values()) > sum(abs(t.weight) for t in names.values())

    def test_max_gross_caps_whatever_the_target_asks_for(self, history):
        d = XsSparse({"vol_mode": "portfolio", "max_leverage": 5.0, "max_gross": 0.3}).decide(_snap(history))
        assert sum(abs(t.weight) for t in d.values()) <= 0.3 + 1e-9


class TestCadence:
    def test_a_slower_cadence_reselects_less_often(self, history):
        """The ladder churns both books every day; what the cadence governs is re-selection."""
        candles, funding = history
        stamps = sorted(candles["ts"].unique())
        start = 24 * 120
        fast, slow = XsSparse({"rebalance_every_weeks": 1}), XsSparse({"rebalance_every_weeks": 4})
        seen = {"fast": set(), "slow": set()}
        for ts in stamps[start : start + 24 * 7 * 9]:
            snap = Snapshot.from_long(ts, SYMS, candles[candles["ts"] <= ts], funding[funding["ts"] <= ts])
            for name, comp in (("fast", fast), ("slow", slow)):
                comp.decide(snap)
                if comp.state()["last_rebalance"]:
                    seen[name].add(comp.state()["last_rebalance"])
        assert len(seen["fast"]) >= 8
        assert 2 <= len(seen["slow"]) <= 3

    def test_cadence_survives_a_state_round_trip(self, history):
        comp = XsSparse({"rebalance_every_weeks": 4})
        comp.decide(_snap(history))
        clone = XsSparse({"rebalance_every_weeks": 4})
        clone.restore_state(comp.state())
        assert clone.state()["last_rebalance"] == comp.state()["last_rebalance"]


class TestTwoSignalPreset:
    def test_it_is_a_named_preset_not_the_default(self, history):
        from arena.competitors.xs_sparse import DEFAULT_SIGNALS, TWO_SIGNALS

        assert set(TWO_SIGNALS) == {"rank_vol_30d", "rank_donchian_position"}
        assert XsSparse()._signals() == DEFAULT_SIGNALS
        assert XsSparse({"signals": TWO_SIGNALS}).decide(_snap(history))


# --------------------------------------------------------------------------- adaptive


class TestXsAdaptive:
    def test_it_stands_aside_until_signals_have_earned_a_vote(self, history):
        from arena.competitors.xs_adaptive import XsAdaptive

        comp = XsAdaptive({"min_symbols": 6})
        assert comp.decide(_snap(history)) == {}  # no trailing history yet
        assert comp.state()["history"]  # but it recorded the panel for later

    def test_it_learns_only_from_the_past(self, history):
        """Walk it weekly; trailing ICs appear only once a horizon has fully elapsed,
        and on data with real drift some candidate eventually clears a low bar."""
        from arena.competitors.xs_adaptive import XsAdaptive

        candles, funding = history
        stamps = sorted(candles["ts"].unique())
        comp = XsAdaptive({"min_symbols": 6, "min_t": 0.5, "lookback_weeks": 20, "horizon_days": 7})
        first_ics = None
        traded = False
        mondays = [t for t in stamps[24 * 100 :] if t.weekday() == 0 and t.hour == 0]  # the rebalance bar
        for i, ts in enumerate(mondays):
            snap = Snapshot.from_long(ts, SYMS, candles[candles["ts"] <= ts], funding[funding["ts"] <= ts])
            ics = comp._trailing_ics(snap)
            if i < 2:
                assert ics == {}  # nothing resolved yet: no look-ahead possible
            if ics and first_ics is None:
                first_ics = ics
            traded = traded or bool(comp.decide(snap))
        assert first_ics, "no trailing IC ever resolved"
        assert traded
        assert len(comp.state()["history"]) >= 8

    def test_state_round_trips(self, history):
        from arena.competitors.xs_adaptive import XsAdaptive

        comp = XsAdaptive({"min_symbols": 6})
        comp.decide(_snap(history))
        clone = XsAdaptive({"min_symbols": 6})
        clone.restore_state(comp.state())
        assert clone.state() == comp.state()

    def test_long_only_carries_no_shorts(self, history):
        d = XsSparse({"long_only": True}).decide(_snap(history))
        assert d and all(t.weight > 0 for t in d.values())


class TestTranchesAndRiskParity:
    def test_five_tranches_reselect_on_five_weekdays(self, history):
        candles, funding = history
        stamps = sorted(candles["ts"].unique())
        comp = XsSparse({"tranches": 5})
        for ts in [t for t in stamps[24 * 100 : 24 * 121] if t.hour == 0]:  # three weeks of midnights
            comp.decide(Snapshot.from_long(ts, SYMS, candles[candles["ts"] <= ts], funding[funding["ts"] <= ts]))
        assert len(comp.state()["tranches"]) == 5

    def test_a_tranched_book_averages_its_sub_books(self, history):
        candles, funding = history
        stamps = sorted(candles["ts"].unique())
        comp = XsSparse({"tranches": 5, "k": 3})
        last = {}
        for ts in [t for t in stamps[24 * 100 : 24 * 121] if t.hour == 0]:
            last = comp.decide(Snapshot.from_long(ts, SYMS, candles[candles["ts"] <= ts], funding[funding["ts"] <= ts]))
        assert last and max(abs(t.weight) for t in last.values()) < XsSparse.default_params["max_weight"] + 1e-9
        assert all("tranches" in t.reason for t in last.values())

    def test_tranche_state_round_trips(self, history):
        comp = XsSparse({"tranches": 3})
        comp.decide(_snap(history))
        clone = XsSparse({"tranches": 3})
        clone.restore_state(comp.state())
        assert clone.state()["tranches"] == comp.state()["tranches"]

    def test_risk_parity_gives_the_volatile_pick_less_dollar_weight(self, history):
        from arena.competitors.ladder import LadderHoldingCompetitor

        panel = build_panel(_snap(history))
        weights = {s: 0.1 for s in panel.index[:4]} | {s: -0.1 for s in panel.index[4:8]}
        parity = LadderHoldingCompetitor._risk_parity(panel, weights)
        longs = [s for s, w in parity.items() if w > 0]
        assert sum(parity[s] for s in longs) == pytest.approx(0.4)
        most, least = (
            max(longs, key=lambda s: panel.loc[s, "vol_30d"]),
            min(longs, key=lambda s: panel.loc[s, "vol_30d"]),
        )
        assert parity[most] < parity[least]

    def test_risk_parity_mode_trades_and_stays_neutral(self, history):
        d = XsSparse({"vol_mode": "riskparity"}).decide(_snap(history))
        assert d and sum(t.weight for t in d.values()) == pytest.approx(0.0, abs=1e-9)

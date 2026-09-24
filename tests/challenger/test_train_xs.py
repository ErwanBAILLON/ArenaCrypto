"""The training pipeline: what the target is, and whether selection can leak."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.challenger import train_xs
from arena.labeling.barriers import RoiLadder
from tests.conftest import make_candles, make_funding

SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "NEAR"]
LADDER = RoiLadder(steps=((0, 0.05), (48, 0.02), (120, 0.0)))


@pytest.fixture(scope="module")
def data():
    candles = make_candles(symbols=SYMS, bars=24 * 260, drift={"BTC": 0.0004, "DOGE": -0.0004}, seed=5)
    funding = make_funding(symbols=SYMS, candles=candles, rate=0.0001, per_symbol={"BTC": 0.0007})
    stamps = sorted(candles["ts"].unique())
    events = {pd.Timestamp(t): SYMS for t in stamps[24 * 100 :: 24 * 7]}
    return candles, funding, events


@pytest.fixture(scope="module")
def dataset(data):
    candles, funding, events = data
    return train_xs.build_dataset(candles, funding, events, ladder=LADDER, stop=0.06, costs=0.002)


class TestDataset:
    def test_it_produces_one_row_per_event_and_symbol(self, dataset):
        assert len(dataset) > 100
        assert len(dataset.features.groupby(["ts", "symbol"])) == len(dataset)

    def test_the_target_is_relative_within_its_own_date(self, dataset):
        """A neutral book trades relative outcomes, so the target is demeaned per date."""
        per_date = dataset.features.assign(y=dataset.target.to_numpy()).groupby("ts")["y"].mean()
        assert per_date.abs().max() < 1e-9

    def test_the_target_is_what_the_exit_rule_delivers_not_the_raw_return(self, dataset):
        """The ladder caps the upside; the target must be capped with it."""
        assert dataset.events["net_return"].max() < 0.5
        assert set(dataset.events["barrier"]) <= {"roi", "stop", "time"}

    def test_unfinished_trades_are_excluded(self, dataset):
        assert "open" not in set(dataset.events["barrier"])

    def test_costs_move_the_labels(self, data):
        candles, funding, events = data
        cheap = train_xs.build_dataset(candles, funding, events, ladder=LADDER, stop=0.06, costs=0.0)
        dear = train_xs.build_dataset(candles, funding, events, ladder=LADDER, stop=0.06, costs=0.03)
        assert (dear.events["label"] == 1).mean() < (cheap.events["label"] == 1).mean()

    def test_the_point_in_time_universe_is_honoured(self, data):
        """Only the symbols passed for a date may appear on that date."""
        candles, funding, events = data
        narrowed = {ts: SYMS[:5] for ts in events}
        ds = train_xs.build_dataset(candles, funding, narrowed, ladder=LADDER, stop=0.06)
        assert set(ds.features["symbol"]) <= set(SYMS[:5])

    def test_weights_average_to_one_and_vary(self, dataset):
        assert dataset.weights.mean() == pytest.approx(1.0)
        assert dataset.weights.std() > 0

    def test_no_label_column_leaks_into_the_features(self, dataset):
        for leaky in ("label", "net_return", "bars_held", "target_hit"):
            assert leaky not in dataset.columns

    def test_an_empty_universe_returns_an_empty_dataset(self, data):
        candles, funding, events = data
        assert len(train_xs.build_dataset(candles, funding, {ts: [] for ts in events})) == 0


class TestSpread:
    def test_a_perfect_ranker_scores_the_full_spread(self):
        y = np.array([3.0, 1.0, -1.0, -3.0] * 3)
        dates = np.repeat(pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]), 4)
        assert train_xs.long_short_spread(y, y, dates, k=1) == pytest.approx(6.0)

    def test_an_inverted_ranker_scores_the_negative_of_it(self):
        y = np.array([3.0, 1.0, -1.0, -3.0] * 3)
        dates = np.repeat(pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]), 4)
        assert train_xs.long_short_spread(-y, y, dates, k=1) == pytest.approx(-6.0)

    def test_it_ignores_the_middle_of_the_cross_section(self):
        """R-squared rewards accuracy about names nobody trades. This does not."""
        y = np.array([5.0, 0.1, -0.1, -5.0])
        dates = np.repeat(pd.to_datetime(["2024-01-01"]), 4)
        good_edges = train_xs.long_short_spread(np.array([2.0, -1.0, 1.0, -2.0]), y, dates, k=1)
        assert good_edges == pytest.approx(10.0)


class TestSelection:
    def test_selection_is_purged_and_leaks_nothing(self, dataset):
        sel = train_xs.select(dataset, n_features=400, gamma=0.02, n_groups=5, n_test=2, k=3)
        assert sel.leakage["max_overlap_ns"] == 0
        assert sel.n_splits == 10  # C(5, 2)

    def test_it_reports_the_whole_shrinkage_path(self, dataset):
        sel = train_xs.select(dataset, n_features=400, gamma=0.02, n_groups=5, n_test=2, k=3)
        assert set(sel.by_lambda) == set(train_xs.rff.DEFAULT_LAMBDAS)
        assert sel.lam in sel.by_lambda
        assert sel.score == max(v for v in sel.by_lambda.values() if np.isfinite(v))

    def test_it_measures_whether_picking_the_best_generalises(self, dataset):
        sel = train_xs.select(dataset, n_features=400, gamma=0.02, n_groups=5, n_test=2, k=3)
        assert 0.0 <= sel.pbo["pbo"] <= 1.0

    def test_noise_does_not_produce_a_positive_spread(self):
        """Pure random walks, no drift anywhere: finding a spread here would be the bug.

        The shared fixture gives BTC and DOGE a drift, so momentum legitimately
        predicts there; this test builds its own driftless panel. The bar is loose
        (0.01) because a spread over ~110 test rows and nine shrinkage values is a
        noisy statistic even on noise.
        """
        candles = make_candles(symbols=SYMS, bars=24 * 260, seed=11)
        funding = make_funding(symbols=SYMS, candles=candles, rate=0.0001)
        stamps = sorted(candles["ts"].unique())
        events = {pd.Timestamp(t): SYMS for t in stamps[24 * 100 :: 24 * 7]}
        data = train_xs.build_dataset(candles, funding, events, ladder=LADDER, stop=0.06, costs=0.002)
        sel = train_xs.select(data, n_features=400, gamma=0.02, n_groups=5, n_test=2, k=3)
        assert sel.score < 0.01

    def test_too_few_events_refuses_rather_than_guesses(self, data):
        candles, funding, events = data
        first = dict(list(sorted(events.items()))[:1])
        tiny = train_xs.build_dataset(candles, funding, first, ladder=LADDER)
        with pytest.raises(ValueError, match="cross-validate"):
            train_xs.select(tiny, n_features=100, n_groups=5, n_test=2)


class TestTrain:
    def test_the_artefact_replays_identically(self, dataset):
        sel = train_xs.select(dataset, n_features=300, gamma=0.02, n_groups=5, n_test=2, k=3)
        model = train_xs.train(dataset, sel)
        replayed = train_xs.rff.RffRidge.from_json(model.to_json())
        assert np.allclose(model.predict(dataset.matrix), replayed.predict(dataset.matrix), atol=1e-4)

    def test_it_carries_the_evidence_that_chose_it(self, dataset):
        sel = train_xs.select(dataset, n_features=300, gamma=0.02, n_groups=5, n_test=2, k=3)
        metrics = train_xs.train(dataset, sel).metrics
        assert metrics["cv_splits"] == 10
        assert metrics["max_overlap_ns"] == 0
        assert "pbo" in metrics and "cv_by_lambda" in metrics


class TestSearch:
    def test_it_sweeps_width_and_bandwidth_not_just_shrinkage(self, dataset):
        out = train_xs.search(dataset, widths=(200, 400), gammas=(0.01, 0.05), n_groups=5, n_test=2, k=3)
        assert len(out.grid) == 4
        assert {(s.n_features, s.gamma) for s in out.grid} == {(200, 0.01), (200, 0.05), (400, 0.01), (400, 0.05)}

    def test_pbo_is_over_the_whole_grid(self, dataset):
        """Nine lambdas are nine near-identical models; the grid is the real test."""
        out = train_xs.search(dataset, widths=(200, 400), gammas=(0.01, 0.05), n_groups=5, n_test=2, k=3)
        assert out.pbo["n_configs"] == 4 * len(train_xs.rff.DEFAULT_LAMBDAS)
        assert 0.0 <= out.pbo["pbo"] <= 1.0
        assert out.best.pbo is out.pbo

    def test_the_table_ranks_configurations(self, dataset):
        out = train_xs.search(dataset, widths=(200,), gammas=(0.01, 0.05), n_groups=5, n_test=2, k=3)
        table = out.table
        assert list(table.columns) == ["n_features", "gamma", "lam", "cv_spread"]
        assert table["cv_spread"].is_monotonic_decreasing

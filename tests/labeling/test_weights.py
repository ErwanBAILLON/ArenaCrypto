"""Overlapping labels must not let one market move vote fifty times."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.labeling.weights import average_uniqueness, concurrency, sample_weights, trend_tstat, trend_tstats

IDX = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")


def _events(spans):
    return pd.DataFrame({"ts": [IDX[a] for a, _ in spans], "exit_ts": [IDX[b] for _, b in spans]})


class TestConcurrency:
    def test_disjoint_labels_never_overlap(self):
        counts = concurrency(_events([(0, 5), (10, 15)]), IDX)
        assert counts.max() == 1
        assert counts.iloc[7] == 0

    def test_overlapping_labels_stack(self):
        counts = concurrency(_events([(0, 20), (5, 25), (10, 30)]), IDX)
        assert counts.iloc[12] == 3
        assert counts.iloc[2] == 1

    def test_no_events_is_all_zero(self):
        assert concurrency(pd.DataFrame(columns=["ts", "exit_ts"]), IDX).sum() == 0


class TestUniqueness:
    def test_a_label_alone_in_the_window_is_fully_unique(self):
        u = average_uniqueness(_events([(0, 10)]), IDX)
        assert u.iloc[0] == pytest.approx(1.0)

    def test_three_simultaneous_labels_are_worth_a_third_each(self):
        u = average_uniqueness(_events([(0, 10), (0, 10), (0, 10)]), IDX)
        assert all(x == pytest.approx(1 / 3) for x in u)

    def test_partial_overlap_is_between_the_two(self):
        u = average_uniqueness(_events([(0, 20), (10, 30)]), IDX)
        assert 0.5 < u.iloc[0] < 1.0

    def test_empty_input(self):
        assert average_uniqueness(pd.DataFrame(columns=["ts", "exit_ts"]), IDX).empty


class TestTrendScanning:
    def test_a_clean_uptrend_has_a_large_positive_t(self):
        prices = pd.Series(100 * np.exp(np.linspace(0, 0.2, 48)), index=IDX)
        t, horizon = trend_tstat(prices)
        assert t > 20 and horizon >= 6

    def test_a_clean_downtrend_is_negative(self):
        prices = pd.Series(100 * np.exp(np.linspace(0, -0.2, 48)), index=IDX)
        assert trend_tstat(prices)[0] < -20

    def test_noise_has_a_small_t(self):
        rng = np.random.default_rng(3)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 48))), index=IDX)
        assert abs(trend_tstat(prices)[0]) < 20

    def test_too_short_to_fit_says_zero(self):
        assert trend_tstat(pd.Series([100.0, 101.0], index=IDX[:2])) == (0.0, 0)

    def test_it_scans_forward_from_each_event(self):
        prices = pd.Series(100 * np.exp(np.linspace(0, 0.3, 48)), index=IDX)
        out = trend_tstats(prices, pd.DatetimeIndex([IDX[0], IDX[20]]))
        assert len(out) == 2 and (out > 0).all()


class TestSampleWeights:
    def test_weights_average_to_one(self):
        w = sample_weights(_events([(0, 10), (0, 10), (20, 30)]), IDX)
        assert w.mean() == pytest.approx(1.0)

    def test_a_crowded_label_weighs_less_than_a_lonely_one(self):
        events = _events([(0, 20), (0, 20), (0, 20), (30, 40)])
        w = sample_weights(events, IDX)
        assert w.iloc[3] > w.iloc[0]

    def test_trend_strength_reweights_without_inflating_the_sample(self):
        events = _events([(0, 10), (20, 30)])
        strong = pd.Series({IDX[0]: 6.0, IDX[20]: 0.2})
        w = sample_weights(events, IDX, trend_t=strong)
        assert w.iloc[0] > w.iloc[1]
        assert w.mean() == pytest.approx(1.0)

    def test_trend_strength_is_clipped_so_one_episode_cannot_dominate(self):
        events = _events([(0, 10), (20, 30)])
        capped = sample_weights(events, IDX, trend_t=pd.Series({IDX[0]: 4.0, IDX[20]: 1.0}))
        absurd = sample_weights(events, IDX, trend_t=pd.Series({IDX[0]: 400.0, IDX[20]: 1.0}))
        assert capped.iloc[0] == pytest.approx(absurd.iloc[0])

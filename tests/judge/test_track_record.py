"""Standard error, PSR, MinTRL, effective sample size and PBO."""

from __future__ import annotations

import numpy as np
import pytest

from arena.judge import metrics as m


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def _overlapping(rng: np.random.Generator, n_bets: int, hold: int) -> np.ndarray:
    """One bet repeated ``hold`` bars: T bars but only ``n_bets`` independent draws."""
    return np.repeat(rng.normal(0.0005, 0.02, n_bets), hold) / hold


class TestEffectiveN:
    def test_iid_series_keeps_its_sample_size(self, rng):
        r = rng.normal(0.0, 0.01, 2000)
        assert m.effective_n(r) == pytest.approx(2000, rel=0.05)

    def test_held_position_is_discounted(self, rng):
        r = _overlapping(rng, 40, 168)
        assert m.effective_n(r) < 0.25 * r.size

    def test_never_exceeds_the_number_of_bars(self, rng):
        alternating = np.tile([1.0, -1.0], 500) * rng.normal(1.0, 0.01, 1000)
        assert m.effective_n(alternating) <= 1000

    def test_short_series_is_returned_as_is(self):
        assert m.effective_n([0.01, -0.01]) == 2.0

    def test_newey_west_lag_grows_slowly(self):
        assert m.newey_west_lag(4) == 0
        assert m.newey_west_lag(100) == 4
        assert m.newey_west_lag(10_000) < 12


class TestSharpeSe:
    def test_four_days_of_hourly_bars_is_worthless(self, rng):
        """The pathology the leaderboard used to rank on: se dwarfs the estimate."""
        r = rng.normal(0.0, 0.005, 98)
        assert m.sharpe_se(r) > 8.0

    def test_se_shrinks_with_the_square_root_of_time(self, rng):
        long = rng.normal(0.0, 0.005, 9800)
        short = rng.normal(0.0, 0.005, 98)
        assert m.sharpe_se(long) < m.sharpe_se(short) / 5.0

    def test_autocorrelation_widens_the_interval(self, rng):
        r = _overlapping(rng, 40, 168)
        assert m.sharpe_se(r, adjust_autocorr=True) > 2.0 * m.sharpe_se(r, adjust_autocorr=False)

    def test_degenerate_series_is_nan(self):
        assert np.isnan(m.sharpe_se([0.01]))


class TestProbabilisticSharpe:
    def test_flat_series_carries_no_opinion(self):
        assert m.probabilistic_sharpe([0.0, 0.0, 0.0, 0.0]) == 0.5

    def test_strong_long_record_is_conclusive(self, rng):
        r = rng.normal(0.0008, 0.005, 20_000)
        assert m.probabilistic_sharpe(r) > 0.99

    def test_losing_record_is_not(self, rng):
        r = rng.normal(-0.0008, 0.005, 20_000)
        assert m.probabilistic_sharpe(r) < 0.01

    def test_benchmark_is_annualised(self, rng):
        r = rng.normal(0.0004, 0.005, 20_000)
        assert m.probabilistic_sharpe(r, sr_benchmark=0.0) > m.probabilistic_sharpe(r, sr_benchmark=10.0)

    def test_overlap_makes_the_same_series_less_convincing(self, rng):
        r = _overlapping(rng, 60, 168)
        assert m.probabilistic_sharpe(r, adjust_autocorr=True) < m.probabilistic_sharpe(r, adjust_autocorr=False)


class TestMinTrackRecordLength:
    def test_a_losing_rule_never_becomes_conclusive(self):
        assert m.min_track_record_length(-1.0) == float("inf")
        assert m.min_track_record_length(1.0, sr_benchmark=2.0) == float("inf")

    def test_a_bigger_edge_needs_less_time(self):
        assert m.min_track_record_length(3.0) < m.min_track_record_length(1.0)

    def test_negative_skew_and_fat_tails_demand_more_bars(self):
        plain = m.min_track_record_length(1.5, skew=0.0, kurt=3.0)
        ugly = m.min_track_record_length(1.5, skew=-1.5, kurt=9.0)
        assert ugly > plain

    def test_a_sharpe_of_one_needs_years_of_hourly_bars(self):
        assert m.min_track_record_length(1.0) > 2 * m.PPY_HOURLY


class TestTrackRecordVerdict:
    def test_a_four_day_arena_can_conclude_nothing(self, rng):
        v = m.track_record_verdict(rng.normal(0.0002, 0.005, 98))
        assert not v["conclusive"]
        assert v["missing"] > 0

    def test_it_reports_the_overlap_it_found(self, rng):
        v = m.track_record_verdict(_overlapping(rng, 40, 168))
        assert v["overlap"] > 3.0
        assert v["effective"] < v["observed"]

    def test_needed_is_expressed_in_bars_not_in_independent_draws(self, rng):
        r = _overlapping(rng, 60, 168)
        v = m.track_record_verdict(r)
        assert v["needed"] == pytest.approx(
            m.min_track_record_length(v["sharpe"], 0.0, *m.skew_kurt(r)) * v["overlap"], rel=1e-6
        )


class TestPbo:
    def test_pure_noise_search_is_around_a_coin_flip(self, rng):
        assert m.pbo_cscv(rng.normal(0.0, 0.01, (2000, 40)))["pbo"] > 0.3

    def test_a_genuine_edge_generalises(self, rng):
        mat = rng.normal(0.0, 0.01, (2000, 40))
        mat[:, 7] += 0.003
        assert m.pbo_cscv(mat)["pbo"] < 0.1

    def test_too_few_configurations_means_no_confidence(self, rng):
        assert m.pbo_cscv(rng.normal(0.0, 0.01, (2000, 1)))["pbo"] == 1.0

    def test_too_few_bars_means_no_confidence(self, rng):
        assert m.pbo_cscv(rng.normal(0.0, 0.01, (10, 40)))["pbo"] == 1.0

    def test_split_count_is_the_symmetric_combination_count(self, rng):
        out = m.pbo_cscv(rng.normal(0.0, 0.01, (2000, 5)), s=6)
        assert out["n_splits"] == 20  # C(6, 3)

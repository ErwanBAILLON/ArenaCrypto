"""Purging and embargo, tested as the leak-prevention they are."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.judge.cpcv import cpcv_splits, group_rows, leakage_report, n_paths, purge

BASE = pd.Timestamp("2024-01-01", tz="UTC")


def _events(n=60, span_hours=10, step_hours=4):
    ts = [BASE + pd.Timedelta(hours=i * step_hours) for i in range(n)]
    return pd.DataFrame({"ts": ts, "exit_ts": [t + pd.Timedelta(hours=span_hours) for t in ts]})


class TestPurge:
    def test_an_overlapping_label_is_removed_from_training(self):
        events = _events(n=10, span_hours=10, step_hours=4)  # each label outlives the next two starts
        train = purge(events, np.array([5]))
        assert 5 not in train
        assert 4 not in train and 6 not in train  # their lifetimes straddle label 5

    def test_a_distant_label_survives(self):
        events = _events(n=10, span_hours=2, step_hours=24)
        assert 0 in purge(events, np.array([5]))

    def test_the_embargo_removes_what_merely_follows(self):
        events = _events(n=20, span_hours=1, step_hours=24)
        no_embargo = purge(events, np.array([5]))
        embargoed = purge(events, np.array([5]), embargo=pd.Timedelta(days=2))
        assert 6 in no_embargo and 6 not in embargoed
        assert 4 in embargoed  # the embargo looks forward only

    def test_test_rows_are_never_in_train(self):
        events = _events()
        test = np.array([10, 11, 12])
        assert not set(test) & set(purge(events, test))

    def test_an_empty_test_block_purges_nothing(self):
        events = _events(n=8)
        assert purge(events, np.array([], dtype=int)).size == 8


class TestGroups:
    def test_groups_are_contiguous_in_time(self):
        events = _events(n=60).sample(frac=1.0, random_state=0).reset_index(drop=True)
        groups = group_rows(events, 6)
        ends = [pd.DatetimeIndex(events["ts"]).asi8[g].max() for g in groups]
        assert ends == sorted(ends)  # ordered even though the input was shuffled

    def test_every_row_lands_in_exactly_one_group(self):
        events = _events(n=61)
        groups = group_rows(events, 6)
        allocated = np.concatenate(groups)
        assert sorted(allocated) == list(range(61))


class TestCpcv:
    def test_one_split_per_combination(self):
        splits = cpcv_splits(_events(n=120, span_hours=1, step_hours=4), n_groups=6, n_test=2)
        assert len(splits) == 15  # C(6, 2)
        assert len({s.test_groups for s in splits}) == 15

    def test_paths_are_many_not_one(self):
        """One backtest path is an anecdote; the combinatorial scheme gives a distribution."""
        assert n_paths(6, 2) == 5
        assert n_paths(10, 3) == 36
        assert n_paths(6, 6) == 0

    def test_no_overlap_survives_the_purge(self):
        events = _events(n=200, span_hours=30, step_hours=4)  # heavily overlapping labels
        splits = cpcv_splits(events, n_groups=6, n_test=2, embargo_frac=0.01)
        report = leakage_report(events, splits)
        assert report["splits"] == 15
        assert report["max_overlap_ns"] == 0  # anything else means the purge is broken

    def test_heavier_overlap_costs_training_data(self):
        """Purging is not free: the more labels overlap, the less is left to learn from."""
        light = leakage_report(*_with_splits(_events(n=200, span_hours=1, step_hours=4)))
        heavy = leakage_report(*_with_splits(_events(n=200, span_hours=100, step_hours=4)))
        assert heavy["mean_train"] < light["mean_train"]

    def test_the_embargo_is_read_in_the_right_unit(self):
        """asi8 gives the index's own unit; Timedelta.value is always ns. Mixing them
        turned a four-hour embargo into 198 days and emptied five splits in fifteen."""
        events = _events(n=120, span_hours=1, step_hours=4)
        wide = cpcv_splits(events, n_groups=6, n_test=2, embargo_frac=0.01)
        none = cpcv_splits(events, n_groups=6, n_test=2, embargo_frac=0.0)
        assert len(wide) == len(none) == 15
        assert all(s.train.size > 0 for s in wide)
        # a 1 % embargo on a 20-day sample costs a little training data, not all of it
        assert 0 < sum(s.train.size for s in none) - sum(s.train.size for s in wide) < sum(s.train.size for s in none)

    def test_degenerate_configurations_return_nothing(self):
        events = _events(n=50)
        assert cpcv_splits(events, n_groups=4, n_test=4) == []
        assert cpcv_splits(events, n_groups=4, n_test=0) == []
        assert cpcv_splits(pd.DataFrame(columns=["ts", "exit_ts"]), n_groups=4, n_test=1) == []

    def test_train_and_test_sizes_are_reported(self):
        splits = cpcv_splits(_events(n=120, span_hours=1, step_hours=4), n_groups=6, n_test=2)
        train, test = splits[0].sizes
        assert train > 0 and test == pytest.approx(40, abs=2)


def _with_splits(events):
    return events, cpcv_splits(events, n_groups=6, n_test=2, embargo_frac=0.0)

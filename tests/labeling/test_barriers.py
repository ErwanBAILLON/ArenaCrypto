"""The ROI ladder as a decaying upper barrier, and what a label is allowed to mean."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.labeling.barriers import DEFAULT_LADDER, RoiLadder, excess_paths, label_event, label_panel

FLAT = RoiLadder(steps=((0, 0.05), (10, 0.0)))


class TestRoiLadder:
    def test_it_reads_like_freqtrade_minimal_roi(self):
        ladder = RoiLadder(steps=((0, 0.06), (12, 0.03), (48, 0.015), (120, 0.0)))
        assert ladder.target_at(0) == 0.06
        assert ladder.target_at(11) == 0.06
        assert ladder.target_at(12) == 0.03
        assert ladder.target_at(47) == 0.03
        assert ladder.target_at(48) == 0.015
        assert ladder.target_at(500) == 0.0

    def test_the_last_rung_is_the_vertical_barrier(self):
        assert DEFAULT_LADDER.horizon == 120

    def test_it_refuses_a_malformed_ladder(self):
        with pytest.raises(ValueError):
            RoiLadder(steps=((5, 0.05), (0, 0.0)))  # not sorted, does not start at 0
        with pytest.raises(ValueError):
            RoiLadder(steps=((0, 0.05), (0, 0.01)))  # duplicate holding time

    def test_as_array_is_the_ladder_bar_by_bar(self):
        assert list(FLAT.as_array(12)) == [0.05] * 9 + [0.0] * 3


class TestLabelEvent:
    def _path(self, *steps):
        return np.cumsum(np.array(steps, dtype=float))

    def test_a_fast_winner_hits_the_top_rung(self):
        out = label_event(self._path(0.03, 0.04), FLAT)
        assert out.label == 1 and out.barrier == "roi" and out.bars_held == 2
        assert out.target_hit == 0.05

    def test_a_slow_winner_is_closed_by_the_deadline_in_profit(self):
        """+2 % never clears the 5 % rung; the 0 % rung is the deadline and it closes up."""
        out = label_event(self._path(*([0.002] * 12)), FLAT)
        assert out.label == 1 and out.barrier == "time" and out.bars_held == 10

    def test_matching_the_market_exactly_is_not_a_win(self):
        """A zero rung accepts equality; reading it as profit makes a flat trade a winner."""
        out = label_event(np.zeros(12), FLAT)
        assert out.label == 0 and out.barrier == "time"

    def test_a_ladder_must_decay(self):
        with pytest.raises(ValueError, match="non-increasing"):
            RoiLadder(steps=((0, 0.01), (10, 0.05)))

    def test_the_stop_wins_when_it_is_touched_first(self):
        out = label_event(self._path(-0.03, -0.04), FLAT, stop=0.05)
        assert out.label == -1 and out.barrier == "stop" and out.bars_held == 2

    def test_costs_are_charged_before_the_barriers_are_read(self):
        path = self._path(0.051)
        assert label_event(path, FLAT, cost=0.0).label == 1
        assert label_event(path, FLAT, cost=0.01).barrier != "roi"  # 5.1 % gross is not 5 % net

    def test_an_unfinished_path_says_open_rather_than_flat(self):
        out = label_event(self._path(0.001, 0.001), FLAT)
        assert out.barrier == "open" and out.label == 0

    def test_a_short_mirrors_the_path_not_the_barriers(self):
        falling = self._path(-0.03, -0.04)
        assert label_event(falling, FLAT, side=-1).label == 1
        assert label_event(falling, FLAT, side=1).label == -1

    def test_an_empty_path_is_open(self):
        assert label_event(np.array([]), FLAT).barrier == "open"

    def test_the_stop_is_read_as_a_magnitude(self):
        path = self._path(-0.06)
        assert label_event(path, FLAT, stop=0.05).label == -1
        assert label_event(path, FLAT, stop=-0.05).label == -1


class TestExcessAndPanel:
    def _frame(self):
        idx = pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC")
        returns = pd.DataFrame({"WIN": 0.01, "LOSE": -0.01, "FLAT": 0.0}, index=idx)
        benchmark = pd.Series(0.0, index=idx)
        return returns, benchmark, idx

    def test_excess_subtracts_the_benchmark_bar_by_bar(self):
        returns, _, idx = self._frame()
        bench = pd.Series(0.01, index=idx)
        ex = excess_paths(returns, bench)
        assert (ex["WIN"] == 0.0).all() and (ex["LOSE"] == -0.02).all()

    def test_beating_the_market_is_what_counts_not_going_up(self):
        """A symbol up 1 %/bar in a market up 1 %/bar has done nothing."""
        returns, _, idx = self._frame()
        ex = excess_paths(returns, pd.Series(0.01, index=idx))
        labels = label_panel(ex, pd.DatetimeIndex([idx[0]]), FLAT)
        assert labels.set_index("symbol").loc["WIN", "label"] != 1

    def test_the_panel_labels_every_pair(self):
        returns, bench, idx = self._frame()
        ex = excess_paths(returns, bench)
        labels = label_panel(ex, pd.DatetimeIndex([idx[0], idx[12]]), FLAT)
        assert len(labels) == 6
        assert set(labels["symbol"]) == {"WIN", "LOSE", "FLAT"}
        by = labels[labels["ts"] == idx[0]].set_index("symbol")
        assert by.loc["WIN", "label"] == 1 and by.loc["LOSE", "label"] == -1

    def test_per_symbol_costs_are_honoured(self):
        returns, bench, idx = self._frame()
        ex = excess_paths(returns, bench)
        cheap = label_panel(ex, pd.DatetimeIndex([idx[0]]), FLAT, costs=0.0)
        dear = label_panel(ex, pd.DatetimeIndex([idx[0]]), FLAT, costs=pd.Series({"WIN": 0.2}))
        assert cheap.set_index("symbol").loc["WIN", "label"] == 1
        assert dear.set_index("symbol").loc["WIN", "label"] != 1

    def test_events_at_the_very_end_are_skipped_not_invented(self):
        returns, bench, idx = self._frame()
        ex = excess_paths(returns, bench)
        assert label_panel(ex, pd.DatetimeIndex([idx[-1]]), FLAT).empty

    def test_exit_timestamps_are_inside_the_window(self):
        returns, bench, idx = self._frame()
        labels = label_panel(excess_paths(returns, bench), pd.DatetimeIndex([idx[0]]), FLAT)
        assert (labels["exit_ts"] > idx[0]).all()
        assert (labels["exit_ts"] <= idx[FLAT.horizon]).all()

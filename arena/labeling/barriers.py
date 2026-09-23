"""Triple-barrier labels with a ROI ladder, on excess returns, net of costs.

The question a label should answer is not "did the price go up" but **"was
there a trade here that paid, after costs, better than the market"**. Three
choices make that concrete:

*Pre-specified exits, never hindsight-optimal ones.* Labelling the local minimum
as "buy" produces a target that has no stable relationship to anything
observable, so the model fits it perfectly in sample and emits noise out of it.
The triple-barrier method (López de Prado, *Advances in Financial Machine
Learning*, 2018) instead fixes a profit barrier, a loss barrier and a deadline
in advance, and labels by whichever is touched first.

*A ROI ladder as the profit barrier.* freqtrade's ``minimal_roi`` is exactly a
profit target that decays with holding time -- take 6 % in the first hours, 3 %
after half a day, settle for anything after five days -- which is a
time-varying upper barrier and drops straight into the framework. The same
ladder labels the history and closes the live position, so training and trading
cannot drift apart.

*Excess over the benchmark, net of costs.* The path is the cumulative return
minus the universe's, and the round-trip cost is subtracted up front. A label
then reads "this beat the market by enough to pay for itself", which is the
only definition of winning that survives a bull run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Barrier = Literal["roi", "stop", "time", "open"]


@dataclass(frozen=True)
class RoiLadder:
    """A decaying profit target, in freqtrade's ``minimal_roi`` shape.

    ``steps`` maps *bars held* to the minimum profit that closes the position
    from that point on, read as "the requirement in force is the one for the
    largest threshold at or below the current holding time". The final step's
    bar count doubles as the vertical barrier: once the ladder reaches zero
    there is nothing left to wait for.
    """

    steps: tuple[tuple[int, float], ...] = ((0, 0.06), (12, 0.03), (48, 0.015), (120, 0.0))

    def __post_init__(self) -> None:
        bars = [b for b, _ in self.steps]
        if not self.steps or bars != sorted(bars) or bars[0] != 0:
            raise ValueError("steps must start at 0 bars and be sorted by holding time")
        if len(set(bars)) != len(bars):
            raise ValueError("duplicate holding times in the ladder")
        profits = [p for _, p in self.steps]
        if profits != sorted(profits, reverse=True):
            raise ValueError("a ROI ladder decays: profit targets must be non-increasing")
        if profits[-1] < 0:
            raise ValueError("the final rung cannot demand a loss")

    def target_at(self, bars_held: int) -> float:
        """Profit required to close after ``bars_held`` bars."""
        target = self.steps[0][1]
        for bars, value in self.steps:
            if bars_held >= bars:
                target = value
            else:
                break
        return float(target)

    @property
    def horizon(self) -> int:
        """Vertical barrier: the holding time at which the ladder bottoms out."""
        return int(self.steps[-1][0])

    def as_array(self, length: int) -> np.ndarray:
        """The ladder evaluated for holding times ``1..length``, for vectorised labelling."""
        return np.array([self.target_at(i) for i in range(1, length + 1)], dtype=float)


DEFAULT_LADDER = RoiLadder()


@dataclass(frozen=True)
class Outcome:
    """What a pre-specified trade opened at this event would have done."""

    label: int  # +1 profit barrier, -1 stop, 0 flat at the deadline
    barrier: Barrier
    bars_held: int
    net_return: float  # excess over benchmark, net of the round trip
    target_hit: float  # the ladder rung in force when it closed

    @property
    def won(self) -> bool:
        return self.label > 0


def label_event(
    excess_path: np.ndarray | pd.Series,
    ladder: RoiLadder = DEFAULT_LADDER,
    stop: float = 0.05,
    cost: float = 0.0,
    side: int = 1,
) -> Outcome:
    """Label one event from the cumulative excess return path that follows it.

    ``excess_path`` is the cumulative return of the symbol minus the benchmark's,
    one entry per bar after entry, *gross*. ``cost`` is the whole round trip and
    is charged immediately, so a move that merely covers the spread is not a win.
    ``side`` is +1 for a long and -1 for a short; the path is mirrored, the
    barriers are not.

    A path that reaches neither barrier before running out returns ``open`` and
    must be dropped from training: it is an unfinished trade, not a flat one.
    """
    path = np.asarray(pd.Series(excess_path, dtype=float).to_numpy(), dtype=float) * float(side)
    if path.size == 0:
        return Outcome(0, "open", 0, 0.0, ladder.target_at(0))
    net = path - float(cost)
    horizon = min(ladder.horizon, net.size)
    targets = ladder.as_array(net.size)

    for i in range(horizon):
        bars = i + 1
        if targets[i] <= 0.0:
            # A zero rung is freqtrade's force-exit: the deadline, not a profit
            # target. Treating it as one labels a symbol that exactly matched
            # the market as a winner, which is the whole failure mode this
            # module exists to avoid.
            return Outcome(int(np.sign(net[i])), "time", bars, float(net[i]), 0.0)
        if net[i] >= targets[i]:
            return Outcome(1, "roi", bars, float(net[i]), float(targets[i]))
        if net[i] <= -abs(stop):
            return Outcome(-1, "stop", bars, float(net[i]), float(targets[i]))
    if net.size < ladder.horizon:
        return Outcome(0, "open", net.size, float(net[-1]), float(targets[net.size - 1]))
    end = horizon - 1
    return Outcome(int(np.sign(net[end])), "time", horizon, float(net[end]), float(targets[end]))


def excess_paths(returns: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    """Per-bar excess returns of every column over ``benchmark``, aligned on the index."""
    bench = benchmark.reindex(returns.index)
    return returns.sub(bench, axis=0)


def _cost_for(costs: pd.Series | float, symbol: str) -> float:
    """Round-trip cost for one symbol; a symbol missing from the series is charged nothing."""
    if isinstance(costs, pd.Series):
        value = costs.get(symbol)
        return float(value) if value is not None and np.isfinite(value) else 0.0
    return float(costs) if costs else 0.0


def label_panel(
    excess: pd.DataFrame,
    events: pd.DatetimeIndex,
    ladder: RoiLadder = DEFAULT_LADDER,
    stop: float = 0.05,
    costs: pd.Series | float = 0.0,
    side: int = 1,
) -> pd.DataFrame:
    """Label every (event, symbol) pair from a frame of per-bar excess returns.

    Returns one row per pair with the outcome, the holding time and the realised
    net excess. Unfinished trades at the end of the sample are labelled ``open``
    and left in, so the caller can drop them deliberately rather than by
    accident -- silently treating them as flat is a small, systematic lie about
    the most recent (and most tempting) part of the history.
    """
    rows: list[dict] = []
    index = excess.index
    for event in events:
        start = index.searchsorted(pd.Timestamp(event), side="right")
        if start >= len(index):
            continue
        window = excess.iloc[start : start + ladder.horizon]
        if window.empty:
            continue
        for sym in excess.columns:
            series = window[sym].dropna()
            if series.empty:
                continue
            cost = _cost_for(costs, sym)
            outcome = label_event(series.cumsum(), ladder, stop, cost, side)
            rows.append(
                {
                    "ts": pd.Timestamp(event),
                    "symbol": sym,
                    "label": outcome.label,
                    "barrier": outcome.barrier,
                    "bars_held": outcome.bars_held,
                    "net_return": outcome.net_return,
                    "target_hit": outcome.target_hit,
                    "exit_ts": series.index[min(outcome.bars_held, len(series)) - 1],
                }
            )
    return pd.DataFrame(
        rows, columns=["ts", "symbol", "label", "barrier", "bars_held", "net_return", "target_hit", "exit_ts"]
    )

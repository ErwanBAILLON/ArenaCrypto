import pandas as pd

from arena.book.book import FeeModel
from arena.competitors.base import Competitor
from arena.core.types import Target
from arena.judge.backtest import HistoryFrames
from arena.judge.walkforward import folds, run_walkforward
from tests.conftest import make_candles


def test_folds_anchored_contiguous_cover_range():
    start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")
    fs = folds(start, end, test_days=90, min_train_days=180)
    assert len(fs) == 2  # 366 days = 180 train + 90 + 90 + 6 (trailing 6 days dropped: < 30)
    assert all(f.train_start == start for f in fs)
    assert fs[0].test_start == start + pd.Timedelta(days=180)
    assert all(f.train_end == f.test_start for f in fs)
    assert all(a.test_end == b.test_start for a, b in zip(fs, fs[1:], strict=False))
    assert all(f.test_end - f.test_start == pd.Timedelta(days=90) for f in fs[:2])
    assert fs[-1].test_end == start + pd.Timedelta(days=180 + 2 * 90)
    assert end - fs[-1].test_end < pd.Timedelta(days=30)


def test_short_trailing_fold_kept_when_at_least_30_days():
    fs = folds("2024-01-01", "2024-12-01", test_days=90, min_train_days=180)
    # 335 days: 180 train, then 90, then 65 (kept, >= 30)
    assert len(fs) == 2
    assert fs[-1].test_end == pd.Timestamp("2024-12-01", tz="UTC")
    assert fs[-1].test_end - fs[-1].test_start == pd.Timedelta(days=65)


def test_no_fold_when_history_too_short():
    assert folds("2024-01-01", "2024-06-01") == []


class Counting:
    instances = 0

    def __init__(self):
        Counting.instances += 1
        self.calls = 0

    def warmup_bars(self) -> int:
        return 48

    def decide(self, snap):
        self.calls += 1
        return {"BTC": Target(weight=0.5)}


def test_fresh_competitor_per_fold_and_test_window_only():
    candles = make_candles(["BTC", "ETH"], bars=24 * 300)
    Counting.instances = 0
    made = []

    def make():
        c = Counting()
        made.append(c)
        return c

    start, end = candles["ts"].min(), candles["ts"].max() + pd.Timedelta(hours=1)
    results = run_walkforward(
        make, HistoryFrames(candles), ["BTC", "ETH"], start, end, FeeModel(), test_days=30, min_train_days=180
    )
    fs = folds(start, end, test_days=30, min_train_days=180)
    assert len(results) == len(fs) == Counting.instances == 4
    for (fold, res), comp in zip(results, made, strict=True):
        assert res.returns.index.min() == fold.test_start  # warm-up satisfied from history before the fold
        assert res.returns.index.max() == fold.test_end - pd.Timedelta(hours=1)
        assert comp.calls == len(res.rows)
    all_ts = pd.concat([res.returns for _, res in results]).index
    assert all_ts.is_unique and all_ts.is_monotonic_increasing


class _BuyAll(Competitor):
    """Buys every symbol it is shown, equally: the membership decides what it can touch."""

    family = "_buy_all"

    def __init__(self):
        super().__init__({})
        self.seen: set[str] = set()

    def decide(self, snap):
        self.seen |= set(snap.symbols)
        return {s: Target(weight=1.0 / len(snap.symbols), conviction=1.0) for s in snap.symbols}


def test_point_in_time_membership_restricts_every_fold():
    candles = make_candles(["BTC", "ETH", "SOL"], bars=24 * 300)
    start, end = candles["ts"].min(), candles["ts"].max() + pd.Timedelta(hours=1)
    made = []

    def make():
        made.append(_BuyAll())
        return made[-1]

    # SOL never listed before the last fold; ETH delisted after the first rebalance
    members_at = {
        start: ["BTC", "ETH"],
        start + pd.Timedelta(days=200): ["BTC"],
        start + pd.Timedelta(days=260): ["BTC", "SOL"],
    }
    run_walkforward(
        make, HistoryFrames(candles), ["BTC", "ETH", "SOL"], start, end, FeeModel(), test_days=30, members_at=members_at
    )
    seen = set().union(*(c.seen for c in made))
    assert seen == {"BTC", "ETH", "SOL"}
    assert "SOL" not in made[0].seen and "ETH" not in made[-1].seen  # each fold saw only that bar's members

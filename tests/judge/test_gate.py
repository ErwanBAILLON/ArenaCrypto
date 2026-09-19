import numpy as np
import pandas as pd
import pytest

from arena.book.book import FeeModel
from arena.core.types import Target
from arena.judge.backtest import BacktestResult, HistoryFrames
from arena.judge.gate import GateConfig, evaluate, null_sharpe_threshold, run_null_distribution
from arena.judge.walkforward import Fold, run_walkforward
from tests.conftest import make_candles

SYMS = ["BTC", "ETH", "SOL"]


class NoiseTrader:
    """Random weights, gross ~ 1, re-drawn every bar."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def warmup_bars(self) -> int:
        return 0

    def decide(self, snap):
        w = self.rng.normal(size=len(snap.symbols))
        w = w / np.abs(w).sum()
        return {s: Target(weight=float(x)) for s, x in zip(snap.symbols, w, strict=True)}


class DriftFollower:
    def warmup_bars(self) -> int:
        return 24

    def decide(self, snap):
        return {"BTC": Target(weight=1.0)}


@pytest.fixture(scope="module")
def noise_history():
    return HistoryFrames(make_candles(SYMS, bars=24 * 300, seed=3))


@pytest.fixture(scope="module")
def drift_history():
    return HistoryFrames(make_candles(SYMS, bars=24 * 300, seed=4, drift={"BTC": 0.0008}))


def _span(h: HistoryFrames):
    return h.candles["ts"].min(), h.candles["ts"].max() + pd.Timedelta(hours=1)


def test_noise_trader_rejected(noise_history):
    start, end = _span(noise_history)
    fees = FeeModel()
    nulls = run_null_distribution(NoiseTrader, noise_history, SYMS, start, end, fees, n=12)
    thr = null_sharpe_threshold(nulls)
    wf = run_walkforward(lambda: NoiseTrader(seed=99), noise_history, SYMS, start, end, fees)
    verdict = evaluate(wf, thr, n_trials=13)
    assert not verdict.admitted
    assert {"sharpe_above_null", "bootstrap_p", "dsr"} & set(verdict.failed)
    assert verdict.metrics["decisions"] >= 30


def test_drift_follower_admitted_and_dsr_deflates(drift_history):
    start, end = _span(drift_history)
    fees = FeeModel()
    nulls = run_null_distribution(NoiseTrader, drift_history, SYMS, start, end, fees, n=12)
    thr = null_sharpe_threshold(nulls)
    wf = run_walkforward(DriftFollower, drift_history, SYMS, start, end, fees)
    v1 = evaluate(wf, thr, n_trials=1)
    assert v1.admitted, v1.failed
    assert v1.metrics["folds_positive_frac"] == 1.0
    assert v1.metrics["sharpe"] > thr
    assert v1.metrics["max_drawdown"] < 0.30
    v500 = evaluate(wf, thr, n_trials=500)
    assert v500.metrics["dsr"] < v1.metrics["dsr"]
    assert v500.metrics["n_trials"] == 500


def _result(r: pd.Series, decisions: int) -> BacktestResult:
    nav = 10_000 * (1 + r).cumprod()
    return BacktestResult(returns=r, nav=nav, rows=[], decisions=decisions, turnover=1.0)


def test_evaluate_edge_cases():
    fold = Fold(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-07-01", tz="UTC"),
        pd.Timestamp("2024-07-01", tz="UTC"),
        pd.Timestamp("2024-10-01", tz="UTC"),
    )
    idx = pd.date_range("2024-07-01", periods=500, freq="1h", tz="UTC")
    good = _result(pd.Series(np.random.default_rng(0).normal(0.002, 0.005, 500), index=idx), decisions=5)
    v = evaluate([(fold, good)], null_threshold=0.0, n_trials=1)
    assert v.failed == ["min_decisions"]
    v = evaluate([(fold, good)], null_threshold=0.0, n_trials=1, cfg=GateConfig(min_decisions=1))
    assert v.admitted
    empty = _result(pd.Series(dtype=float), decisions=0)
    v = evaluate([(fold, empty)], null_threshold=0.0, n_trials=1)
    assert not v.admitted and "min_decisions" in v.failed and v.metrics["sharpe"] == 0.0
    assert null_sharpe_threshold([]) == 0.0


def test_null_distribution_parallel_matches_serial():
    import numpy as np
    import pandas as pd

    from arena.book.book import FeeModel
    from arena.judge.backtest import HistoryFrames
    from arena.judge.gate import run_null_distribution
    from tests.conftest import make_candles

    class Coin:
        def __init__(self, seed):
            self.seed = seed

        def warmup_bars(self):
            return 1

        def decide(self, snap):
            from arena.core.types import Target

            rng = np.random.default_rng(self.seed + int(snap.ts.timestamp()) // (3600 * 168))
            return {"BTC": Target(float(rng.uniform(-0.5, 0.5)))}

    c = make_candles(["BTC"], bars=24 * 40)
    h = HistoryFrames(candles=c)
    end = c["ts"].max()
    start = end - pd.Timedelta(days=10)
    serial = run_null_distribution(Coin, h, ["BTC"], start, end, FeeModel(), n=4, workers=1)
    parallel = run_null_distribution(Coin, h, ["BTC"], start, end, FeeModel(), n=4, workers=2)
    for a, b in zip(serial, parallel, strict=True):
        pd.testing.assert_series_equal(a.returns, b.returns)

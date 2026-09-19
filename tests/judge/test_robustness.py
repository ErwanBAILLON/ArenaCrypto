import numpy as np
import pandas as pd
import pytest

from arena.book.book import FeeModel
from arena.core.types import Target
from arena.explain.verdicts import explain_robustness
from arena.judge.backtest import BacktestResult, HistoryFrames
from arena.judge.gate import GateConfig, evaluate, robust_regimes
from arena.judge.robustness import label_window, run_robustness, sample_windows, summarise
from arena.judge.walkforward import Fold
from tests.conftest import make_candles

SYMS = ["BTC", "ETH", "SOL"]
DAY = pd.Timedelta(days=1)


class LongBTC:
    def warmup_bars(self) -> int:
        return 24

    def decide(self, snap):
        return {"BTC": Target(weight=1.0)}


def two_regime_candles(bars: int = 24 * 400, seed: int = 7) -> pd.DataFrame:
    """Random walk where BTC drifts +0.1 %/bar in the first half and -0.1 %/bar in the second."""
    c = make_candles(SYMS, bars=bars, seed=seed)
    half = bars // 2
    drift = np.where(np.arange(bars) < half, 0.001, -0.001)
    factor = np.exp(np.cumsum(drift))
    is_btc = c["symbol"] == "BTC"
    for col in ("open", "high", "low", "close"):
        c.loc[is_btc, col] = c.loc[is_btc, col].to_numpy() * factor
    return c


@pytest.fixture(scope="module")
def regime_history() -> HistoryFrames:
    return HistoryFrames(two_regime_candles())


def _span(h: HistoryFrames):
    return h.candles["ts"].min(), h.candles["ts"].max() + pd.Timedelta(hours=1)


def test_sample_windows_deterministic_and_bounded():
    start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")
    a = sample_windows(start, end, n=50, seed=3)
    b = sample_windows(start, end, n=50, seed=3)
    c = sample_windows(start, end, n=50, seed=4)
    assert a == b and a != c and len(a) == 50
    for ws, we in a:
        assert start <= ws < we <= end
        assert 60 * DAY <= we - ws <= 180 * DAY
        assert ws.tz is not None and (ws - start) % pd.Timedelta(hours=1) == pd.Timedelta(0)
    lengths = {we - ws for ws, we in a}
    assert len(lengths) > 10  # lengths actually vary


def test_sample_windows_clamps_and_rejects_short_span():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    short = sample_windows(start, start + 90 * DAY, n=20, seed=0)  # max_days clamped to 90
    assert all(60 * DAY <= we - ws <= 90 * DAY for ws, we in short)
    with pytest.raises(ValueError):
        sample_windows(start, start + 30 * DAY, n=3)
    assert sample_windows(start, start + 200 * DAY, n=0) == []


def test_label_window_thresholds():
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    s, e = idx[0], idx[-1]
    assert label_window(pd.Series([100, 105, 110, 108, 111.0], index=idx), s, e) == "bull"
    assert label_window(pd.Series([100, 95, 90, 92, 89.0], index=idx), s, e) == "bear"
    assert label_window(pd.Series([100, 105, 95, 101, 109.0], index=idx), s, e) == "range"
    assert label_window(pd.Series([100, 90, 95, 101, 91.0], index=idx), s, e) == "range"
    assert label_window(pd.Series([100.0], index=idx[:1]), s, e) == "range"  # one close: no move
    naive = pd.Series([100, 120.0], index=idx[:2].tz_localize(None))
    assert label_window(naive, s, idx[1]) == "bull"
    assert label_window(pd.Series([100, 120.0], index=idx[:2]), idx[3], e) == "range"  # empty slice


def test_run_robustness_separates_regimes(regime_history):
    start, end = _span(regime_history)
    rob = run_robustness(LongBTC, regime_history, SYMS, start, end, FeeModel(), n=30, seed=1, workers=1)
    assert rob["n_windows"] == 30
    assert sum(rob["n_by_regime"].values()) == 30
    assert rob["n_by_regime"]["bull"] >= 5 and rob["n_by_regime"]["bear"] >= 5
    assert rob["win_rate_by_regime"]["bull"] >= 0.8
    assert rob["win_rate_by_regime"]["bear"] <= 0.2
    assert 0.0 <= rob["win_rate_overall"] <= 1.0
    assert rob["worst_return"] < 0  # the bear windows lose money
    assert rob["min_days"] == 60 and rob["max_days"] == 180
    for v in (rob["median_sharpe"], rob["median_return"], rob["worst_return"], rob["win_rate_overall"]):
        assert type(v) is float  # plain JSON types, no numpy scalars


def test_run_robustness_parallel_matches_serial(regime_history):
    start, end = _span(regime_history)
    kw = dict(n=6, seed=2, min_days=60, max_days=70)
    serial = run_robustness(LongBTC, regime_history, SYMS, start, end, FeeModel(), workers=1, **kw)
    parallel = run_robustness(LongBTC, regime_history, SYMS, start, end, FeeModel(), workers=2, **kw)
    assert serial == parallel


def test_summarise_missing_regime_is_none():
    rob = summarise([(1.0, 0.1), (0.5, -0.02), (2.0, 0.3)], ["bull", "bull", "range"], 60, 180)
    assert rob["win_rate_by_regime"] == {"bull": 0.5, "bear": None, "range": 1.0}
    assert rob["n_by_regime"] == {"bull": 2, "bear": 0, "range": 1}
    assert rob["win_rate_overall"] == pytest.approx(2 / 3)
    assert rob["worst_return"] == -0.02 and rob["median_sharpe"] == 1.0
    empty = summarise([], [], 60, 180)
    assert empty["median_sharpe"] is None and empty["worst_return"] is None and empty["win_rate_overall"] == 0.0


def _good_fold():
    fold = Fold(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-07-01", tz="UTC"),
        pd.Timestamp("2024-07-01", tz="UTC"),
        pd.Timestamp("2024-10-01", tz="UTC"),
    )
    idx = pd.date_range("2024-07-01", periods=500, freq="1h", tz="UTC")
    r = pd.Series(np.random.default_rng(0).normal(0.002, 0.005, 500), index=idx)
    return fold, BacktestResult(returns=r, nav=10_000 * (1 + r).cumprod(), rows=[], decisions=50, turnover=1.0)


def _rob(bull, bear, rng, n=(41, 29, 50)):
    return {
        "n_windows": sum(n),
        "win_rate_overall": 0.6,
        "win_rate_by_regime": {"bull": bull, "bear": bear, "range": rng},
        "n_by_regime": dict(zip(("bull", "bear", "range"), n, strict=True)),
        "median_sharpe": 1.0,
        "median_return": 0.05,
        "worst_return": -0.1,
    }


def test_evaluate_robust_regimes_criterion():
    folds = [_good_fold()]
    ok = evaluate(folds, null_threshold=0.0, n_trials=1, robustness=_rob(0.82, 0.38, 0.71))
    assert ok.admitted, ok.failed
    assert ok.metrics["win_rate_overall"] == 0.6
    assert ok.metrics["robustness"]["regimes_positive"] == 2 and ok.metrics["robustness"]["passed"] is True

    bad = evaluate(folds, null_threshold=0.0, n_trials=1, robustness=_rob(0.82, 0.38, 0.45))
    assert bad.failed == ["robust_regimes"]
    assert bad.metrics["robustness"]["regimes_positive"] == 1

    # a regime under 5 windows is not judged: bear 100 % on 3 windows does not count
    few = evaluate(folds, null_threshold=0.0, n_trials=1, robustness=_rob(0.9, 1.0, 0.2, n=(40, 3, 50)))
    assert "robust_regimes" in few.failed and few.metrics["robustness"]["regimes_judged"] == 2

    strict = GateConfig(min_regimes_positive=3)
    assert "robust_regimes" in evaluate(folds, 0.0, 1, cfg=strict, robustness=_rob(0.82, 0.38, 0.71)).failed
    lax = GateConfig(min_regime_win_rate=0.3)
    assert evaluate(folds, 0.0, 1, cfg=lax, robustness=_rob(0.82, 0.38, 0.71)).admitted

    none = evaluate(folds, null_threshold=0.0, n_trials=1)
    assert "robustness" not in none.metrics and "robust_regimes" not in none.failed
    assert robust_regimes({"win_rate_by_regime": {"bull": None}, "n_by_regime": {"bull": 10}}) == (False, 0, 0)


def test_explain_robustness_renders_percentages():
    rob = {**_rob(0.82, 0.38, 0.71), "n_windows": 120, "win_rate_overall": 0.74, "min_days": 60, "max_days": 180}
    rob.update(regimes_positive=2, regimes_judged=3, n_regimes_required=2, min_regime_win_rate=0.5, passed=True)
    lines = explain_robustness(rob)
    assert lines[0] == "Sur 120 périodes de 2 à 6 mois tirées au hasard : gagne dans 74 % des cas."
    assert lines[1] == "Par type de marché : haussier 82 % (41 périodes), baissier 38 % (29), sans tendance 71 % (50)."
    assert lines[-1].startswith("✓ Tient dans 2 types de marché sur 3 (il en faut 2")
    assert all("'" not in line and '"' not in line and "’" not in line for line in lines)

    missing = explain_robustness({**_rob(0.9, None, 0.6, n=(30, 0, 20)), "regimes_positive": 2, "passed": False})
    assert "baissier : aucune période" in missing[1] and missing[-1].startswith("✗ ")
    assert explain_robustness({}) == []

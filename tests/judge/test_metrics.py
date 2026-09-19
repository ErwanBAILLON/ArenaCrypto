import time

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from arena.judge import metrics as m


def test_sharpe_zero_std_guard():
    assert m.sharpe(pd.Series([0.01] * 50)) == 0.0
    assert m.sharpe(pd.Series([0.01])) == 0.0


def test_sharpe_hand_computed():
    r = np.array([0.01, -0.01, 0.02])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(8760)
    assert m.sharpe(pd.Series(r)) == pytest.approx(expected)
    assert m.sharpe(pd.Series(r), ppy=1) == pytest.approx(r.mean() / r.std(ddof=1))


def test_sortino():
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    dd = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0) ** 2))
    assert m.sortino(r, ppy=1) == pytest.approx(r.mean() / dd)
    assert m.sortino(pd.Series([0.01, 0.02])) == 0.0


def test_max_drawdown_exact():
    # nav: 1 -> 1.1 -> 0.55 -> 0.605 -> 0.6655 ; trough 0.55 from peak 1.1 = 50 %
    assert m.max_drawdown(pd.Series([0.1, -0.5, 0.1, 0.1])) == pytest.approx(0.5)
    assert m.max_drawdown(pd.Series([-0.2, 0.1])) == pytest.approx(0.2)  # first bar counts vs nav0=1
    assert m.max_drawdown(pd.Series([0.01, 0.02])) == 0.0


def test_profit_factor():
    assert m.profit_factor(pd.Series([0.02, -0.01, 0.03, -0.04])) == pytest.approx(0.05 / 0.05)
    assert m.profit_factor(pd.Series([0.01, 0.02])) == 100.0
    assert m.profit_factor(pd.Series([0.0, 0.0])) == 0.0
    assert m.profit_factor(pd.Series([-0.01])) == 0.0


def test_total_return():
    assert m.total_return(pd.Series([0.1, 0.1])) == pytest.approx(0.21)


def test_skew_kurt_normal_like():
    rng = np.random.default_rng(0)
    s, k = m.skew_kurt(pd.Series(rng.normal(size=100_000)))
    assert abs(s) < 0.05 and abs(k - 3) < 0.1
    assert m.skew_kurt(pd.Series([1.0, 1.0, 1.0])) == (0.0, 3.0)


def test_deflated_sharpe_single_trial_is_psr():
    sr, T = 0.05, 1000
    assert m.deflated_sharpe(sr, 1, T, skew=0.0, kurt=1.0) == pytest.approx(norm.cdf(sr * np.sqrt(T - 1)))
    # with kurt=3 the (kurt-1)/4*SR^2 term is tiny for small SR
    assert m.deflated_sharpe(sr, 1, T) == pytest.approx(norm.cdf(sr * np.sqrt(T - 1)), rel=1e-3)


def test_deflated_sharpe_monotone_in_trials():
    vals = [m.deflated_sharpe(0.03, n, 5000) for n in (1, 2, 10, 100, 1000)]
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False))
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_deflated_sharpe_guards():
    assert m.deflated_sharpe(0.1, 10, 1) == 0.0
    assert m.deflated_sharpe(0.03, 0, 100) == m.deflated_sharpe(0.03, 1, 100)


def test_block_bootstrap_p_noise_and_drift():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, 4000)
    noise -= noise.mean()  # exactly zero sample Sharpe -> p centred on 0.5
    p = m.block_bootstrap_p(pd.Series(noise), n=500)
    assert 0.3 < p < 0.7
    drift = pd.Series(rng.normal(0.002, 0.01, 2000))
    assert m.block_bootstrap_p(drift, n=500) < 0.01
    assert m.block_bootstrap_p(pd.Series([0.01] * 10)) == 1.0


def test_block_bootstrap_speed():
    r = pd.Series(np.random.default_rng(0).normal(0.0001, 0.01, 8760))
    t0 = time.perf_counter()
    m.block_bootstrap_p(r, block=24, n=1000)
    assert time.perf_counter() - t0 < 2.0

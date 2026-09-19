import numpy as np
import pandas as pd
import pytest

from arena.runner.allocator import compute, raw_weights


def _returns(n=24 * 60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        1: rng.normal(0.0010, 0.005, n),   # good
        2: rng.normal(-0.0003, 0.005, n),  # bad
        3: rng.normal(0.0001, 0.005, n),   # meh
    }, index=idx)


def test_negative_sharpe_gets_zero_and_weights_sum_to_one():
    w = raw_weights(_returns())
    assert 2 not in w
    assert sum(w.values()) == pytest.approx(1.0)
    assert w[1] > w[3]


def test_short_history_excluded():
    r = _returns(n=100)
    assert raw_weights(r) == {}


def test_ema_smoothing_moves_slowly():
    r = _returns()
    prev = {1: 0.0, 3: 1.0}
    w = compute(r, prev, alpha=0.1)
    assert 0 < w[1] < 0.2
    assert 0.8 < w[3] < 1.0


def test_all_negative_means_cash():
    r = _returns()
    r[1] = -abs(r[1]); r[3] = -abs(r[3])
    assert compute(r, {}, alpha=1.0) == {}

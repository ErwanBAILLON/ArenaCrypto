import numpy as np
import pandas as pd
import pytest

from arena.competitors.features import atr, ema, pct_return, realised_vol, zscore


def test_ema_matches_pandas_ewm():
    s = pd.Series(np.random.default_rng(0).normal(size=300)).cumsum()
    pd.testing.assert_series_equal(ema(s, 20), s.ewm(span=20, adjust=False).mean())


def test_atr_on_constructed_candles():
    n = 30
    df = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0}, index=range(n))
    a = atr(df, window=14)
    assert np.isnan(a.iloc[12])
    assert a.iloc[-1] == pytest.approx(4.0)
    # a gap up: previous close far below today's low widens the true range
    df.loc[n - 1, ["high", "low", "close"]] = [120.0, 118.0, 119.0]
    assert atr(df, window=14).iloc[-1] > 4.0


def test_realised_vol_scaling():
    rng = np.random.default_rng(1)
    closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 5000))))
    v_hourly = realised_vol(closes, 1000, bars_per_year=8760).iloc[-1]
    v_daily = realised_vol(closes, 1000, bars_per_year=365).iloc[-1]
    assert v_hourly == pytest.approx(v_daily * np.sqrt(8760 / 365))
    assert v_hourly == pytest.approx(0.01 * np.sqrt(8760), rel=0.1)


def test_pct_return_and_nan_handling():
    closes = pd.Series([100.0, 110.0, 121.0])
    assert pct_return(closes, 1) == pytest.approx(0.10)
    assert pct_return(closes, 2) == pytest.approx(0.21)
    assert np.isnan(pct_return(closes, 3))
    assert np.isnan(pct_return(pd.Series(dtype=float), 1))


def test_zscore_last_value():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0])
    z = zscore(s, 5)
    assert z.iloc[-1] == pytest.approx((10 - s.mean()) / s.std())

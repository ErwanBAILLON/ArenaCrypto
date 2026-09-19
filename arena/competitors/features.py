"""Shared indicator helpers used by every rule-based competitor.

Pure pandas, no state, no I/O. Keeping indicators in one place means two
families that both talk about "30-day realised vol" compute the same number,
which is what makes their parameters comparable in the challenger search.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with the classic alpha = 2 / (span + 1)."""
    return s.ewm(span=span, adjust=False).mean()


def atr(candles_df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range: rolling mean of max(H-L, |H-prevC|, |L-prevC|)."""
    prev_close = candles_df["close"].shift(1)
    tr = pd.concat(
        [
            candles_df["high"] - candles_df["low"],
            (candles_df["high"] - prev_close).abs(),
            (candles_df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def realised_vol(closes: pd.Series, window: int, bars_per_year: int = 8760) -> pd.Series:
    """Annualised rolling standard deviation of log returns."""
    log_ret = np.log(closes).diff()
    return log_ret.rolling(window).std() * np.sqrt(bars_per_year)


def pct_return(closes: pd.Series, bars: int) -> float:
    """Simple return of the last close over the close ``bars`` bars earlier; nan if too short."""
    if bars < 0 or len(closes) <= bars:
        return float("nan")
    ref = closes.iloc[-1 - bars]
    if ref == 0 or np.isnan(ref):
        return float("nan")
    return float(closes.iloc[-1] / ref - 1.0)


def zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling mean) / rolling std."""
    roll = s.rolling(window)
    return (s - roll.mean()) / roll.std()

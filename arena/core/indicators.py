"""Shared indicator helpers used by every rule-based competitor.

Pure pandas, no state, no I/O. Keeping indicators in one place means two
families that both talk about "30-day realised vol" compute the same number,
which is what makes their parameters comparable in the challenger search.
"""

from __future__ import annotations

from datetime import timedelta

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


# ---------------------------------------------------------------------------
# "last value" helpers in plain numpy: competitors call them once per symbol
# per bar, where pandas' per-call overhead (not the data size) dominates.


def last_ema(s: pd.Series, span: int) -> float:
    """Last EMA value (alpha = 2/(span+1)), from the trailing ``10*span`` closes.

    Equals ``ema(s, span).iloc[-1]`` up to the truncated warm-up (<1e-9 relative
    for 10 spans); with adjust=False the first value seeds the recursion.
    """
    x = np.asarray(s.to_numpy(dtype=float)[-span * 10 :])
    alpha = 2.0 / (span + 1)
    n = len(x)
    if n == 0:
        return float("nan")
    w = (1 - alpha) ** np.arange(n - 1, -1, -1)  # oldest gets highest power
    w[1:] *= alpha  # seed keeps full weight (1-alpha)^(n-1), the rest alpha(1-alpha)^k
    return float(np.dot(w, x))


def last_atr(candles_df: pd.DataFrame, window: int = 14) -> float:
    """Last ATR value (rolling mean of the true range over ``window`` bars)."""
    tail = candles_df.iloc[-(window + 1) :]
    h, lo, c = (tail[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    if len(c) < window + 1:
        return float("nan")
    prev = c[:-1]
    tr = np.maximum.reduce([h[1:] - lo[1:], np.abs(h[1:] - prev), np.abs(lo[1:] - prev)])
    return float(tr.mean())


def last_realised_vol(closes: pd.Series, window: int, bars_per_year: int = 8760) -> float:
    """Last annualised realised volatility over ``window`` log returns (ddof=1, as pandas)."""
    x = closes.to_numpy(dtype=float)[-(window + 1) :]
    if len(x) < window + 1:
        return float("nan")
    r = np.diff(np.log(x))
    return float(r.std(ddof=1) * np.sqrt(bars_per_year))


MIN_FUNDING_DISPERSION = 1e-6
"""Smallest 8h funding spread worth calling crowding (~0.1 %/year across the cross-section).

Also the guard against dividing by numerical dust: on a perfectly flat
cross-section the sample standard deviation is not exactly zero but ~1e-20, and
the resulting z-scores are floating-point noise that looks like a signal.
"""


def funding_zscores(
    snap, lookback_days: int, min_symbols: int = 4, min_dispersion: float = MIN_FUNDING_DISPERSION
) -> dict[str, float]:
    """Cross-sectional z-score of each symbol's mean 8h funding over ``lookback_days``.

    The **level** of funding says what the market is doing: in a bull run every
    perp pays, and ranking on the level then just ranks beta. The z-score
    across the universe says who is paying *more than the others* to hold the
    same kind of leveraged exposure, which is the crowding measure.

    Empty when fewer than ``min_symbols`` carry funding, or when the spread
    across them is below ``min_dispersion``: an undifferentiated cross-section
    has nothing to say about who is crowded.
    """
    since = snap.ts - timedelta(days=lookback_days)
    means: dict[str, float] = {}
    for sym in snap.symbols:
        f = snap.funding(sym)
        f = f[f.index > since]
        if not f.empty:
            means[sym] = float(f.mean())
    if len(means) < min_symbols:
        return {}
    values = np.array(list(means.values()), dtype=float)
    sd = float(values.std(ddof=1))
    if not np.isfinite(sd) or sd < min_dispersion:
        return {}
    mu = float(values.mean())
    return {sym: (m - mu) / sd for sym, m in means.items()}

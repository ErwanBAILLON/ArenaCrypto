"""Performance statistics on a series of simple per-period returns.

Every function takes a ``pd.Series`` (or array-like) ``r`` of simple returns,
one value per bar. Bars are hourly by default, hence ``ppy = 8760`` periods
per year for annualisation. All functions are pure and NaN-safe (NaNs are
dropped first).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PPY_HOURLY = 8760
EULER_GAMMA = 0.5772156649015329
PROFIT_FACTOR_CAP = 100.0


def _arr(r) -> np.ndarray:
    a = np.asarray(pd.Series(r, dtype=float).dropna().to_numpy(), dtype=float)
    return a


def sharpe(r, ppy: int = PPY_HOURLY) -> float:
    """Annualised Sharpe ratio ``mean(r) / std(r) * sqrt(ppy)`` (sample std, ddof=1).

    Returns 0 when fewer than two observations or when ``std == 0``.
    """
    a = _arr(r)
    if a.size < 2:
        return 0.0
    sd = a.std(ddof=1)
    if sd == 0.0 or not np.isfinite(sd):
        return 0.0
    return float(a.mean() / sd * np.sqrt(ppy))


def sortino(r, ppy: int = PPY_HOURLY) -> float:
    """Annualised Sortino ratio ``mean(r) / downside_dev * sqrt(ppy)``.

    ``downside_dev = sqrt(mean(min(r, 0)^2))``. Returns 0 when there is no
    downside (no losing bar) or fewer than two observations.
    """
    a = _arr(r)
    if a.size < 2:
        return 0.0
    dd = np.sqrt(np.mean(np.minimum(a, 0.0) ** 2))
    if dd == 0.0:
        return 0.0
    return float(a.mean() / dd * np.sqrt(ppy))


def max_drawdown(r) -> float:
    """Maximum peak-to-trough drawdown of the compounded NAV, as a positive fraction.

    ``nav_t = prod_{i<=t}(1 + r_i)`` with ``nav_0 = 1``;
    ``mdd = max_t (1 - nav_t / max_{s<=t} nav_s)``.
    """
    a = _arr(r)
    if a.size == 0:
        return 0.0
    nav = np.concatenate([[1.0], np.cumprod(1.0 + a)])
    peak = np.maximum.accumulate(nav)
    return float(np.max(1.0 - nav / peak))


def profit_factor(r) -> float:
    """``sum(r[r>0]) / |sum(r[r<0])|``; capped at 100 when there are no losses (0 if no gains either)."""
    a = _arr(r)
    gains = a[a > 0].sum()
    losses = -a[a < 0].sum()
    if losses == 0.0:
        return PROFIT_FACTOR_CAP if gains > 0 else 0.0
    return float(min(gains / losses, PROFIT_FACTOR_CAP))


def total_return(r) -> float:
    """Compounded return ``prod(1 + r) - 1``."""
    a = _arr(r)
    return float(np.prod(1.0 + a) - 1.0) if a.size else 0.0


def skew_kurt(r) -> tuple[float, float]:
    """Sample skewness and (non-excess) kurtosis; a normal series has ``(0, 3)``.

    Returns ``(0, 3)`` when the series is degenerate (fewer than 3 points or zero variance).
    """
    a = _arr(r)
    if a.size < 3 or a.std() == 0.0:
        return 0.0, 3.0
    return float(stats.skew(a, bias=False)), float(stats.kurtosis(a, fisher=False, bias=False))


def deflated_sharpe(
    sr_hat: float,
    n_trials: int,
    T: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sr_var_across_trials: float | None = None,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014): P(true SR > 0 | N trials).

    All Sharpe ratios are **per period** (not annualised). With ``V`` the
    variance of the SR estimates across the ``N`` trials, the expected maximum
    SR of ``N`` pure-noise strategies is::

        E[max SR] = sqrt(V) * ((1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N e)))    γ = 0.5772…

    and the probabilistic Sharpe ratio evaluated against that benchmark is::

        DSR = Φ( (SR̂ - E[max SR]) * sqrt(T - 1) / sqrt(1 - skew·SR̂ + (kurt - 1)/4 · SR̂²) )

    When ``sr_var_across_trials`` is not given, the SR estimator variance
    ``V = (1 - skew·SR̂ + (kurt - 1)/4 · SR̂²) / T`` is used. ``N <= 1`` means no
    selection bias, so ``E[max SR] = 0`` and the DSR reduces to the PSR against 0.
    """
    if T < 2:
        return 0.0
    denom_sq = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    if denom_sq <= 0.0:
        return 0.0
    if n_trials <= 1:
        e_max = 0.0
    else:
        v = denom_sq / T if sr_var_across_trials is None else float(sr_var_across_trials)
        e_max = np.sqrt(v) * (
            (1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n_trials)
            + EULER_GAMMA * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )
    z = (sr_hat - e_max) * np.sqrt(T - 1.0) / np.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def block_bootstrap_p(r, block: int = 24, n: int = 1000, seed: int = 0) -> float:
    """One-sided p-value ``P(Sharpe <= 0)`` under a stationary block bootstrap.

    Politis & Romano (1994): resampled series of the same length ``T`` are built
    from blocks whose start is uniform on ``[0, T)`` and whose length is
    geometric with mean ``block`` (each step continues the current block with
    probability ``1 - 1/block``, wrapping around circularly). The statistic is
    the per-period Sharpe ``mean/std`` of each resample; the p-value is the
    fraction of the ``n`` resampled Sharpes that are ``<= 0``.

    Fully vectorised: an ``(n, T)`` index matrix is built with a cumulative max
    trick (``k_t`` = position of the last block start at or before ``t``, index
    ``= start[k_t] + (t - k_t) mod T``).
    """
    a = _arr(r)
    T = a.size
    if T < 2 or np.ptp(a) == 0.0:  # degenerate series: no evidence either way
        return 1.0
    rng = np.random.default_rng(seed)
    p_new = 1.0 / max(block, 1)
    t = np.arange(T, dtype=np.int32)
    starts = rng.integers(0, T, size=(n, T), dtype=np.int32)
    new_block = rng.random((n, T)) < p_new
    new_block[:, 0] = True
    k = np.maximum.accumulate(np.where(new_block, t, 0).astype(np.int32), axis=1)
    idx = (np.take_along_axis(starts, k, axis=1) + (t - k)) % T
    samples = a[idx]
    mu = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mu / sd, 0.0)
    return float(np.mean(sr <= 0.0))

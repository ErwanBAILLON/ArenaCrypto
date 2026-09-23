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
    # "nearly identical" values make the third and fourth moments numerically
    # meaningless (scipy warns about catastrophic cancellation); a series that
    # flat has no skew and no tails worth naming.
    if a.size < 3 or a.std() == 0.0 or np.ptp(a) <= 1e-12 * max(1.0, float(np.abs(a).max())):
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


# ---------------------------------------------------------------------------
# Effective sample size: a position held for four days is not 96 observations
# ---------------------------------------------------------------------------


def newey_west_lag(n: int) -> int:
    """Automatic truncation lag ``floor(4 (n/100)^(2/9))`` (Newey & West, 1994)."""
    if n < 8:
        return 0
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def autocorrelations(r, max_lag: int) -> np.ndarray:
    """Sample autocorrelations ``ρ_1..ρ_max_lag`` (0 when the series is too short or flat)."""
    a = _arr(r)
    if a.size < 3 or max_lag < 1:
        return np.zeros(0)
    a = a - a.mean()
    denom = float(a @ a)
    if denom <= 0.0:
        return np.zeros(max_lag)
    lags = min(max_lag, a.size - 2)
    return np.array([float(a[k:] @ a[:-k]) / denom for k in range(1, lags + 1)])


def effective_n(r, max_lag: int | None = None) -> float:
    """Autocorrelation-adjusted sample size ``T / (1 + 2 Σ_k (1 - k/(L+1)) ρ_k)``.

    A competitor that re-decides once a week and holds in between produces
    hourly book returns that repeat the same bet 168 times; counting them as
    168 independent observations is how a backtest convinces itself. The
    Bartlett-weighted sum over a Newey-West bandwidth discounts them back.
    Clamped to ``[2, T]``: strong negative autocorrelation cannot manufacture
    more evidence than there are bars.
    """
    a = _arr(r)
    t = a.size
    if t < 3:
        return float(t)
    lag = newey_west_lag(t) if max_lag is None else int(max_lag)
    rho = autocorrelations(a, lag)
    if rho.size == 0:
        return float(t)
    weights = 1.0 - np.arange(1, rho.size + 1) / (rho.size + 1.0)
    factor = 1.0 + 2.0 * float(weights @ rho)
    if not np.isfinite(factor) or factor <= 0.0:
        return 2.0
    return float(min(max(t / factor, 2.0), t))


# ---------------------------------------------------------------------------
# How long before a track record means anything (Bailey & López de Prado, 2012)
# ---------------------------------------------------------------------------


def _psr_denominator(sr_hat: float, skew: float, kurt: float) -> float:
    """``1 - γ₃·SR̂ + (γ₄-1)/4·SR̂²``: the variance factor of the Sharpe estimator.

    Non-normality is what makes a Sharpe estimate wider than the textbook
    ``1/√T``: negative skew and fat tails inflate it. Returns NaN when the
    factor is non-positive (a degenerate series the estimator cannot describe).
    """
    d = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    return d if d > 0.0 else float("nan")


def sharpe_se(r, ppy: int = PPY_HOURLY, adjust_autocorr: bool = True) -> float:
    """Standard error of the **annualised** Sharpe, for skew, kurtosis and overlap.

    ``se(SR̂) = sqrt((1 - γ₃·SR̂ + (γ₄-1)/4·SR̂²) / (n_eff - 1))`` per period,
    scaled by ``sqrt(ppy)``, with ``n_eff`` the autocorrelation-adjusted sample
    size. This is the number that says whether a leaderboard line is
    information or decoration: on four days of hourly bars it is worth about
    nine Sharpe points, which is why the arena refuses to rank on it.
    """
    a = _arr(r)
    if a.size < 3:
        return float("nan")
    sr_p = sharpe(a, 1)
    d = _psr_denominator(sr_p, *skew_kurt(a))
    if not np.isfinite(d):
        return float("nan")
    n = effective_n(a) if adjust_autocorr else float(a.size)
    if n < 2:
        return float("nan")
    return float(np.sqrt(d / (n - 1.0)) * np.sqrt(ppy))


def probabilistic_sharpe(r, sr_benchmark: float = 0.0, ppy: int = PPY_HOURLY, adjust_autocorr: bool = True) -> float:
    """``P(true Sharpe > sr_benchmark)`` given the observed series (PSR).

    ``sr_benchmark`` is **annualised**, like every Sharpe the arena displays;
    it is converted to per-period internally. The sample size is the
    autocorrelation-adjusted one, so holding a position for a week does not
    count as a week of independent evidence. Returns 0.5 (no information) on a
    series too short or too degenerate to say anything.
    """
    a = _arr(r)
    if a.size < 3:
        return 0.5
    sr_p = sharpe(a, 1)
    sk, ku = skew_kurt(a)
    d = _psr_denominator(sr_p, sk, ku)
    if not np.isfinite(d):
        return 0.5
    n = effective_n(a) if adjust_autocorr else float(a.size)
    if n < 2:
        return 0.5
    bench_p = float(sr_benchmark) / np.sqrt(ppy)
    z = (sr_p - bench_p) * np.sqrt(n - 1.0) / np.sqrt(d)
    return float(stats.norm.cdf(z))


def min_track_record_length(
    sr_hat: float,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurt: float = 3.0,
    conf: float = 0.95,
    ppy: int = PPY_HOURLY,
) -> float:
    """Bars needed before ``SR > sr_benchmark`` can be claimed at ``conf`` (MinTRL).

    ``n* = 1 + (1 - γ₃·SR̂ + (γ₄-1)/4·SR̂²) · (z_conf / (SR̂ - SR*))²``, with both
    Sharpes **annualised** on input and converted to per-period. Infinite when
    the observed Sharpe does not beat the benchmark at all: no amount of
    waiting turns a losing rule into a winning one.
    """
    sr_p = float(sr_hat) / np.sqrt(ppy)
    bench_p = float(sr_benchmark) / np.sqrt(ppy)
    edge = sr_p - bench_p
    if edge <= 0.0:
        return float("inf")
    d = _psr_denominator(sr_p, skew, kurt)
    if not np.isfinite(d):
        return float("inf")
    return float(1.0 + d * (stats.norm.ppf(conf) / edge) ** 2)


def track_record_verdict(r, sr_benchmark: float = 0.0, conf: float = 0.95, ppy: int = PPY_HOURLY) -> dict:
    """``{sharpe, se, psr, observed, needed, missing, conclusive}`` for one live series.

    ``needed`` is the MinTRL implied by the Sharpe observed so far, so it moves
    as the evidence accumulates; ``conclusive`` is ``psr >= conf``, which is the
    only statement the arena is allowed to make about a competitor.
    """
    a = _arr(r)
    sr = sharpe(a, ppy)
    sk, ku = skew_kurt(a)
    n_eff = effective_n(a) if a.size >= 3 else float(a.size)
    needed_eff = min_track_record_length(sr, sr_benchmark, sk, ku, conf, ppy)
    # MinTRL is expressed in independent observations; the arena counts bars,
    # so it is inflated back by the overlap factor the series actually shows.
    overlap = (a.size / n_eff) if n_eff > 0 else 1.0
    needed = needed_eff * overlap if np.isfinite(needed_eff) else float("inf")
    psr = probabilistic_sharpe(a, sr_benchmark, ppy)
    return {
        "sharpe": sr,
        "se": sharpe_se(a, ppy),
        "psr": psr,
        "observed": int(a.size),
        "effective": float(n_eff),
        "overlap": float(overlap),
        "needed": needed,
        "missing": float("inf") if not np.isfinite(needed) else max(0.0, needed - a.size),
        "conclusive": bool(psr >= conf),
    }


# ---------------------------------------------------------------------------
# Probability of backtest overfitting (Bailey, Borwein, López de Prado, Zhu 2017)
# ---------------------------------------------------------------------------


def pbo_cscv(returns_matrix, s: int = 8, seed: int = 0) -> dict:
    """Probability of Backtest Overfitting by combinatorially symmetric cross-validation.

    ``returns_matrix`` is ``(T, N)``: one column of per-bar returns per
    configuration tried during a parameter search. The series are cut into
    ``s`` contiguous blocks; for each of the ``C(s, s/2)`` ways of splitting
    them into an in-sample and an out-of-sample half, the configuration that
    wins in-sample is located in the out-of-sample ranking. ``PBO`` is the
    share of splits where the in-sample winner lands in the bottom half
    out-of-sample — that is, the probability that the winner of the search is
    a winner only because of the search.

    Where the deflated Sharpe asks "is this one Sharpe too good for the number
    of tries?", PBO asks "does picking the best actually generalise?". The two
    fail differently, which is why the gate uses both. Returns
    ``{pbo, n_splits, n_configs, logits}``; ``pbo = 1.0`` (no confidence) when
    there is not enough material to split.
    """
    from itertools import combinations

    m = np.asarray(pd.DataFrame(returns_matrix).to_numpy(dtype=float))
    if m.ndim != 2 or m.shape[1] < 2:
        return {"pbo": 1.0, "n_splits": 0, "n_configs": int(m.shape[1]) if m.ndim == 2 else 0, "logits": []}
    s = max(2, s - (s % 2))
    t, n = m.shape
    if t < 2 * s:
        return {"pbo": 1.0, "n_splits": 0, "n_configs": n, "logits": []}
    blocks = np.array_split(np.arange(t), s)
    logits: list[float] = []
    for train_ids in combinations(range(s), s // 2):
        test_ids = [i for i in range(s) if i not in train_ids]
        tr = np.concatenate([blocks[i] for i in train_ids])
        te = np.concatenate([blocks[i] for i in test_ids])
        sr_in = np.array([sharpe(m[tr, j], 1) for j in range(n)])
        sr_out = np.array([sharpe(m[te, j], 1) for j in range(n)])
        best = int(np.argmax(sr_in))
        # relative rank of the in-sample winner in the out-of-sample ranking
        rank = float(stats.rankdata(sr_out)[best])
        omega = rank / (n + 1.0)
        omega = min(max(omega, 1e-6), 1.0 - 1e-6)
        logits.append(float(np.log(omega / (1.0 - omega))))
    if not logits:
        return {"pbo": 1.0, "n_splits": 0, "n_configs": n, "logits": []}
    return {
        "pbo": float(np.mean(np.asarray(logits) <= 0.0)),
        "n_splits": len(logits),
        "n_configs": n,
        "logits": logits,
    }

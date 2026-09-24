"""Training for the cross-sectional families: dataset, purged selection, artefact.

The target is the **realised net excess return of the pre-specified trade** --
what a position opened at this event and closed by the ROI ladder or the stop
would actually have earned, after costs, relative to the universe.

That choice matters more than the estimator. Regressing on the next period's
raw return teaches a model to forecast a number nobody can collect: the ladder
caps the upside, the stop truncates the downside, and fees take a slice of both.
Regressing on what the exit rule delivers teaches it to forecast the trade that
will actually be placed. The model and the book then want the same thing.

Model selection happens outside the fit, in combinatorial purged
cross-validation, and is scored on the long-short spread a dollar-neutral book
would have earned -- not on R², which in this signal-to-noise regime is
negative for models that make money.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from arena.core.snapshot import Snapshot
from arena.features import build as build_panel
from arena.features import feature_columns
from arena.judge.cpcv import cpcv_splits, leakage_report
from arena.judge.metrics import pbo_cscv
from arena.labeling.barriers import DEFAULT_LADDER, RoiLadder, label_panel
from arena.labeling.weights import sample_weights, trend_tstats
from arena.models import rff

log = logging.getLogger(__name__)
WINSOR = 0.01  # crypto tails would otherwise decide the fit on their own


@dataclass
class Dataset:
    features: pd.DataFrame  # one row per (event, symbol)
    events: pd.DataFrame  # ts, symbol, exit_ts, label, net_return
    target: pd.Series
    weights: pd.Series
    columns: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.features)

    @property
    def matrix(self) -> np.ndarray:
        return self.features[self.columns].to_numpy(dtype=float)


def _winsorise(y: pd.Series, q: float = WINSOR) -> pd.Series:
    if y.empty:
        return y
    lo, hi = y.quantile(q), y.quantile(1.0 - q)
    return y.clip(lower=lo, upper=hi)


def build_dataset(
    candles: pd.DataFrame,
    funding: pd.DataFrame | None,
    symbols_at: dict[pd.Timestamp, list[str]],
    ladder: RoiLadder = DEFAULT_LADDER,
    stop: float = 0.08,
    costs: pd.Series | float = 0.0,
    bar_hours: int = 1,
    use_trend_weights: bool = True,
) -> Dataset:
    """Panel features and barrier labels for every rebalance date in ``symbols_at``.

    ``symbols_at`` is the point-in-time universe: the symbols tradable **on that
    date**, not the ones tradable today. Feeding today's survivors here is the
    survivorship bias the whole design exists to avoid, and no amount of
    cross-validation downstream can undo it.
    """
    wide = candles.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last").sort_index()
    returns = wide.pct_change()
    benchmark = returns.mean(axis=1)
    excess = returns.sub(benchmark, axis=0)

    feature_rows: list[pd.DataFrame] = []
    label_rows: list[pd.DataFrame] = []
    for ts in sorted(symbols_at):
        names = [s for s in symbols_at[ts] if s in wide.columns]
        if len(names) < 4:
            continue
        window = candles[candles["symbol"].isin(names) & (candles["ts"] <= ts)]
        if window.empty:
            continue
        fund = funding[funding["symbol"].isin(names) & (funding["ts"] <= ts)] if funding is not None else None
        snap = Snapshot.from_long(ts, names, window, fund, bar_hours=bar_hours)
        panel = build_panel(snap)
        if panel.empty:
            continue
        labels = label_panel(excess[names], pd.DatetimeIndex([ts]), ladder, stop, costs)
        labels = labels[labels["barrier"] != "open"]
        if labels.empty:
            continue
        panel = panel.loc[panel.index.intersection(labels["symbol"])]
        if panel.empty:
            continue
        panel = panel.assign(ts=ts, symbol=panel.index)
        feature_rows.append(panel.reset_index(drop=True))
        label_rows.append(labels[labels["symbol"].isin(panel["symbol"])])

    if not feature_rows:
        return Dataset(
            pd.DataFrame(),
            pd.DataFrame(columns=["ts", "symbol", "exit_ts"]),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    features = pd.concat(feature_rows, ignore_index=True)
    events = pd.concat(label_rows, ignore_index=True)
    merged = features.merge(events, on=["ts", "symbol"], how="inner", suffixes=("", "_label"))
    if merged.empty:
        return Dataset(
            pd.DataFrame(),
            pd.DataFrame(columns=["ts", "symbol", "exit_ts"]),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    # Winsorise first, demean second. The other order leaves a residual mean per
    # date, which is a market-timing component this family is not allowed to
    # trade -- a dollar-neutral book cannot express it, so learning it is waste
    # at best and a hidden beta bet at worst.
    tamed = _winsorise(merged["net_return"])
    target = tamed.groupby(merged["ts"]).transform(lambda s: s - s.mean())

    event_table = merged[["ts", "symbol", "exit_ts", "label", "net_return", "bars_held", "barrier"]].copy()
    trend = None
    if use_trend_weights:
        market = (1.0 + benchmark.fillna(0.0)).cumprod()
        trend = trend_tstats(market, pd.DatetimeIndex(sorted(set(event_table["ts"]))))
    weights = sample_weights(event_table, pd.DatetimeIndex(wide.index), trend_t=trend)

    columns = [c for c in feature_columns(merged) if c not in {"label", "net_return", "bars_held", "target_hit"}]
    return Dataset(
        features=merged,
        events=event_table,
        target=target.rename("target"),
        weights=weights,
        columns=columns,
    )


# --------------------------------------------------------------------------- selection


def long_short_spread(predictions: np.ndarray, actual: np.ndarray, dates: np.ndarray, k: int = 8) -> float:
    """Mean per-date spread between the top-k and bottom-k predicted names.

    This is the metric because it is the strategy: a dollar-neutral book holds
    the extremes and collects the difference. R² would reward a model that is
    accurate about the middle of the cross-section, which is never traded.
    """
    frame = pd.DataFrame({"p": predictions, "y": actual, "d": dates})
    spreads = []
    for _, group in frame.groupby("d"):
        if len(group) < 2 * 2:
            continue
        width = max(1, min(k, len(group) // 2))
        ranked = group.sort_values("p", ascending=False)
        spreads.append(float(ranked["y"].head(width).mean() - ranked["y"].tail(width).mean()))
    return float(np.mean(spreads)) if spreads else float("nan")


@dataclass
class Selection:
    lam: float
    n_features: int
    gamma: float
    score: float
    by_lambda: dict[float, float]
    pbo: dict
    leakage: dict
    n_splits: int


def select(
    data: Dataset,
    n_features: int = 4_000,
    gamma: float = 0.02,
    lambdas=rff.DEFAULT_LAMBDAS,
    n_groups: int = 6,
    n_test: int = 2,
    embargo_frac: float = 0.02,
    k: int = 8,
    seed: int = 0,
) -> Selection:
    """Choose the shrinkage out of sample, under purging and embargo.

    Also returns the probability of backtest overfitting over the shrinkage
    path: if picking the best λ does not generalise, the winner is the search's
    artefact rather than the data's.
    """
    splits = cpcv_splits(data.events, n_groups=n_groups, n_test=n_test, embargo_frac=embargo_frac)
    if not splits:
        raise ValueError("not enough events to cross-validate")
    x, y = data.matrix, data.target.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(data.events["ts"]).to_numpy()
    w = data.weights.to_numpy(dtype=float)

    per_lambda: dict[float, list[float]] = {float(v): [] for v in lambdas}
    paths: dict[float, list[np.ndarray]] = {float(v): [] for v in lambdas}
    for split in splits:
        models = rff.fit(x[split.train], y[split.train], data.columns, n_features, gamma, lambdas, w[split.train], seed)
        for lam, model in models.items():
            predicted = model.predict(x[split.test])
            per_lambda[lam].append(long_short_spread(predicted, y[split.test], dates[split.test], k))
            paths[lam].append(predicted * np.sign(y[split.test]).astype(float))

    means = {lam: float(np.nanmean(v)) if v else float("nan") for lam, v in per_lambda.items()}
    best = max(means, key=lambda lam: means[lam] if np.isfinite(means[lam]) else -np.inf)
    width = min(len(p) for p in paths.values() if p) if any(paths.values()) else 0
    matrix = np.column_stack([np.concatenate(paths[lam])[: width and None] for lam in lambdas]) if width else None
    return Selection(
        lam=float(best),
        n_features=int(n_features),
        gamma=float(gamma),
        score=means[best],
        by_lambda=means,
        pbo=pbo_cscv(matrix) if matrix is not None and matrix.shape[1] >= 2 else {"pbo": 1.0, "n_configs": 0},
        leakage=leakage_report(data.events, splits),
        n_splits=len(splits),
    )


def train(data: Dataset, selection: Selection, seed: int = 0) -> rff.RffRidge:
    """Refit on everything at the chosen shrinkage, carrying the selection evidence."""
    model = rff.fit(
        data.matrix,
        data.target.to_numpy(dtype=float),
        data.columns,
        selection.n_features,
        selection.gamma,
        weights=data.weights.to_numpy(dtype=float),
        seed=seed,
        lam=selection.lam,
    )[selection.lam]
    return rff.RffRidge(
        columns=model.columns,
        standardiser=model.standardiser,
        features=model.features,
        beta=model.beta,
        lam=model.lam,
        seed=model.seed,
        metrics={
            **model.metrics,
            "cv_spread": selection.score,
            "cv_by_lambda": selection.by_lambda,
            "cv_splits": selection.n_splits,
            "pbo": selection.pbo.get("pbo"),
            "max_overlap_ns": selection.leakage.get("max_overlap_ns"),
        },
    )


# --------------------------------------------------------------------------- the search over the search


@dataclass
class Search:
    best: Selection
    grid: list[Selection]
    pbo: dict

    @property
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"n_features": s.n_features, "gamma": s.gamma, "lam": s.lam, "cv_spread": s.score} for s in self.grid]
        ).sort_values("cv_spread", ascending=False)


def search(
    data: Dataset,
    widths=(500, 2_000),
    gammas=(0.005, 0.02, 0.08),
    lambdas=rff.DEFAULT_LAMBDAS,
    n_groups: int = 6,
    n_test: int = 2,
    embargo_frac: float = 0.02,
    k: int = 8,
    seed: int = 0,
) -> Search:
    """Sweep width and bandwidth as well as shrinkage, and score the *whole* sweep.

    The first round swept only λ and read a PBO of 0.00 over it. That was a weak
    test: nine shrinkage values yield nine nearly identical models, so the
    in-sample winner is almost bound to rank well out of sample. Bandwidth γ was
    never swept at all, and it decides whether the random features are nearly
    linear or nearly noise. This sweeps both and computes PBO across every
    (width, γ, λ) configuration, which is the question PBO exists to answer:
    does picking the best of *these* generalise?
    """
    splits = cpcv_splits(data.events, n_groups=n_groups, n_test=n_test, embargo_frac=embargo_frac)
    if not splits:
        raise ValueError("not enough events to cross-validate")
    x, y = data.matrix, data.target.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(data.events["ts"]).to_numpy()
    w = data.weights.to_numpy(dtype=float)

    grid: list[Selection] = []
    paths: dict[tuple, list[np.ndarray]] = {}
    for width in widths:
        for gamma in gammas:
            per_lambda: dict[float, list[float]] = {float(v): [] for v in lambdas}
            for split in splits:
                models = rff.fit(
                    x[split.train], y[split.train], data.columns, width, gamma, lambdas, w[split.train], seed
                )
                for lam, model in models.items():
                    predicted = model.predict(x[split.test])
                    per_lambda[lam].append(long_short_spread(predicted, y[split.test], dates[split.test], k))
                    paths.setdefault((width, gamma, lam), []).append(predicted * np.sign(y[split.test]))
            means = {lam: float(np.nanmean(v)) for lam, v in per_lambda.items()}
            best_lam = max(means, key=lambda lam: means[lam] if np.isfinite(means[lam]) else -np.inf)
            grid.append(
                Selection(
                    lam=float(best_lam),
                    n_features=int(width),
                    gamma=float(gamma),
                    score=means[best_lam],
                    by_lambda=means,
                    pbo={},
                    leakage=leakage_report(data.events, splits),
                    n_splits=len(splits),
                )
            )
    keys = sorted(paths)
    width_ = min(len(np.concatenate(paths[key])) for key in keys)
    matrix = np.column_stack([np.concatenate(paths[key])[:width_] for key in keys])
    pbo = pbo_cscv(matrix) if matrix.shape[1] >= 2 else {"pbo": 1.0, "n_configs": matrix.shape[1]}
    pbo.pop("logits", None)
    best = max(grid, key=lambda s: s.score if np.isfinite(s.score) else -np.inf)
    best.pbo = pbo
    return Search(best=best, grid=grid, pbo=pbo)

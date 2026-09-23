"""Random Fourier features and ridge: the "virtue of complexity" machinery.

[Kelly, Malamud and Zhou (*Journal of Finance* 79(1), 2024)](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13298)
show that in the regime where parameters vastly outnumber observations, a
ridge fit on random nonlinear features can beat the parsimonious model it
contains -- benign overfitting, with the shrinkage doing the work. Random
Fourier features (Rahimi & Recht) approximate an RBF kernel: draw `P/2` random
projections and pair each with its cosine and sine, and a linear fit on the
result is a nonlinear fit on the original.

Two deliberate departures from the paper, both because of what is being
modelled here:

*The panel, not the time series.* KMZ time the market, where `T` is a few
hundred months of one series. This arena's observations are symbol-weeks, so
`T` is the cross-section times the history. That is where the data actually is.

*Costs and a standing objection.* The paper's results exclude transaction costs,
and [Nagel (2025)](https://voices.uchicago.edu/stefannagel/files/2025/07/Complexity_2.pdf)
argues the gain largely reduces to volatility-timed momentum. Both are why
``xs_sparse`` exists as a control and why nothing here is believed without it.

The estimator is deliberately boring: standardise, project, ridge over a grid,
pick the shrinkage out of sample. Everything needed to reproduce a prediction
serialises to one JSON string, because the arena replays competitors from
stored artefacts and a model it cannot replay is a model it cannot judge.
"""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass

import numpy as np

ARTIFACT_VERSION = 1
DEFAULT_LAMBDAS: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)


def _pack(array: np.ndarray) -> str:
    """Compact, lossless float32 encoding: a 20 000-feature projection is megabytes as JSON."""
    a = np.ascontiguousarray(array, dtype=np.float32)
    return base64.b64encode(zlib.compress(a.tobytes(), 6)).decode()


def _unpack(blob: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(blob.encode()))
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float64)


@dataclass(frozen=True)
class Standardiser:
    """Column means and scales, **fitted on training rows only**.

    Fitting these on the whole sample is the quietest leak available: the test
    rows then help decide what "average" means.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Standardiser:
        mean = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0, ddof=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return cls(mean=np.nan_to_num(mean), scale=scale)

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = (np.asarray(x, dtype=float) - self.mean) / self.scale
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)


@dataclass(frozen=True)
class RandomFeatures:
    """``z(x) = sqrt(2/P) · [cos(Wx), sin(Wx)]`` with ``W ~ N(0, 2γ)``.

    The cosine/sine pair is the lower-variance form of the RBF approximation and
    needs no random phase, which also makes the artefact smaller.
    """

    weights: np.ndarray  # (d, P/2)
    gamma: float

    @classmethod
    def draw(cls, n_inputs: int, n_features: int, gamma: float, seed: int = 0) -> RandomFeatures:
        half = max(1, int(n_features) // 2)
        rng = np.random.default_rng(seed)
        return cls(weights=rng.normal(0.0, np.sqrt(2.0 * gamma), size=(n_inputs, half)), gamma=float(gamma))

    @property
    def n_features(self) -> int:
        return 2 * self.weights.shape[1]

    def transform(self, x: np.ndarray) -> np.ndarray:
        projected = np.asarray(x, dtype=float) @ self.weights
        return np.sqrt(2.0 / self.n_features) * np.hstack([np.cos(projected), np.sin(projected)])


def ridge_path(z: np.ndarray, y: np.ndarray, lambdas, weights: np.ndarray | None = None) -> dict[float, np.ndarray]:
    """Ridge coefficients for every shrinkage in ``lambdas``, from one SVD.

    ``beta(λ) = V diag(s/(s²+λ)) Uᵀ y``. Sweeping a grid is then nearly free,
    which matters because in this regime the grid *is* the model: the same
    features at λ=1e-4 and λ=1e4 are two different estimators.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if weights is not None:
        w = np.sqrt(np.clip(np.asarray(weights, dtype=float).ravel(), 0.0, None))
        z, y = z * w[:, None], y * w
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    uty = u.T @ y
    out: dict[float, np.ndarray] = {}
    for lam in lambdas:
        out[float(lam)] = vt.T @ (s / (s**2 + float(lam)) * uty)
    return out


@dataclass(frozen=True)
class RffRidge:
    """A fitted model: standardiser, projection, coefficients, and how to replay it."""

    columns: tuple[str, ...]
    standardiser: Standardiser
    features: RandomFeatures
    beta: np.ndarray
    lam: float
    seed: int
    metrics: dict

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.features.transform(self.standardiser.apply(x)) @ self.beta

    def predict_frame(self, frame) -> np.ndarray:
        """Predict from a DataFrame, reindexed to the training columns.

        A column the model never saw is ignored; one it expects and cannot find
        is zero *after* standardisation, i.e. the training mean, which is the
        only neutral answer available.
        """
        import pandas as pd

        aligned = pd.DataFrame(frame).reindex(columns=list(self.columns))
        values = np.array(aligned.to_numpy(dtype=float), copy=True)
        missing = np.isnan(values).all(axis=0)
        values[:, missing] = self.standardiser.mean[missing]
        return self.predict(values)

    # ------------------------------------------------------------------ artefact

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": ARTIFACT_VERSION,
                "columns": list(self.columns),
                "mean": _pack(self.standardiser.mean),
                "scale": _pack(self.standardiser.scale),
                "weights": _pack(self.features.weights),
                "weights_shape": list(self.features.weights.shape),
                "gamma": self.features.gamma,
                "beta": _pack(self.beta),
                "lam": self.lam,
                "seed": self.seed,
                "metrics": self.metrics,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, blob: str) -> RffRidge:
        d = json.loads(blob)
        if int(d.get("version", 0)) != ARTIFACT_VERSION:
            raise ValueError(f"unsupported rff artefact version {d.get('version')}")
        shape = tuple(d["weights_shape"])
        n_inputs = len(d["columns"])
        return cls(
            columns=tuple(d["columns"]),
            standardiser=Standardiser(_unpack(d["mean"], (n_inputs,)), _unpack(d["scale"], (n_inputs,))),
            features=RandomFeatures(_unpack(d["weights"], shape), float(d["gamma"])),
            beta=_unpack(d["beta"], (2 * shape[1],)),
            lam=float(d["lam"]),
            seed=int(d["seed"]),
            metrics=dict(d.get("metrics") or {}),
        )


def fit(
    x: np.ndarray,
    y: np.ndarray,
    columns,
    n_features: int = 4_000,
    gamma: float = 0.02,
    lambdas=DEFAULT_LAMBDAS,
    weights: np.ndarray | None = None,
    seed: int = 0,
    lam: float | None = None,
) -> dict[float, RffRidge]:
    """Fit one model per shrinkage value. Choosing between them is the caller's job.

    Returning the whole path rather than a winner keeps model selection where it
    belongs -- in the cross-validation that can purge and embargo -- instead of
    hiding it inside a fit that has already seen everything.
    """
    x = np.asarray(x, dtype=float)
    standardiser = Standardiser.fit(x)
    rf = RandomFeatures.draw(x.shape[1], n_features, gamma, seed)
    z = rf.transform(standardiser.apply(x))
    grid = [lam] if lam is not None else list(lambdas)
    betas = ridge_path(z, y, grid, weights)
    return {
        value: RffRidge(
            columns=tuple(columns),
            standardiser=standardiser,
            features=rf,
            beta=beta,
            lam=value,
            seed=seed,
            metrics={"n_train": int(x.shape[0]), "n_features": rf.n_features, "gamma": gamma},
        )
        for value, beta in betas.items()
    }

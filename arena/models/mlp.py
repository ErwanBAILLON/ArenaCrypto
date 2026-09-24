"""A small multilayer perceptron, the way Gu, Kelly and Xiu found it should be built.

[Gu, Kelly and Xiu (*RFS* 2020)](https://academic.oup.com/rfs/article/33/5/2223/5758276)
ran the definitive horse race of machine-learning return predictors on thirty
thousand stocks and sixty years, and their best neural network had one hidden
layer of 32 units; deeper did worse. With three orders of magnitude less data
than they had, the only defensible trial here is a *smaller* one, with the
linear model as the control: ``hidden=0`` is plain ridge on the same inputs,
so the comparison measures the hidden layer and nothing else.

Deliberately plain: numpy, ReLU, Adam, L2, early stopping on a validation
split the caller provides (so it can be purged), ensembling over seeds because
a single small network's fit is mostly its initialisation. No framework
dependency -- the arena replays models from stored artefacts, and an artefact
that needs a GPU runtime is one the tick cannot load.
"""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass, field

import numpy as np

from arena.models.rff import Standardiser

ARTIFACT_VERSION = 1


def _pack(a: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(a, dtype=np.float32).tobytes(), 6)).decode()


def _unpack(blob: str, shape) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(blob)), dtype=np.float32).reshape(shape).astype(float)


@dataclass
class Net:
    """One network: weights for ``hidden`` ReLU units (or none) and a linear head."""

    w1: np.ndarray | None
    b1: np.ndarray | None
    w2: np.ndarray
    b2: float

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x if self.w1 is None else np.maximum(x @ self.w1 + self.b1, 0.0)
        return h @ self.w2 + self.b2


def _init(n_in: int, hidden: int, rng: np.random.Generator) -> Net:
    if hidden <= 0:
        return Net(None, None, rng.normal(0, 1.0 / np.sqrt(n_in), n_in), 0.0)
    return Net(
        rng.normal(0, np.sqrt(2.0 / n_in), (n_in, hidden)),
        np.zeros(hidden),
        rng.normal(0, 1.0 / np.sqrt(hidden), hidden),
        0.0,
    )


def _train_one(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    hidden: int,
    l2: float,
    lr: float,
    epochs: int,
    patience: int,
    batch: int,
    seed: int,
) -> tuple[Net, dict]:
    rng = np.random.default_rng(seed)
    net = _init(x.shape[1], hidden, rng)
    params = [p for p in (net.w1, net.b1, net.w2) if p is not None]
    m_ = [np.zeros_like(p) for p in params]
    v_ = [np.zeros_like(p) for p in params]
    mb2, vb2 = 0.0, 0.0
    best, best_state, bad, step = np.inf, None, 0, 0
    n = x.shape[0]

    def loss(net_: Net, xx, yy, ww):
        pred = net_.forward(xx)
        return float(np.mean(ww * (pred - yy) ** 2))

    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            xb, yb, wb = x[idx], y[idx], w[idx]
            step += 1
            # forward
            if net.w1 is not None:
                z = xb @ net.w1 + net.b1
                h = np.maximum(z, 0.0)
            else:
                h = xb
            pred = h @ net.w2 + net.b2
            g = 2.0 * wb * (pred - yb) / len(idx)  # d loss / d pred
            gw2 = h.T @ g + l2 * net.w2
            gb2 = float(g.sum())
            grads = []
            if net.w1 is not None:
                gh = np.outer(g, net.w2) * (z > 0)
                gw1 = xb.T @ gh + l2 * net.w1
                gb1 = gh.sum(axis=0)
                grads = [gw1, gb1, gw2]
            else:
                grads = [gw2]
            # adam
            b1, b2c, eps = 0.9, 0.999, 1e-8
            for i, (p, gp) in enumerate(zip(params, grads, strict=True)):
                m_[i] = b1 * m_[i] + (1 - b1) * gp
                v_[i] = b2c * v_[i] + (1 - b2c) * gp * gp
                p -= lr * (m_[i] / (1 - b1**step)) / (np.sqrt(v_[i] / (1 - b2c**step)) + eps)
            mb2 = b1 * mb2 + (1 - b1) * gb2
            vb2 = b2c * vb2 + (1 - b2c) * gb2 * gb2
            net.b2 -= lr * (mb2 / (1 - b1**step)) / (np.sqrt(vb2 / (1 - b2c**step)) + eps)
        val = loss(net, xv, yv, np.ones(len(yv)))
        if val < best - 1e-9:
            best, bad = val, 0
            best_state = Net(
                None if net.w1 is None else net.w1.copy(),
                None if net.b1 is None else net.b1.copy(),
                net.w2.copy(),
                net.b2,
            )
        else:
            bad += 1
            if bad >= patience:
                break
    return best_state or net, {"val_loss": best, "epochs": epoch + 1}


@dataclass(frozen=True)
class MlpModel:
    columns: tuple[str, ...]
    standardiser: Standardiser
    nets: list[Net]
    hidden: int
    metrics: dict = field(default_factory=dict)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.standardiser.apply(x)
        return np.mean([n.forward(z) for n in self.nets], axis=0)

    def predict_frame(self, frame) -> np.ndarray:
        import pandas as pd

        aligned = pd.DataFrame(frame).reindex(columns=list(self.columns))
        values = np.array(aligned.to_numpy(dtype=float), copy=True)
        missing = np.isnan(values).all(axis=0)
        values[:, missing] = self.standardiser.mean[missing]
        return self.predict(values)

    def to_json(self) -> str:
        nets = []
        for n in self.nets:
            nets.append(
                {
                    "w1": None if n.w1 is None else _pack(n.w1),
                    "b1": None if n.b1 is None else _pack(n.b1),
                    "w2": _pack(n.w2),
                    "b2": float(n.b2),
                }
            )
        return json.dumps(
            {
                "version": ARTIFACT_VERSION,
                "kind": "mlp",
                "columns": list(self.columns),
                "mean": _pack(self.standardiser.mean),
                "scale": _pack(self.standardiser.scale),
                "hidden": self.hidden,
                "nets": nets,
                "metrics": self.metrics,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, blob: str) -> MlpModel:
        d = json.loads(blob)
        if int(d.get("version", 0)) != ARTIFACT_VERSION or d.get("kind") != "mlp":
            raise ValueError("unsupported mlp artefact")
        n_in, hidden = len(d["columns"]), int(d["hidden"])
        nets = []
        for n in d["nets"]:
            nets.append(
                Net(
                    None if n["w1"] is None else _unpack(n["w1"], (n_in, hidden)),
                    None if n["b1"] is None else _unpack(n["b1"], (hidden,)),
                    _unpack(n["w2"], (hidden if hidden > 0 else n_in,)),
                    float(n["b2"]),
                )
            )
        return cls(
            columns=tuple(d["columns"]),
            standardiser=Standardiser(_unpack(d["mean"], (n_in,)), _unpack(d["scale"], (n_in,))),
            nets=nets,
            hidden=hidden,
            metrics=dict(d.get("metrics") or {}),
        )


def fit(
    x: np.ndarray,
    y: np.ndarray,
    columns,
    val_mask: np.ndarray,
    hidden: int = 16,
    l2: float = 1e-3,
    lr: float = 1e-3,
    epochs: int = 200,
    patience: int = 15,
    batch: int = 256,
    n_seeds: int = 5,
    weights: np.ndarray | None = None,
    seed: int = 0,
) -> MlpModel:
    """Fit an ensemble of small networks; ``val_mask`` marks the rows used only for early stopping.

    The caller decides which rows are validation, because the caller is the one
    who knows which labels overlap which: early stopping on rows whose lifetime
    overlaps the training rows is the same leak purged cross-validation exists
    to remove, one level down.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float).ravel()
    val_mask = np.asarray(val_mask, dtype=bool)
    std = Standardiser.fit(x[~val_mask])
    z = std.apply(x)
    nets, logs = [], []
    for s in range(n_seeds):
        net, log = _train_one(
            z[~val_mask],
            y[~val_mask],
            w[~val_mask],
            z[val_mask],
            y[val_mask],
            hidden,
            l2,
            lr,
            epochs,
            patience,
            batch,
            seed + s,
        )
        nets.append(net)
        logs.append(log)
    return MlpModel(
        columns=tuple(columns),
        standardiser=std,
        nets=nets,
        hidden=int(hidden),
        metrics={
            "hidden": int(hidden),
            "l2": l2,
            "n_train": int((~val_mask).sum()),
            "n_val": int(val_mask.sum()),
            "val_loss": float(np.mean([log["val_loss"] for log in logs])),
            "epochs": float(np.mean([log["epochs"] for log in logs])),
        },
    )

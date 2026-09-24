"""A small network against its own linear control."""

from __future__ import annotations

import numpy as np
import pytest

from arena.models import mlp

D = 12


def _data(n=1500, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, D))
    y = np.sin(1.5 * x[:, 0]) * x[:, 1] + 0.3 * x[:, 2] ** 2 + rng.normal(0, noise, n)
    return x, y, [f"f{i}" for i in range(D)]


def _ic(model, x, y):
    return float(np.corrcoef(model.predict(x), y)[0, 1])


def _split(n, frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, int(n * frac), replace=False)] = True
    return mask


class TestControl:
    def test_hidden_zero_is_a_linear_model(self):
        x, y, cols = _data()
        model = mlp.fit(x[:1000], y[:1000], cols, _split(1000), hidden=0, epochs=60, n_seeds=2)
        assert model.hidden == 0 and all(n.w1 is None for n in model.nets)
        # linear cannot see sin(x0)*x1 or x2^2
        assert abs(_ic(model, x[1000:], y[1000:])) < 0.25

    def test_a_small_hidden_layer_learns_what_the_linear_one_cannot(self):
        x, y, cols = _data()
        linear = mlp.fit(x[:1000], y[:1000], cols, _split(1000), hidden=0, epochs=60, n_seeds=2)
        small = mlp.fit(x[:1000], y[:1000], cols, _split(1000), hidden=16, epochs=200, n_seeds=3, lr=3e-3)
        assert _ic(small, x[1000:], y[1000:]) > _ic(linear, x[1000:], y[1000:]) + 0.2

    def test_early_stopping_uses_only_the_validation_rows(self):
        x, y, cols = _data(n=600)
        mask = _split(600)
        model = mlp.fit(x, y, cols, mask, hidden=8, epochs=50, n_seeds=1)
        assert model.metrics["n_train"] == int((~mask).sum()) and model.metrics["n_val"] == int(mask.sum())
        assert model.metrics["epochs"] <= 50

    def test_sample_weights_change_the_fit(self):
        x, y, cols = _data(n=600)
        mask = _split(600)
        flat = mlp.fit(x, y, cols, mask, hidden=8, epochs=30, n_seeds=1)
        skew = mlp.fit(x, y, cols, mask, hidden=8, epochs=30, n_seeds=1, weights=np.linspace(0.1, 2.0, 600))
        assert not np.allclose(flat.predict(x[:50]), skew.predict(x[:50]))


class TestArtifact:
    def test_round_trip(self):
        x, y, cols = _data(n=600)
        model = mlp.fit(x, y, cols, _split(600), hidden=8, epochs=30, n_seeds=2)
        back = mlp.MlpModel.from_json(model.to_json())
        assert np.allclose(model.predict(x[:100]), back.predict(x[:100]), atol=1e-4)

    def test_linear_round_trip(self):
        x, y, cols = _data(n=600)
        model = mlp.fit(x, y, cols, _split(600), hidden=0, epochs=30, n_seeds=1)
        back = mlp.MlpModel.from_json(model.to_json())
        assert np.allclose(model.predict(x[:100]), back.predict(x[:100]), atol=1e-4)

    def test_frame_prediction_aligns_columns(self):
        import pandas as pd

        x, y, cols = _data(n=600)
        model = mlp.fit(x, y, cols, _split(600), hidden=8, epochs=20, n_seeds=1)
        frame = pd.DataFrame(x[:50], columns=cols)
        assert np.allclose(model.predict_frame(frame), model.predict_frame(frame[list(reversed(cols))]))

    def test_wrong_artefact_is_refused(self):
        with pytest.raises(ValueError):
            mlp.MlpModel.from_json('{"version": 1, "kind": "rff"}')

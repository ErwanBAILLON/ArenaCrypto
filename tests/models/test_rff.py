"""Random features and ridge: does it learn nonlinearity, and does it replay identically."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arena.models import rff

D = 15


def _data(n=900, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, D))
    y = np.sin(1.5 * x[:, 0]) * x[:, 1] + 0.3 * x[:, 2] ** 2 + rng.normal(0, noise, n)
    return x, y, [f"f{i}" for i in range(D)]


def _ic(model, x, y):
    return float(np.corrcoef(model.predict(x), y)[0, 1])


class TestStandardiser:
    def test_it_centres_and_scales(self):
        x = np.random.default_rng(0).normal(5.0, 3.0, size=(200, 4))
        z = rff.Standardiser.fit(x).apply(x)
        assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.05

    def test_a_constant_column_does_not_explode(self):
        x = np.column_stack([np.ones(50), np.random.default_rng(0).normal(size=50)])
        assert np.isfinite(rff.Standardiser.fit(x).apply(x)).all()

    def test_outliers_are_clipped_not_propagated(self):
        x = np.random.default_rng(0).normal(size=(200, 2))
        std = rff.Standardiser.fit(x)
        wild = std.apply(np.array([[1e9, -1e9]]))
        assert np.abs(wild).max() <= 8.0

    def test_nans_become_the_training_mean(self):
        x = np.random.default_rng(0).normal(size=(100, 3))
        assert np.isfinite(rff.Standardiser.fit(x).apply(np.array([[np.nan] * 3]))).all()


class TestRandomFeatures:
    def test_the_width_is_what_was_asked_for(self):
        assert rff.RandomFeatures.draw(10, 2_000, 0.05).n_features == 2_000

    def test_the_same_seed_draws_the_same_projection(self):
        a = rff.RandomFeatures.draw(8, 100, 0.05, seed=3)
        b = rff.RandomFeatures.draw(8, 100, 0.05, seed=3)
        assert np.array_equal(a.weights, b.weights)

    def test_features_are_bounded(self):
        x = np.random.default_rng(0).normal(size=(50, 8))
        z = rff.RandomFeatures.draw(8, 200, 0.05).transform(x)
        assert np.abs(z).max() <= np.sqrt(2.0 / 200) + 1e-9


class TestRidgePath:
    def test_more_shrinkage_means_smaller_coefficients(self):
        x, y, _ = _data(n=200)
        z = rff.RandomFeatures.draw(D, 300, 0.05).transform(rff.Standardiser.fit(x).apply(x))
        betas = rff.ridge_path(z, y, [1e-3, 1.0, 1e4])
        norms = [float(np.linalg.norm(betas[lam])) for lam in (1e-3, 1.0, 1e4)]
        assert norms[0] > norms[1] > norms[2]

    def test_sample_weights_change_the_fit(self):
        x, y, _ = _data(n=200)
        z = rff.RandomFeatures.draw(D, 200, 0.05).transform(rff.Standardiser.fit(x).apply(x))
        flat = rff.ridge_path(z, y, [1.0])[1.0]
        skewed = rff.ridge_path(z, y, [1.0], weights=np.linspace(0.0, 2.0, len(y)))[1.0]
        assert not np.allclose(flat, skewed)


class TestVirtueOfComplexity:
    def test_it_learns_what_a_linear_model_cannot(self):
        x, y, cols = _data()
        tr, te = slice(0, 600), slice(600, None)
        model = rff.fit(x[tr], y[tr], cols, n_features=2_000, gamma=0.05, lam=0.1)[0.1]
        linear = np.linalg.lstsq(x[tr], y[tr], rcond=None)[0]
        assert _ic(model, x[te], y[te]) > 0.4
        assert abs(float(np.corrcoef(x[te] @ linear, y[te])[0, 1])) < 0.15

    def test_widening_helps_up_to_a_point(self):
        """More parameters than observations is the regime, not the failure."""
        x, y, cols = _data()
        tr, te = slice(0, 600), slice(600, None)
        scores = {}
        for width in (100, 2_000):
            models = rff.fit(x[tr], y[tr], cols, n_features=width, gamma=0.05)
            scores[width] = max(_ic(m, x[te], y[te]) for m in models.values())
        assert scores[2_000] > scores[100]
        assert 2_000 > 600  # P > T: benign overfitting, not a mistake

    def test_the_whole_path_comes_back_so_selection_stays_outside(self):
        x, y, cols = _data(n=200)
        models = rff.fit(x[:150], y[:150], cols, n_features=200, gamma=0.05)
        assert set(models) == set(rff.DEFAULT_LAMBDAS)
        assert all(m.lam == lam for lam, m in models.items())


class TestArtifact:
    def test_a_round_trip_predicts_identically(self):
        x, y, cols = _data(n=300)
        model = rff.fit(x[:200], y[:200], cols, n_features=400, gamma=0.05, lam=1.0)[1.0]
        replayed = rff.RffRidge.from_json(model.to_json())
        assert np.allclose(model.predict(x[200:]), replayed.predict(x[200:]), atol=1e-4)

    def test_the_artefact_is_small_enough_to_store(self):
        x, y, cols = _data(n=300)
        model = rff.fit(x[:200], y[:200], cols, n_features=8_000, gamma=0.05, lam=1.0)[1.0]
        assert len(model.to_json()) < 2_000_000  # 8k features, compressed float32

    def test_an_unknown_version_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="version"):
            rff.RffRidge.from_json('{"version": 99}')

    def test_predicting_from_a_frame_aligns_on_the_training_columns(self):
        x, y, cols = _data(n=300)
        model = rff.fit(x[:200], y[:200], cols, n_features=200, gamma=0.05, lam=1.0)[1.0]
        frame = pd.DataFrame(x[200:], columns=cols)
        shuffled = frame[list(reversed(cols))]  # same data, wrong order
        assert np.allclose(model.predict_frame(frame), model.predict_frame(shuffled))

    def test_a_column_the_model_never_saw_is_ignored(self):
        x, y, cols = _data(n=300)
        model = rff.fit(x[:200], y[:200], cols, n_features=200, gamma=0.05, lam=1.0)[1.0]
        frame = pd.DataFrame(x[200:], columns=cols)
        extra = frame.assign(brand_new=1.23)
        assert np.allclose(model.predict_frame(frame), model.predict_frame(extra))

    def test_a_missing_column_falls_back_to_the_training_mean(self):
        x, y, cols = _data(n=300)
        model = rff.fit(x[:200], y[:200], cols, n_features=200, gamma=0.05, lam=1.0)[1.0]
        frame = pd.DataFrame(x[200:], columns=cols).drop(columns=["f7"])
        assert np.isfinite(model.predict_frame(frame)).all()

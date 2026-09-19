import numpy as np
import pytest

from arena.competitors.meta_label import FEATURE_COLUMNS, MetaLabel, build_features, combine_bases
from arena.core.snapshot import Snapshot
from arena.core.types import Target
from tests.conftest import SYMBOLS, make_candles, make_funding


@pytest.fixture
def trending_snapshot():
    c = make_candles(bars=24 * 420, drift={"BTC": 0.0008, "ETH": 0.0006, "SOL": -0.0008}, seed=3)
    ts = c["ts"].max()
    return Snapshot.from_long(ts, SYMBOLS, c, make_funding(candles=c))


def test_combine_bases_averages_and_counts():
    d1 = {"BTC": Target(0.4), "ETH": Target(0.2)}
    d2 = {"BTC": Target(0.2), "SOL": Target(0.1, kind="carry")}
    out = combine_bases([d1, d2])
    assert out["BTC"] == (pytest.approx(0.3), 2)
    assert out["ETH"] == (pytest.approx(0.1), 1)
    assert "SOL" not in out  # carry legs are not meta-labelled


def test_pass_through_without_model(trending_snapshot):
    ml = MetaLabel()
    d = ml.decide(trending_snapshot)
    combined, _ = ml.base_signals(trending_snapshot)
    assert set(d) == set(combined)
    assert sum(abs(t.weight) for t in d.values()) <= 1.0 + 1e-9
    assert all(t.reason["mode"] == "pass_through" for t in d.values())


def test_features_have_all_columns(trending_snapshot):
    f = build_features(trending_snapshot, "BTC", 0.3, 2, "bull_calm")
    assert set(f) == set(FEATURE_COLUMNS)
    assert f["regime_bull_calm"] == 1.0 and f["regime_bear"] == 0.0
    assert 0 <= f["hour"] < 24


def test_model_scales_weights(trending_snapshot):
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(FEATURE_COLUMNS)))
    y = np.ones(200)  # a model that has only ever seen winners predicts ~1
    y[:5] = 0
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "min_data_in_leaf": 5}, lgb.Dataset(X, y), num_boost_round=5
    )
    ml = MetaLabel({"model_str": booster.model_to_string(), "threshold": 0.5})
    plain = MetaLabel().decide(trending_snapshot)
    scaled = ml.decide(trending_snapshot)
    assert set(scaled) <= set(plain)
    for sym, t in scaled.items():
        assert abs(t.weight) <= abs(plain[sym].weight) + 1e-9
        assert 0.5 < t.conviction <= 1.0


def test_high_threshold_flattens(trending_snapshot):
    import lightgbm as lgb

    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, len(FEATURE_COLUMNS)))
    y = rng.integers(0, 2, 200)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "min_data_in_leaf": 5}, lgb.Dataset(X, y), num_boost_round=3
    )
    ml = MetaLabel({"model_str": booster.model_to_string(), "threshold": 0.999})
    assert ml.decide(trending_snapshot) == {}

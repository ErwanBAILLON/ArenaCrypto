import pytest

import arena.competitors  # noqa: F401  populates REGISTRY
from arena.competitors.base import REGISTRY, build, cap_gross
from arena.competitors.carry import Carry
from arena.core.types import CompetitorSpec, Target

FAMILIES = {"null_cash", "null_random", "bench_btc_hold", "bench_carry_equal",
            "carry", "trend_ts", "xs_momentum", "regime"}


def test_registry_has_all_founding_families():
    assert FAMILIES <= set(REGISTRY)


def test_build_from_spec_merges_params_and_seed():
    spec = CompetitorSpec(id=None, name="c1", family="carry", version=1, params={"k": 2, "seed": 7})
    comp = build(spec)
    assert isinstance(comp, Carry)
    assert comp.params["k"] == 2
    assert comp.params["lookback_days"] == 14
    assert comp.seed == 7
    assert comp.warmup_bars() == 72


def test_cap_gross_scales_only_when_needed():
    d = {"BTC": Target(0.3), "ETH": Target(-0.2)}
    assert cap_gross(d) == d
    big = {"BTC": Target(1.0, conviction=0.9, kind="carry", reason={"x": 1}), "ETH": Target(-1.0)}
    capped = cap_gross(big)
    assert sum(abs(t.weight) for t in capped.values()) == pytest.approx(1.0)
    assert capped["BTC"].weight == pytest.approx(0.5) and capped["ETH"].weight == pytest.approx(-0.5)
    assert capped["BTC"].kind == "carry" and capped["BTC"].conviction == 0.9 and capped["BTC"].reason == {"x": 1}

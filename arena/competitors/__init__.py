"""Competing models. Importing this package populates ``base.REGISTRY``."""

from arena.competitors import (  # noqa: E402,F401
    carry,
    crowded_trend,
    funding_skew,
    meta_label,
    news,
    nulls,
    price_action,
    regime,
    trend_ts,
    xs_complex,
    xs_momentum,
    xs_sparse,
)
from arena.competitors.base import REGISTRY, Competitor, build, cap_gross, register

__all__ = ["REGISTRY", "Competitor", "build", "cap_gross", "register"]

"""Competing models. Importing this package populates ``base.REGISTRY``."""

from arena.competitors import (  # noqa: E402,F401
    carry,
    meta_label,
    news,
    nulls,
    price_action,
    regime,
    trend_ts,
    xs_momentum,
)
from arena.competitors.base import REGISTRY, Competitor, build, cap_gross, register

__all__ = ["REGISTRY", "Competitor", "build", "cap_gross", "register"]

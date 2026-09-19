"""Competing models. Importing this package populates ``base.REGISTRY``."""

from arena.competitors import carry, meta_label, news, nulls, regime, trend_ts, xs_momentum  # noqa: E402,F401
from arena.competitors.base import REGISTRY, Competitor, build, cap_gross, register

__all__ = ["REGISTRY", "Competitor", "build", "cap_gross", "register"]

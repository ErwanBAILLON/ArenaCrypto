"""Backwards-compatible alias for the shared indicators.

These helpers were never specific to competitors -- ``arena.features.panel``
needs them too, and importing them from here dragged the whole
``arena.competitors`` package into the import, which imports ``ladder``, which
imports ``arena.features.panel``. They now live in :mod:`arena.core.indicators`,
one layer below anything that decides anything.
"""

from arena.core.indicators import (  # noqa: F401
    MIN_FUNDING_DISPERSION,
    atr,
    ema,
    funding_zscores,
    last_atr,
    last_ema,
    last_realised_vol,
    pct_return,
    realised_vol,
    zscore,
)

__all__ = [
    "MIN_FUNDING_DISPERSION",
    "atr",
    "ema",
    "funding_zscores",
    "last_atr",
    "last_ema",
    "last_realised_vol",
    "pct_return",
    "realised_vol",
    "zscore",
]

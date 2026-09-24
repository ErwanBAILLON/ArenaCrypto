"""Cross-sectional predictor on random Fourier features: the complexity arm.

Same universe, same labels, same barriers, same costs and the same portfolio
construction as ``xs_sparse``. The **only** difference is the predictor: a ridge
over thousands of random nonlinear features instead of an unweighted average of
five ranks. That is deliberate, so the comparison measures complexity and
nothing else.

What the extra capacity is for is interactions a linear composite cannot state:
reversal paying when funding is extreme, momentum paying when the cross-section
is dispersed, both dying when the volatility regime shifts. Not "more
parameters" as a virtue in itself.

The model is trained offline and replayed here from a stored artefact, like
``meta_label``: a competitor must be a pure function of ``(params, Snapshot)``,
and training inside a decision would make every backtest unreproducible. With
no artefact the family stands flat rather than falling back on something else,
because a silent fallback is a different model wearing this one's name.
"""

from __future__ import annotations

import pandas as pd

from arena.competitors.base import register
from arena.competitors.ladder import LadderHoldingCompetitor
from arena.core.snapshot import Snapshot
from arena.core.types import Decision
from arena.features import build as build_panel
from arena.features import feature_columns
from arena.models.rff import RffRidge


@register
class XsComplex(LadderHoldingCompetitor):
    family = "xs_complex"
    rebalance_weekday = 0

    default_params = {
        "k": 8,
        "max_weight": 0.08,
        "target_vol": 0.20,
        "min_symbols": 10,
        "min_abs_score": 0.0,  # ignore predictions too small to pay for a trade
        "stop": 0.08,
        "roi_steps": None,
        "hedge": "picks",
        "vol_mode": "names",
        "max_gross": 1.0,
        "max_leverage": 1.0,
        "rebalance_every_weeks": 1,
        "model_str": None,  # the RffRidge artefact, injected by the tick from `models`
    }

    def __init__(self, params: dict | None = None, seed: int = 0, bar_hours: int = 1):
        super().__init__(params, seed, bar_hours)
        self._model: RffRidge | None = None
        blob = self.params.get("model_str")
        if blob:
            self._model = RffRidge.from_json(blob if isinstance(blob, str) else blob.decode())

    def warmup_bars(self) -> int:
        return 95 * self.bars_per_day

    @property
    def trained(self) -> bool:
        return self._model is not None

    def score(self, panel: pd.DataFrame) -> pd.Series:
        """Predicted excess outcome per symbol, demeaned across the cross-section.

        Demeaning is what makes the score a *relative* statement. A model that
        is uniformly optimistic one week would otherwise buy everything, which
        is a market call this family does not claim to make.
        """
        if self._model is None or panel.empty:
            return pd.Series(dtype=float)
        columns = [c for c in feature_columns(panel) if c in self._model.columns]
        if not columns:
            return pd.Series(dtype=float)
        raw = pd.Series(self._model.predict_frame(panel), index=panel.index, dtype=float)
        return (raw - raw.mean()).dropna()

    def select(self, snap: Snapshot) -> Decision:
        if self._model is None:
            return {}
        panel = build_panel(snap)
        if len(panel) < int(self.params["min_symbols"]):
            return {}
        scores = self.score(panel)
        threshold = float(self.params.get("min_abs_score") or 0.0)
        if threshold > 0:
            scores = scores[scores.abs() >= threshold]
        if scores.empty:
            return {}
        model = self._model
        return self.size_book(
            snap,
            panel,
            scores,
            lambda sym: {
                "predicted": round(float(scores[sym]), 5),
                "lambda": model.lam,
                "features": model.features.n_features,
            },
        )

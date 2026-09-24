"""A composite that decides, every rebalance, which signals have recently earned a vote.

The five-signal composite beat the two-signal one across ten quarters even
though three of its signals were negative in the bear year: signals change
sign with the regime, and a fixed set is a bet that the regime does not change.
The pre-registered two-signal rule lost money for exactly that reason.

This family keeps a wide candidate list and, at each rebalance, measures the
rank correlation between every candidate and the realised excess return over
the **trailing** ``lookback_weeks`` -- past data only, nothing from the period
being traded -- keeps the candidates whose t-statistic clears ``min_t``, and
weights them by that trailing IC. It is a regime model with roughly one free
parameter per signal, which is all a hundred and fifty weeks can afford, and
it is the only kind of nonlinearity the data has been willing to pay for.

The candidate list is wide on purpose: every column here is a hypothesis, and
the trailing-IC gate is what stops a hundred hypotheses from becoming a hundred
overfit signals. A candidate that never clears the bar simply never votes.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from arena.competitors.base import register
from arena.competitors.ladder import LadderHoldingCompetitor
from arena.core.snapshot import Snapshot
from arena.core.types import Decision
from arena.features import build as build_panel

# rank column -> prior sign (which way "high" should predict outperformance). The
# adaptive weights may flip a sign; the prior only orients the initial reading.
CANDIDATES: dict[str, float] = {
    "rank_vol_30d": -1.0,
    "rank_vol_7d": -1.0,
    "rank_idio_vol": -1.0,
    "rank_max_daily_ret": -1.0,
    "rank_max5_daily_ret": -1.0,
    "rank_skew_30d": -1.0,
    "rank_kurt_30d": -1.0,
    "rank_extreme_share_30d": -1.0,
    "rank_ret_7d": -1.0,
    "rank_ret_30d_skip_7d": 1.0,
    "rank_ret_90d": 1.0,
    "rank_donchian_position": 1.0,
    "rank_efficiency_ratio": 1.0,
    "rank_adx": 1.0,
    "rank_hurst_proxy": 1.0,
    "rank_funding_crowding_z": -1.0,
    "rank_funding_7d": -1.0,
    "rank_funding_slope": -1.0,
    "rank_amihud": -1.0,
    "rank_volume_surge": -1.0,
    "rank_drawdown_90d": 1.0,
    "rank_beta_btc": -1.0,
    "rank_rsi": -1.0,
    "rank_bollinger_b": -1.0,
    "rank_up_day_share_30d": -1.0,
}


@register
class XsAdaptive(LadderHoldingCompetitor):
    family = "xs_adaptive"
    rebalance_weekday = 0
    history_cap: ClassVar[int] = 120  # weeks of panels kept in state

    default_params = {
        "k": 8,
        "max_weight": 0.08,
        "target_vol": 0.20,
        "min_symbols": 10,
        "stop": 0.08,
        "roi_steps": None,
        "hedge": "picks",
        "vol_mode": "names",
        "max_gross": 1.0,
        "max_leverage": 1.0,
        "rebalance_every_weeks": 1,
        "lookback_weeks": 52,
        "min_t": 2.0,
        "max_signals": 8,
        "horizon_days": 7,
        "candidates": None,  # None -> CANDIDATES
    }

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0, bar_hours: int = 1):
        super().__init__(params, seed, bar_hours)
        # ts -> {"panel": {col: {sym: rank}}, "close": {sym: price}}; resolved into ICs as prices arrive
        self._history: dict[str, dict[str, Any]] = {}
        self._weights: dict[str, float] = {}

    def warmup_bars(self) -> int:
        return 95 * self.bars_per_day

    def _candidates(self) -> dict[str, float]:
        c = self.params.get("candidates")
        return {str(k): float(v) for k, v in c.items()} if c else dict(CANDIDATES)

    # ------------------------------------------------------------------ trailing ICs

    def _record(self, snap: Snapshot, panel: pd.DataFrame) -> None:
        cols = [c for c in self._candidates() if c in panel.columns]
        self._history[snap.ts.isoformat()] = {
            "panel": {c: panel[c].dropna().to_dict() for c in cols},
            "close": {s: snap.last_close(s) for s in panel.index},
        }
        for key in sorted(self._history)[: -self.history_cap]:
            del self._history[key]

    def _trailing_ics(self, snap: Snapshot) -> dict[str, tuple[float, float]]:
        """``{column: (mean IC, t-stat)}`` from past panels whose horizon has fully elapsed."""
        horizon = pd.Timedelta(days=int(self.params["horizon_days"]))
        since = snap.ts - pd.Timedelta(weeks=int(self.params["lookback_weeks"]))
        ics: dict[str, list[float]] = {}
        for key, rec in self._history.items():
            t0 = pd.Timestamp(key)
            if t0 < since or t0 + horizon > snap.ts:
                continue
            # realised excess return over the horizon: price then -> price at the first snapshot after
            later = [k for k in self._history if pd.Timestamp(k) >= t0 + horizon]
            if not later:
                continue
            p1 = self._history[min(later, key=lambda k: pd.Timestamp(k))]["close"]
            p0 = rec["close"]
            rets = {s: p1[s] / p0[s] - 1.0 for s in p0 if s in p1 and p0[s] and p1[s] == p1[s] and p0[s] == p0[s]}
            if len(rets) < 8:
                continue
            fwd = pd.Series(rets)
            fwd = fwd - fwd.mean()
            for col, values in rec["panel"].items():
                x = pd.Series(values).reindex(fwd.index).dropna()
                if len(x) < 8 or x.nunique() < 3:
                    continue
                ic = x.corr(fwd.reindex(x.index), method="spearman")
                if ic == ic:
                    ics.setdefault(col, []).append(float(ic))
        out = {}
        for col, values in ics.items():
            if len(values) < 8:
                continue
            arr = np.asarray(values)
            sd = arr.std(ddof=1)
            t = arr.mean() / (sd / np.sqrt(arr.size)) if sd > 0 else 0.0
            out[col] = (float(arr.mean()), float(t))
        return out

    def score(self, panel: pd.DataFrame, ics: dict[str, tuple[float, float]]) -> pd.Series:
        """IC-weighted sum of the centred ranks that cleared the gate; empty when none did."""
        min_t = float(self.params["min_t"])
        chosen = sorted(
            ((c, ic, t) for c, (ic, t) in ics.items() if abs(t) >= min_t and c in panel.columns),
            key=lambda x: -abs(x[2]),
        )[: int(self.params["max_signals"])]
        self._weights = {c: ic for c, ic, _ in chosen}
        if not chosen:
            return pd.Series(dtype=float)
        total = sum(abs(ic) for _, ic, _ in chosen)
        acc = pd.Series(0.0, index=panel.index)
        for col, ic, _ in chosen:
            acc = acc.add((panel[col] - 0.5) * (ic / total), fill_value=0.0)
        return acc.dropna()

    def select(self, snap: Snapshot) -> Decision:
        panel = build_panel(snap)
        if len(panel) < int(self.params["min_symbols"]):
            return {}
        ics = self._trailing_ics(snap)
        self._record(snap, panel)
        scores = self.score(panel, ics)
        if scores.empty:
            return {}  # nothing has earned a vote: stand aside rather than guess
        return self.size_book(
            snap, panel, scores, lambda sym: {"score": round(float(scores[sym]), 4), "signals": len(self._weights)}
        )

    # ------------------------------------------------------------------ state

    def state(self) -> dict[str, Any]:
        return {**super().state(), "history": self._history, "weights": self._weights}

    def restore_state(self, state: dict[str, Any]) -> None:
        super().restore_state(state)
        self._history = dict(state.get("history") or {})
        self._weights = {str(k): float(v) for k, v in (state.get("weights") or {}).items()}

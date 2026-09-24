"""A competitor that closes on the same ROI ladder its labels were built with.

``HoldingCompetitor`` re-decides on a schedule and holds in between. That is the
right default for a fee-sensitive rule, but it means a position can only be
closed at the next rebalance, however far it has already run. The ROI ladder
fixes that: the profit target that labelled the history also closes the live
position, checked on *every* bar rather than on rebalance days.

The point is not the extra exit. It is that training and trading then describe
the same trade. A model trained on "this reached +3 % within two days, net of
costs, against the market" is only telling the truth if the live book actually
takes that profit when it appears. Otherwise the label is a story about a trade
nobody made.

Exits are measured on the **excess** return over the equal-weight universe, so
a position is not congratulated for rising in a market that rose more.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from arena.competitors.base import HoldingCompetitor
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target
from arena.labeling.barriers import DEFAULT_LADDER, RoiLadder

VOL_FLOOR = 0.05


class LadderHoldingCompetitor(HoldingCompetitor):
    """Holds like its parent, but takes profit and stops out on the ladder.

    Subclasses implement :meth:`compute` as usual. State carries the open
    positions' entry marks and a short cooldown, so a symbol closed on its
    target is not bought straight back at the next bar.
    """

    ladder: ClassVar[RoiLadder] = DEFAULT_LADDER
    stop: ClassVar[float] = 0.08
    cooldown_bars: ClassVar[int] = 24

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0, bar_hours: int = 1):
        super().__init__(params, seed, bar_hours)
        self._entries: dict[str, dict] = {}
        self._cooldown: dict[str, int] = {}
        self._basis: dict[str, dict[str, float]] = {}
        self._last_rebalance: str | None = None

    # ------------------------------------------------------------------ ladder

    def _ladder(self) -> RoiLadder:
        steps = self.params.get("roi_steps")
        return RoiLadder(steps=tuple(tuple(s) for s in steps)) if steps else self.ladder

    def _stop(self) -> float:
        return abs(float(self.params.get("stop", self.stop)))

    # ------------------------------------------------------------------ cadence

    def rebalance_due(self, snap: Snapshot) -> bool:
        """True when a full re-selection is allowed at this bar.

        ``rebalance_every_weeks`` slows the parent's weekly schedule down. On
        this data capacity, not signal, is the binding constraint, and a slow
        signal such as low volatility does not need refreshing every Monday:
        every extra rebalance is turnover paid for nothing.
        """
        every = int(self.params.get("rebalance_every_weeks", 1) or 1)
        if every <= 1 or self._last_rebalance is None:
            return True
        return snap.ts - pd.Timestamp(self._last_rebalance) >= pd.Timedelta(weeks=every)

    def compute(self, snap: Snapshot) -> Decision:
        """Cadence gate around :meth:`select`, which subclasses implement."""
        if not self.rebalance_due(snap):
            return dict(self._held)
        self._last_rebalance = snap.ts.isoformat()
        return self.select(snap)

    def select(self, snap: Snapshot) -> Decision:  # pragma: no cover - abstract by convention
        raise NotImplementedError

    # ------------------------------------------------------------------ sizing shared by the families

    def size_book(self, snap: Snapshot, panel: pd.DataFrame, scores: pd.Series, reason_of) -> Decision:
        """Turn a cross-sectional score into a sized, hedged, vol-targeted book.

        Shared by every family in this module so that a comparison between them
        measures the predictor and nothing else.
        """
        p = self.params
        weights = neutral_book(
            scores,
            int(p["k"]),
            float(p["max_weight"]),
            hedge=str(p.get("hedge", "picks")),
            hedge_symbols=list(p.get("hedge_symbols") or ["BTCUSDT", "ETHUSDT", "BTC", "ETH"]),
        )
        if not weights:
            return {}
        scale = self._vol_scale(snap, panel, weights)
        gross = sum(abs(w) for w in weights.values()) * scale
        cap = float(p.get("max_gross", 1.0))
        if gross > cap > 0:
            scale *= cap / gross
        span = float(scores.abs().max()) or 1.0
        out: Decision = {}
        for sym, weight in weights.items():
            conviction = float(np.clip(abs(float(scores.get(sym, 0.0))) / span, 0.0, 1.0))
            out[sym] = Target(
                weight=weight * scale,
                conviction=conviction,
                reason={**reason_of(sym), "vol_scale": round(scale, 3)},
            )
        return out

    def _vol_scale(self, snap: Snapshot, panel: pd.DataFrame, weights: dict[str, float]) -> float:
        p = self.params
        target = float(p["target_vol"])
        mode = str(p.get("vol_mode", "names"))
        if mode == "portfolio":
            closes = snap.closes()
            window = int(p.get("vol_window_days", 30)) * self.bars_per_day
            rets = closes.pct_change().tail(window)
            vol = portfolio_vol(weights, rets, self.bars_per_year)
        else:
            vols = panel.get("vol_30d")
            held = [s for s in weights if vols is not None and s in vols.index]
            vol = float(np.nanmean([max(float(vols[s]), VOL_FLOOR) for s in held])) if held else float("nan")
        if not np.isfinite(vol) or vol <= 0:
            return 1.0
        return float(min(float(p.get("max_leverage", 1.0)), target / max(vol, VOL_FLOOR)))

    def _basket_return(self, snap: Snapshot, basis: dict[str, float]) -> float:
        """Equal-weight return of the basket priced at entry, from entry to now.

        The basket is frozen when the position opens, so this is O(symbols) with
        one binary search each rather than a recomputed index over the whole
        history. It is also the right object: the benchmark a position is judged
        against should be the universe as it stood when the position was taken.
        """
        moves = []
        for sym, entry_price in basis.items():
            if not entry_price:
                continue
            now = snap.last_close(sym)
            if now == now:  # not NaN
                moves.append(now / entry_price - 1.0)
        return float(sum(moves) / len(moves)) if moves else 0.0

    def _excess_since(
        self, snap: Snapshot, symbol: str, entry: dict[str, float], basis: dict[str, float]
    ) -> float | None:
        """Excess return of ``symbol`` over the frozen basket since entry."""
        price = snap.last_close(symbol)
        entry_price = entry.get("price")
        if price != price or not price or not entry_price:
            return None
        symbol_leg = price / entry_price - 1.0
        return float(entry.get("side", 1.0)) * (symbol_leg - self._basket_return(snap, basis))

    def decide(self, snap: Snapshot) -> Decision:
        base = super().decide(snap)
        ladder, stop = self._ladder(), self._stop()

        for sym in list(self._cooldown):
            self._cooldown[sym] -= 1
            if self._cooldown[sym] <= 0:
                del self._cooldown[sym]

        out: Decision = {}
        for sym, target in base.items():
            if target.weight == 0.0:
                continue
            if sym in self._cooldown:
                continue
            side = 1.0 if target.weight > 0 else -1.0
            entry = self._entries.get(sym)
            if entry is None or entry.get("side", side) != side:
                price = snap.last_close(sym)
                if price != price or not price:
                    continue
                basis_key = snap.ts.isoformat()
                if basis_key not in self._basis:
                    self._basis[basis_key] = {s: c for s in snap.symbols if (c := snap.last_close(s)) == c and c}
                self._entries[sym] = {"price": float(price), "bars": 0.0, "side": side, "basis": basis_key}
                out[sym] = target
                continue

            entry["bars"] = float(entry.get("bars", 0.0)) + 1.0
            held = int(entry["bars"])
            excess = self._excess_since(snap, sym, entry, self._basis.get(str(entry.get("basis")), {}))
            if excess is None:
                out[sym] = target
                continue
            if held >= ladder.horizon or excess >= ladder.target_at(held) > 0.0 or excess <= -stop:
                reason = "roi" if excess > 0 else ("stop" if excess <= -stop else "deadline")
                self._close(sym, reason)
                continue
            out[sym] = replace(target, reason={**target.reason, "held_bars": held, "excess": round(excess, 5)})

        for sym in list(self._entries):
            if sym not in out:
                self._entries.pop(sym, None)
        live = {str(e.get("basis")) for e in self._entries.values()}
        for key in list(self._basis):
            if key not in live:
                del self._basis[key]  # nothing references this basket any more
        return out

    def _close(self, symbol: str, reason: str) -> None:
        """Close on the ladder and forget the position until the next re-selection.

        Leaving it in the parent's held book meant it came straight back after
        the cooldown at its old weight, without anyone having re-selected it:
        close, wait a day, re-enter, close again -- turnover paid for nothing.
        """
        self._entries.pop(symbol, None)
        self._held.pop(symbol, None)
        self._cooldown[symbol] = int(self.cooldown_bars)

    # ------------------------------------------------------------------ state

    def state(self) -> dict[str, Any]:
        return {
            **super().state(),
            "entries": {k: dict(v) for k, v in self._entries.items()},
            "cooldown": dict(self._cooldown),
            "basis": {k: dict(v) for k, v in self._basis.items()},
            "last_rebalance": self._last_rebalance,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        super().restore_state(state)
        self._entries = {
            str(k): {kk: (vv if kk == "basis" else float(vv)) for kk, vv in v.items()}
            for k, v in (state.get("entries") or {}).items()
        }
        self._cooldown = {str(k): int(v) for k, v in (state.get("cooldown") or {}).items()}
        self._basis = {
            str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in (state.get("basis") or {}).items()
        }
        self._last_rebalance = state.get("last_rebalance")


def neutral_book(
    scores: pd.Series,
    k: int,
    max_weight: float,
    long_only: bool = False,
    hedge: str = "picks",
    hedge_symbols: list[str] | None = None,
) -> dict[str, float]:
    """Dollar-neutral weights from a score. Three ways to be neutral, one choice.

    ``hedge="picks"`` shorts the *k* lowest-scored names -- the classic
    long-short book. On this data the short leg had negative alpha: shorting
    the highest-volatility perpetuals is shorting lottery tickets with positive
    skew, and they do not underperform reliably, they occasionally explode.

    ``hedge="index"`` keeps the long picks and shorts the whole visible universe
    equally, so the book is long a factor and short the market rather than
    short a handful of idiosyncratic bets. ``hedge="anchor"`` shorts only
    ``hedge_symbols`` (BTC and ETH, the deepest books), which is cheapest to
    execute and carries a little basis risk in exchange.
    """
    ranked = scores.dropna().sort_values(ascending=False)
    if ranked.empty:
        return {}
    k = max(1, min(int(k), len(ranked) // (1 if long_only else 2) or 1))
    longs = list(ranked.index[:k])
    out: dict[str, float] = {}
    long_weight = min(float(max_weight), (1.0 if long_only else 0.5) / len(longs))
    for sym in longs:
        out[sym] = long_weight
    if long_only:
        return out
    long_gross = long_weight * len(longs)

    if hedge == "picks":
        shorts = [s for s in ranked.index[-k:] if s not in longs]
        if shorts:
            weight = min(float(max_weight), 0.5 / len(shorts))
            for sym in shorts:
                out[sym] = -weight
    elif hedge == "index":
        universe = [s for s in ranked.index if s not in longs]
        if universe:
            each = long_gross / len(universe)
            for sym in universe:
                out[sym] = -each
    elif hedge == "anchor":
        anchors = [s for s in (hedge_symbols or []) if s in ranked.index and s not in longs]
        if anchors:
            each = long_gross / len(anchors)
            for sym in anchors:
                out[sym] = -each
    else:
        raise ValueError(f"unknown hedge mode {hedge!r}")
    return out


def portfolio_vol(weights: dict[str, float], returns: pd.DataFrame, bars_per_year: int) -> float:
    """Annualised volatility of the book, from the covariance of its members.

    Scaling on the *average* volatility of the names held ignores that a
    neutral book diversifies most of it away: on this data the book realised
    2.3 % against a 20 % target, running on a ninth of its risk budget. Scaling
    on the portfolio's own covariance is what a target volatility means.
    """
    cols = [s for s in weights if s in returns.columns]
    if len(cols) < 2:
        return float("nan")
    frame = returns[cols].dropna(how="all").fillna(0.0)
    if len(frame) < 10:
        return float("nan")
    w = np.array([weights[s] for s in cols], dtype=float)
    cov = frame.cov().to_numpy()
    var = float(w @ cov @ w)
    return float(np.sqrt(max(var, 0.0) * bars_per_year))

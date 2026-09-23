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

import pandas as pd

from arena.competitors.base import HoldingCompetitor
from arena.core.snapshot import Snapshot
from arena.core.types import Decision
from arena.labeling.barriers import DEFAULT_LADDER, RoiLadder


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

    # ------------------------------------------------------------------ ladder

    def _ladder(self) -> RoiLadder:
        steps = self.params.get("roi_steps")
        return RoiLadder(steps=tuple(tuple(s) for s in steps)) if steps else self.ladder

    def _stop(self) -> float:
        return abs(float(self.params.get("stop", self.stop)))

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
        self._entries.pop(symbol, None)
        self._cooldown[symbol] = int(self.cooldown_bars)

    # ------------------------------------------------------------------ state

    def state(self) -> dict[str, Any]:
        return {
            **super().state(),
            "entries": {k: dict(v) for k, v in self._entries.items()},
            "cooldown": dict(self._cooldown),
            "basis": {k: dict(v) for k, v in self._basis.items()},
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


def neutral_book(
    scores: pd.Series,
    k: int,
    max_weight: float,
    long_only: bool = False,
) -> dict[str, float]:
    """Dollar-neutral long-short weights from a score, highest long and lowest short.

    Gross is split evenly between the two sides so the book carries no market
    exposure by construction. A family that can win by being long crypto in a
    bull run has demonstrated nothing the buy-and-hold benchmark has not.
    """
    ranked = scores.dropna().sort_values(ascending=False)
    if ranked.empty:
        return {}
    k = max(1, min(int(k), len(ranked) // (1 if long_only else 2) or 1))
    longs = list(ranked.index[:k])
    shorts = [] if long_only else [s for s in ranked.index[-k:] if s not in longs]
    out: dict[str, float] = {}
    if longs:
        weight = min(float(max_weight), (1.0 if long_only else 0.5) / len(longs))
        for sym in longs:
            out[sym] = weight
    if shorts:
        weight = min(float(max_weight), 0.5 / len(shorts))
        for sym in shorts:
            out[sym] = -weight
    return out

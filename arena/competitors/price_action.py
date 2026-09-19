"""Classic technical analysis made mechanical (supports, resistances, breakouts, Fibonacci pullbacks).

The chartist's toolbox, without the chartist: swing highs and lows are local
extrema over a ±``pivot_days`` window, the resistance is the highest confirmed
swing high of the lookback and the support the lowest swing low. We enter on a
close beyond a level with a volume expansion and a candle in the direction of
the move (breakout / breakdown), or on a Fibonacci retracement of the last
impulse when a reversal candle (hammer or engulfing) prints in the zone. Every
position carries a mechanical stop at the extreme of the last ``stop_days``.

One family, two horizons picked by ``style``: ``swing`` re-decides daily on a
10-day pivot / 60-day lookback, ``position`` re-decides on Mondays only with a
30-day pivot / 180-day lookback. Works on 1h crypto and 1d classic bars alike
because every horizon is expressed in days and converted with
:meth:`Competitor.days`.

Held positions are kept until their stop is hit (setups are one-bar events; a
stateless re-check would exit the day after every entry). The held book is
already part of :class:`HoldingCompetitor`'s state, so replays stay identical.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from arena.competitors.base import HoldingCompetitor, cap_gross, register
from arena.competitors.features import last_atr
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target

CONVICTION_ATR = 2.0  # a close 2 ATR beyond the level counts as full conviction
HORIZON_KEYS = ("pivot_days", "level_lookback_days", "stop_days")
PRESETS: dict[str, dict[str, int]] = {
    "swing": {"pivot_days": 10, "level_lookback_days": 60, "stop_days": 20},
    "position": {"pivot_days": 30, "level_lookback_days": 180, "stop_days": 60},
}


def presets(style: str) -> dict[str, int]:
    """Horizon parameters of a ``style``; unknown styles fall back to ``swing``."""
    return dict(PRESETS.get(style, PRESETS["swing"]))


def _pivots(values: np.ndarray, w: int, kind: str) -> list[int]:
    """Indices ``i`` such that ``values[i]`` is the max (``kind="high"``) or min of ``values[i-w:i+w+1]``.

    Only pivots with ``w`` bars on both sides are returned: a bar within the
    last ``w`` bars cannot be confirmed without looking ahead.
    """
    n = len(values)
    out: list[int] = []
    for i in range(w, n - w):
        window = values[i - w : i + w + 1]
        if kind == "high" and values[i] >= window.max():
            out.append(i)
        elif kind == "low" and values[i] <= window.min():
            out.append(i)
    return out


def _is_hammer(o: float, h: float, lo: float, c: float) -> bool:
    body = abs(c - o)
    return c >= o and (min(o, c) - lo) >= 2.0 * max(body, 1e-12) and (h - max(o, c)) <= body + 1e-12


def _is_shooting_star(o: float, h: float, lo: float, c: float) -> bool:
    body = abs(c - o)
    return c <= o and (h - max(o, c)) >= 2.0 * max(body, 1e-12) and (min(o, c) - lo) <= body + 1e-12


def _is_bull_engulfing(po: float, pc: float, o: float, c: float) -> bool:
    return c > o and pc < po and o <= pc and c >= po


def _is_bear_engulfing(po: float, pc: float, o: float, c: float) -> bool:
    return c < o and pc > po and o >= pc and c <= po


@register
class PriceAction(HoldingCompetitor):
    family = "price_action"
    default_params: ClassVar[dict[str, Any]] = {
        "style": "swing",
        "pivot_days": 10,
        "level_lookback_days": 60,
        "breakout_buffer": 0.002,
        "min_volume_ratio": 1.2,
        "fib_low": 0.382,
        "fib_high": 0.618,
        "stop_days": 20,
        "max_weight": 0.25,
        "k": 4,
    }

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0, bar_hours: int = 1):
        given = dict(params or {})
        style = str(given.get("style", self.default_params["style"]))
        # style presets apply to every horizon key the caller left at its default
        for key, val in presets(style).items():
            if given.get(key, self.default_params[key]) == self.default_params[key]:
                given[key] = val
        super().__init__(given, seed, bar_hours)
        # ``rebalance_weekday`` is a ClassVar; the position style re-decides on Mondays via this flag
        self._weekly = style == "position"

    def decide(self, snap: Snapshot) -> Decision:
        if self._weekly and self._held_ts is not None and snap.ts.weekday() != 0:
            return dict(self._held)
        return super().decide(snap)

    def warmup_bars(self) -> int:
        p = self.params
        return self.days(p["level_lookback_days"]) + self.days(p["pivot_days"]) + 1

    # ----------------------------------------------------------------- signal
    def _setup(self, sym: str, snap: Snapshot) -> tuple[int, float, dict[str, Any]] | None:
        """(direction, conviction, reason) of the best setup on ``sym`` at this bar, or None."""
        p = self.params
        c = snap.candles(sym, "1h")
        if len(c) < self.warmup_bars():
            return None
        w, lb = self.days(p["pivot_days"]), self.days(p["level_lookback_days"])
        tail = c.iloc[-(lb + w) :]
        high, low = tail["high"].to_numpy(float), tail["low"].to_numpy(float)
        o, h, lo, cl = (float(c[k].iloc[-1]) for k in ("open", "high", "low", "close"))
        po, pc = float(c["open"].iloc[-2]), float(c["close"].iloc[-2])
        highs, lows = _pivots(high, w, "high"), _pivots(low, w, "low")
        if not highs or not lows:
            return None
        resistance = float(high[highs].max())
        support = float(low[lows].min())
        atr = last_atr(c)
        if not np.isfinite(atr) or atr <= 0:
            atr = 0.01 * cl
        vol = tail["volume"].to_numpy(float)
        med = float(np.median(vol[-lb - 1 : -1]))
        vol_ratio = float(vol[-1] / med) if med > 0 else None
        vol_ok = vol_ratio is not None and vol_ratio > float(p["min_volume_ratio"])
        buf = float(p["breakout_buffer"])
        stop_long = self._stop(c, "low")
        stop_short = self._stop(c, "high")

        if cl > resistance * (1 + buf) and vol_ok and cl > o:
            strength = (cl - resistance) / atr
            candle = "engulfing" if _is_bull_engulfing(po, pc, o, cl) else "bullish"
            return (
                1,
                float(np.clip(strength / CONVICTION_ATR, 0.0, 1.0)),
                self._reason("breakout", resistance, cl, vol_ratio, candle, stop_long),
            )
        if cl < support * (1 - buf) and vol_ok and cl < o:
            strength = (support - cl) / atr
            candle = "engulfing" if _is_bear_engulfing(po, pc, o, cl) else "bearish"
            return (
                -1,
                float(np.clip(strength / CONVICTION_ATR, 0.0, 1.0)),
                self._reason("breakdown", support, cl, vol_ratio, candle, stop_short),
            )

        # Fibonacci pullback on the last impulse: swing low then swing high (up) or the reverse (down)
        i_hi, i_lo = highs[-1], lows[-1]
        fib_lo, fib_hi = float(p["fib_low"]), float(p["fib_high"])
        mid = 0.5 * (fib_lo + fib_hi)
        half = max(0.5 * (fib_hi - fib_lo), 1e-9)
        if i_hi > i_lo:
            prior_lows = [i for i in lows if i < i_hi]
            if not prior_lows:
                return None
            imp_lo, imp_hi = float(low[prior_lows[-1]]), float(high[i_hi])
            rng = imp_hi - imp_lo
            if rng <= 0:
                return None
            zone_lo, zone_hi = imp_hi - fib_hi * rng, imp_hi - fib_lo * rng
            if zone_lo <= cl <= zone_hi:
                if _is_hammer(o, h, lo, cl):
                    candle = "hammer"
                elif _is_bull_engulfing(po, pc, o, cl):
                    candle = "engulfing"
                else:
                    return None
                retrace = (imp_hi - cl) / rng
                conviction = float(np.clip(1.0 - abs(retrace - mid) / half, 0.0, 1.0))
                r = self._reason("fib_pullback_long", imp_lo, cl, None, candle, stop_long)
                r.update({"retrace": retrace, "zone_low": zone_lo, "zone_high": zone_hi, "impulse_high": imp_hi})
                return 1, conviction, r
            return None
        prior_highs = [i for i in highs if i < i_lo]
        if not prior_highs:
            return None
        imp_hi, imp_lo = float(high[prior_highs[-1]]), float(low[i_lo])
        rng = imp_hi - imp_lo
        if rng <= 0:
            return None
        zone_lo, zone_hi = imp_lo + fib_lo * rng, imp_lo + fib_hi * rng
        if zone_lo <= cl <= zone_hi:
            if _is_shooting_star(o, h, lo, cl):
                candle = "hammer"
            elif _is_bear_engulfing(po, pc, o, cl):
                candle = "engulfing"
            else:
                return None
            retrace = (cl - imp_lo) / rng
            conviction = float(np.clip(1.0 - abs(retrace - mid) / half, 0.0, 1.0))
            r = self._reason("fib_pullback_short", imp_hi, cl, None, candle, stop_short)
            r.update({"retrace": retrace, "zone_low": zone_lo, "zone_high": zone_hi, "impulse_low": imp_lo})
            return -1, conviction, r
        return None

    def _stop(self, c, col: str) -> float:
        """Lowest low (``col="low"``) or highest high of the ``stop_days`` bars before the current one."""
        n = self.days(self.params["stop_days"])
        x = c[col].to_numpy(float)[-(n + 1) : -1]
        return float(x.min() if col == "low" else x.max())

    @staticmethod
    def _reason(setup: str, level: float, close: float, vol_ratio: float | None, candle: str, stop: float) -> dict:
        return {
            "setup": setup,
            "level": float(level),
            "close": float(close),
            "volume_ratio": None if vol_ratio is None else float(vol_ratio),
            "candle": candle,
            "stop": float(stop),
        }

    # ---------------------------------------------------------------- decision
    def compute(self, snap: Snapshot) -> Decision:
        p = self.params
        k, w_max = int(p["k"]), float(p["max_weight"])
        out: Decision = {}
        # 1. held positions: keep unless the stop is hit (the stop trails the last stop_days extreme)
        for sym, old in self._held.items():
            if sym not in snap.symbols:
                continue
            c = snap.candles(sym, "1h")
            if len(c) < self.warmup_bars() or old.weight == 0:
                continue
            cl = float(c["close"].iloc[-1])
            stop = self._stop(c, "low" if old.weight > 0 else "high")
            if (old.weight > 0 and cl < stop) or (old.weight < 0 and cl > stop):
                continue
            out[sym] = Target(
                weight=old.weight, conviction=old.conviction, reason={**old.reason, "close": cl, "stop": stop}
            )
        # 2. new setups, best conviction first, until k positions
        candidates: list[tuple[float, str, int, dict[str, Any]]] = []
        for sym in snap.symbols:
            if sym in out:
                continue
            found = self._setup(sym, snap)
            if found is None:
                continue
            direction, conviction, reason = found
            candidates.append((conviction, sym, direction, reason))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        for conviction, sym, direction, reason in candidates:
            if len(out) >= k:
                break
            out[sym] = Target(weight=direction * w_max, conviction=conviction, reason=reason)
        return cap_gross(out)

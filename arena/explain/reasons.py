"""Turn a Target's ``reason`` dict into one French sentence with the live values."""

from __future__ import annotations

from typing import Any

from arena.core.types import Target

REGIME_FR = {
    "bull_calm": "haussier calme",
    "bull_vol": "haussier volatil",
    "bear": "baissier",
    "range": "sans tendance",
    "unknown": "indéterminé",
}


def _pct(v: Any, digits: int = 1) -> str:
    return f"{float(v) * 100:+.{digits}f} %"


def _num(v: Any) -> str:
    """Price rendering: two decimals under 1 000, grouped thousands above (3 120)."""
    x = float(v)
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", " ")
    return f"{x:.2f}"


CANDLE_FR = {
    ("bullish", 1): "chandelier haussier",
    ("bearish", -1): "chandelier baissier",
    ("engulfing", 1): "englobante haussière",
    ("engulfing", -1): "englobante baissière",
    ("hammer", 1): "marteau de retournement",
    ("hammer", -1): "étoile filante de retournement",
}


def _price_action(symbol: str, t: Target, r: dict[str, Any], w: str) -> str:
    side = 1 if t.weight > 0 else -1
    setup = str(r.get("setup", ""))
    candle = CANDLE_FR.get((str(r.get("candle", "")), side), str(r.get("candle", "")))
    stop = f"stop {'sous' if side > 0 else 'au-dessus de'} {_num(r.get('stop', 0.0))}"
    head = f"{direction(t).capitalize()} {symbol} ({w})"
    if setup in ("breakout", "breakdown"):
        level = "de la résistance" if setup == "breakout" else "du support"
        vr = r.get("volume_ratio")
        vol = f" avec un volume {float(vr):.1f}× la normale" if vr is not None else ""
        return f"{head} : cassure {level} {_num(r.get('level', 0.0))}{vol}, {candle} ; {stop}."
    if setup in ("fib_pullback_long", "fib_pullback_short"):
        impulse = "haussière" if setup.endswith("long") else "baissière"
        retrace = float(r.get("retrace", 0.5)) * 100
        zone = f"zone {_num(r.get('zone_low', 0.0))}–{_num(r.get('zone_high', 0.0))}"
        return f"{head} : repli à {retrace:.0f} % de la dernière impulsion {impulse} ({zone}), {candle} ; {stop}."
    return f"{head} : niveau {_num(r.get('level', 0.0))}, {candle} ; {stop}."


def direction(t: Target) -> str:
    if t.kind == "carry":
        return "carry" if t.weight > 0 else "à plat"
    if t.weight > 0:
        return "achat"
    if t.weight < 0:
        return "vente"
    return "à plat"


def explain_target(family: str, symbol: str, t: Target) -> str:
    r = t.reason or {}
    w = f"{abs(t.weight) * 100:.0f} % du capital"
    if r.get("exit"):
        return f"Sort de {symbol} : les conditions d'entrée ne tiennent plus."
    if family == "carry" or (family == "regime" and "mean_funding_8h" in r):
        rate = float(r.get("mean_funding_8h", 0.0))
        return (
            f"Carry sur {symbol} ({w}) : funding moyen {rate * 100:.4f} % par 8 h, soit ≈ {rate * 3 * 365 * 100:.1f} %/an, "
            f"rang {int(r.get('rank', 0)) + 1} des mieux payés."
        )
    if family == "trend_ts":
        side = "monte" if t.weight > 0 else "baisse"
        return (
            f"{direction(t).capitalize()} {symbol} ({w}) : {symbol} {side} depuis 30 jours ({_pct(r.get('r_short', 0))}) et 90 jours "
            f"({_pct(r.get('r_long', 0))}), EMA 50 {'au-dessus' if t.weight > 0 else 'en dessous'} de l'EMA 200, volatilité {float(r.get('vol', 0)) * 100:.0f} %/an."
        )
    if family == "xs_momentum":
        return (
            f"{direction(t).capitalize()} {symbol} ({w}) : classé {int(r.get('rank', 0)) + 1} sur 30 jours avec {_pct(r.get('score', 0))}"
            f"{' (parmi les plus forts)' if t.weight > 0 else ' (parmi les plus faibles)'}."
        )
    if family == "regime":
        return f"{direction(t).capitalize()} {symbol} ({w}) : marché jugé {REGIME_FR.get(str(r.get('regime')), r.get('regime'))} sur le BTC."
    if family == "meta_label":
        if r.get("mode") == "pass_through":
            return f"{direction(t).capitalize()} {symbol} ({w}) : reprend le signal de {r.get('n_agree', 1)} base(s), sans filtre appris."
        return (
            f"{direction(t).capitalize()} {symbol} ({w}) : signal de base retenu, probabilité de gain estimée {float(r.get('p_win', 0)) * 100:.0f} %, "
            f"marché {REGIME_FR.get(str(r.get('regime')), '')}."
        )
    if family == "price_action":
        return _price_action(symbol, t, r, w)
    if family == "news":
        return f"Achat {symbol} ({w}) : sentiment 7 jours {float(r.get('sent_7d', 0)):+.2f} et en hausse, {int(r.get('n_7d', 0))} articles."
    if family.startswith("null") or family.startswith("bench"):
        return f"{direction(t).capitalize()} {symbol} ({w}) : position de repère."
    return f"{direction(t).capitalize()} {symbol} ({w})."


def explain_decision(family: str, decision: dict[str, Target]) -> list[str]:
    return [explain_target(family, s, t) for s, t in sorted(decision.items(), key=lambda kv: -abs(kv[1].weight))]

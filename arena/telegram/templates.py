"""Plain-text Telegram message templates, in French, with the "why" behind each line.

Pure functions returning ``str``. No Markdown ``parse_mode`` is used, so no
escaping is needed; every message is capped at ``MAX_LEN`` characters. All the
wording about families, decisions and verdicts comes from ``arena.explain`` so
the dashboard and Telegram tell the same story.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from arena.core.types import Target
from arena.explain.families import STATUS_FR, card
from arena.explain.reasons import REGIME_FR, explain_target
from arena.explain.verdicts import explain_gate, money

MAX_LEN = 4000
NAV0 = 10_000.0
MIN_DAYS = 42
MIN_DECISIONS = 100

_EVENT_EMOJI = {
    "promotion": "🏆",
    "rejected": "🚫",
    "stale": "⏳",
    "error": "❗",
    "drift": "📉",
    "info": "ℹ️",
}
_EVENT_LABEL = {
    "promotion": "Promotion",
    "rejected": "Refusé à l'entrée",
    "stale": "Données en retard",
    "error": "Erreur",
    "drift": "Dérive",
}
_CRITERIA_FR = {
    "folds_positive": "gagner dans deux périodes de test sur trois",
    "sharpe_above_null": "battre le 95e centile des modèles aléatoires",
    "dsr": "un Sharpe déflaté crédible à 90 % vu le nombre d'essais",
    "bootstrap_p": "moins de 10 % de chances d'obtenir ça en mélangeant ses rendements",
    "max_drawdown": "un pire creux sous 30 %",
    "min_decisions": "au moins 30 décisions",
}


def _cap(text: str, limit: int = MAX_LEN) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fr_num(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def _fr_money(v: float) -> str:
    return money(v, NAV0).replace(".", ",")


def split_name(name: str) -> tuple[str, int | None]:
    """``"carry_v1"`` -> ``("carry", 1)``; a name without ``_vN`` keeps itself as family."""
    m = re.fullmatch(r"(.+)_v(\d+)", name)
    if not m:
        return name, None
    return m.group(1), int(m.group(2))


def display_name(family: str, version: int | None) -> str:
    return f"{family} v{version}" if version is not None else family


# --------------------------------------------------------------------------- signal alert


def signal_alert(
    name: str,
    symbol: str,
    weight: float,
    conviction: float,
    kind: str,
    regime: str | None,
    funding_8h: float | None,
    reason: dict[str, Any],
    family: str | None = None,
    version: int | None = None,
    status: str | None = None,
    nav0: float = NAV0,
) -> str:
    """Three lines: what model speaks, what it does and why, in which market.

    ``⚔️ Carry de funding (carry v1, champion)``
    ``Carry sur SUI (20 % du capital) : funding moyen 0,0120 % par 8 h, soit ≈ 13 %/an, rang 1 des mieux payés.``
    ``Marché : haussier calme. Capital virtuel : 10 000 €.``
    """
    fam, ver = split_name(name)
    family = family or fam
    version = version if version is not None else ver
    who = display_name(family, version)
    if status:
        who += f", {STATUS_FR.get(status, status)}"
    header = f"⚔️ {card(family).title} ({who})"

    r = dict(reason or {})
    if weight == 0.0 and not r.get("exit"):
        r["exit"] = True  # a flat target on a champion is always an exit
    target = Target(weight=weight, conviction=conviction, kind=kind, reason=r)  # type: ignore[arg-type]
    why = _fr_dec(explain_target(family, symbol, target))

    market = f"Marché : {REGIME_FR.get(regime or 'unknown', regime or 'indéterminé')}."
    if funding_8h is not None and family != "carry":
        market += f" Funding actuel sur {symbol} : {_fr_num(funding_8h * 100, 4)} % par 8 h."
    market += f" Capital virtuel : {nav0:,.0f} €.".replace(",", " ")
    return _cap("\n".join([header, why, market]))


def _fr_dec(text: str) -> str:
    """French decimal comma on numbers inside a sentence (``0.0120 %`` -> ``0,0120 %``)."""
    return re.sub(r"(\d)\.(\d)", r"\1,\2", text)


# --------------------------------------------------------------------------- digest views


@dataclass
class PositionView:
    symbol: str
    weight: float
    conviction: float = 0.5
    kind: str = "perp"
    reason: dict[str, Any] = field(default_factory=dict)

    def target(self) -> Target:
        return Target(weight=self.weight, conviction=self.conviction, kind=self.kind, reason=self.reason)  # type: ignore[arg-type]


@dataclass
class ChampionView:
    name: str
    family: str
    version: int | None = None
    positions: list[PositionView] = field(default_factory=list)
    pnl_1d: float | None = None  # fractions of NAV0
    pnl_7d: float | None = None
    pnl_30d: float | None = None
    btc_30d: float | None = None  # "garder du BTC" over the same 30 days


@dataclass
class ChallengerView:
    name: str
    family: str
    version: int | None = None
    days: int = 0
    decisions: int = 0
    days_required: int = MIN_DAYS
    decisions_required: int = MIN_DECISIONS
    pnl_30d: float | None = None
    sharpe_30d: float | None = None
    champion_sharpe_30d: float | None = None


@dataclass
class DigestContext:
    date_str: str
    champions: list[ChampionView] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)
    challengers: list[ChallengerView] = field(default_factory=list)
    btc_30d_eur: float | None = None  # already in euros
    null95: float | None = None
    last_tick: datetime | None = None
    evaluated: int | None = None
    drift_lines: list[str] = field(default_factory=list)


def _money_or_na(v: float | None) -> str:
    return "n/d" if v is None else _fr_money(v)


def _vs_btc(pnl_30d: float | None, btc_30d: float | None) -> str:
    if pnl_30d is None or btc_30d is None:
        return ""
    diff = pnl_30d - btc_30d
    if abs(diff) < 1e-9:
        return " ; à égalité avec « garder du BTC »"
    verb = "devant" if diff > 0 else "derrière"
    return f" ; {verb} « garder du BTC » de {_fr_money(abs(diff)).lstrip('+-')}"


def _champion_block(c: ChampionView) -> list[str]:
    who = display_name(c.family, c.version)
    lines = [f"• {card(c.family).title} ({who})"]
    if c.positions:
        for p in sorted(c.positions, key=lambda p: -abs(p.weight)):
            lines.append("  " + _fr_dec(explain_target(c.family, p.symbol, p.target())))
    else:
        lines.append("  À plat : aucune position, il attend un signal.")
    lines.append(
        f"  P&L hier {_money_or_na(c.pnl_1d)} / 7 j {_money_or_na(c.pnl_7d)} / 30 j {_money_or_na(c.pnl_30d)}"
        + _vs_btc(c.pnl_30d, c.btc_30d)
        + "."
    )
    return lines


def _challenger_block(c: ChallengerView, null95: float | None) -> list[str]:
    who = display_name(c.family, c.version)
    head = f"• {who} ({card(c.family).title}) : {c.days}/{c.days_required} jours, "
    head += f"{c.decisions}/{c.decisions_required} décisions"
    lines = [head + "."]
    tail = f"  30 j : {_money_or_na(c.pnl_30d)}"
    if c.sharpe_30d is not None and c.champion_sharpe_30d is not None:
        tail += " ; " + ("en avance" if c.sharpe_30d > c.champion_sharpe_30d else "en retard") + " sur le champion"
    elif c.sharpe_30d is not None:
        tail += " ; pas de champion à battre dans sa famille"
    if c.sharpe_30d is not None and null95 is not None:
        tail += " ; " + ("au-dessus" if c.sharpe_30d > null95 else "en dessous") + " de la chance"
    lines.append(tail + ".")
    return lines


def daily_digest(ctx: DigestContext) -> str:
    lines: list[str] = [f"📊 L'arène ce matin — {ctx.date_str}", ""]

    lines.append("Qui parle :")
    if ctx.champions:
        for c in ctx.champions:
            lines.extend(_champion_block(c))
    else:
        lines.append("• Personne : aucun champion en lice.")
    lines.append("")

    lines.append("Ce qui a changé depuis hier :")
    if ctx.changes:
        lines.extend(f"• {ch}" for ch in ctx.changes)
    else:
        lines.append("• rien")
    lines.append("")

    lines.append("Les prétendants :")
    if ctx.challengers:
        for c in ctx.challengers:
            lines.extend(_challenger_block(c, ctx.null95))
    else:
        lines.append("• aucun")
    lines.append("")

    btc = "n/d" if ctx.btc_30d_eur is None else f"{ctx.btc_30d_eur:+,.0f} €".replace(",", " ")
    chance = "n/d" if ctx.null95 is None else f"Sharpe {_fr_num(ctx.null95)}"
    lines.append(f"Repères : garder du BTC 30 j : {btc} ; la chance (95e centile) : {chance}.")
    lines.append("")

    tick = "aucun passage enregistré" if ctx.last_tick is None else f"dernier passage {ctx.last_tick:%H:%M} UTC"
    evaluated = "" if ctx.evaluated is None else f", {ctx.evaluated} modèles évalués"
    if ctx.drift_lines:
        lines.append(f"Santé : {tick}{evaluated}, dérives :")
        lines.extend(f"• {d}" for d in ctx.drift_lines)
    else:
        lines.append(f"Santé : {tick}{evaluated}, dérives : aucune")

    return _cap("\n".join(lines))


# --------------------------------------------------------------------------- event alerts

_RE_STALE = re.compile(r"last closed bar (.+?) UTC, lag ([\d.]+)h")
_RE_BLEED = re.compile(r"^(.+?): 30d Sharpe (-?[\d.]+), return (-?[\d.]+%)$")
_RE_SILENT = re.compile(r"^(.+?): no non-zero target for (\d+)\+ days$")
_RE_NEW_CH = re.compile(r"^new challenger (\S+): (.*)$")
_RE_REJECT_GATE = re.compile(r"^(\S+) rejected at gate \((.+?)\)(?:; enters as challenger)?$")
_RE_REJECT_OPT = re.compile(r"^(\S+) optimize best rejected: (.+)$")
_RE_REJECT_RETRAIN = re.compile(r"^meta_label retrain rejected: wf AUC ([\d.]+) < ([\d.]+)$")
_RE_NAV = re.compile(r"^(.+?): nav is (.+)$")


def criteria_fr(failed: list[str]) -> str:
    return " ; ".join(_CRITERIA_FR.get(f.strip(), f.strip()) for f in failed if f.strip())


def detail_fr(kind: str, detail: str) -> str:
    """Best-effort French rendering of the English ``detail`` strings the runner produces."""
    d = detail.strip()
    if kind == "stale":
        if d == "no candles at all":
            return "aucune bougie en base, le marché n'a jamais été ingéré."
        m = _RE_STALE.search(d)
        if m:
            return f"dernière bougie close {m.group(1)} UTC, {_fr_num(float(m.group(2)), 1)} h de retard."
    elif kind == "drift":
        m = _RE_BLEED.match(d)
        if m:
            return (
                f"{m.group(1)} perd de l'argent en direct : Sharpe 30 j {_fr_dec(m.group(2))}, "
                f"rendement {_fr_dec(m.group(3))}."
            )
        m = _RE_SILENT.match(d)
        if m:
            return f"{m.group(1)} n'a pris aucune position depuis {m.group(2)} jours ou plus."
    elif kind == "info":
        m = _RE_NEW_CH.match(d)
        if m:
            return f"Nouveau prétendant {m.group(1)} : {m.group(2)}"
    elif kind == "rejected":
        m = _RE_REJECT_GATE.match(d)
        if m:
            tail = " Il entre quand même comme prétendant, jugé en direct." if "enters as challenger" in d else ""
            return f"{m.group(1)} (critères manqués : {criteria_fr(m.group(2).split(','))}).{tail}"
        m = _RE_REJECT_OPT.match(d)
        if m:
            missed = criteria_fr(m.group(2).split(","))
            return f"meilleur essai d'optimisation {m.group(1)} (critères manqués : {missed})."
        m = _RE_REJECT_RETRAIN.match(d)
        if m:
            return (
                f"nouveau modèle méta-étiquetage : AUC hors échantillon {_fr_dec(m.group(1))}, "
                f"sous le minimum {_fr_dec(m.group(2))}."
            )
    elif kind == "error":
        m = _RE_NAV.match(d)
        if m:
            return f"{m.group(1)} : capital virtuel invalide ({m.group(2)}), livre cassé."
    return d


def promotion_text(
    challenger: str,
    family: str,
    old_champion: str | None,
    challenger_sharpe: float | None,
    champion_sharpe: float | None,
    null95: float | None,
) -> str:
    was = f" (était {old_champion})" if old_champion and old_champion != "none" else " (la famille n'en avait pas)"
    text = f"{challenger} devient champion de {card(family).title}{was}."
    why: list[str] = []
    if challenger_sharpe is not None:
        if champion_sharpe is not None:
            why.append(f"Sharpe en direct {_fr_num(challenger_sharpe)} contre {_fr_num(champion_sharpe)}")
        else:
            why.append(f"Sharpe en direct {_fr_num(challenger_sharpe)}")
    if null95 is not None:
        why.append(f"au-dessus de la chance ({_fr_num(null95)})")
    if why:
        text += " Raison : " + ", ".join(why) + "."
    return text


def rejected_text(detail: str, payload: dict[str, Any] | None) -> str:
    p = payload or {}
    metrics = p.get("metrics")
    failed = list(p.get("failed") or [])
    if isinstance(metrics, dict) and metrics:
        who = detail.split(" rejected", 1)[0] if " rejected" in detail else detail
        lines = [ln for ln in explain_gate(metrics, failed) if ln.startswith("✗")] or explain_gate(metrics, failed)
        return f"{who}\n" + "\n".join(_fr_dec(ln) for ln in lines)
    if failed:
        who = detail.split(" rejected", 1)[0] if " rejected" in detail else detail
        return f"{who} (critères manqués : {criteria_fr(failed)})."
    return detail_fr("rejected", detail)


def event_alert(kind: str, text: str, payload: dict[str, Any] | None = None) -> str:
    """``kind`` picks the emoji and French label; ``payload`` adds the "why" when the runner stored it."""
    emoji = _EVENT_EMOJI.get(kind, _EVENT_EMOJI["info"])
    label = _EVENT_LABEL.get(kind)
    if kind == "promotion" and payload and "family" in payload:
        body = promotion_text(
            str(payload.get("challenger", text)),
            str(payload["family"]),
            payload.get("old_champion"),
            payload.get("challenger_sharpe"),
            payload.get("champion_sharpe"),
            payload.get("null95"),
        )
    elif kind == "rejected":
        body = rejected_text(text, payload)
    else:
        body = detail_fr(kind, text)
    return _cap(f"{emoji} {label} : {body}" if label else f"{emoji} {body}")

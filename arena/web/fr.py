"""French labels and formatting for the dashboard.

Wording about families, decisions and verdicts lives in ``arena.explain``;
this module only maps the store's enumerations (alert kinds, trial kinds,
verdicts, statuses) to plain French and formats numbers, money and delays.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from arena.core.types import Target
from arena.explain.families import ROLE_FR, STATUS_FR
from arena.explain.reasons import explain_target
from arena.explain.verdicts import money

# Alert kinds written by the runner, cli and challenger modules.
KIND_FR: dict[str, str] = {
    "signal": "Nouveau signal",
    "promotion": "Promotion",
    "drift": "Dérive",
    "stale": "Données en retard",
    "error": "Erreur",
    "rejected": "Refusé au test",
    "info": "Information",
}

TRIAL_KIND_FR: dict[str, str] = {
    "backtest": "test sur l'historique",
    "walkforward": "test glissant (walk-forward)",
    "optimize": "recherche de paramètres",
    "retrain": "ré-entraînement",
}

VERDICT_FR: dict[str | None, str] = {"admitted": "admis", "rejected": "refusé", None: "en cours"}

# Data sources shown in « Sur quoi l'arène se base ».
SOURCE_FR: dict[str, str] = {
    "candles": "Bougies Binance (cours horaires)",
    "funding": "Funding Binance (toutes les 8 h)",
    "open_interest": "Intérêt ouvert Binance",
    "hl_snapshots": "Instantané Hyperliquid",
    "articles": "Articles (flux RSS crypto)",
    "macro_events": "Calendrier macro",
}

REGIME_FR_SHORT = {"bull": "haussier", "bear": "baissier", "range": "sans tendance"}

FRESH_HOURS = 2.0
WARM_HOURS = 6.0


def kind_fr(kind: str) -> str:
    return KIND_FR.get(kind, kind)


def trial_kind_fr(kind: str) -> str:
    return TRIAL_KIND_FR.get(kind, kind)


def verdict_fr(verdict: str | None) -> str:
    return VERDICT_FR.get(verdict, verdict or "en cours")


def status_fr(role: str, status: str) -> str:
    """« champion », « prétendant »… for models; « repère » for benchmarks and nulls."""
    if role != "competitor":
        return ROLE_FR.get(role, role)
    return STATUS_FR.get(status, status)


def status_css(role: str, status: str) -> str:
    return status if role == "competitor" else role


def freshness_level(age: timedelta | None, limit_hours: float | None = None) -> str:
    """``ok`` / ``warn`` / ``bad`` for the age of a source's last successful fetch.

    With ``limit_hours`` (what ``arena.runner.drift`` would alert on), the
    thresholds follow that source's own cadence: green below two thirds of it,
    orange up to it, red past it. Without, the flat 2 h / 6 h reading is kept.
    """
    if age is None:
        return "bad"
    hours = age.total_seconds() / 3600
    if limit_hours is not None and limit_hours > 0:
        if hours < 2.0 / 3.0 * limit_hours:
            return "ok"
        return "warn" if hours <= limit_hours else "bad"
    if hours < FRESH_HOURS:
        return "ok"
    if hours < WARM_HOURS:
        return "warn"
    return "bad"


def ago_fr(ts: datetime | None, now: datetime) -> str:
    if ts is None:
        return "jamais"
    secs = int((now - ts).total_seconds())
    if secs < 90:
        return "il y a moins d'une minute"
    if secs < 5400:
        return f"il y a {secs // 60} min"
    if secs < 172800:
        return f"il y a {secs // 3600} h"
    return f"il y a {secs // 86400} j"


def heure(ts: datetime | None) -> str:
    return "" if ts is None else ts.strftime("%H:%M")


def date_heure(ts: datetime | None) -> str:
    return "" if ts is None else ts.strftime("%d/%m %H:%M")


def date_fr(ts: datetime | None) -> str:
    return "" if ts is None else ts.strftime("%d/%m/%Y")


def euros(v: float | None) -> str:
    """Signed euros with a thin space as thousands separator (``+1 234 €``)."""
    if v is None:
        return "n/a"
    return f"{v:+,.0f} €".replace(",", " ")


def euros_frac(frac: float | None) -> str:
    return "n/a" if frac is None else money(float(frac))


def pct_fr(v: float | None, digits: int = 1) -> str:
    return "n/a" if v is None else f"{float(v) * 100:+.{digits}f} %".replace(".", ",")


def pct_plain(v: float | None, digits: int = 0) -> str:
    return "n/a" if v is None else f"{float(v) * 100:.{digits}f} %".replace(".", ",")


def num_fr(v: Any, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (list, tuple)):
        return ", ".join(num_fr(x, digits) for x in v)
    return f"{float(v):.{digits}f}".replace(".", ",") if isinstance(v, (int, float)) else str(v)


def sign_css(v: float | None) -> str:
    if v is None or v == 0:
        return ""
    return "pos" if v > 0 else "neg"


def position_sentence(family: str, symbol: str, weight: float, kind: str, reason: dict[str, Any] | None) -> str:
    """Wrapper around ``explain_target`` from raw target columns."""
    t = Target(weight=float(weight), kind=kind or "perp", reason=dict(reason or {}))
    return explain_target(family, symbol, t)


def exit_sentence(family: str, symbol: str) -> str:
    return explain_target(family, symbol, Target(0.0, reason={"exit": True}))


def alert_sentence(kind: str, family: str | None, symbol: str | None, payload: dict[str, Any]) -> str:
    """One French line for an alert: signals go through ``explain_target``, others use ``detail``."""
    if kind == "signal" and family and symbol:
        weight = float(payload.get("weight", 0.0))
        reason = payload.get("reason") or {}
        if weight == 0.0 or reason.get("exit"):
            return exit_sentence(family, symbol)
        return position_sentence(family, symbol, weight, str(payload.get("kind", "perp")), reason)
    detail = payload.get("detail")
    if detail:
        return str(detail)
    return ", ".join(f"{k} : {v}" for k, v in payload.items())


def days_fr(n: int) -> str:
    return f"{n} jour" if n == 1 else f"{n} jours"


def duration_fr(delta: timedelta) -> str:
    """« 3 j », « 14 h » or « 25 min » for how long a position has been held."""
    secs = int(delta.total_seconds())
    if secs >= 172800:
        return f"{secs // 86400} j"
    if secs >= 3600:
        return f"{secs // 3600} h"
    return f"{max(1, secs // 60)} min"


UNIVERSE_FR = {"crypto": "Crypto", "classic": "Marchés classiques"}


def universe_fr(name: str | None) -> str:
    return UNIVERSE_FR.get(str(name), str(name or "Crypto"))

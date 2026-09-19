"""Plain-text Telegram message templates.

Pure functions returning ``str``. No Markdown ``parse_mode`` is used, so no
escaping is needed; every message is capped at ``MAX_LEN`` characters.
"""

from __future__ import annotations

from typing import Any

MAX_LEN = 4000
MAX_REASON_PAIRS = 6

_STATUS_MARK = {"champion": "🏆", "challenger": "🧪", "candidate": "🧪"}
_EVENT_EMOJI = {
    "promotion": "🏆",
    "rejected": "🚫",
    "stale": "⏳",
    "error": "❗",
    "info": "ℹ️",
}


def _cap(text: str, limit: int = MAX_LEN) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt_num(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        return f"{v:.3g}"
    return str(v)


def _direction(weight: float, kind: str) -> str:
    if kind == "carry":
        return "CARRY" if weight > 0 else "FLAT"
    if weight > 0:
        return "LONG"
    if weight < 0:
        return "SHORT"
    return "FLAT"


def _reason_str(reason: dict[str, Any] | None) -> str:
    if not reason:
        return "-"
    pairs = [f"{k}={_fmt_num(v)}" for k, v in list(reason.items())[:MAX_REASON_PAIRS]]
    return ", ".join(pairs)


def signal_alert(
    name: str,
    symbol: str,
    weight: float,
    conviction: float,
    kind: str,
    regime: str | None,
    funding_8h: float | None,
    reason: dict[str, Any],
) -> str:
    """One-line signal alert, e.g.
    ``⚔️ trend_ts v3 → LONG ETH 0.35 (conv 0.71) | regime bull_vol | fund +0.012%/8h | why: ema50=1, r30=0.18``
    """
    direction = _direction(weight, kind)
    parts = [f"⚔️ {name} → {direction} {symbol} {abs(weight):.2f} (conv {conviction:.2f})"]
    parts.append(f"regime {regime}" if regime else "regime n/a")
    if funding_8h is not None:
        parts.append(f"fund {funding_8h * 100:+.3f}%/8h")
    parts.append(f"why: {_reason_str(reason)}")
    return _cap(" | ".join(parts))


def _pct(v: Any) -> str:
    return "  n/a" if v is None else f"{v * 100:+.1f}%"


def _num(v: Any, fmt: str = "{:.2f}") -> str:
    return "n/a" if v is None else fmt.format(v)


def daily_digest(
    date_str: str,
    leaderboard: list[dict[str, Any]],
    allocation: list[tuple[str, float]],
    null95: float | None,
    challengers: list[dict[str, Any]],
    drift_alerts: list[str],
) -> str:
    lines: list[str] = [f"📊 Arena digest {date_str}", ""]

    lines.append("Leaderboard (30d)")
    if leaderboard:
        name_w = max(4, min(18, max(len(str(r.get("name", ""))) for r in leaderboard)))
        lines.append(f"   {'name':<{name_w}} {'sharpe':>6} {'ret':>7} {'mdd':>7} {'nav':>9}")
        for r in leaderboard:
            role = r.get("role")
            mark = "⚪" if role in ("null", "benchmark") else _STATUS_MARK.get(str(r.get("status")), "•")
            nm = str(r.get("name", ""))[:name_w]
            lines.append(
                f"{mark} {nm:<{name_w}} {_num(r.get('sharpe_30d')):>6} {_pct(r.get('ret_30d')):>7} "
                f"{_pct(r.get('mdd_30d')):>7} {_num(r.get('nav'), '{:.0f}'):>9}"
            )
    else:
        lines.append("  (empty)")
    lines.append("")

    lines.append("Allocator opinion")
    if allocation:
        for nm, w in allocation:
            lines.append(f"  {nm} → {w * 100:.0f}%")
    else:
        lines.append("  none")
    lines.append("")

    lines.append(f"Null 95th pct Sharpe: {_num(null95)}")
    lines.append("")

    lines.append("Challengers")
    if challengers:
        for c in challengers:
            progress = c.get("progress") or (
                f"{c.get('days_in_arena', 0)}/{c.get('days_required', 42)}d, "
                f"{c.get('decisions', 0)}/{c.get('decisions_required', 100)} dec"
            )
            lines.append(f"  🧪 {c.get('name', '?')}: {progress}")
    else:
        lines.append("  none")
    lines.append("")

    lines.append("Drift")
    if drift_alerts:
        lines.extend(f"  • {a}" for a in drift_alerts)
    else:
        lines.append("  none")

    return _cap("\n".join(lines))


def event_alert(kind: str, text: str) -> str:
    emoji = _EVENT_EMOJI.get(kind, _EVENT_EMOJI["info"])
    return _cap(f"{emoji} {text}")

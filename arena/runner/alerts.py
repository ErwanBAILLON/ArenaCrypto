"""Turn queued alerts into Telegram messages, under the daily cap."""

from __future__ import annotations

import re
from datetime import datetime

import psycopg

from arena.core.types import Alert, CompetitorSpec
from arena.settings import Settings
from arena.store import books as bstore
from arena.store import registry
from arena.telegram import sender, templates

ALWAYS_SEND = {"promotion", "stale", "error", "live_exit"}  # never folded into the digest
_RE_WAS = re.compile(r"\(was (.+?)\)")


def _spec_map(conn: psycopg.Connection) -> dict[int, CompetitorSpec]:
    return {s.id: s for s in registry.list_competitors(conn) if s.id is not None}


def render(alert: Alert, names: dict[int, str], specs: dict[int, CompetitorSpec] | None = None) -> str:
    p = alert.payload
    spec = (specs or {}).get(alert.competitor_id or -1)
    name = names.get(alert.competitor_id or -1, str(alert.competitor_id))
    if alert.kind == "signal":
        return templates.signal_alert(
            name=name,
            symbol=alert.symbol or "?",
            weight=float(p.get("weight", 0.0)),
            conviction=float(p.get("conviction", 0.0)),
            kind=str(p.get("kind", "perp")),
            regime=p.get("regime"),
            funding_8h=p.get("funding_8h"),
            reason=dict(p.get("reason", {})),
            family=spec.family if spec else None,
            version=spec.version if spec else None,
            status=spec.status if spec else None,
        )
    detail = str(p.get("detail", p))
    if alert.kind == "promotion":
        m = _RE_WAS.search(detail)
        family = spec.family if spec else templates.split_name(name)[0]
        payload = {
            **p,
            "challenger": name if spec else detail.split(" promoted", 1)[0],
            "family": family,
            "old_champion": m.group(1) if m else None,
        }
        return templates.event_alert("promotion", detail, payload)
    return templates.event_alert(alert.kind, detail, p)


def flush(conn: psycopg.Connection, settings: Settings, now: datetime, names: dict[int, str]) -> int:
    """Send unsent alerts; surplus beyond the daily cap stays queued for the digest."""
    budget = settings.max_alerts_per_day - bstore.alerts_sent_today(conn, now)
    pending = bstore.unsent_alerts(conn)
    specs = _spec_map(conn) if pending else {}
    sent_ids: list[int] = []
    for alert_id, alert, _ts in pending:
        if alert.kind not in ALWAYS_SEND and budget <= 0:
            continue
        if sender.send(settings, render(alert, names, specs)):
            sent_ids.append(alert_id)
            if alert.kind not in ALWAYS_SEND:
                budget -= 1
    bstore.mark_sent(conn, sent_ids)
    conn.commit()
    return len(sent_ids)

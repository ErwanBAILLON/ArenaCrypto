"""Turn queued alerts into Telegram messages, under the daily cap."""

from __future__ import annotations

from datetime import datetime

import psycopg

from arena.core.types import Alert
from arena.settings import Settings
from arena.store import books as bstore
from arena.telegram import sender, templates

ALWAYS_SEND = {"promotion", "stale", "error"}  # never folded into the digest


def render(alert: Alert, names: dict[int, str]) -> str:
    p = alert.payload
    if alert.kind == "signal":
        return templates.signal_alert(
            name=names.get(alert.competitor_id or -1, str(alert.competitor_id)),
            symbol=alert.symbol or "?",
            weight=float(p.get("weight", 0.0)),
            conviction=float(p.get("conviction", 0.0)),
            kind=str(p.get("kind", "perp")),
            regime=p.get("regime"),
            funding_8h=p.get("funding_8h"),
            reason=dict(p.get("reason", {})),
        )
    return templates.event_alert(alert.kind, str(p.get("detail", p)))


def flush(conn: psycopg.Connection, settings: Settings, now: datetime, names: dict[int, str]) -> int:
    """Send unsent alerts; surplus beyond the daily cap stays queued for the digest."""
    budget = settings.max_alerts_per_day - bstore.alerts_sent_today(conn, now)
    sent_ids: list[int] = []
    for alert_id, alert, _ts in bstore.unsent_alerts(conn):
        if alert.kind not in ALWAYS_SEND and budget <= 0:
            continue
        if sender.send(settings, render(alert, names)):
            sent_ids.append(alert_id)
            if alert.kind not in ALWAYS_SEND:
                budget -= 1
    bstore.mark_sent(conn, sent_ids)
    conn.commit()
    return len(sent_ids)

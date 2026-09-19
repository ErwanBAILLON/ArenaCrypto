"""Telegram sendMessage wrapper. Never raises; rate limiting is the caller's job."""

from __future__ import annotations

import logging

import httpx

from arena.settings import Settings

log = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/sendMessage"
TIMEOUT = 10.0


def send(settings: Settings, text: str, client: httpx.Client | None = None) -> bool:
    """Send ``text`` to the configured chat. Returns True on success.

    In dry-run mode, or when no bot token is configured, the message is printed
    and True is returned.
    """
    if settings.dry_run or not settings.telegram_bot_token:
        print(f"[telegram dry] {text}")
        return True

    url = API.format(token=settings.telegram_bot_token)
    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    own = client is None
    c = client or httpx.Client(timeout=TIMEOUT)
    try:
        r = c.post(url, json=payload, timeout=TIMEOUT)
        if r.status_code != 200:
            log.warning("telegram sendMessage HTTP %s: %s", r.status_code, r.text[:200])
            return False
        ok = bool(r.json().get("ok", False))
        if not ok:
            log.warning("telegram sendMessage not ok: %s", r.text[:200])
        return ok
    except Exception as exc:  # noqa: BLE001 - never raise from alerting
        log.warning("telegram sendMessage failed: %s", exc)
        return False
    finally:
        if own:
            c.close()

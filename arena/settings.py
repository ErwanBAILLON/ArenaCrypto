"""Runtime settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    telegram_bot_token: str
    telegram_chat_id: str
    dry_run: bool
    universe_path: str | None
    max_alerts_per_day: int = 20

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=os.environ.get("DATABASE_URL", ""),
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
            dry_run=os.environ.get("DRY_RUN", "false").lower() in ("1", "true", "yes"),
            universe_path=os.environ.get("UNIVERSE_PATH") or None,
            max_alerts_per_day=int(os.environ.get("MAX_ALERTS_PER_DAY", "20")),
        )

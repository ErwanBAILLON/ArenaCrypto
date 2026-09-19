"""High-impact macro events for the current week (ForexFactory public JSON mirror)."""

from __future__ import annotations

from datetime import datetime

import httpx
import pandas as pd

from arena.data.http import get_json

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
KEEP_IMPACT = {"High"}
COLUMNS = ["ts", "currency", "title", "impact"]


def calendar(client: httpx.Client, now: datetime) -> pd.DataFrame:
    """This week's High-impact events as DataFrame[ts, currency, title, impact], ts UTC.

    `now` is accepted for call-site symmetry with the other adapters (the
    caller stamps `fetched_at`); the endpoint itself has no time parameter.
    """
    events = get_json(client, CALENDAR_URL)
    kept = [e for e in events if e.get("impact") in KEEP_IMPACT]
    if not kept:
        return pd.DataFrame(
            {
                "ts": pd.Series(dtype="datetime64[ns, UTC]"),
                "currency": pd.Series(dtype="object"),
                "title": pd.Series(dtype="object"),
                "impact": pd.Series(dtype="object"),
            }
        )
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime([e["date"] for e in kept], utc=True),
            "currency": [e.get("country", "") for e in kept],
            "title": [e.get("title", "") for e in kept],
            "impact": [e["impact"] for e in kept],
        }
    )
    return out.sort_values("ts").reset_index(drop=True)[COLUMNS]

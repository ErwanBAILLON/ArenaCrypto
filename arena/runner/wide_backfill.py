"""Building the wide, survivorship-free universe: probe daily, then fetch hourly.

Downloading hourly history for all 864 archived perpetuals would be tens of
thousands of requests for a universe of fifty. So it happens in two passes:

1. **Probe.** Daily bars for every candidate, which is one archive file per
   symbol-month and cheap. Enough to rank by dollar volume at every rebalance
   date and decide who was ever a member.
2. **Fetch.** Hourly bars only for the union of everyone who made the cut.

The order matters: the probe must cover *dead* symbols too, or the ranking is
computed among survivors and the whole exercise collapses back into the bias it
exists to remove.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from arena.core.membership import DEFAULT_RULE, Member, MembershipRule, rebalance_dates, select
from arena.data import binance_archive as archive
from arena.data.http import make_client

log = logging.getLogger(__name__)
DAILY_BARS_PER_DAY = 1


@dataclass
class WideBuild:
    members_by_date: dict[pd.Timestamp, list[Member]]
    probed: int
    candidates: int

    @property
    def symbols(self) -> list[str]:
        """Everyone who was a member at least once -- the hourly download list."""
        out: set[str] = set()
        for members in self.members_by_date.values():
            out.update(m.symbol for m in members)
        return sorted(out)

    @property
    def churn(self) -> float:
        """Mean share of the universe replaced between consecutive rebalances."""
        dates = sorted(self.members_by_date)
        if len(dates) < 2:
            return 0.0
        rates = []
        for a, b in zip(dates, dates[1:], strict=False):
            before = {m.symbol for m in self.members_by_date[a]}
            after = {m.symbol for m in self.members_by_date[b]}
            if before:
                rates.append(len(before - after) / len(before))
        return float(sum(rates) / len(rates)) if rates else 0.0


def _fetch_one(symbol: str, start: datetime, end: datetime, interval: str, timeout: float) -> tuple[str, pd.DataFrame]:
    with make_client(timeout=timeout) as client:
        try:
            return symbol, archive.history(client, symbol, start, end, interval)
        except Exception:
            log.exception("archive history failed for %s", symbol)
            return symbol, pd.DataFrame()


def fetch_many(
    symbols: list[str],
    start: datetime,
    end: datetime,
    interval: str = "1d",
    workers: int = 12,
    timeout: float = 60.0,
    on_done=None,
) -> dict[str, pd.DataFrame]:
    """Archive history for many symbols, in parallel, skipping the ones that fail."""
    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_one, s, start, end, interval, timeout) for s in symbols]
        for i, future in enumerate(futures, 1):
            symbol, frame = future.result()
            if not frame.empty:
                out[symbol] = frame
            if on_done:
                on_done(i, len(symbols), symbol, len(frame))
    return out


def to_wide(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(closes, volumes)`` wide frames from per-symbol histories."""
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    closes = pd.DataFrame({s: f.set_index("ts")["close"] for s, f in frames.items()}).sort_index()
    volumes = pd.DataFrame({s: f.set_index("ts")["volume"] for s, f in frames.items()}).sort_index()
    return closes, volumes


def build_membership(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    start: datetime,
    end: datetime,
    rule: MembershipRule = DEFAULT_RULE,
    bars_per_day: int = DAILY_BARS_PER_DAY,
) -> dict[pd.Timestamp, list[Member]]:
    """Weekly point-in-time membership from the daily probe."""
    out: dict[pd.Timestamp, list[Member]] = {}
    for ts in rebalance_dates(start, end):
        members = select(closes, volumes, ts, rule, bars_per_day)
        if members:
            out[ts] = members
    return out


def probe_and_select(
    start: datetime,
    end: datetime,
    rule: MembershipRule = DEFAULT_RULE,
    quote: str = "USDT",
    workers: int = 12,
    on_done=None,
) -> WideBuild:
    """Enumerate every archived perpetual, probe it daily, and decide membership."""
    with make_client(timeout=60.0) as client:
        candidates = archive.tradable_symbols(client, quote)
    frames = fetch_many(candidates, start, end, "1d", workers, on_done=on_done)
    closes, volumes = to_wide(frames)
    members = build_membership(closes, volumes, start, end, rule)
    return WideBuild(members_by_date=members, probed=len(frames), candidates=len(candidates))

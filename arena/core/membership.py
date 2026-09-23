"""Point-in-time universe membership: who was tradable, as judged on the day.

Taking today's most liquid perps and backtesting them over two years selects
the survivors. Binance's archive holds 864 USDT perpetual symbols and 525 still
trade, so that choice silently discards ~40 % of the cross-section -- and the
discarded part is the part that went to zero.

Membership here is decided only from bars dated at or before the decision
timestamp. A symbol qualifies when it actually traded through the trailing
window and cleared a dollar-volume floor; it stops qualifying the moment its
bars stop arriving, which is what being delisted looks like from the inside.

The rule is deliberately dumb and deterministic: rank by trailing dollar volume,
take the top N above a floor. Anything cleverer is a parameter that would need
its own trial counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from arena.core.costs import SymbolLiquidity


@dataclass(frozen=True)
class MembershipRule:
    top_n: int = 50
    min_adv_usd: float = 5_000_000.0
    window_days: int = 30
    min_coverage: float = 0.9  # share of the trailing window that must carry bars
    max_stale_bars: int = 6  # a symbol silent longer than this at ts is not trading


DEFAULT_RULE = MembershipRule()


@dataclass(frozen=True)
class Member:
    symbol: str
    rank: int
    adv_usd: float
    daily_vol: float

    @property
    def liquidity(self) -> SymbolLiquidity:
        return SymbolLiquidity(adv_usd=self.adv_usd, daily_vol=self.daily_vol)


def _utc(ts: datetime) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def select(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    ts: datetime,
    rule: MembershipRule = DEFAULT_RULE,
    bars_per_day: int = 24,
) -> list[Member]:
    """Members of the universe as of ``ts``, ranked by trailing dollar volume.

    ``closes`` and ``volumes`` are wide frames indexed by bar timestamp with one
    column per symbol. Everything after ``ts`` is dropped before anything else
    happens, so this function cannot see the future even if handed it.
    """
    at = _utc(ts)
    closes = closes.loc[closes.index <= at]
    volumes = volumes.reindex(closes.index)
    if closes.empty:
        return []
    window = max(int(rule.window_days * bars_per_day), 2)
    closes = closes.tail(window)
    volumes = volumes.tail(window)
    needed = max(int(rule.min_coverage * len(closes)), 2)

    rows: list[Member] = []
    for sym in closes.columns:
        close = closes[sym].dropna()
        if len(close) < needed:
            continue  # too young, or already gone
        last_gap = len(closes) - closes.index.get_indexer([close.index[-1]])[0] - 1
        if last_gap > rule.max_stale_bars:
            continue  # bars stopped arriving before ts: not trading any more
        volume = volumes[sym].reindex(close.index).fillna(0.0)
        days = max(len(close) / bars_per_day, 1e-9)
        adv = float((close * volume).sum() / days)
        if not np.isfinite(adv) or adv < rule.min_adv_usd:
            continue
        per_bar = np.log(close).diff().dropna()
        daily_vol = float(per_bar.std(ddof=1) * np.sqrt(bars_per_day)) if len(per_bar) > 1 else float("nan")
        if not np.isfinite(daily_vol) or daily_vol <= 0:
            continue
        rows.append(Member(symbol=sym, rank=0, adv_usd=adv, daily_vol=daily_vol))

    rows.sort(key=lambda m: (-m.adv_usd, m.symbol))
    return [Member(m.symbol, i, m.adv_usd, m.daily_vol) for i, m in enumerate(rows[: rule.top_n])]


def rebalance_dates(start: datetime, end: datetime, weekday: int = 0, hour: int = 0) -> list[pd.Timestamp]:
    """Weekly membership review dates in ``[start, end]`` (Mondays 00:00 UTC by default)."""
    days = pd.date_range(_utc(start).normalize(), _utc(end).normalize(), freq="1D", tz="UTC")
    return [d + pd.Timedelta(hours=hour) for d in days if d.weekday() == weekday]


def exits(previous: list[Member] | list[str], current: list[Member] | list[str]) -> list[str]:
    """Symbols that were members and no longer are.

    A delisting is an exit, not a disappearance: the caller must close these at
    the last observed price and take the loss. Dropping them silently is how a
    backtest collects returns nobody could have collected.
    """

    def names(xs):
        return {x if isinstance(x, str) else x.symbol for x in xs}

    return sorted(names(previous) - names(current))

"""Backtest a competitor on stored history with the live ``Book`` accounting.

One ``Snapshot`` is built for the whole history and re-viewed at each bar with
``Snapshot.at(ts)`` (a cheap view limited to ``ts``), so the only per-bar cost
is the competitor's ``decide``. Prices and funding are pre-aligned to the
hourly bar grid as wide frames; the per-bar work is dict construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd

from arena.book.book import Book, FeeModel
from arena.core.snapshot import Snapshot
from arena.core.types import BookRow, Decision

REFERENCE_SYMBOL = "BTC"


class Competitor(Protocol):
    """Duck-typed contract; concrete families live in ``arena.competitors``."""

    def warmup_bars(self) -> int: ...

    def decide(self, snap: Snapshot) -> Decision: ...


@dataclass
class HistoryFrames:
    """Long frames as returned by the store (columns include ``symbol`` and ``ts``)."""

    candles: pd.DataFrame  # symbol, ts, open, high, low, close, volume (1h)
    funding: pd.DataFrame | None = None  # symbol, ts, rate (8h stamps)
    open_interest: pd.DataFrame | None = None  # symbol, ts, oi
    news: pd.DataFrame | None = None  # symbol, ts, sent_24h, sent_7d, n_24h, n_7d, shock
    macro_events: pd.DatetimeIndex | None = None


@dataclass
class BacktestResult:
    returns: pd.Series  # index ts (UTC), simple period return per 1h bar
    nav: pd.Series
    rows: list[BookRow]
    decisions: int  # bars where the decision had >= 1 non-zero target
    turnover: float  # sum of per-bar turnover


def _utc(ts: datetime | str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _wide_closes(candles: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """``ts x symbol`` frame of closes over every bar present in ``candles``."""
    c = candles[candles["symbol"].isin(symbols)]
    wide = c.pivot(index="ts", columns="symbol", values="close").sort_index()
    idx = pd.DatetimeIndex(wide.index)
    wide.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return wide.astype(float)


def _wide_funding(funding: pd.DataFrame | None, bars: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """Funding accrued per bar: rate stamped at ``f.ts`` is booked on the first bar ``>= f.ts``.

    Bar ``i`` therefore carries ``sum(rate for prev_bar < f.ts <= bar_i)``. Stamps
    after the last bar are dropped; stamps at or before the first bar land on
    it, where the book holds no position yet.
    """
    out = pd.DataFrame(0.0, index=bars, columns=symbols)
    if funding is None or funding.empty:
        return out
    f = funding[funding["symbol"].isin(symbols)]
    if f.empty:
        return out
    fts = pd.DatetimeIndex(f["ts"])
    fts = fts.tz_localize("UTC") if fts.tz is None else fts.tz_convert("UTC")
    pos = bars.searchsorted(fts, side="left")
    keep = pos < len(bars)
    acc = (
        pd.DataFrame(
            {
                "i": pos[keep],
                "symbol": f["symbol"].to_numpy()[keep],
                "rate": f["rate"].to_numpy(dtype=float)[keep],
            }
        )
        .groupby(["i", "symbol"])["rate"]
        .sum()
    )
    for (i, sym), rate in acc.items():
        out.iat[int(i), out.columns.get_loc(sym)] += rate
    return out


def _row_dict(cols: list[str], values: np.ndarray) -> dict[str, float]:
    return {s: float(v) for s, v in zip(cols, values, strict=True) if not np.isnan(v)}


def _members_on(schedule: list, table: dict | None, ts):
    """The entry in force at ``ts``: the latest scheduled date at or before it."""
    if not table or not schedule:
        return None
    i = int(np.searchsorted(np.array(schedule), ts, side="right")) - 1
    return table[schedule[i]] if i >= 0 else None


def _has_targets(decision: Decision) -> bool:
    return any(t.weight != 0.0 for t in decision.values())


def run(
    competitor: Any,
    history: HistoryFrames,
    symbols: list[str],
    start: datetime | str,
    end: datetime | str,
    fees: FeeModel,
    nav0: float = 10_000.0,
    bar_hours: int = 1,
    members_at: dict | None = None,
    liquidity_at: dict | None = None,
) -> BacktestResult:
    """Step one ``Book`` over every hourly bar in ``[start, end]`` present in the candles.

    Bars are skipped until warm-up is satisfied: every symbol that has data
    must expose at least ``competitor.warmup_bars()`` bars, or the reference
    symbol (``BTC`` when in the universe, else the first symbol) does. Bars
    before ``start`` still count toward warm-up, which is what lets a
    walk-forward fold decide from its first test bar.
    """
    start_ts, end_ts = _utc(start), _utc(end)
    closes = _wide_closes(history.candles, symbols)
    bars = pd.DatetimeIndex(closes.index)
    cols = [str(c) for c in closes.columns]
    close_arr = closes.to_numpy()
    fund_arr = _wide_funding(history.funding, bars, cols).to_numpy()
    counts = np.cumsum(~np.isnan(close_arr), axis=0)  # bars available per symbol at each bar
    has_data = counts[-1] > 0 if len(bars) else np.zeros(len(cols), dtype=bool)
    # the reference is whatever has the longest history, so warm-up is not
    # gated by a symbol that listed last month
    if REFERENCE_SYMBOL in cols:
        ref_i = cols.index(REFERENCE_SYMBOL)
    elif cols:
        ref_i = int(np.argmax(counts[-1])) if len(bars) else None
    else:
        ref_i = None
    member_days = sorted(members_at) if members_at else []
    liquidity_days = sorted(liquidity_at) if liquidity_at else []

    full = Snapshot.from_long(
        end_ts,
        symbols,
        history.candles,
        history.funding,
        history.open_interest,
        news=history.news,
        macro_events=history.macro_events,
        bar_hours=bar_hours,
    )
    warm = int(competitor.warmup_bars())
    book = Book(nav=nav0, fees=fees)
    rows: list[BookRow] = []
    decisions = 0

    lo = int(bars.searchsorted(start_ts, side="left"))
    hi = int(bars.searchsorted(end_ts, side="right"))
    for i in range(lo, hi):
        all_ready = bool(np.all(counts[i][has_data] >= warm)) if has_data.any() else False
        ref_ready = ref_i is not None and counts[i][ref_i] >= warm
        if not (all_ready or ref_ready):
            continue
        ts = bars[i]
        visible = _members_on(member_days, members_at, ts)
        decision = competitor.decide(full.at(ts, visible))
        if _has_targets(decision):
            decisions += 1
        prices = _row_dict(cols, close_arr[i])
        prev_prices = _row_dict(cols, close_arr[i - 1]) if i > 0 else {}
        funding = {s: float(v) for s, v in zip(cols, fund_arr[i], strict=True) if v != 0.0}
        liquidity = _members_on(liquidity_days, liquidity_at, ts)
        rows.append(book.step(ts.to_pydatetime(), prices, prev_prices, funding, decision, liquidity))

    idx = pd.DatetimeIndex([r.ts for r in rows], tz="UTC", name="ts")
    returns = pd.Series([r.ret for r in rows], index=idx, dtype=float, name="ret")
    nav = pd.Series([r.nav for r in rows], index=idx, dtype=float, name="nav")
    return BacktestResult(
        returns=returns,
        nav=nav,
        rows=rows,
        decisions=decisions,
        turnover=float(sum(r.turnover for r in rows)),
    )

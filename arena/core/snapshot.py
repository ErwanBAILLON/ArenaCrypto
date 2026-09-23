"""Point-in-time view of the market handed to competitors.

Every accessor returns data with ``ts <= snapshot.ts`` only. Competitors never
touch the database; they only see a Snapshot. Frames are stored in long format
once and sliced on access, so building one Snapshot per bar in a backtest is
cheap.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

import pandas as pd

CANDLE_COLS = ["open", "high", "low", "close", "volume"]
MAX_BARS = 6000  # 250 days of 1h bars: above the longest warm-up (regime, 5041), bounds per-bar indicator cost
NEWS_COLS = ["sent_24h", "sent_7d", "n_24h", "n_7d", "shock"]
_TF_RULE = {"1h": "1h", "4h": "4h", "1d": "1D"}


def _utc(ts: datetime) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _empty_candles() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype=float) for c in CANDLE_COLS}, index=idx)


def _empty_series() -> pd.Series:
    return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC", name="ts"))


def resample_candles(c1h: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate 1h candles into 4h or 1d bars labelled by their *close* time.

    A bar labelled 04:00 contains the 1h candles labelled 01:00..04:00 (each 1h
    candle is itself labelled by its close). Partial trailing bars are dropped.
    """
    if tf == "1h" or c1h.empty:
        return c1h
    rule = _TF_RULE[tf]
    # candles are labelled by close time -> shift back one hour so that a
    # candle labelled 04:00 (covering 03:00-04:00) falls in the 00:00-04:00 bin.
    shifted = c1h.copy()
    shifted.index = shifted.index - pd.Timedelta(hours=1)
    agg = shifted.resample(rule, label="right", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    counts = shifted["close"].resample(rule, label="right", closed="left").count()
    per_bar = {"4h": 4, "1d": 24}[tf]
    agg = agg[counts == per_bar]
    agg.index.name = "ts"
    return agg.dropna()


class Snapshot:
    def __init__(
        self,
        ts: datetime,
        symbols: Iterable[str],
        candles_1h: dict[str, pd.DataFrame],
        funding: dict[str, pd.Series] | None = None,
        open_interest: dict[str, pd.Series] | None = None,
        hl_funding: dict[str, float] | None = None,
        news: dict[str, pd.DataFrame] | None = None,
        macro_events: pd.DatetimeIndex | None = None,
        bar_hours: int = 1,
    ):
        self.ts = _utc(ts)
        self.bar_hours = int(bar_hours)
        self.symbols = list(symbols)
        self._c1h = candles_1h
        self._funding = funding or {}
        self._oi = open_interest or {}
        self._hl = hl_funding or {}
        self._news = news or {}
        self._macro = macro_events if macro_events is not None else pd.DatetimeIndex([], tz="UTC")
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_long(
        cls,
        ts: datetime,
        symbols: Iterable[str],
        candles: pd.DataFrame,
        funding: pd.DataFrame | None = None,
        open_interest: pd.DataFrame | None = None,
        hl_funding: dict[str, float] | None = None,
        news: pd.DataFrame | None = None,
        macro_events: pd.DatetimeIndex | None = None,
        bar_hours: int = 1,
    ) -> Snapshot:
        """Build from long frames (columns include ``symbol`` and ``ts``).

        candles: symbol, ts, open, high, low, close, volume (1h only)
        funding: symbol, ts, rate ; open_interest: symbol, ts, oi
        news: symbol, ts, sent_24h, sent_7d, n_24h, n_7d, shock
        """
        c1h: dict[str, pd.DataFrame] = {}
        if candles is not None and not candles.empty:
            for sym, g in candles.groupby("symbol"):
                f = g.set_index("ts")[CANDLE_COLS].sort_index()
                f.index = (
                    pd.DatetimeIndex(f.index).tz_convert("UTC")
                    if f.index.tz
                    else pd.DatetimeIndex(f.index).tz_localize("UTC")
                )
                f.index.name = "ts"
                c1h[str(sym)] = f.astype(float)
        fund = {}
        if funding is not None and not funding.empty:
            for sym, g in funding.groupby("symbol"):
                s = g.set_index("ts")["rate"].sort_index().astype(float)
                s.index = _to_utc_index(s.index)
                fund[str(sym)] = s
        oi = {}
        if open_interest is not None and not open_interest.empty:
            for sym, g in open_interest.groupby("symbol"):
                s = g.set_index("ts")["oi"].sort_index().astype(float)
                s.index = _to_utc_index(s.index)
                oi[str(sym)] = s
        nw = {}
        if news is not None and not news.empty:
            for sym, g in news.groupby("symbol"):
                f = g.set_index("ts")[NEWS_COLS].sort_index().astype(float)
                f.index = _to_utc_index(f.index)
                nw[str(sym)] = f
        return cls(ts, symbols, c1h, fund, oi, hl_funding, nw, macro_events, bar_hours=bar_hours)

    def at(self, ts: datetime, symbols: list[str] | None = None) -> Snapshot:
        """Cheap view of the same data at another decision time, optionally narrowed.

        ``symbols`` restricts what the competitor can see, which is how a
        point-in-time universe is enforced: on a bar where a symbol was not a
        member, it simply is not in the snapshot, so no rule can trade it and
        none has to know the membership rule exists.
        """
        s = Snapshot.__new__(Snapshot)
        s.ts = _utc(ts)
        s.bar_hours = self.bar_hours
        s.symbols = list(symbols) if symbols is not None else self.symbols
        s._c1h, s._funding, s._oi, s._hl, s._news, s._macro = (
            self._c1h,
            self._funding,
            self._oi,
            self._hl,
            self._news,
            self._macro,
        )
        s._cache = {}
        return s

    # ----------------------------------------------------------------- accessors
    @property
    def bars_per_day(self) -> int:
        return 24 // self.bar_hours

    @property
    def reference(self) -> str:
        return "BTC" if "BTC" in self.symbols else (self.symbols[0] if self.symbols else "BTC")

    @property
    def bars_per_year(self) -> int:
        return 365 * self.bars_per_day

    def candles(self, symbol: str, tf: str = "1h") -> pd.DataFrame:
        """Bars at ``tf``; ``tf="1h"`` means "the universe's native bar" (1h or 1d)."""
        if tf not in _TF_RULE:
            raise ValueError(f"unknown tf {tf}")
        native = {1: "1h", 24: "1d"}[self.bar_hours]
        if tf == native or tf == "1h":
            tf = "1h"  # native bars are stored under the 1h key whatever their size
        elif self.bar_hours == 24:
            raise ValueError(f"tf {tf} is finer than the daily bars of this universe")
        key = (symbol, tf)
        if key in self._cache:
            return self._cache[key]
        base = self._c1h.get(symbol)
        if base is None or base.empty:
            out = _empty_candles()
        else:
            cut = base.loc[: self.ts].tail(MAX_BARS)
            out = resample_candles(cut, tf) if tf != "1h" else cut
        self._cache[key] = out
        return out

    def funding(self, symbol: str) -> pd.Series:
        s = self._funding.get(symbol)
        return _empty_series() if s is None else s.loc[: self.ts]

    def hl_funding(self, symbol: str) -> float | None:
        return self._hl.get(symbol)

    def open_interest(self, symbol: str) -> pd.Series:
        s = self._oi.get(symbol)
        return _empty_series() if s is None else s.loc[: self.ts]

    def news(self, symbol: str) -> pd.DataFrame:
        f = self._news.get(symbol)
        if f is None:
            idx = pd.DatetimeIndex([], tz="UTC", name="ts")
            return pd.DataFrame({c: pd.Series(dtype=float) for c in NEWS_COLS}, index=idx)
        return f.loc[: self.ts]

    def macro_today(self, window_hours: int = 12) -> bool:
        if len(self._macro) == 0:
            return False
        lo = self.ts - timedelta(hours=window_hours)
        hi = self.ts + timedelta(hours=window_hours)
        return bool(((self._macro >= lo) & (self._macro <= hi)).any())

    def close(self, symbol: str) -> float:
        c = self.candles(symbol)
        return float(c["close"].iloc[-1]) if not c.empty else float("nan")

    def closes(self) -> pd.DataFrame:
        cols = {}
        for sym in self.symbols:
            c = self.candles(sym)
            if not c.empty:
                cols[sym] = c["close"]
        if not cols:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts"))
        return pd.DataFrame(cols).sort_index()

    def has(self, symbol: str, min_bars: int) -> bool:
        return len(self.candles(symbol)) >= min_bars


def _to_utc_index(idx) -> pd.DatetimeIndex:
    di = pd.DatetimeIndex(idx)
    di = di.tz_localize("UTC") if di.tz is None else di.tz_convert("UTC")
    di.name = "ts"
    return di

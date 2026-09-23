"""Binance public data archive (data.binance.vision): bulk history, including delisted symbols.

The REST klines endpoint only answers for symbols that still exist, so a
universe built from it is a universe of survivors. The public archive keeps the
dead ones: of the 864 USDT perpetual symbols it holds, only about 525 still
trade. Selecting today's most liquid perps and backtesting them over two years
therefore discards ~40 % of the cross-section, and the discarded part is the
part that went to zero.

This module is the bulk path used to build history and a point-in-time
universe. ``arena.data.binance`` stays the incremental path for the live tick.

Layout: ``data/futures/um/monthly/klines/<SYMBOL>/1h/<SYMBOL>-1h-YYYY-MM.zip``,
each a CSV of the same twelve columns the REST endpoint returns.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from datetime import date, datetime

import httpx
import pandas as pd

from arena.data.http import get_json

ARCHIVE = "https://data.binance.vision"
LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FAPI = "https://fapi.binance.com"
KLINE_PREFIX = "data/futures/um/monthly/klines/"
COLUMNS = ["ts", "open", "high", "low", "close", "volume"]

log = logging.getLogger(__name__)
_PREFIX_RE = re.compile(r"<Prefix>" + re.escape(KLINE_PREFIX) + r"([^/]+)/</Prefix>")


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="datetime64[ns, UTC]" if c == "ts" else "float64") for c in COLUMNS})


def archived_symbols(client: httpx.Client, quote: str = "USDT") -> list[str]:
    """Every symbol the archive has ever held, live or dead, ending in ``quote``.

    Pages the S3 listing API by common prefix. This is the only source that
    knows a symbol existed after Binance has stopped serving it.
    """
    out: list[str] = []
    marker = ""
    while True:
        resp = client.get(
            LISTING, params={"delimiter": "/", "prefix": KLINE_PREFIX, "max-keys": 1000, "marker": marker}
        )
        resp.raise_for_status()
        found = _PREFIX_RE.findall(resp.text)
        if not found:
            break
        out.extend(found)
        if "<IsTruncated>true</IsTruncated>" not in resp.text:
            break
        marker = f"{KLINE_PREFIX}{found[-1]}/"
    return sorted(s for s in out if s.endswith(quote))


def crypto_perpetuals(client: httpx.Client, quote: str = "USDT") -> set[str]:
    """Symbols currently listed as *crypto* perpetuals, from ``exchangeInfo``.

    Binance now also lists equity and commodity perpetuals (201 of them, e.g.
    ``AAPLUSDT``) under ``contractType == TRADIFI_PERPETUAL``. They are not this
    arena's market and are excluded by ``underlyingType == COIN``.
    """
    info = get_json(client, f"{FAPI}/fapi/v1/exchangeInfo")
    return {
        s["symbol"]
        for s in info.get("symbols", [])
        if s.get("contractType") == "PERPETUAL" and s.get("underlyingType") == "COIN" and s.get("quoteAsset") == quote
    }


def tradable_symbols(client: httpx.Client, quote: str = "USDT") -> list[str]:
    """Archived symbols minus the ones we know are not crypto perpetuals.

    A symbol present only in the archive is delisted, so ``exchangeInfo`` cannot
    classify it. Those are kept: Binance launched TradFi perpetuals long after
    the arena's history window begins, so an unclassifiable symbol in that
    window is a dead crypto perp, which is exactly what must not be dropped.
    """
    archived = archived_symbols(client, quote)
    info = get_json(client, f"{FAPI}/fapi/v1/exchangeInfo")
    known = {s["symbol"]: s for s in info.get("symbols", [])}
    keep = []
    for sym in archived:
        meta = known.get(sym)
        if meta is None:
            keep.append(sym)  # delisted: predates the TradFi listings
        elif meta.get("contractType") == "PERPETUAL" and meta.get("underlyingType") == "COIN":
            keep.append(sym)
    return keep


def _utc_ts(ts: datetime) -> pd.Timestamp:
    """Naive datetimes are read as UTC; the archive is UTC and nothing else is meaningful here."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _months(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def monthly_klines(client: httpx.Client, symbol: str, year: int, month: int, interval: str = "1h") -> pd.DataFrame:
    """One month of klines from the archive, or an empty frame when absent.

    A missing month is normal and not an error: it means the symbol was not
    trading then, which is information the point-in-time universe needs.
    """
    url = f"{ARCHIVE}/data/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}.zip"
    resp = client.get(url)
    if resp.status_code == 404:
        return _empty()
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode("utf-8")
    rows = [r for r in csv.reader(io.StringIO(text)) if r and not r[0].startswith("open_time")]
    if not rows:
        return _empty()
    step_ms = {"1h": 3_600_000, "1d": 86_400_000}[interval]
    open_ms = [int(float(r[0])) for r in rows]
    # Archives from 2025 onward stamp microseconds; normalise to milliseconds.
    open_ms = [t // 1000 if t > 10**14 else t for t in open_ms]
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(pd.Series(open_ms, dtype="int64") + step_ms, unit="ms", utc=True),
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [float(r[5]) for r in rows],
        }
    )
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def monthly_funding(client: httpx.Client, symbol: str, year: int, month: int) -> pd.DataFrame:
    """One month of realised funding from the archive as DataFrame[ts, rate].

    Columns are ``calc_time, funding_interval_hours, last_funding_rate``. The
    interval is carried through because Binance moved several symbols from 8h
    to 4h or 1h funding, and averaging rates of different intervals without
    noticing is a silent factor-of-two error in every carry signal.
    """
    url = f"{ARCHIVE}/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year:04d}-{month:02d}.zip"
    resp = client.get(url)
    if resp.status_code == 404:
        return pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns, UTC]"), "rate": pd.Series(dtype="float64")})
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        text = zf.read(zf.namelist()[0]).decode("utf-8")
    rows = [r for r in csv.reader(io.StringIO(text)) if r and not r[0].startswith("calc_time")]
    if not rows:
        return pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns, UTC]"), "rate": pd.Series(dtype="float64")})
    stamps = [int(float(r[0])) for r in rows]
    stamps = [t // 1000 if t > 10**14 else t for t in stamps]
    hours = [float(r[1]) if len(r) > 1 and r[1] else 8.0 for r in rows]
    rates = [float(r[2]) for r in rows]
    # normalise to an 8h-equivalent rate so signals are comparable across symbols
    normalised = [rate * (8.0 / h) if h else rate for rate, h in zip(rates, hours, strict=True)]
    out = pd.DataFrame(
        {"ts": pd.to_datetime(pd.Series(stamps, dtype="int64"), unit="ms", utc=True), "rate": normalised}
    )
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def funding_history(client: httpx.Client, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Concatenated monthly funding archives covering ``[start, end]``."""
    frames = []
    for year, month in _months(start.date(), end.date()):
        try:
            frame = monthly_funding(client, symbol, year, month)
        except Exception:
            log.exception("funding archive failed for %s %04d-%02d", symbol, year, month)
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns, UTC]"), "rate": pd.Series(dtype="float64")})
    out = pd.concat(frames, ignore_index=True)
    lo, hi = _utc_ts(start), _utc_ts(end)
    out = out[(out["ts"] >= lo) & (out["ts"] <= hi)]
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def history(client: httpx.Client, symbol: str, start: datetime, end: datetime, interval: str = "1h") -> pd.DataFrame:
    """Concatenated monthly archives covering ``[start, end]`` for one symbol."""
    frames = []
    for year, month in _months(start.date(), end.date()):
        try:
            frame = monthly_klines(client, symbol, year, month, interval)
        except Exception:  # one bad month must not lose the symbol
            log.exception("archive fetch failed for %s %04d-%02d", symbol, year, month)
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return _empty()
    out = pd.concat(frames, ignore_index=True)
    lo, hi = _utc_ts(start), _utc_ts(end)
    out = out[(out["ts"] >= lo) & (out["ts"] <= hi)]
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

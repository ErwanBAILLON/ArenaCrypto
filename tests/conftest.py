"""Shared fixtures: synthetic market history and an optional Postgres."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from arena.core.snapshot import Snapshot

SYMBOLS = ["BTC", "ETH", "SOL"]


def make_candles(
    symbols=SYMBOLS,
    bars: int = 24 * 400,
    seed: int = 0,
    drift: dict[str, float] | None = None,
    start: str = "2024-01-01T01:00:00Z",
) -> pd.DataFrame:
    """Long-format synthetic 1h candles (random walk, optional per-symbol drift per bar)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=bars, freq="1h", tz="UTC")
    frames = []
    for i, sym in enumerate(symbols):
        mu = (drift or {}).get(sym, 0.0)
        r = rng.normal(mu, 0.01, size=bars)
        close = 100.0 * (1 + i) * np.exp(np.cumsum(r))
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, bars))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, bars))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "ts": idx,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": rng.uniform(1e5, 1e6, bars),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_funding(
    symbols=SYMBOLS,
    candles: pd.DataFrame | None = None,
    rate: float = 0.0001,
    per_symbol: dict[str, float] | None = None,
) -> pd.DataFrame:
    ts = (
        pd.DatetimeIndex(sorted(candles["ts"].unique()))
        if candles is not None
        else pd.date_range("2024-01-01", periods=1200, freq="8h", tz="UTC")
    )
    ts8 = ts[ts.hour % 8 == 0]
    rows = []
    for sym in symbols:
        r = (per_symbol or {}).get(sym, rate)
        rows.append(pd.DataFrame({"symbol": sym, "ts": ts8, "rate": r}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def candles() -> pd.DataFrame:
    return make_candles()


@pytest.fixture
def snapshot(candles) -> Snapshot:
    ts = candles["ts"].max()
    return Snapshot.from_long(ts, SYMBOLS, candles, make_funding(candles=candles))


@pytest.fixture
def pg_url() -> str:
    url = os.environ.get("PG_TEST_URL")
    if not url:
        pytest.skip("PG_TEST_URL not set")
    return url


@pytest.fixture
def conn(pg_url):
    from arena.store.db import connect, run_migrations

    c = connect(pg_url)
    with c.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    c.commit()
    run_migrations(c)
    yield c
    c.rollback()
    c.close()

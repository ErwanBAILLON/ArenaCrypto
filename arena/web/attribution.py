"""Where a competitor wins and where it loses, derived from what it already stored.

No new write path. A position episode is a maximal run of consecutive bars on
which a competitor held a non-zero weight in one symbol, and everything else --
entry, exit, holding time, gross and excess return, which slice it belongs to --
is reconstructed from ``targets`` and ``candles``. That means attribution works
retroactively for every family the arena has ever run, and cannot introduce a
failure mode into the hourly tick.

What it answers, in order of usefulness:

* **the journal** -- what was predicted, how strongly, what happened;
* **the decomposition** -- P&L by symbol, by conviction decile, by holding time,
  by side, by volatility. A model that earns everything on three symbols is not
  a model, and a model that only earns on its low-conviction trades does not
  know what it knows;
* **calibration** -- when it says 0.7, does it happen 70 % of the time? This is
  the diagnostic almost nobody draws, and it is the one that decides whether
  conviction may be used for sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import psycopg

DECILES = 5  # quintiles: deciles are noise at the sample sizes this arena has


@dataclass(frozen=True)
class Episode:
    """One held position, from the bar it opened to the bar it closed."""

    symbol: str
    entry_ts: datetime
    exit_ts: datetime
    bars_held: int
    weight: float  # average weight over the episode, signed
    conviction: float
    gross_return: float  # the symbol's own return over the episode
    excess_return: float  # the same, minus the universe's
    pnl: float  # excess_return * weight: what the position contributed

    @property
    def side(self) -> str:
        return "long" if self.weight > 0 else "short"

    @property
    def won(self) -> bool:
        return self.pnl > 0


def _targets(conn: psycopg.Connection, competitor_id: int, since: datetime | None) -> pd.DataFrame:
    sql = "SELECT ts, symbol, weight, conviction FROM targets WHERE competitor_id = %s"
    params: list[Any] = [competitor_id]
    if since is not None:
        sql += " AND ts >= %s"
        params.append(since)
    with conn.cursor() as cur:
        cur.execute(sql + " ORDER BY ts", params)
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["ts", "symbol", "weight", "conviction"]) if rows else pd.DataFrame()


def _closes(conn: psycopg.Connection, symbols: list[str] | None, since: datetime | None) -> pd.DataFrame:
    """Wide closes. ``symbols=None`` means every symbol stored, which is the benchmark."""
    sql = "SELECT ts, symbol, close FROM candles"
    clauses, params = [], []
    if symbols is not None:
        if not symbols:
            return pd.DataFrame()
        clauses.append("symbol = ANY(%s)")
        params.append(symbols)
    if since is not None:
        clauses.append("ts >= %s")
        params.append(since)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    with conn.cursor() as cur:
        cur.execute(sql + " ORDER BY ts", params)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=["ts", "symbol", "close"])
    return frame.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last").sort_index()


def episodes(conn: psycopg.Connection, competitor_id: int, since: datetime | None = None) -> list[Episode]:
    """Every position episode this competitor held, with its contribution.

    Returns are measured against an equal-weight index of the symbols in the
    book's own price history, so a long that rose less than the market counts
    as the loss it was.
    """
    targets = _targets(conn, competitor_id, since)
    if targets.empty:
        return []
    targets = targets[targets["weight"].astype(float) != 0.0]
    if targets.empty:
        return []
    closes = _closes(conn, sorted(set(targets["symbol"])), since)
    if closes.empty:
        return []
    # The benchmark is the *universe*, not the handful of names this competitor
    # happened to hold. Averaging only the traded symbols makes a single-name
    # position its own benchmark, and its excess return identically zero.
    universe = _closes(conn, None, since)
    reference = universe if not universe.empty else closes
    benchmark = (1.0 + reference.pct_change().mean(axis=1).fillna(0.0)).cumprod()

    out: list[Episode] = []
    for symbol, group in targets.groupby("symbol"):
        if symbol not in closes.columns:
            continue
        group = group.sort_values("ts")
        stamps = pd.DatetimeIndex(group["ts"])
        # a gap longer than the usual spacing ends an episode
        spacing = pd.Series(stamps).diff().median() or pd.Timedelta(hours=1)
        breaks = np.flatnonzero(pd.Series(stamps).diff() > spacing * 1.5)
        for chunk in np.split(np.arange(len(group)), breaks):
            if chunk.size == 0:
                continue
            block = group.iloc[chunk]
            entry, exit_ = pd.Timestamp(block["ts"].iloc[0]), pd.Timestamp(block["ts"].iloc[-1])
            price = closes[symbol]
            if entry not in price.index or exit_ not in price.index:
                continue
            p0, p1 = float(price.loc[entry]), float(price.loc[exit_])
            b0, b1 = float(benchmark.loc[entry]), float(benchmark.loc[exit_])
            if not p0 or not b0:
                continue
            weight = float(block["weight"].astype(float).mean())
            gross = p1 / p0 - 1.0
            excess = gross - (b1 / b0 - 1.0)
            out.append(
                Episode(
                    symbol=str(symbol),
                    entry_ts=entry.to_pydatetime(),
                    exit_ts=exit_.to_pydatetime(),
                    bars_held=int(len(block)),
                    weight=weight,
                    conviction=float(block["conviction"].astype(float).mean()),
                    gross_return=gross,
                    excess_return=excess,
                    pnl=excess * weight,
                )
            )
    return sorted(out, key=lambda e: e.entry_ts)


def to_frame(eps: list[Episode]) -> pd.DataFrame:
    if not eps:
        return pd.DataFrame(
            columns=[
                "symbol",
                "entry_ts",
                "exit_ts",
                "bars_held",
                "weight",
                "conviction",
                "excess_return",
                "pnl",
                "side",
            ]
        )
    return pd.DataFrame([{**e.__dict__, "side": e.side} for e in eps])


def decompose(eps: list[Episode], by: str, buckets: int = DECILES) -> list[dict[str, Any]]:
    """P&L split along one dimension: ``symbol``, ``side``, ``conviction``, ``bars_held``.

    Continuous dimensions are cut into quantile buckets. Every row carries the
    count as well as the total, because a bucket of three trades is an anecdote
    however large its number.
    """
    frame = to_frame(eps)
    if frame.empty:
        return []
    if by in {"symbol", "side"}:
        keys = frame[by]
    elif by in frame.columns:
        try:
            keys = pd.qcut(frame[by], q=min(buckets, frame[by].nunique()), duplicates="drop")
        except (ValueError, IndexError):
            return []
        keys = keys.astype(str)
    else:
        return []
    grouped = frame.groupby(keys, observed=True)["pnl"]
    rows = [
        {
            "bucket": str(name),
            "trades": int(len(values)),
            "pnl": float(values.sum()),
            "mean": float(values.mean()),
            "win_rate": float((values > 0).mean()),
        }
        for name, values in grouped
    ]
    return sorted(rows, key=lambda r: -r["pnl"])


def reliability(eps: list[Episode], buckets: int = DECILES) -> list[dict[str, Any]]:
    """Conviction against realised win rate: the diagram that catches a bluffer.

    A competitor whose 0.8-conviction trades win as often as its 0.2-conviction
    ones has a conviction number that means nothing, and sizing on it is
    sizing on noise.
    """
    frame = to_frame(eps)
    if frame.empty or frame["conviction"].nunique() < 2:
        return []
    try:
        cut = pd.qcut(frame["conviction"], q=min(buckets, frame["conviction"].nunique()), duplicates="drop")
    except (ValueError, IndexError):
        return []
    rows = []
    for name, group in frame.groupby(cut, observed=True):
        rows.append(
            {
                "bucket": str(name),
                "stated": float(group["conviction"].mean()),
                "observed": float((group["pnl"] > 0).mean()),
                "trades": int(len(group)),
                "pnl": float(group["pnl"].sum()),
            }
        )
    return sorted(rows, key=lambda r: r["stated"])


def calibration_error(eps: list[Episode], buckets: int = DECILES) -> float:
    """Mean absolute gap between stated conviction and realised win rate (0 is perfect)."""
    rows = reliability(eps, buckets)
    if not rows:
        return float("nan")
    weight = sum(r["trades"] for r in rows)
    return float(sum(abs(r["stated"] - r["observed"]) * r["trades"] for r in rows) / weight) if weight else float("nan")


def summary(eps: list[Episode]) -> dict[str, Any]:
    """Headline numbers, plus the concentration figure that deflates most of them."""
    frame = to_frame(eps)
    if frame.empty:
        return {"trades": 0, "pnl": 0.0, "win_rate": 0.0, "concentration": 0.0, "calibration_error": float("nan")}
    pnl = frame["pnl"]
    winners = pnl[pnl > 0].sum()
    by_symbol = frame.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
    # only positive contributions count toward concentration: adding a loser's
    # negative total to the numerator would make a concentrated book look diverse
    top3 = float(by_symbol.head(3).clip(lower=0.0).sum())
    return {
        "trades": int(len(frame)),
        "pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()),
        "mean_bars": float(frame["bars_held"].mean()),
        "best_symbol": str(by_symbol.index[0]) if len(by_symbol) else None,
        "worst_symbol": str(by_symbol.index[-1]) if len(by_symbol) else None,
        # share of all profit coming from the three best names: at 1.0 the strategy is those names
        "concentration": float(min(top3 / winners, 1.0)) if winners > 0 else 0.0,
        "calibration_error": calibration_error(eps),
    }


def worst(eps: list[Episode], n: int = 10) -> list[Episode]:
    """The trades that cost the most, which is where the next fix usually is."""
    return sorted(eps, key=lambda e: e.pnl)[:n]

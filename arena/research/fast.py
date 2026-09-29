"""The fast harness: weekly weight rules priced on hourly closes, as-paid funding and impact costs, in seconds.

Every walk-forward before this rebuilt a Snapshot and a 120-column panel at
each bar it decided on: a quarter of an hour per rule on five years. A weekly
rule only ever reads the panel at its rebalance dates, so :func:`build_store`
computes those panels once, for that date's members, and keeps them with the
hourly closes and the funding as paid. :func:`run` then holds a rule's weights
from one rebalance to the next, compounds the hourly P&L (price, minus funding
paid, minus the arena's impact cost of the rebalance at a deployment size) and
:func:`summarise` reports per quarter what the slow engine reports, plus the
decomposition gross / funding / fees that says *why* a rule lands where it does.

Validated against the slow Book engine on the same rules before anything was
believed (2026-09-28: equal-weight 0.09 vs 0.04, composite long-short -0.33
both sides). It is a research instrument, not a competitor: nothing here
decides anything in the arena.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from arena.book.book import FeeModel
from arena.core.costs import ImpactModel, SymbolLiquidity
from arena.core.snapshot import Snapshot
from arena.features import build as build_panel
from arena.judge import metrics as m

PPY_HOURLY = 8760
PANEL_LOOKBACK = pd.Timedelta(days=130)  # the panel's longest window plus slack

Rule = Callable[..., dict[str, float] | pd.Series]


@dataclass
class Store:
    """Everything a weekly rule needs to be priced: the panel per rebalance date, hourly closes, funding, liquidity."""

    panel: pd.DataFrame  # one row per (ts, symbol), the feature columns plus ts and symbol
    dates: list[pd.Timestamp]
    closes: pd.DataFrame  # index hourly ts, columns symbols
    funding: pd.DataFrame  # index funding ts, columns symbols, rate as paid
    liquidity: dict[pd.Timestamp, dict[str, SymbolLiquidity]] = field(default_factory=dict)

    def by_date(self) -> dict[pd.Timestamp, pd.DataFrame]:
        return {ts: g.set_index("symbol") for ts, g in self.panel.groupby("ts")}


def build_store(
    candles: pd.DataFrame,
    funding: pd.DataFrame | None,
    members: dict[datetime, list],
    log: Callable[[str], None] | None = None,
) -> Store:
    """Compute the panel at each rebalance date for that date's members.

    ``members`` maps a rebalance timestamp to that date's ``Member`` list (the
    membership store's schedule). ``candles`` and ``funding`` are long frames.
    """
    candles = candles.copy()
    candles["ts"] = pd.to_datetime(candles["ts"], utc=True)
    if funding is not None and not funding.empty:
        funding = funding[["symbol", "ts", "rate"]].copy()
        funding["ts"] = pd.to_datetime(funding["ts"], utc=True)
    else:
        funding = pd.DataFrame(columns=["symbol", "ts", "rate"])
    dates = sorted(pd.Timestamp(d) for d in members)
    panels = []
    liquidity: dict[pd.Timestamp, dict[str, SymbolLiquidity]] = {}
    for i, ts in enumerate(dates):
        roster = members[ts] if ts in members else members[ts.to_pydatetime()]
        names = [x.symbol for x in roster]
        liquidity[ts] = {x.symbol: x.liquidity for x in roster}
        lo = ts - PANEL_LOOKBACK
        win = candles[candles["symbol"].isin(names) & (candles["ts"] <= ts) & (candles["ts"] > lo)]
        fwin = funding[funding["symbol"].isin(names) & (funding["ts"] <= ts) & (funding["ts"] > lo)]
        if win.empty:
            continue
        try:
            panel = build_panel(Snapshot.from_long(ts, names, win, fwin if not fwin.empty else None))
        except Exception as exc:  # a thin week is data, not a reason to stop
            if log:
                log(f"skip {ts.date()} {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if panel.empty:
            continue
        panels.append(panel.assign(ts=ts, symbol=panel.index).reset_index(drop=True))
        if log and i % 26 == 0:
            log(f"{ts.date()}: {len(panels)} dates")
    if not panels:
        raise ValueError("no panel could be built: not enough history for any rebalance date")
    return Store(
        panel=pd.concat(panels, ignore_index=True),
        dates=dates,
        closes=candles.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last").sort_index(),
        funding=funding.pivot_table(index="ts", columns="symbol", values="rate", aggfunc="sum").sort_index()
        if not funding.empty
        else pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC")),
        liquidity=liquidity,
    )


def fee_model(capacity_nav: float | None) -> FeeModel:
    """The arena's perp costs at a deployment size (``None`` = flat fees, no impact)."""
    impact = ImpactModel(k=1.0, capacity_nav=float(capacity_nav)) if capacity_nav else None
    return FeeModel(0.0005, 0.0002, 0.0010, impact)


def run(store: Store, rule: Rule, fees: FeeModel, every: int = 1, start=None, end=None, **kw) -> pd.DataFrame:
    """Hourly returns of ``rule`` rebalanced every ``every`` rebalance dates (1 = weekly).

    Returns DataFrame[ret, price, fund, fee] indexed by hour. ``kw`` are the
    rule's own parameters, passed as keywords.
    """
    by_date = store.by_date()
    ret_h = store.closes.pct_change()
    fund_h = (
        store.funding.groupby(store.funding.index.floor("h")).sum().reindex(store.closes.index, fill_value=0.0)
        if not store.funding.empty
        else pd.DataFrame(0.0, index=store.closes.index, columns=store.closes.columns)
    )
    dates = [d for d in store.dates if d in by_date and (start is None or d >= start) and (end is None or d < end)]
    w = pd.Series(dtype=float)
    out = []
    ctx: dict = {"prev": w}
    for i, d in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else store.closes.index[-1]
        cost = 0.0
        if i % every == 0:
            new = rule(by_date[d], {**ctx, "date": d}, **kw)
            new = pd.Series(new, dtype=float).dropna() if not isinstance(new, pd.Series) else new.dropna()
            gross = float(new.abs().sum())
            if gross > 1.0:
                new = new / gross
            liq = store.liquidity.get(d, {})
            for s in set(new.index) | set(w.index):
                dw = abs(float(new.get(s, 0.0)) - float(w.get(s, 0.0)))
                if dw > 1e-12:
                    cost += dw * fees.cost("perp", dw, liq.get(s))
            w = new
            ctx["prev"] = w
        mask = (ret_h.index > d) & (ret_h.index <= nxt)
        if w.empty:
            idx = ret_h.index[mask]
            out.append(pd.DataFrame({"ret": 0.0, "price": 0.0, "fund": 0.0, "fee": 0.0}, index=idx))
            continue
        cols = [s for s in w.index if s in ret_h.columns]
        w = w[cols]
        seg = ret_h.loc[mask, cols].fillna(0.0)
        fund = fund_h.reindex(columns=cols).loc[seg.index].fillna(0.0)
        price = seg.to_numpy() @ w.to_numpy()
        funding = -(fund.to_numpy() @ w.to_numpy())  # longs pay a positive rate
        fee = np.zeros(len(seg))
        if len(fee):
            fee[0] = cost
        frame = {"ret": price + funding - fee, "price": price, "fund": funding, "fee": fee}
        out.append(pd.DataFrame(frame, index=seg.index))
    return pd.concat(out) if out else pd.DataFrame(columns=["ret", "price", "fund", "fee"])


def quarters(store: Store, warmup_days: int = 100) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Calendar quarters covering the store, the first one starting after the warm-up."""
    first = store.dates[0] + pd.Timedelta(days=warmup_days)
    edges = pd.date_range(first.normalize(), store.dates[-1], freq="QS", tz="UTC")
    return [(a, b) for a, b in zip(edges, list(edges[1:]) + [store.dates[-1]], strict=False) if (b - a).days > 60]


def summarise(res: pd.DataFrame, qs: Iterable[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    """Per-quarter return, Sharpe, drawdown and the gross / funding / fee decomposition."""
    rows = []
    for a, b in qs:
        r = res.loc[(res.index >= a) & (res.index < b)]
        if len(r) < 200:
            continue
        rows.append(
            {
                "q": a.strftime("%Y-%m"),
                "ret": m.total_return(r["ret"]),
                "sharpe": m.sharpe(r["ret"], PPY_HOURLY),
                "mdd": m.max_drawdown(r["ret"]),
                "price": float(r["price"].sum()),
                "fund": float(r["fund"].sum()),
                "fee": float(r["fee"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=["q", "ret", "sharpe", "mdd", "price", "fund", "fee"])


def headline(name: str, f: pd.DataFrame) -> str:
    """One line per rule, the way the night report tabulated them."""
    if f.empty:
        return f"{name:34} no full quarter"
    sh = f["sharpe"]
    years = len(f) / 4
    positive = int((sh > 0).sum())
    return (
        f"{name:34} q {len(f):2d}  sharpe med {sh.median():5.2f} mean {sh.mean():5.2f}  >0 {positive:2d}/{len(sh)}"
        f"  ann {f['ret'].mean() * 4:+6.1%}  mdd {f['mdd'].abs().max():6.1%}"
        f"  | gross/yr {f['price'].sum() / years:+.1%} fund {f['fund'].sum() / years:+.1%}"
        f" fees {f['fee'].sum() / years:.1%}"
    )


# ---------------------------------------------------------------- rules: f(panel_at_date, ctx, **params) -> weights


def majors(panel: pd.DataFrame, n: int = 5) -> pd.Index:
    return panel.sort_values("dollar_volume_30d", ascending=False).index[:n]


def _vol_size(panel: pd.DataFrame, s: str, n: int, target_vol: float, vol_col: str = "vol_30d") -> float:
    vol = max(float(panel.loc[s, vol_col]), 0.05)
    return min(1.0 / n, (target_vol / vol) / n)


def rule_long_majors(panel, ctx, n=5, target_vol=0.20):
    """The null every long-only trend rule must beat: always long the deepest names, vol-targeted, no timing."""
    return {s: _vol_size(panel, s, n, target_vol) for s in majors(panel, n)}


def rule_tsmom(panel, ctx, n=5, lookback="ret_30d", target_vol=0.20, long_short=False):
    """Time-series momentum on the deepest names: long what rose over the lookback, flat (or short) otherwise."""
    w = {}
    for s in majors(panel, n):
        r = panel.loc[s, lookback] if lookback in panel.columns else np.nan
        if not np.isfinite(r):
            continue
        side = 1.0 if r > 0 else (-1.0 if long_short else 0.0)
        w[s] = side * _vol_size(panel, s, n, target_vol)
    return w


def rule_ema_trend(panel, ctx, n=5, target_vol=0.20, ema="ema_ratio_50", conf="ret_30d"):
    """Trend by alignment: price above its EMA and a positive lookback return; long or flat, vol-targeted."""
    w = {}
    for s in majors(panel, n):
        up = ema in panel.columns and panel.loc[s, ema] > 0 and panel.loc[s, conf] > 0
        w[s] = _vol_size(panel, s, n, target_vol) if up else 0.0
    return w


def rule_lowvol(panel, ctx, k=10, hedge="none", gross=0.5):
    """Long the k calmest names; hedged with the equal-weight rest ("index"), BTC ("btc") or not at all."""
    v = panel["vol_30d"].dropna().sort_values()
    picks = list(v.index[:k])
    if hedge == "none":
        return {s: gross / k for s in picks}
    w = {s: gross / 2 / k for s in picks}
    if hedge == "index":
        others = [s for s in v.index if s not in picks]
        for s in others:
            w[s] = -gross / 2 / len(others)
    elif hedge == "btc" and "BTCUSDT" in panel.index:
        w["BTCUSDT"] = w.get("BTCUSDT", 0.0) - gross / 2
    return w


def rule_ew(panel, ctx):
    return {s: 1.0 / len(panel.index) for s in panel.index}


RULES: dict[str, tuple[Rule, dict]] = {
    "ew": (rule_ew, {}),
    "null_long2": (rule_long_majors, {"n": 2}),
    "null_long5": (rule_long_majors, {}),
    "tsmom2_30d": (rule_tsmom, {"n": 2}),
    "tsmom5_30d": (rule_tsmom, {}),
    "tsmom5_90d": (rule_tsmom, {"lookback": "ret_90d"}),
    "tsmom5_90d_ls": (rule_tsmom, {"lookback": "ret_90d", "long_short": True}),
    "ema_trend2": (rule_ema_trend, {"n": 2}),
    "ema_trend5": (rule_ema_trend, {}),
    "lowvol10_long": (rule_lowvol, {}),
    "lowvol10_index": (rule_lowvol, {"hedge": "index"}),
    "lowvol10_btc": (rule_lowvol, {"hedge": "btc"}),
}

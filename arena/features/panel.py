"""The feature panel: one row per symbol, computed from a point-in-time Snapshot.

Everything here reads a ``Snapshot``, which is already truncated to
``ts <= decision_ts``, so no feature can see the future unless it does its own
arithmetic wrong. The tests check that property directly rather than trusting it.

Two families of column come out:

* **raw** -- levels and ratios, comparable within a symbol across time;
* **rank** -- the cross-sectional percentile of each raw column at this bar.

The ranks matter more than they look. A raw 30-day return is not stationary:
+40 % means something different in March 2024 and in September 2026. Its rank
among the 50 symbols trading that day is, and a cross-sectional model can only
use what is comparable across the cross-section. Every raw column therefore
ships with a ``rank_`` twin, which roughly doubles the width for free and is
where most of the usable signal tends to live.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arena.core.indicators import ema, funding_zscores, realised_vol
from arena.core.snapshot import Snapshot

RANK_PREFIX = "rank_"
EPS = 1e-12


@dataclass(frozen=True)
class PanelConfig:
    """Lookbacks, in days unless named otherwise. Widening these widens the model."""

    return_days: tuple[int, ...] = (1, 3, 7, 14, 30, 60, 90)
    skip_days: int = 7  # for the momentum that drops the most recent week
    ema_spans: tuple[int, ...] = (10, 20, 50, 100, 200)  # in bars
    vol_days: tuple[int, ...] = (7, 30, 90)
    extreme_days: tuple[int, ...] = (30, 90)
    volume_days: tuple[int, ...] = (1, 7, 30)
    funding_days: tuple[int, ...] = (1, 7, 14, 30)
    rsi_bars: int = 14
    atr_bars: int = 14
    bollinger_bars: int = 20
    donchian_days: int = 20
    beta_days: int = 30
    min_history_days: int = 8  # below this a symbol is dropped rather than imputed


DEFAULT_CONFIG = PanelConfig()


# --------------------------------------------------------------------------- helpers


def _safe(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def _pct_return(close: pd.Series, bars: int) -> float:
    if len(close) <= bars or bars <= 0:
        return float("nan")
    past = float(close.iloc[-1 - bars])
    return _safe(float(close.iloc[-1]) / past - 1.0) if past else float("nan")


def _rsi(close: pd.Series, bars: int) -> float:
    if len(close) < bars + 1:
        return float("nan")
    delta = close.diff().tail(bars)
    gain = float(delta.clip(lower=0).mean())
    loss = float(-delta.clip(upper=0).mean())
    if loss <= EPS:
        return 100.0 if gain > 0 else 50.0
    return _safe(100.0 - 100.0 / (1.0 + gain / loss))


def _parkinson(candles: pd.DataFrame, bars: int, bars_per_year: int) -> float:
    """Range-based volatility: uses the high and low a close-to-close estimator throws away."""
    window = candles.tail(bars)
    if len(window) < 3:
        return float("nan")
    ratio = np.log(window["high"] / window["low"].replace(0, np.nan)).dropna()
    if ratio.empty:
        return float("nan")
    var = float((ratio**2).mean()) / (4.0 * np.log(2.0))
    return _safe(np.sqrt(max(var, 0.0) * bars_per_year))


def _garman_klass(candles: pd.DataFrame, bars: int, bars_per_year: int) -> float:
    window = candles.tail(bars)
    if len(window) < 3:
        return float("nan")
    hl = np.log(window["high"] / window["low"].replace(0, np.nan))
    co = np.log(window["close"] / window["open"].replace(0, np.nan))
    var = float((0.5 * hl**2 - (2 * np.log(2) - 1) * co**2).dropna().mean())
    return _safe(np.sqrt(max(var, 0.0) * bars_per_year))


def _drawdown(close: pd.Series, bars: int) -> float:
    window = close.tail(bars)
    if window.empty:
        return float("nan")
    peak = float(window.max())
    return _safe(float(window.iloc[-1]) / peak - 1.0) if peak else float("nan")


def _donchian_position(close: pd.Series, bars: int) -> float:
    """Where the last close sits in its recent range: 0 at the low, 1 at the high."""
    window = close.tail(bars)
    if len(window) < 3:
        return float("nan")
    lo, hi = float(window.min()), float(window.max())
    return _safe((float(window.iloc[-1]) - lo) / (hi - lo)) if hi > lo else 0.5


def _bollinger_b(close: pd.Series, bars: int) -> float:
    window = close.tail(bars)
    if len(window) < 3:
        return float("nan")
    mean, sd = float(window.mean()), float(window.std(ddof=1))
    return _safe((float(close.iloc[-1]) - mean) / (2.0 * sd) + 0.5) if sd > EPS else 0.5


def _atr_over_price(candles: pd.DataFrame, bars: int) -> float:
    window = candles.tail(bars + 1)
    if len(window) < 3:
        return float("nan")
    prev = window["close"].shift(1)
    tr = pd.concat(
        [window["high"] - window["low"], (window["high"] - prev).abs(), (window["low"] - prev).abs()], axis=1
    ).max(axis=1)
    last = float(window["close"].iloc[-1])
    return _safe(float(tr.tail(bars).mean()) / last) if last else float("nan")


def _amihud(close: pd.Series, volume: pd.Series, bars: int) -> float:
    """Illiquidity: absolute return per dollar traded. High means the price moves on little flow."""
    ret = close.pct_change().abs().tail(bars)
    dollar = (close * volume).tail(bars)
    joined = pd.concat([ret, dollar], axis=1).dropna()
    joined = joined[joined.iloc[:, 1] > 0]
    if joined.empty:
        return float("nan")
    return _safe(float((joined.iloc[:, 0] / joined.iloc[:, 1]).mean()) * 1e9)


def _beta_and_idio(close: pd.Series, market: pd.Series, bars: int) -> tuple[float, float]:
    """OLS beta to the market and the residual volatility left over."""
    a = close.pct_change().tail(bars)
    b = market.pct_change().reindex(a.index).tail(bars)
    joined = pd.concat([a.rename("x"), b.rename("m")], axis=1).dropna()
    if len(joined) < 10:
        return float("nan"), float("nan")
    var = float(joined["m"].var(ddof=1))
    if var <= EPS:
        return float("nan"), float(joined["x"].std(ddof=1))
    beta = float(joined["x"].cov(joined["m"]) / var)
    resid = joined["x"] - beta * joined["m"]
    return _safe(beta), _safe(float(resid.std(ddof=1)))


# --------------------------------------------------------------------------- per symbol


def symbol_features(snap: Snapshot, symbol: str, market: pd.Series, cfg: PanelConfig = DEFAULT_CONFIG) -> dict:
    """Every raw feature for one symbol, or an empty dict when its history is too short."""
    candles = snap.candles(symbol)
    bars_per_day = snap.bars_per_day
    if len(candles) < cfg.min_history_days * bars_per_day:
        return {}
    close, volume = candles["close"], candles["volume"]
    bpy = snap.bars_per_year
    out: dict[str, float] = {}

    for days in cfg.return_days:
        out[f"ret_{days}d"] = _pct_return(close, days * bars_per_day)
    skip = cfg.skip_days * bars_per_day
    long_bars = 30 * bars_per_day + skip
    if len(close) > long_bars:
        recent, older = float(close.iloc[-1 - skip]), float(close.iloc[-1 - long_bars])
        out["ret_30d_skip_7d"] = _safe(recent / older - 1.0) if older else float("nan")

    last = float(close.iloc[-1])
    for span in cfg.ema_spans:
        if len(close) > span:
            value = float(ema(close, span).iloc[-1])
            out[f"ema_ratio_{span}"] = _safe(last / value - 1.0) if value else float("nan")
    if len(close) > max(cfg.ema_spans):
        fast, slow = float(ema(close, 12).iloc[-1]), float(ema(close, 26).iloc[-1])
        out["macd"] = _safe((fast - slow) / last) if last else float("nan")

    for days in cfg.vol_days:
        bars = days * bars_per_day
        if len(close) > bars:
            out[f"vol_{days}d"] = _safe(float(realised_vol(close, bars, bpy).iloc[-1]))
    v7, v30 = out.get("vol_7d"), out.get("vol_30d")
    if v7 and v30 and np.isfinite(v7) and np.isfinite(v30) and v30 > EPS:
        out["vol_term_structure"] = _safe(v7 / v30)
    out["vol_parkinson_30d"] = _parkinson(candles, 30 * bars_per_day, bpy)
    out["vol_garman_klass_30d"] = _garman_klass(candles, 30 * bars_per_day, bpy)
    downside = close.pct_change().tail(30 * bars_per_day).clip(upper=0.0)
    out["vol_downside_30d"] = _safe(float(np.sqrt((downside**2).mean()) * np.sqrt(bpy)))
    vols = [out.get(f"vol_{d}d") for d in cfg.vol_days]
    finite = [v for v in vols if v is not None and np.isfinite(v)]
    out["vol_of_vol"] = _safe(float(np.std(finite, ddof=1))) if len(finite) > 1 else float("nan")

    for days in cfg.extreme_days:
        out[f"drawdown_{days}d"] = _drawdown(close, days * bars_per_day)
    out["donchian_position"] = _donchian_position(close, cfg.donchian_days * bars_per_day)
    out["bollinger_b"] = _bollinger_b(close, cfg.bollinger_bars)
    out["rsi"] = _rsi(close, cfg.rsi_bars)
    out["atr_over_price"] = _atr_over_price(candles, cfg.atr_bars)

    for days in cfg.volume_days:
        bars = days * bars_per_day
        window = (close * volume).tail(bars)
        out[f"dollar_volume_{days}d"] = _safe(float(window.sum()) / max(days, 1))
    dv1, dv30 = out.get("dollar_volume_1d"), out.get("dollar_volume_30d")
    if dv1 and dv30 and np.isfinite(dv1) and np.isfinite(dv30) and dv30 > EPS:
        out["volume_surge"] = _safe(dv1 / dv30)
    out["amihud"] = _amihud(close, volume, 30 * bars_per_day)

    funding = snap.funding(symbol)
    if len(funding):
        for days in cfg.funding_days:
            window = funding[funding.index > snap.ts - pd.Timedelta(days=days)]
            out[f"funding_{days}d"] = _safe(float(window.mean())) if len(window) else float("nan")
        out["funding_cumulative_30d"] = _safe(float(funding[funding.index > snap.ts - pd.Timedelta(days=30)].sum()))
        recent = funding.tail(90)
        out["funding_vol"] = _safe(float(recent.std(ddof=1))) if len(recent) > 2 else float("nan")
        f7, f30 = out.get("funding_7d"), out.get("funding_30d")
        if f7 is not None and f30 is not None:
            out["funding_slope"] = _safe(f7 - f30)
    hl = snap.hl_funding(symbol)
    if hl is not None and out.get("funding_7d") is not None and np.isfinite(out["funding_7d"]):
        out["funding_venue_spread"] = _safe(float(hl) * 8.0 - out["funding_7d"])

    oi = snap.open_interest(symbol)
    if len(oi) > bars_per_day:
        now_oi, then_oi = float(oi.iloc[-1]), float(oi.iloc[-1 - bars_per_day])
        out["oi_change_1d"] = _safe(now_oi / then_oi - 1.0) if then_oi else float("nan")
        dv = out.get("dollar_volume_1d")
        out["oi_over_volume"] = _safe(now_oi * last / dv) if dv and dv > EPS else float("nan")

    news = snap.news(symbol)
    if len(news):
        row = news.iloc[-1]
        for col in ("sent_24h", "sent_7d", "n_24h", "shock"):
            if col in news.columns:
                out[f"news_{col}"] = _safe(float(row[col]))
        if "n_24h" in news.columns and "n_7d" in news.columns:
            weekly = float(row["n_7d"]) / 7.0
            out["news_attention"] = _safe(float(row["n_24h"]) / weekly) if weekly > EPS else float("nan")

    beta, idio = _beta_and_idio(close, market, cfg.beta_days * bars_per_day)
    out["beta_market"] = beta
    out["idio_vol"] = idio
    out["days_of_history"] = float(len(close) / bars_per_day)
    return out


# --------------------------------------------------------------------------- the panel


def market_series(snap: Snapshot, symbols: list[str]) -> pd.Series:
    """Equal-weight index of the universe, the benchmark every excess return is taken against."""
    frames = [snap.candles(s)["close"].pct_change() for s in symbols if not snap.candles(s).empty]
    if not frames:
        return pd.Series(dtype=float)
    returns = pd.concat(frames, axis=1).mean(axis=1).fillna(0.0)
    return (1.0 + returns).cumprod()


def build(snap: Snapshot, symbols: list[str] | None = None, cfg: PanelConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Feature panel for the universe at ``snap.ts``: raw columns plus their cross-sectional ranks.

    Symbols with too little history are dropped, never imputed: a model that
    learns from a filled-in zero learns that newly listed symbols are average,
    which is the opposite of true.
    """
    names = list(symbols or snap.symbols)
    market = market_series(snap, names)
    rows = {sym: symbol_features(snap, sym, market, cfg) for sym in names}
    rows = {k: v for k, v in rows.items() if v}
    if not rows:
        return pd.DataFrame()
    panel = pd.DataFrame.from_dict(rows, orient="index").sort_index()

    zscores = funding_zscores(snap, 7)
    panel["funding_crowding_z"] = pd.Series(zscores).reindex(panel.index)

    ranked = panel.rank(pct=True, numeric_only=True)
    ranked.columns = [f"{RANK_PREFIX}{c}" for c in ranked.columns]
    panel = pd.concat([panel, ranked], axis=1)

    # cross-sectional state: identical on every row, but it is what conditions the rest
    panel["xs_dispersion"] = float(panel["ret_7d"].std(ddof=1)) if "ret_7d" in panel else np.nan
    panel["xs_breadth"] = float((panel["ret_7d"] > 0).mean()) if "ret_7d" in panel else np.nan
    if "funding_7d" in panel:
        panel["xs_funding_mean"] = float(panel["funding_7d"].mean())
    panel["xs_n_symbols"] = float(len(panel))
    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Model inputs: everything numeric except the bookkeeping column."""
    return [c for c in panel.columns if c != "days_of_history" and pd.api.types.is_numeric_dtype(panel[c])]

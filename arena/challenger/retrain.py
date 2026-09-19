"""Retraining of the meta-labeling model on a rolling window → new challenger.

Dataset: every base signal emitted over the window (replayed from history with
the champions' base parameters), labelled 1 when the signal's 24h forward
return net of fees was positive. Validation is walk-forward (train on the past,
score on the next block) to report an honest AUC before the model is trusted.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg

from arena.competitors.meta_label import FEATURE_COLUMNS, MetaLabel, build_features
from arena.core.snapshot import Snapshot
from arena.core.types import Alert, CompetitorSpec
from arena.judge.backtest import HistoryFrames
from arena.store import books as bstore
from arena.store import registry

log = logging.getLogger(__name__)
HORIZON = 24
ROUND_TRIP_COST = 2 * 0.0007
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 0}


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), no sklearn dependency."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    vals = np.concatenate([pos, neg])
    for v in np.unique(vals):
        idx = np.where(vals == v)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def build_dataset(meta: MetaLabel, history: HistoryFrames, symbols: list[str], start: datetime, end: datetime, step_hours: int = 4) -> pd.DataFrame:
    """Replay the bases over ``[start, end]`` every ``step_hours`` and label each signal."""
    full = Snapshot.from_long(end, symbols, history.candles, history.funding, history.open_interest, None, history.news, history.macro_events)
    closes = full.closes()
    bars = closes.index[(closes.index >= pd.Timestamp(start)) & (closes.index <= pd.Timestamp(end) - pd.Timedelta(hours=HORIZON))]
    rows: list[dict[str, Any]] = []
    for ts in bars[:: step_hours]:
        snap = full.at(ts)
        combined, regime = meta.base_signals(snap)
        if not combined:
            continue
        pos = closes.index.get_loc(ts)
        for sym, (w, n) in combined.items():
            if sym not in closes.columns or pos + HORIZON >= len(closes):
                continue
            fwd = closes[sym].iloc[pos + HORIZON] / closes[sym].iloc[pos] - 1.0
            pnl = np.sign(w) * fwd - ROUND_TRIP_COST
            feats = build_features(snap, sym, w, n, regime)
            feats.update({"ts": ts, "symbol": sym, "label": int(pnl > 0)})
            rows.append(feats)
    return pd.DataFrame(rows)


def walk_forward_auc(df: pd.DataFrame, n_blocks: int = 4) -> float:
    df = df.sort_values("ts")
    edges = np.linspace(0, len(df), n_blocks + 1, dtype=int)
    aucs = []
    for i in range(1, n_blocks):
        train, test = df.iloc[: edges[i]], df.iloc[edges[i]: edges[i + 1]]
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        booster = lgb.train(LGB_PARAMS, lgb.Dataset(train[FEATURE_COLUMNS], train["label"]), num_boost_round=200)
        aucs.append(roc_auc(test["label"].to_numpy(), booster.predict(test[FEATURE_COLUMNS])))
    return float(np.nanmean(aucs)) if aucs else float("nan")


def run(conn: psycopg.Connection, history: HistoryFrames, symbols: list[str], end: datetime, window_days: int = 365,
        bases: list[dict] | None = None, min_auc: float = 0.52) -> CompetitorSpec | None:
    """Train, validate, and insert a new meta_label challenger carrying the model."""
    champions = registry.list_competitors(conn, statuses=["champion"])
    champion = next((c for c in champions if c.family == "meta_label"), None)
    base_params = bases or (champion.params.get("bases") if champion else None) or MetaLabel.default_params["bases"]
    meta = MetaLabel({"bases": base_params})
    start = pd.Timestamp(end) - timedelta(days=window_days)
    df = build_dataset(meta, history, symbols, start, end)
    trial_id = registry.add_trial(conn, "meta_label", "retrain", {"bases": base_params, "window_days": window_days}, {"rows": len(df)}, None)
    conn.commit()
    if len(df) < 200 or df["label"].nunique() < 2:
        registry.finish_trial(conn, trial_id, {"rows": len(df)}, "rejected")
        conn.commit()
        return None
    auc = walk_forward_auc(df)
    booster = lgb.train(LGB_PARAMS, lgb.Dataset(df[FEATURE_COLUMNS], df["label"]), num_boost_round=200)
    metrics = {"rows": len(df), "wf_auc": auc, "positive_rate": float(df["label"].mean())}
    if not np.isfinite(auc) or auc < min_auc:
        registry.finish_trial(conn, trial_id, metrics, "rejected")
        bstore.add_alert(conn, Alert(kind="rejected", payload={"detail": f"meta_label retrain rejected: wf AUC {auc:.3f} < {min_auc}"}))
        conn.commit()
        return None
    registry.finish_trial(conn, trial_id, metrics, "admitted")
    with conn.cursor() as cur:
        cur.execute("SELECT coalesce(max(version), 0) AS v FROM competitors WHERE family = 'meta_label'")
        version = int(cur.fetchone()["v"]) + 1
    spec = CompetitorSpec(None, f"meta_label_v{version}", "meta_label", version, {"bases": base_params, "threshold": 0.55},
                          status="challenger", parent_id=champion.id if champion else None,
                          rationale=f"retrain on {window_days}d, {len(df)} events, wf AUC {auc:.3f}; trial {trial_id}")
    cid = registry.insert_competitor(conn, spec)
    registry.save_model(conn, cid, booster.model_to_string().encode(), metrics)
    conn.commit()
    return CompetitorSpec(**{**spec.__dict__, "id": cid})

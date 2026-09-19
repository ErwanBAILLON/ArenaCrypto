"""Optuna search spaces per rule family. Deliberately small: every evaluation is a counted trial."""

from __future__ import annotations

from typing import Any, Callable

import optuna

Space = Callable[[optuna.Trial], dict[str, Any]]


def carry(t: optuna.Trial) -> dict[str, Any]:
    return {
        "k": t.suggest_int("k", 2, 6),
        "min_rate": t.suggest_float("min_rate", 0.0, 0.0002),
        "lookback_days": t.suggest_int("lookback_days", 3, 30),
        "exit_ratio": t.suggest_float("exit_ratio", 0.0, 0.8),
        "rank_band": t.suggest_int("rank_band", 0, 10),
    }


def trend_ts(t: optuna.Trial) -> dict[str, Any]:
    fast = t.suggest_int("fast", 20, 80)
    return {
        "fast": fast,
        "slow": t.suggest_int("slow", 100, 300),
        "lb_short_days": t.suggest_int("lb_short_days", 14, 45),
        "lb_long_days": t.suggest_int("lb_long_days", 60, 120),
        "target_vol": t.suggest_float("target_vol", 0.10, 0.30),
        "atr_stop_mult": t.suggest_float("atr_stop_mult", 2.0, 5.0),
    }


def xs_momentum(t: optuna.Trial) -> dict[str, Any]:
    return {
        "k": t.suggest_int("k", 2, 5),
        "lookback_days": t.suggest_int("lookback_days", 14, 60),
        "skip_days": t.suggest_int("skip_days", 0, 3),
        "band": t.suggest_int("band", 0, 3),
        "long_only": t.suggest_categorical("long_only", [False, True]),
    }


def regime(t: optuna.Trial) -> dict[str, Any]:
    return {
        "vol_window": t.suggest_int("vol_window", 24 * 10, 24 * 45),
        "trend_fast": t.suggest_int("trend_fast", 20, 80),
        "trend_slow": t.suggest_int("trend_slow", 100, 300),
        "bull_weight": t.suggest_float("bull_weight", 0.3, 0.8),
        "range_carry_weight": t.suggest_float("range_carry_weight", 0.2, 0.6),
    }


SPACES: dict[str, Space] = {"carry": carry, "trend_ts": trend_ts, "xs_momentum": xs_momentum, "regime": regime}

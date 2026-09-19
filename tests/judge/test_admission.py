"""``admit`` end to end against Postgres (needs ``PG_TEST_URL``), including the robustness step."""

import pandas as pd

from arena.book.book import FeeModel
from arena.core.types import Target, Verdict
from arena.judge.admission import _jsonable, admit
from arena.judge.backtest import HistoryFrames
from arena.store import registry
from tests.conftest import make_candles

SYMS = ["BTC", "ETH", "SOL"]


class LongBTC:
    def warmup_bars(self) -> int:
        return 24

    def decide(self, snap):
        return {"BTC": Target(weight=1.0)}


def test_jsonable_converts_numpy_and_nan():
    import numpy as np

    out = _jsonable({"a": np.float64(1.5), "b": float("nan"), "c": [np.int64(2), np.bool_(True)], "d": {"e": None}})
    assert out == {"a": 1.5, "b": None, "c": [2, True], "d": {"e": None}}
    assert type(out["a"]) is float and type(out["c"][0]) is int and type(out["c"][1]) is bool


def test_admit_with_robustness_records_trial(conn):
    candles = make_candles(SYMS, bars=24 * 300, seed=4, drift={"BTC": 0.0008})
    history = HistoryFrames(candles)
    start, end = candles["ts"].min(), candles["ts"].max() + pd.Timedelta(hours=1)
    adm = admit(
        conn,
        "trend_ts",
        {"k": 1},
        history,
        SYMS,
        start,
        end,
        FeeModel(),
        null_thr=0.0,
        make_competitor=LongBTC,
        robustness_n=6,
        notes="test",
    )
    assert isinstance(adm.verdict, Verdict)
    rob = adm.verdict.metrics["robustness"]
    assert rob["n_windows"] == 6 and "passed" in rob
    assert adm.verdict.metrics["win_rate_overall"] == rob["win_rate_overall"]
    with conn.cursor() as cur:
        cur.execute("SELECT metrics, verdict FROM trials WHERE id = %s", (adm.trial_id,))
        row = cur.fetchone()
    assert row["verdict"] in ("admitted", "rejected")
    assert row["metrics"]["robustness"]["n_windows"] == 6
    assert row["metrics"]["failed"] == adm.verdict.failed
    assert registry.count_trials(conn, "trend_ts") == 1


def test_admit_robustness_disabled(conn):
    candles = make_candles(SYMS, bars=24 * 300, seed=4)
    start, end = candles["ts"].min(), candles["ts"].max() + pd.Timedelta(hours=1)
    adm = admit(
        conn, "trend_ts", {}, HistoryFrames(candles), SYMS, start, end, FeeModel(), 0.0, LongBTC, robustness_n=0
    )
    assert "robustness" not in adm.verdict.metrics and "robust_regimes" not in adm.verdict.failed

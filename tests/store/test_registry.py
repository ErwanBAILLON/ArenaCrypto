import psycopg
import pytest

from arena.core.types import CompetitorSpec
from arena.store import registry as repo


def spec(name: str, **kw) -> CompetitorSpec:
    base = dict(id=None, name=name, family="carry", version=1, params={"k": 3, "threshold": 0.0001})
    base.update(kw)
    return CompetitorSpec(**base)


def test_insert_get_round_trip(conn):
    cid = repo.insert_competitor(conn, spec("carry_v1", rationale="founding"))
    got = repo.get_competitor(conn, "carry_v1")
    assert got == spec("carry_v1", id=cid, rationale="founding")
    assert got.params["threshold"] == 0.0001
    assert repo.get_competitor(conn, "nope") is None


def test_name_is_unique(conn):
    repo.insert_competitor(conn, spec("dup"))
    with pytest.raises(psycopg.errors.UniqueViolation):
        repo.insert_competitor(conn, spec("dup"))
    conn.rollback()


def test_list_filters_and_status(conn):
    a = repo.insert_competitor(conn, spec("null_flat", family="null", role="null", status="champion"))
    b = repo.insert_competitor(conn, spec("carry_v1", status="champion"))
    c = repo.insert_competitor(conn, spec("carry_v2", version=2, parent_id=b))
    assert [s.id for s in repo.list_competitors(conn)] == [a, b, c]
    assert [s.id for s in repo.list_competitors(conn, statuses=["champion"])] == [a, b]
    assert [s.id for s in repo.list_competitors(conn, statuses=["champion"], roles=["competitor"])] == [b]
    assert repo.get_competitor(conn, "carry_v2").parent_id == b

    repo.set_status(conn, c, "retired")
    assert repo.get_competitor(conn, "carry_v2").status == "retired"
    assert repo.list_competitors(conn, statuses=["candidate"]) == []


def test_trials_lifecycle_and_count(conn):
    cid = repo.insert_competitor(conn, spec("carry_v1"))
    assert repo.count_trials(conn, "carry") == 0
    t1 = repo.add_trial(conn, "carry", "backtest", {"k": 3}, {}, None, competitor_id=cid, notes="default params")
    t2 = repo.add_trial(conn, "carry", "optimize", {"k": 4}, {"sharpe": 0.4}, "rejected")
    repo.add_trial(conn, "trend_ts", "backtest", {}, {}, None)
    assert repo.count_trials(conn, "carry") == 2

    repo.finish_trial(conn, t1, {"sharpe": 1.2, "mdd": -0.1}, "admitted")
    with conn.cursor() as cur:
        cur.execute("SELECT metrics, verdict, finished_at, competitor_id FROM trials WHERE id = %s", (t1,))
        row = cur.fetchone()
    assert row["metrics"] == {"sharpe": 1.2, "mdd": -0.1}
    assert row["verdict"] == "admitted" and row["finished_at"] is not None and row["competitor_id"] == cid
    with conn.cursor() as cur:
        cur.execute("SELECT finished_at, params FROM trials WHERE id = %s", (t2,))
        row = cur.fetchone()
    assert row["finished_at"] is None and row["params"] == {"k": 4}


def test_models_latest(conn):
    cid = repo.insert_competitor(conn, spec("meta_v1", family="meta_label"))
    assert repo.latest_model(conn, cid) is None
    repo.save_model(conn, cid, b"first", {"auc": 0.55})
    m2 = repo.save_model(conn, cid, b"second", {"auc": 0.61})
    with conn.cursor() as cur:  # make ordering unambiguous even within one transaction
        cur.execute("UPDATE models SET trained_at = trained_at + interval '1 minute' WHERE id = %s", (m2,))
    artifact, metrics = repo.latest_model(conn, cid)
    assert artifact == b"second" and metrics == {"auc": 0.61}


def test_cached_null_threshold_reuses_trial(conn, monkeypatch):
    import pandas as pd

    from arena.book.book import FeeModel
    from arena.judge import admission
    from arena.judge.backtest import HistoryFrames
    from tests.conftest import make_candles

    c = make_candles(["BTC"], bars=24 * 20)
    h = HistoryFrames(candles=c)
    end = c["ts"].max()
    start = end - pd.Timedelta(days=5)
    calls = {"n": 0}

    def fake_sharpes(*a, **k):
        calls["n"] += 1
        return [0.1, 0.2, 0.3, 0.4]

    monkeypatch.setattr(admission, "null_sharpes", fake_sharpes)
    t1 = admission.cached_null_threshold(conn, h, ["BTC"], start, end, FeeModel(), n=4)
    t2 = admission.cached_null_threshold(conn, h, ["BTC"], start, end, FeeModel(), n=4)
    assert t1 == t2 and calls["n"] == 1

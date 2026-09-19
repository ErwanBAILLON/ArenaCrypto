from datetime import datetime, timezone

import numpy as np
import pandas as pd

from arena.core.types import CompetitorSpec
from arena.runner import drift
from arena.runner.promotion import Candidate, should_promote, surplus_challengers

NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _spec(i, family="trend_ts", status="challenger", role="competitor"):
    return CompetitorSpec(i, f"{family}_v{i}", family, i, {}, role=role, status=status)


def _series(mu, n=24 * 50, seed=0, start="2024-04-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.Series(rng.normal(mu, 0.004, n), index=idx)


def test_stale_detection():
    assert drift.stale_data(datetime(2024, 5, 31, 20, tzinfo=timezone.utc), NOW) is not None
    assert drift.stale_data(datetime(2024, 5, 31, 23, tzinfo=timezone.utc), NOW) is None
    assert drift.stale_data(None, NOW).kind == "stale"


def test_bleeding_champion_flagged():
    bad = _series(-0.0008)
    assert drift.champion_bleeding(_spec(1, status="champion"), bad) is not None
    good = _series(0.0005)
    assert drift.champion_bleeding(_spec(1, status="champion"), good) is None


def test_silent_and_broken():
    assert drift.silent_competitor(_spec(1), None, NOW) is not None
    assert drift.silent_competitor(_spec(1, role="null"), None, NOW) is None
    assert drift.broken_book(_spec(1), float("nan")).kind == "error"
    assert drift.broken_book(_spec(1), 10_000.0) is None


def _nulls():
    idx = pd.date_range("2024-04-01", periods=24 * 50, freq="1h", tz="UTC")
    rng = np.random.default_rng(9)
    return pd.DataFrame({100 + i: rng.normal(0, 0.004, len(idx)) for i in range(5)}, index=idx)


def test_promotion_requires_age_and_edge():
    start = datetime(2024, 4, 1, tzinfo=timezone.utc)
    strong = Candidate(_spec(2), start, 50, _series(0.0008))
    weak_champion = Candidate(_spec(1, status="champion"), start, 300, _series(0.0, seed=3))
    ok, ev = should_promote(strong, weak_champion, _nulls(), NOW)
    assert ok and ev["challenger_sharpe"] > ev["champion_sharpe"]
    young = Candidate(_spec(3), datetime(2024, 5, 25, tzinfo=timezone.utc), 10, _series(0.0008, start="2024-05-25", n=24 * 6))
    assert should_promote(young, weak_champion, _nulls(), NOW) == (False, {"reason": "not_ready"})
    noise = Candidate(_spec(4), start, 200, _series(0.0, seed=5))
    ok, ev = should_promote(noise, weak_champion, _nulls(), NOW)
    assert not ok and ev["reason"] in ("below_null", "below_champion")


def test_surplus_challengers_oldest_first():
    ch = [_spec(i) for i in (5, 3, 9, 7, 1)]
    surplus = surplus_challengers(ch)
    assert [s.id for s in surplus] == [1, 3]

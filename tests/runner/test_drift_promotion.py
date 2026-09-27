from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from arena.core.types import CompetitorSpec
from arena.runner import drift, promotion
from arena.runner.promotion import (
    MIN_NULL_SAMPLES,
    Candidate,
    gate_cleared,
    null_sharpes,
    null_threshold,
    should_promote,
    surplus_challengers,
)

NOW = datetime(2024, 6, 1, tzinfo=UTC)
START = datetime(2024, 4, 1, tzinfo=UTC)


def _spec(i, family="trend_ts", status="challenger", role="competitor", gate_admitted=True):
    return CompetitorSpec(i, f"{family}_v{i}", family, i, {}, role=role, status=status, gate_admitted=gate_admitted)


def _series(mu, n=24 * 50, seed=0, start="2024-04-01", sd=0.004):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.Series(rng.normal(mu, sd, n), index=idx)


def _nulls(k=MIN_NULL_SAMPLES + 5, n=24 * 50, start="2024-04-01"):
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(9)
    return pd.DataFrame({100 + i: rng.normal(0, 0.004, len(idx)) for i in range(k)}, index=idx)


def _candidate(i, mu, *, decisions=200, seed=0, status="challenger", gate_admitted=True, **kw):
    return Candidate(
        _spec(i, status=status, gate_admitted=gate_admitted),
        START,
        decisions,
        _series(mu, seed=seed, **kw),
    )


# --------------------------------------------------------------------------- drift


def test_stale_detection():
    assert drift.stale_data(datetime(2024, 5, 31, 20, tzinfo=UTC), NOW) is not None
    assert drift.stale_data(datetime(2024, 5, 31, 23, tzinfo=UTC), NOW) is None
    assert drift.stale_data(None, NOW).kind == "stale"


def test_bleeding_champion_flagged():
    assert drift.champion_bleeding(_spec(1, status="champion"), _series(-0.0008)) is not None
    assert drift.champion_bleeding(_spec(1, status="champion"), _series(0.0005)) is None


def test_bleeding_window_follows_the_bar_size():
    """30 daily bars is a month in the classic arena; 30 hourly bars is nothing."""
    daily = pd.Series(
        np.random.default_rng(1).normal(-0.004, 0.02, 40),
        index=pd.date_range("2024-04-01", periods=40, freq="1D", tz="UTC"),
    )
    assert drift.champion_bleeding(_spec(1, status="champion"), daily, bars_per_day=1) is not None
    assert drift.champion_bleeding(_spec(1, status="champion"), daily, bars_per_day=24) is None


def test_silent_and_broken():
    assert drift.silent_competitor(_spec(1), None, NOW) is not None
    assert drift.silent_competitor(_spec(1, role="null"), None, NOW) is None
    assert drift.broken_book(_spec(1), float("nan")).kind == "error"
    assert drift.broken_book(_spec(1), 10_000.0) is None


# --------------------------------------------------------------------------- the null threshold


class TestNullThreshold:
    def test_counts_only_the_series_that_cover_the_window(self):
        nulls = _nulls(k=6)
        nulls.iloc[: int(0.9 * len(nulls)), 0] = np.nan  # a null registered late
        assert len(null_sharpes(nulls, START)) == 5

    def test_empty_frame_is_a_zero_threshold(self):
        assert null_threshold(pd.DataFrame(), START) == 0.0


# --------------------------------------------------------------------------- promotion


class TestPromotion:
    def test_a_clear_winner_is_promoted(self):
        ok, ev = should_promote(_candidate(2, 0.0012), _candidate(1, 0.0, seed=3, status="champion"), _nulls(), NOW)
        assert ok
        assert ev["psr_vs_null"] >= 0.95 and ev["psr_vs_champion"] >= 0.95

    def test_a_young_challenger_waits(self):
        young = Candidate(_spec(3), datetime(2024, 5, 25, tzinfo=UTC), 10, _series(0.0008, start="2024-05-25", n=144))
        assert should_promote(young, None, _nulls(), NOW) == (False, {"reason": "not_ready"})

    def test_noise_is_not_promoted(self):
        ok, ev = should_promote(
            _candidate(4, 0.0, seed=5), _candidate(1, 0.0, seed=3, status="champion"), _nulls(), NOW
        )
        assert not ok and ev["reason"] in ("below_null", "below_champion")

    def test_a_model_the_gate_refused_is_never_promoted(self):
        """The loophole: a rejected founder in a family with no champion used to win by default."""
        refused = _candidate(5, 0.0012, gate_admitted=False)
        assert should_promote(refused, None, _nulls(), NOW) == (False, {"reason": "never_admitted"})

    def test_the_same_model_is_promotable_once_it_clears_the_gate(self):
        ok, _ = should_promote(_candidate(5, 0.0012, gate_admitted=True), None, _nulls(), NOW)
        assert ok

    def test_forward_only_families_are_exempt_by_name(self):
        news = Candidate(_spec(6, family="news", gate_admitted=False), START, 200, _series(0.0012))
        assert gate_cleared(news.spec)
        ok, _ = should_promote(news, None, _nulls(), NOW)
        assert ok

    def test_a_quantile_of_five_nulls_is_refused(self):
        ok, ev = should_promote(_candidate(7, 0.0012), None, _nulls(k=5), NOW)
        assert not ok and ev["reason"] == "null_underpowered"
        assert ev["null_samples"] == 5

    def test_beating_the_champion_on_a_point_estimate_is_not_enough(self):
        """Two strong books that share their market moves: only the paired difference decides."""
        common = _series(0.0012, seed=0)
        jitter = pd.Series(np.random.default_rng(12).normal(5e-6, 2e-4, len(common)), index=common.index)
        champion = Candidate(_spec(1, status="champion"), START, 300, common)
        challenger = Candidate(_spec(2), START, 300, common + jitter)  # a hair better on average, noisily
        ok, ev = should_promote(challenger, champion, _nulls(), NOW)
        assert ev["psr_vs_null"] >= 0.95  # it clears luck
        assert ev["challenger_sharpe"] > ev["champion_sharpe"]  # and it leads on the point estimate
        assert not ok and ev["reason"] == "below_champion"  # but the paired test says the lead is noise
        assert ev["paired_bars"] == len(common)

    def test_daily_bars_are_annualised_as_daily_bars(self):
        idx = pd.date_range("2023-01-01", periods=400, freq="1D", tz="UTC")
        rng = np.random.default_rng(4)
        nulls = pd.DataFrame({100 + i: rng.normal(0, 0.01, len(idx)) for i in range(25)}, index=idx)
        cand = Candidate(_spec(8), idx[0].to_pydatetime(), 300, pd.Series(rng.normal(0.002, 0.01, len(idx)), index=idx))
        now = idx[-1].to_pydatetime()
        hourly_ppy = should_promote(cand, None, nulls, now, ppy=8760)[1]["challenger_sharpe"]
        daily_ppy = should_promote(cand, None, nulls, now, ppy=365)[1]["challenger_sharpe"]
        assert daily_ppy == pytest.approx(hourly_ppy / np.sqrt(24), rel=1e-6)


def test_surplus_challengers_oldest_first():
    ch = [_spec(i) for i in (5, 3, 9, 7, 1)]
    assert [s.id for s in surplus_challengers(ch)] == [1, 3]


class TestStaleFeeds:
    def _row(self, source, hours_ago, ok=True, detail=""):
        return {
            "source": source,
            "last_ok_at": None if hours_ago is None else NOW - pd.Timedelta(hours=hours_ago),
            "ok": ok,
            "detail": detail,
        }

    def test_a_fresh_feed_is_quiet(self):
        assert drift.stale_feeds([self._row("candles", 1)], NOW) == []

    def test_a_dead_feed_is_reported_by_name(self):
        alerts = drift.stale_feeds([self._row("funding", 48)], NOW)
        assert len(alerts) == 1 and alerts[0].kind == "stale" and alerts[0].symbol == "funding"
        assert "48.0h ago" in alerts[0].payload["detail"]

    def test_funding_is_not_the_only_watched_source(self):
        """Only candles used to be watched; the arena could trade on a dead funding feed."""
        rows = [self._row(s, 48) for s in ("candles", "funding", "open_interest", "hl_snapshots", "macro_events")]
        assert {a.symbol for a in drift.stale_feeds(rows, NOW)} == {
            "candles",
            "funding",
            "open_interest",
            "hl_snapshots",
            "macro_events",
        }

    def test_never_fetched_is_reported(self):
        alerts = drift.stale_feeds([self._row("articles", None)], NOW)
        assert "never fetched" in alerts[0].payload["detail"]

    def test_a_daily_arena_is_not_declared_dead_every_hour(self):
        rows = [self._row("candles", 20)]
        assert drift.stale_feeds(rows, NOW, bar_hours=1) != []
        assert drift.stale_feeds(rows, NOW, bar_hours=24) == []

    def test_the_failure_detail_travels_with_the_alert(self):
        alerts = drift.stale_feeds([self._row("candles", 10, ok=False, detail="15/15 symbols failed")], NOW)
        assert "15/15 symbols failed" in alerts[0].payload["detail"]


class TestRetirement:
    def _cand(self, rets, days=60):
        idx = pd.date_range(START, periods=len(rets), freq="1h", tz="UTC")
        spec = CompetitorSpec(7, "x_v1", "trend_ts", 1, {}, status="challenger")
        return promotion.Candidate(spec=spec, first_ts=START, decisions=200, returns=pd.Series(rets, index=idx))

    def test_not_before_maturity(self):
        rng = np.random.default_rng(0)
        c = self._cand(rng.normal(-0.001, 0.005, 24 * 10))
        c = promotion.Candidate(spec=c.spec, first_ts=c.first_ts, decisions=10, returns=c.returns)
        assert promotion.should_retire(c, START + timedelta(days=10))[1]["reason"] == "not_ready"

    def test_a_mature_loser_leaves_and_a_coin_flip_stays(self):
        rng = np.random.default_rng(1)
        loser = self._cand(rng.normal(-0.0004, 0.004, 24 * 60))
        gone, why = promotion.should_retire(loser, START + timedelta(days=60))
        assert gone and why["reason"] in ("below_zero", "drawdown")
        flat = self._cand(rng.normal(0.0, 0.001, 24 * 60))
        gone, why = promotion.should_retire(flat, START + timedelta(days=60))
        assert not gone and why["reason"] == "holds"

    def test_a_deep_drawdown_is_enough_on_its_own(self):
        rets = np.concatenate([np.full(24 * 20, 0.002), np.full(24 * 10, -0.0015), np.full(24 * 30, 0.0)])
        c = self._cand(rets)
        gone, why = promotion.should_retire(c, START + timedelta(days=60))
        assert gone and why["reason"] == "drawdown" and why["max_drawdown"] >= 0.25
        alert = promotion.retirement_alert(c.spec, why)
        assert alert.kind == "retired" and "drawdown" in alert.payload["detail"]


class TestRedundancy:
    def _cand(self, cid, name, rets, first=START, status="challenger"):
        idx = pd.date_range(first, periods=len(rets), freq="1h", tz="UTC")
        spec = CompetitorSpec(cid, name, "trend_ts", 1, {}, status=status)
        return promotion.Candidate(spec=spec, first_ts=first, decisions=100, returns=pd.Series(rets, index=idx))

    def test_a_copy_leaves_and_the_incumbent_stays(self):
        rng = np.random.default_rng(3)
        base = rng.normal(0.0002, 0.004, 24 * 40)
        old = self._cand(1, "old_v1", base)
        # the copy starts five days later but moves with the same market on the same hours
        late = base[24 * 5 :] * 0.98 + rng.normal(0, 0.0001, len(base) - 24 * 5)
        copy = self._cand(2, "copy_v2", late, first=START + timedelta(days=5))
        other = self._cand(3, "other_v1", rng.normal(0.0, 0.004, 24 * 40))
        gone = promotion.redundant([old, copy, other], START + timedelta(days=40))
        assert [(loser.name, twin.name) for loser, twin, _ in gone] == [("copy_v2", "old_v1")]
        assert gone[0][2] >= 0.95

    def test_a_champion_is_never_the_one_removed(self):
        rng = np.random.default_rng(4)
        base = rng.normal(0.0002, 0.004, 24 * 40)
        champ = self._cand(1, "c_v2", base, first=START + timedelta(days=3), status="champion")
        older = self._cand(2, "o_v1", base * 1.01)
        assert promotion.redundant([champ, older], START + timedelta(days=40)) == []

    def test_too_young_to_tell(self):
        rng = np.random.default_rng(5)
        base = rng.normal(0.0002, 0.004, 24 * 10)
        assert (
            promotion.redundant([self._cand(1, "a", base), self._cand(2, "b", base)], START + timedelta(days=10)) == []
        )

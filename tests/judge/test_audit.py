"""The arena-level question the per-family gate cannot ask."""

from __future__ import annotations

import pytest

from arena.judge import audit
from arena.store import registry


def _gate_trial(conn, family, verdict, p, *, sr_period=0.02, obs=8000, dsr=0.95, universe="crypto"):
    return registry.add_trial(
        conn,
        family,
        "walkforward",
        {},
        {"bootstrap_p": p, "dsr": dsr, "sharpe": 1.5, "sr_period": sr_period, "observations": obs},
        verdict,
        universe=universe,
        finished=True,
    )


class TestBenjaminiHochberg:
    def test_nothing_survives_when_every_test_is_weak(self):
        threshold, rejected = audit.benjamini_hochberg([0.4, 0.5, 0.6], 0.10)
        assert threshold is None and not any(rejected)

    def test_one_very_strong_test_survives_among_many(self):
        threshold, rejected = audit.benjamini_hochberg([0.0001] + [0.5] * 19, 0.10)
        assert threshold == pytest.approx(0.0001)
        assert rejected[0] and not any(rejected[1:])

    def test_the_bar_rises_with_the_number_of_tests(self):
        """p = 0.02 is a discovery among 5 tests and not among 200."""
        assert audit.benjamini_hochberg([0.02] + [0.5] * 4, 0.10)[1][0]
        assert not audit.benjamini_hochberg([0.02] + [0.5] * 199, 0.10)[1][0]

    def test_empty_batch(self):
        assert audit.benjamini_hochberg([], 0.10) == (None, [])


class TestAudit:
    def test_it_counts_the_tests_behind_the_admissions(self, conn):
        _gate_trial(conn, "carry", "admitted", 0.001)
        _gate_trial(conn, "trend_ts", "rejected", 0.4)
        for _ in range(30):
            registry.add_trial(conn, "carry", "optimize", {}, {}, "scored", universe="crypto", finished=True)
        conn.commit()
        report = audit.run(conn)
        assert report.tests == 2 and report.admitted == 1 and report.search_evaluations == 30
        assert report.expected_false_positives == pytest.approx(0.2)
        assert "30 search evaluations behind them" in report.headline

    def test_a_marginal_admission_does_not_survive_a_wide_search(self, conn):
        """Admitted at p = 0.09 against its family; indefensible against forty tests."""
        _gate_trial(conn, "carry", "admitted", 0.09)
        for i in range(40):
            _gate_trial(conn, f"fam_{i}", "rejected", 0.5)
        conn.commit()
        report = audit.run(conn)
        assert report.admitted == 1 and report.survivors == 0
        row = next(r for r in report.rows if r.verdict == "admitted")
        assert row.survives_bh is False

    def test_a_strong_admission_survives(self, conn):
        _gate_trial(conn, "carry", "admitted", 0.0005)
        for i in range(40):
            _gate_trial(conn, f"fam_{i}", "rejected", 0.5)
        conn.commit()
        report = audit.run(conn)
        assert report.survivors == 1

    def test_the_deflated_sharpe_is_recomputed_against_the_whole_arena(self, conn):
        _gate_trial(conn, "carry", "admitted", 0.001, sr_period=0.02, obs=8000)
        conn.commit()
        alone = next(r for r in audit.run(conn).rows if r.verdict == "admitted").dsr_arena
        for i in range(200):
            _gate_trial(conn, f"fam_{i}", "rejected", 0.5)
        conn.commit()
        crowded = next(r for r in audit.run(conn).rows if r.trial_id is not None and r.verdict == "admitted")
        assert crowded.dsr_arena < alone  # the same result, paying for everything tried around it

    def test_arenas_are_audited_apart(self, conn):
        _gate_trial(conn, "carry", "admitted", 0.001, universe="crypto")
        _gate_trial(conn, "trend_ts", "admitted", 0.001, universe="classic")
        conn.commit()
        assert audit.run(conn, "crypto").tests == 1
        assert audit.run(conn, "classic").tests == 1

    def test_a_trial_without_a_p_value_is_treated_as_no_evidence(self, conn):
        registry.add_trial(conn, "carry", "walkforward", {}, {}, "admitted", universe="crypto", finished=True)
        conn.commit()
        report = audit.run(conn)
        assert report.rows[0].bootstrap_p is None and report.survivors == 0

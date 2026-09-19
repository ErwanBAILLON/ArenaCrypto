"""HoldingCompetitor: decide once per rebalance slot, hold in between, dead band, exits, state round trip.

The subclass under test returns a scripted sequence of decisions so that every
assertion is about the holding logic itself, not about a signal.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from arena.competitors.base import HoldingCompetitor
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target

SYMBOLS = ["BTC", "ETH"]


def snap_at(ts: str) -> Snapshot:
    """A Snapshot with no market data: the holding logic only reads ``snap.ts``."""
    return Snapshot(datetime.fromisoformat(ts).replace(tzinfo=UTC), SYMBOLS, {})


class Scripted(HoldingCompetitor):
    """Returns the scripted decisions in order and counts how often it was asked."""

    family = "scripted_holding_test"

    def __init__(self, script: list[Decision], **kwargs):
        super().__init__(**kwargs)
        self.script = list(script)
        self.calls = 0

    def compute(self, snap: Snapshot) -> Decision:
        self.calls += 1
        if len(self.script) > 1:
            return self.script.pop(0)
        return dict(self.script[0])


class WeeklyScripted(Scripted):
    family = "scripted_weekly_holding_test"
    rebalance_weekday = 0  # Mondays 00:00 UTC


D1: Decision = {"BTC": Target(0.5, conviction=0.8, reason={"n": 1}), "ETH": Target(-0.3, conviction=0.6)}
D2: Decision = {"BTC": Target(-0.5, conviction=0.9, reason={"n": 2}), "ETH": Target(0.3, conviction=0.7)}


def test_first_call_decides_whatever_the_hour():
    c = Scripted([D1])
    out = c.decide(snap_at("2025-03-05T13:00:00"))  # Wednesday, 13:00 UTC
    assert out == D1
    assert c.calls == 1


def test_holds_between_rebalance_hours_and_redecides_at_midnight():
    c = Scripted([D1, D2])
    c.decide(snap_at("2025-03-05T13:00:00"))
    for hour in ("14", "15", "23"):
        assert c.decide(snap_at(f"2025-03-05T{hour}:00:00")) == D1
    assert c.calls == 1, "no recomputation outside rebalance_hour"
    assert c.decide(snap_at("2025-03-06T00:00:00")) == D2
    assert c.calls == 2
    assert c.decide(snap_at("2025-03-06T01:00:00")) == D2
    assert c.calls == 2


def test_same_bar_twice_does_not_recompute():
    c = Scripted([D1, D2])
    ts = snap_at("2025-03-06T00:00:00")
    first = c.decide(ts)
    assert c.decide(ts) == first
    assert c.calls == 1


def test_band_ignores_small_weight_changes_but_takes_new_conviction_and_reason():
    small = {"BTC": Target(0.55, conviction=0.2, reason={"n": 9}), "ETH": Target(-0.3, conviction=0.6)}
    c = Scripted([D1, small])
    c.decide(snap_at("2025-03-05T00:00:00"))
    out = c.decide(snap_at("2025-03-06T00:00:00"))
    assert out["BTC"].weight == pytest.approx(0.5), "|dw| = 0.05 < band: old weight kept"
    assert out["BTC"].conviction == 0.2
    assert out["BTC"].reason == {"n": 9}
    assert out["ETH"] == D1["ETH"]


def test_band_applies_large_changes_and_kind_changes():
    big = {"BTC": Target(0.65, conviction=0.8), "ETH": Target(0.3, kind="carry", conviction=0.6)}
    c = Scripted([D1, big])
    c.decide(snap_at("2025-03-05T00:00:00"))
    out = c.decide(snap_at("2025-03-06T00:00:00"))
    assert out["BTC"].weight == pytest.approx(0.65), "|dw| = 0.15 >= band: new weight"
    assert out["ETH"].kind == "carry" and out["ETH"].weight == pytest.approx(0.3), "kind change bypasses the band"


def test_dropped_symbol_exits():
    c = Scripted([D1, {"BTC": Target(0.5)}])
    c.decide(snap_at("2025-03-05T00:00:00"))
    out = c.decide(snap_at("2025-03-06T00:00:00"))
    assert "ETH" not in out
    assert set(out) == {"BTC"}
    # and the exit is held afterwards
    assert set(c.decide(snap_at("2025-03-06T07:00:00"))) == {"BTC"}


def test_weekly_variant_only_redecides_on_monday_midnight():
    c = WeeklyScripted([D1, D2])
    assert c.decide(snap_at("2025-03-05T00:00:00")) == D1  # Wednesday: first call decides
    assert c.decide(snap_at("2025-03-06T00:00:00")) == D1  # Thursday 00:00: hold
    assert c.decide(snap_at("2025-03-09T00:00:00")) == D1  # Sunday 00:00: hold
    assert c.calls == 1
    assert c.decide(snap_at("2025-03-10T00:00:00")) == D2  # Monday 00:00: re-decide
    assert c.calls == 2
    assert c.decide(snap_at("2025-03-10T05:00:00")) == D2  # Monday 05:00: hold
    assert c.decide(snap_at("2025-03-17T00:00:00")) == D2  # next Monday: compute again, script exhausted -> D2
    assert c.calls == 3


def test_state_roundtrip_reproduces_held_decision_in_fresh_instance():
    a = Scripted([D1, D2])
    ts = snap_at("2025-03-05T00:00:00")
    held = a.decide(ts)
    st = a.state()
    assert st["held_ts"] == ts.ts.isoformat()
    assert set(st["held"]) == set(D1)

    b = Scripted([D2])  # would decide D2 if it ever computed
    b.restore_state(st)
    assert b.decide(ts) == held, "same bar: replay, no compute"
    assert b.decide(snap_at("2025-03-05T09:00:00")) == held, "non-rebalance hour: hold"
    assert b.calls == 0
    assert b.state() == st

    assert b.decide(snap_at("2025-03-06T00:00:00")) == D2
    assert b.calls == 1


def test_restore_state_tolerates_missing_optional_fields():
    b = Scripted([D2])
    b.restore_state({"held_ts": "2025-03-05T00:00:00+00:00", "held": {"BTC": {"weight": 0.4}}})
    out = b.decide(snap_at("2025-03-05T03:00:00"))
    assert out == {"BTC": Target(0.4, conviction=0.5, kind="perp", reason={})}
    assert b.calls == 0

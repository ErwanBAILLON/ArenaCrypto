"""Membership is stored once and read back, never recomputed with hindsight."""

from __future__ import annotations

import pandas as pd
import pytest

from arena.core.membership import Member
from arena.store import membership as mstore

U = "wide"
T1 = pd.Timestamp("2024-01-01", tz="UTC")
T2 = pd.Timestamp("2024-01-08", tz="UTC")


def _members(*names):
    return [Member(n, i, 1e7 - i, 0.05) for i, n in enumerate(names)]


def test_write_and_read_back(conn):
    assert mstore.write_members(conn, U, T1, _members("BTCUSDT", "ETHUSDT")) == 2
    conn.commit()
    got = mstore.members_at(conn, U, T1)
    assert [m.symbol for m in got] == ["BTCUSDT", "ETHUSDT"]
    assert got[0].rank == 0 and got[0].liquidity.usable


def test_membership_in_force_is_the_last_rebalance_at_or_before(conn):
    mstore.write_members(conn, U, T1, _members("A", "B"))
    mstore.write_members(conn, U, T2, _members("A", "C"))
    conn.commit()
    mid = T1 + pd.Timedelta(days=3)
    assert [m.symbol for m in mstore.members_at(conn, U, mid)] == ["A", "B"]
    assert [m.symbol for m in mstore.members_at(conn, U, T2 + pd.Timedelta(days=3))] == ["A", "C"]


def test_before_the_first_rebalance_there_is_no_universe(conn):
    mstore.write_members(conn, U, T2, _members("A"))
    conn.commit()
    assert mstore.members_at(conn, U, T1) == []


def test_rewriting_a_date_replaces_it(conn):
    mstore.write_members(conn, U, T1, _members("A", "B", "C"))
    mstore.write_members(conn, U, T1, _members("A"))
    conn.commit()
    assert [m.symbol for m in mstore.members_at(conn, U, T1)] == ["A"]


def test_universes_do_not_leak_into_each_other(conn):
    mstore.write_members(conn, "crypto", T1, _members("BTCUSDT"))
    mstore.write_members(conn, "wide", T1, _members("DOGEUSDT"))
    conn.commit()
    assert [m.symbol for m in mstore.members_at(conn, "crypto", T1)] == ["BTCUSDT"]
    assert [m.symbol for m in mstore.members_at(conn, "wide", T1)] == ["DOGEUSDT"]


def test_coverage_summarises_a_build(conn):
    mstore.write_members(conn, U, T1, _members("A", "B", "C"))
    mstore.write_members(conn, U, T2, _members("A", "D"))
    conn.commit()
    cov = mstore.coverage(conn, U)
    assert cov["dates"] == 2 and cov["symbols"] == 4 and cov["rows"] == 5
    assert cov["avg_size"] == pytest.approx(2.5)
    assert mstore.rebalance_timestamps(conn, U) == [T1, T2]


def test_membership_frame_is_ordered_for_backtests(conn):
    mstore.write_members(conn, U, T2, _members("A", "B"))
    mstore.write_members(conn, U, T1, _members("C"))
    conn.commit()
    frame = mstore.membership_frame(conn, U)
    assert list(frame["ts"]) == [T1, T2, T2]
    assert list(frame["symbol"]) == ["C", "A", "B"]

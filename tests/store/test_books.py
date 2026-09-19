from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from arena.core.types import Alert, BookRow, CompetitorSpec, Target
from arena.store import books as repo
from arena.store.registry import insert_competitor

T0 = datetime(2024, 6, 1, tzinfo=timezone.utc)
H = timedelta(hours=1)


@pytest.fixture
def cids(conn) -> list[int]:
    return [
        insert_competitor(conn, CompetitorSpec(None, f"c{i}", "carry", 1, {}, status="champion"))
        for i in range(2)
    ]


def row(ts: datetime, nav: float, ret: float) -> BookRow:
    return BookRow(ts=ts, nav=nav, ret=ret, gross=0.5, turnover=0.1, fees=0.0001, funding_pnl=0.0)


def test_targets_write_skips_zero_and_last_targets(conn, cids):
    cid = cids[0]
    assert repo.last_targets(conn, cid) is None
    repo.write_targets(conn, cid, T0, {
        "BTC": Target(0.4, 0.9, "perp", {"z": 1.2}),
        "ETH": Target(0.0),
        "SOL": Target(0.3, kind="carry"),
    })
    ts, targets = repo.last_targets(conn, cid)
    assert ts == T0 and ts.tzinfo is not None
    assert targets == {"BTC": ("perp", 0.4), "SOL": ("carry", 0.3)}

    repo.write_targets(conn, cid, T0, {"BTC": Target(-0.2)})  # same ts: overwrite, not duplicate
    _, targets = repo.last_targets(conn, cid)
    assert targets["BTC"] == ("perp", -0.2) and "SOL" in targets

    repo.write_targets(conn, cid, T0 + H, {"ETH": Target(0.1)})
    ts, targets = repo.last_targets(conn, cid)
    assert ts == T0 + H and targets == {"ETH": ("perp", 0.1)}
    assert repo.last_targets(conn, cids[1]) is None


def test_book_rows_last_returns_nav(conn, cids):
    a, b = cids
    assert repo.last_book_row(conn, a) is None
    repo.write_book_row(conn, a, row(T0, 10000.0, 0.0))
    repo.write_book_row(conn, a, row(T0 + H, 10100.0, 0.01))
    repo.write_book_row(conn, a, row(T0 + H, 10050.0, 0.005))  # rerun overwrites
    repo.write_book_row(conn, b, row(T0, 10000.0, 0.0))

    last = repo.last_book_row(conn, a)
    assert last == row(T0 + H, 10050.0, 0.005) and last.ts.tzinfo is not None

    wide = repo.read_returns(conn, [a, b], T0, T0 + 2 * H)
    assert list(wide.columns) == [a, b]
    assert list(wide.index) == [pd.Timestamp(T0), pd.Timestamp(T0 + H)]
    assert str(wide.index.tz) == "UTC"
    assert wide.loc[pd.Timestamp(T0 + H), a] == 0.005
    assert pd.isna(wide.loc[pd.Timestamp(T0 + H), b])

    nav = repo.read_nav(conn, a, T0, T0 + H)
    assert nav.tolist() == [10000.0, 10050.0] and str(nav.index.tz) == "UTC"

    empty = repo.read_returns(conn, [a], T0 + 5 * H, T0 + 6 * H)
    assert empty.empty and list(empty.columns) == [a]
    assert repo.read_nav(conn, b, T0 + 5 * H, T0 + 6 * H).empty


def test_allocations(conn, cids):
    a, b = cids
    assert repo.last_allocations(conn) == {}
    repo.write_allocations(conn, T0, {a: 0.7, b: 0.3})
    repo.write_allocations(conn, T0, {a: 0.6, b: 0.4})
    assert repo.last_allocations(conn) == {a: 0.6, b: 0.4}
    repo.write_allocations(conn, T0 + H, {a: 1.0})
    assert repo.last_allocations(conn) == {a: 1.0}


def test_alerts_lifecycle(conn, cids):
    a = cids[0]
    now = datetime.now(timezone.utc)
    i1 = repo.add_alert(conn, Alert("signal", {"weight": 0.4}, competitor_id=a, symbol="BTC"))
    i2 = repo.add_alert(conn, Alert("digest", {"text": "daily"}))

    pending = repo.unsent_alerts(conn)
    assert [p[0] for p in pending] == [i1, i2]
    _, alert, ts = pending[0]
    assert alert == Alert("signal", {"weight": 0.4}, competitor_id=a, symbol="BTC")
    assert ts.tzinfo is not None
    assert pending[1][1].competitor_id is None and pending[1][1].symbol is None

    assert repo.alerts_sent_today(conn, now) == 0
    repo.mark_sent(conn, [i1])
    assert [p[0] for p in repo.unsent_alerts(conn)] == [i2]
    assert repo.alerts_sent_today(conn, now) == 1
    assert repo.alerts_sent_today(conn, now + timedelta(days=1)) == 0
    repo.mark_sent(conn, [])
    repo.mark_sent(conn, [i2])
    assert repo.unsent_alerts(conn) == []
    assert repo.alerts_sent_today(conn, now) == 2


def test_recent_alert_exists_matches_null_fields(conn, cids):
    a = cids[0]
    now = datetime.now(timezone.utc)
    repo.add_alert(conn, Alert("drift", {}, competitor_id=a))
    assert repo.recent_alert_exists(conn, "drift", a, None, now - H)
    assert not repo.recent_alert_exists(conn, "drift", a, "BTC", now - H)
    assert not repo.recent_alert_exists(conn, "drift", None, None, now - H)
    assert not repo.recent_alert_exists(conn, "signal", a, None, now - H)
    assert not repo.recent_alert_exists(conn, "drift", a, None, now + H)

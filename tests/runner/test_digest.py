"""Digest built from a seeded Postgres: two competitors, books and targets now and 25 h ago."""

from datetime import UTC, datetime, timedelta

from psycopg.types.json import Jsonb

from arena.core.types import BookRow, CompetitorSpec, Target
from arena.runner import digest
from arena.store import books as bstore
from arena.store import registry

NOW = datetime(2026, 9, 20, 6, 0, tzinfo=UTC)


def _book(conn, cid: int, start: datetime, end: datetime, nav0: float, hourly: float) -> None:
    ts, nav = start, nav0
    while ts <= end:
        bstore.write_book_row(conn, cid, BookRow(ts, nav, hourly, 1.0, 0.0, 0.0, 0.0))
        nav *= 1 + hourly
        ts += timedelta(hours=1)


def _seed(conn):
    carry = registry.insert_competitor(conn, CompetitorSpec(None, "carry_v1", "carry", 1, {}, status="champion"))
    xs = registry.insert_competitor(
        conn, CompetitorSpec(None, "xs_momentum_v1", "xs_momentum", 1, {}, status="challenger")
    )
    carry2 = registry.insert_competitor(conn, CompetitorSpec(None, "carry_v2", "carry", 2, {}, status="challenger"))
    btc = registry.insert_competitor(
        conn, CompetitorSpec(None, "bench_btc_hold", "bench_btc_hold", 1, {}, role="benchmark", status="champion")
    )
    start = NOW - timedelta(days=31)
    _book(conn, carry, start, NOW, 10_000.0, 0.0001)
    _book(conn, xs, start, NOW, 10_000.0, -0.00005)
    _book(conn, carry2, start, NOW, 10_000.0, -0.00002)
    _book(conn, btc, start, NOW, 10_000.0, 0.00005)
    yesterday = NOW - timedelta(hours=25)
    bstore.write_targets(
        conn,
        carry,
        yesterday,
        {
            "ETH": Target(0.2, 0.7, "carry", {"mean_funding_8h": 0.0001, "rank": 0}),
            "SOL": Target(0.1, 0.6, "carry", {"mean_funding_8h": 0.00008, "rank": 1}),
        },
    )
    bstore.write_targets(
        conn,
        carry,
        NOW,
        {
            "SUI": Target(0.2, 0.8, "carry", {"mean_funding_8h": 0.00012, "rank": 0}),
            "SOL": Target(0.3, 0.6, "carry", {"mean_funding_8h": 0.00009, "rank": 1}),
        },
    )
    bstore.write_targets(conn, xs, NOW, {"BTC": Target(-0.3, 0.5, "perp", {"rank": 5, "score": -0.1})})
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO tick_runs (bar_ts, started_at, finished_at, booked, skipped, ingested)"
            " VALUES (%s, %s, %s, 3, 0, %s)",
            (NOW, NOW + timedelta(minutes=3), NOW + timedelta(minutes=4), Jsonb({})),
        )
        cur.execute(
            "INSERT INTO alerts (ts, kind, competitor_id, payload) VALUES (%s, 'drift', %s, %s)",
            (NOW - timedelta(hours=2), xs, Jsonb({"detail": "xs_momentum_v1: no non-zero target for 7+ days"})),
        )
    conn.commit()


def test_build_has_four_french_sections_and_changes(conn):
    _seed(conn)
    text = digest.build(conn, NOW + timedelta(minutes=10))
    assert text.startswith("📊 L'arène ce matin — 20/09/2026")
    for header in ("Qui parle :", "Ce qui a changé depuis hier :", "Les prétendants :", "Repères :", "Santé :"):
        assert header in text, header

    # who speaks: the carry champion, its positions explained, P&L in euros vs BTC
    assert "• Carry de funding (carry v1)" in text
    assert "Carry sur SOL (30 % du capital) : funding moyen 0,0090 % par 8 h" in text
    assert "Carry sur SUI (20 % du capital)" in text
    assert "P&L hier +" in text and "devant « garder du BTC »" in text

    # changes over 24 h: entry, exit, resize
    assert "• carry v1 entre sur SUI (20 % du capital)." in text
    assert "• carry v1 sort de ETH." in text
    assert "• carry v1 renforce SOL : 10 % -> 30 % du capital." in text
    assert "• rien" not in text

    # challengers: progress in days/decisions, verdict in words
    assert "• xs_momentum v1 (Momentum relatif (classement)) : 31/42 jours, 1/100 décisions." in text
    assert "pas de champion à battre dans sa famille" in text  # no xs_momentum champion
    assert "• carry v2 (Carry de funding) : 31/42 jours, 0/100 décisions." in text
    assert "en retard sur le champion" in text  # carry_v2 loses while carry_v1 wins

    # benchmarks and health
    assert "Repères : garder du BTC 30 j : +" in text
    assert "la chance (95e centile) : n/d" in text  # no null_random seeded
    assert "Santé : dernier passage 06:04 UTC, 3 modèles évalués, dérives :\n• xs_momentum_v1" in text
    assert "• xs_momentum_v1 n'a pris aucune position depuis 7 jours ou plus." in text


def test_build_without_changes_says_rien(conn):
    carry = registry.insert_competitor(conn, CompetitorSpec(None, "carry_v1", "carry", 1, {}, status="champion"))
    _book(conn, carry, NOW - timedelta(days=2), NOW, 10_000.0, 0.0)
    pos = {"ETH": Target(0.2, 0.7, "carry", {"mean_funding_8h": 0.0001, "rank": 0})}
    bstore.write_targets(conn, carry, NOW - timedelta(hours=25), pos)
    bstore.write_targets(conn, carry, NOW, pos)
    conn.commit()
    ctx = digest.build_context(conn, NOW)
    assert ctx.changes == []
    assert ctx.last_tick is None
    text = digest.build(conn, NOW)
    assert "Ce qui a changé depuis hier :\n• rien" in text
    assert "Les prétendants :\n• aucun" in text
    assert "P&L hier +0 € / 7 j +0 € / 30 j +0 €." in text
    assert "Santé : aucun passage enregistré" in text

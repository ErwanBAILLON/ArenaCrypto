from arena.store.db import run_migrations


def test_migrations_idempotent(conn):
    assert run_migrations(conn) == []  # already applied by fixture
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        tables = {r["table_name"] for r in cur.fetchall()}
    for t in ("candles", "funding", "competitors", "trials", "targets", "books", "alerts", "models", "articles", "article_scores"):
        assert t in tables

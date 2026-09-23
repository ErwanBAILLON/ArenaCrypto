from arena.store.db import run_migrations


def test_migrations_idempotent(conn):
    assert run_migrations(conn) == []  # already applied by fixture
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        tables = {r["table_name"] for r in cur.fetchall()}
    for t in (
        "candles",
        "funding",
        "competitors",
        "trials",
        "targets",
        "books",
        "alerts",
        "models",
        "articles",
        "article_scores",
    ):
        assert t in tables


def test_gate_admitted_backfill_reads_the_rationale(conn):
    """0005 must not silently make every legacy competitor promotable."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM competitors")
        cur.execute("ALTER TABLE competitors DROP COLUMN IF EXISTS gate_admitted")
        for name, role, rationale in [
            ("admitted_one", "competitor", "founder; gate admitted; trial 1"),
            ("refused_one", "competitor", "founder; gate rejected: dsr, folds_positive; trial 2"),
            ("a_null", "null", "null model"),
        ]:
            cur.execute(
                "INSERT INTO competitors (name, family, version, role, status, rationale)"
                " VALUES (%s, 'trend_ts', 1, %s, 'challenger', %s)",
                (name, role, rationale),
            )
        cur.execute("DELETE FROM schema_migrations WHERE name = '0005_gate_admitted.sql'")
    conn.commit()

    from arena.store.db import run_migrations

    assert "0005_gate_admitted.sql" in run_migrations(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT name, gate_admitted FROM competitors ORDER BY name")
        flags = {r["name"]: r["gate_admitted"] for r in cur.fetchall()}
    assert flags == {"admitted_one": True, "refused_one": False, "a_null": False}


def test_migrate_reports_what_it_applied(conn, monkeypatch, capsys):
    """ "applied: nothing" while creating the whole schema is a lie you read during an incident."""
    from typer.testing import CliRunner

    from arena import cli

    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    conn.commit()
    monkeypatch.setenv("DATABASE_URL", conn.info.dsn)
    monkeypatch.setenv("DRY_RUN", "true")

    runner = CliRunner()
    first = runner.invoke(cli.app, ["migrate"])
    assert first.exit_code == 0
    assert "0001_core.sql" in first.stdout and "0006_feed_health.sql" in first.stdout

    second = runner.invoke(cli.app, ["migrate"])
    assert second.exit_code == 0 and "nothing (schema already current)" in second.stdout

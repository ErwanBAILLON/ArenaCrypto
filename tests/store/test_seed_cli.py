"""`arena seed`: a challenger without the gate, once, with the family defaults merged."""

from typer.testing import CliRunner

from arena import cli
from arena.store import registry


def test_seed_inserts_an_ungated_challenger_once(conn, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", conn.info.dsn)
    monkeypatch.setenv("DRY_RUN", "true")
    runner = CliRunner()
    first = runner.invoke(cli.app, ["seed", "xs_sparse", "xs_sparse_sig_v1", "--params", '{"stop_sigma": 2.5}'])
    assert first.exit_code == 0, first.stdout
    spec = registry.get_competitor(conn, "xs_sparse_sig_v1")
    assert spec is not None and spec.status == "challenger" and spec.gate_admitted is False
    assert spec.params["stop_sigma"] == 2.5 and spec.params["k"] == 8  # override on top of the family defaults
    again = runner.invoke(cli.app, ["seed", "xs_sparse", "xs_sparse_sig_v1"])
    assert again.exit_code == 0 and "already present" in again.stdout
    bad = runner.invoke(cli.app, ["seed", "no_such_family", "x"])
    assert bad.exit_code == 2

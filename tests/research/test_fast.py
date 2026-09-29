"""The fast harness: a store built from long frames, rules priced with costs, parameters that reach the rule."""

import numpy as np
import pandas as pd

from arena.core.membership import Member
from arena.research import fast
from tests.conftest import make_candles, make_funding

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]


def _store():
    candles = make_candles(symbols=SYMS, bars=24 * 260, drift={"BTCUSDT": 0.0006, "DOGEUSDT": -0.0006}, seed=11)
    funding = make_funding(symbols=SYMS, candles=candles, rate=0.0001)
    stamps = sorted(candles["ts"].unique())
    mondays = [pd.Timestamp(t) for t in stamps[24 * 140 :: 24 * 7]]
    members = {t: [Member(s, i + 1, 5e8 / (i + 1), 0.04) for i, s in enumerate(SYMS)] for t in mondays}
    return fast.build_store(candles, funding, members), candles


def test_store_has_a_panel_per_rebalance_date_and_wide_frames():
    store, candles = _store()
    assert set(store.panel["ts"].unique()) <= set(store.dates)
    assert len(store.panel["ts"].unique()) >= len(store.dates) - 1
    assert sorted(store.closes.columns) == sorted(SYMS) and len(store.closes) == 24 * 260
    assert "dollar_volume_30d" in store.panel.columns and "vol_30d" in store.panel.columns


def test_one_name_fully_long_reproduces_its_own_hourly_returns():
    store, candles = _store()
    res = fast.run(store, lambda panel, ctx: {"BTCUSDT": 1.0}, fast.fee_model(None))
    first, last = store.dates[0], store.closes.index[-1]
    btc = store.closes["BTCUSDT"].pct_change()
    expected = btc.loc[(btc.index > first) & (btc.index <= last)].sum()
    assert abs(res["price"].sum() - expected) < 1e-9
    assert res["fee"].sum() > 0 and res["fund"].sum() < 0  # one entry paid; positive funding paid by the long
    assert (res["fee"] > 0).sum() == 1  # the weight never changes after the first rebalance: no more turnover


def test_parameters_reach_the_rule_so_variants_differ():
    store, _ = _store()
    fees = fast.fee_model(1_000_000.0)
    a = fast.run(store, fast.rule_tsmom, fees, n=2, lookback="ret_30d")
    b = fast.run(store, fast.rule_tsmom, fees, n=4, lookback="ret_90d")
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_summary_covers_quarters_and_every_rule_in_the_library_runs():
    store, _ = _store()
    qs = fast.quarters(store, warmup_days=30)
    assert qs
    for name, (rule, kw) in fast.RULES.items():
        res = fast.run(store, rule, fast.fee_model(200_000.0), **kw)
        f = fast.summarise(res, qs)
        assert not f.empty, name
        assert fast.headline(name, f).startswith(name)


def test_research_run_cli_prints_one_line_per_rule(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from arena import cli

    store, _ = _store()
    path = tmp_path / "store.pkl"
    pd.to_pickle(store, path)
    monkeypatch.setenv("DRY_RUN", "true")
    res = CliRunner().invoke(
        cli.app,
        ["research-run", str(path), "--rules", "null_long2,tsmom2_30d", "--capacity", "0", "--warmup-days", "30"],
    )
    assert res.exit_code == 0, res.stdout
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    assert len(lines) == 2 and lines[0].startswith("null_long2") and "sharpe med" in lines[1]
    bad = CliRunner().invoke(cli.app, ["research-run", str(path), "--rules", "nope"])
    assert bad.exit_code == 2

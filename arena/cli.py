"""Command line entry point (``arena <command>``); every CronJob runs one of these."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd
import typer

from arena.competitors.base import REGISTRY
from arena.core.types import Alert, CompetitorSpec
from arena.core.universe import load_universe
from arena.judge import admission
from arena.judge.gate import DEFAULT_GATE
from arena.settings import Settings
from arena.store import books as bstore
from arena.store import registry
from arena.store.db import connect, run_migrations
from arena.telegram import sender, templates

app = typer.Typer(add_completion=False, help="Signal-only crypto research arena.")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("arena")
logging.getLogger("httpx").setLevel(logging.WARNING)

# (family, name, params overriding the family defaults); one family may seed several founders
FOUNDERS: list[tuple[str, str, dict]] = [
    ("carry", "carry_v1", {}),
    ("trend_ts", "trend_ts_v1", {}),
    ("xs_momentum", "xs_momentum_v1", {}),
    ("regime", "regime_v1", {}),
    ("meta_label", "meta_label_v1", {}),
    ("price_action", "price_action_swing_v1", {"style": "swing"}),
    ("price_action", "price_action_position_v1", {"style": "position"}),
    ("funding_skew", "funding_skew_v1", {}),
    ("crowded_trend", "crowded_trend_v1", {}),
]
FUNDING_FAMILIES = {"carry", "bench_carry_equal", "funding_skew", "crowded_trend"}  # need perp funding: crypto only

# The forward promotion test compares a challenger to the NULL_Q quantile of the null
# models running beside it. A quantile of five numbers is a coin flip with extra steps;
# promotion.MIN_NULL_SAMPLES refuses to promote below this many, so the arena carries them.
N_NULL_COMPETITORS = 30


def _uname(universe, name: str) -> str:
    """Competitor names are global: suffix them outside the historical crypto universe."""
    return name if universe.name == "crypto" else f"{name}_{universe.name}"


# Founders of a point-in-time arena. Two rounds of evaluation (a held-out year,
# then ten and seventeen quarters walked forward) could not beat the five-signal
# picks-hedged weekly xs_sparse; the portfolio-sized variant trades Sharpe for
# return. xs_complex lost money in twelve quarters of seventeen and is not seeded.
WIDE_FOUNDERS: list[tuple[str, str, dict]] = [
    ("xs_sparse", "xs_sparse_v1", {}),
    ("xs_sparse", "xs_sparse_pvol_v1", {"vol_mode": "portfolio", "max_leverage": 3.0, "max_gross": 0.6}),
]


def founders_for(universe) -> list[tuple[str, str, dict]]:
    if universe.membership.enabled:
        return list(WIDE_FOUNDERS)
    return [f for f in FOUNDERS if universe.exchange == "binance" or f[0] not in FUNDING_FAMILIES]


def nulls_for(universe) -> list[CompetitorSpec]:
    """Null models and benchmarks of a universe (the classic one has no funding, hence no carry benchmark)."""
    u = universe.name
    mk = lambda name, fam, params, role: CompetitorSpec(  # noqa: E731
        None, _uname(universe, name), fam, 1, params, role=role, status="champion", universe=u
    )
    specs = [mk("null_cash", "null_cash", {}, "null")]
    specs += [mk(f"null_random_{i}", "null_random", {"seed": i}, "null") for i in range(N_NULL_COMPETITORS)]
    if universe.membership.enabled:
        # a point-in-time arena runs cross-sectional books; their fair null is neutral
        specs += [mk(f"null_neutral_{i}", "null_neutral", {"seed": i}, "null") for i in range(N_NULL_COMPETITORS)]
    if universe.exchange == "binance":
        specs.append(mk("bench_btc_hold", "bench_btc_hold", {}, "benchmark"))
        specs.append(mk("bench_carry_equal", "bench_carry_equal", {}, "benchmark"))
    else:
        specs.append(mk("bench_hold", "bench_hold", {"symbol": universe.reference}, "benchmark"))
    return specs


_LAST_MIGRATIONS: list[str] = []


def _ctx(report_migrations: bool = False):
    settings = Settings.from_env()
    if not settings.database_url:
        raise typer.BadParameter("DATABASE_URL is required")
    conn = connect(settings.database_url)
    applied = run_migrations(conn)  # idempotent and cheap: every command runs on a current schema
    if report_migrations:
        _LAST_MIGRATIONS[:] = applied
    return settings, conn, load_universe(settings.universe_path)


def _now() -> datetime:
    return datetime.now(UTC)


@app.command()
def migrate() -> None:
    """Apply pending SQL migrations and report what was applied.

    ``_ctx`` already migrates -- every sub-command runs on a current schema --
    so this reads what that call did instead of running it again. The previous
    version re-ran it, always found nothing left, and printed "applied:
    nothing" even when it had just created the whole schema, which is an
    actively misleading thing to read during an incident.
    """
    _, conn, _ = _ctx(report_migrations=True)
    applied = _LAST_MIGRATIONS
    typer.echo(f"applied: {', '.join(applied) if applied else 'nothing (schema already current)'}")


@app.command()
def backfill(since: str = typer.Option("2024-01-01", help="ISO date for the first candle")) -> None:
    """Load candles, funding, open interest, news and macro from ``since`` (idempotent)."""
    from arena.data.http import make_client
    from arena.runner.ingest import ingest_macro, ingest_market, ingest_news

    _, conn, universe = _ctx()
    start = pd.Timestamp(since, tz="UTC").to_pydatetime()
    with make_client(timeout=30.0) as client:
        counts = ingest_market(conn, client, universe, _now(), since=start)
        counts["news"] = ingest_news(conn, client, _now())["inserted"]
        counts["macro"] = ingest_macro(conn, client, _now())
    typer.echo(f"backfill: {counts}")


@app.command()
def news() -> None:
    """Fetch RSS feeds, store new articles and score them."""
    from arena.data.http import make_client
    from arena.runner.ingest import ingest_macro, ingest_news

    _, conn, _ = _ctx()
    with make_client() as client:
        out = ingest_news(conn, client, _now())
        out["macro"] = ingest_macro(conn, client, _now())
    typer.echo(f"news: {out}")


@app.command()
def tick(no_ingest: bool = typer.Option(False, help="Skip market ingestion (replay stored data)")) -> None:
    """Hourly tick: ingest, decide, book, allocate, drift, promote, alert."""
    from arena.data.http import make_client
    from arena.runner import alerts
    from arena.runner.tick import run as run_tick

    settings, conn, universe = _ctx()
    now = _now()
    with make_client(timeout=30.0) as client:
        rep = run_tick(conn, settings, universe, now, client=None if no_ingest else client, ingest=not no_ingest)
    names = {s.id: s.name for s in registry.list_competitors(conn)}
    sent = alerts.flush(conn, settings, now, names)
    typer.echo(
        f"tick {rep.ts}: booked={len(rep.booked)} skipped={len(rep.skipped)} failed={rep.failed} "
        f"ingested={rep.ingested} alerts_sent={sent}"
    )


@app.command()
def digest() -> None:
    """Send the daily leaderboard digest."""
    from arena.runner.digest import build

    settings, conn, _ = _ctx()
    text = build(conn, _now())
    ok = sender.send(settings, text)
    typer.echo(text if settings.dry_run else f"digest sent: {ok}")


def _history_and_null(conn, universe, end: datetime):
    from arena.runner.history import load_history
    from arena.runner.tick import fees_of

    history = load_history(conn, universe, universe.history_start, end)
    if history.candles.empty:
        raise typer.Exit(
            code=typer.echo(f"no candles stored for universe {universe.name}: run `arena backfill` first") or 2
        )
    fees = fees_of(universe)
    start = pd.Timestamp(universe.history_start)
    thr = admission.cached_null_threshold(
        conn, history, universe.symbols, start, end, fees, bar_hours=universe.bar_hours
    )
    return history, fees, start, thr


@app.command()
def judge(
    family: str = typer.Argument(..., help="Family to gate with default params (or NAME of a competitor)"),
) -> None:
    """Run the entry gate and print the verdict (records a trial).

    Given the NAME of a competitor already in the arena, this is also how a
    model the gate once refused becomes promotable again: an admitted verdict
    sets its ``gate_admitted`` flag (see ``arena.runner.promotion``).
    """
    _, conn, universe = _ctx()
    spec = registry.get_competitor(conn, family)
    fam, params = (spec.family, spec.params) if spec else (family, {})
    end = _now()
    registry.abandon_stale_trials(conn)
    conn.commit()
    history, fees, start, thr = _history_and_null(conn, universe, end)
    adm = admission.admit(
        conn,
        fam,
        params,
        history,
        universe.symbols,
        start,
        end,
        fees,
        thr,
        notes="cli judge",
        bar_hours=universe.bar_hours,
        universe=universe.name,
    )
    v = adm.verdict
    typer.echo(f"{fam}: {'ADMITTED' if v.admitted else 'REJECTED'} failed={v.failed}")
    for k, val in v.metrics.items():
        typer.echo(f"  {k}: {val:.4f}" if isinstance(val, float) else f"  {k}: {val}")
    if spec is not None and spec.id is not None and spec.gate_admitted != v.admitted:
        registry.set_gate_admitted(conn, spec.id, v.admitted)
        conn.commit()
        typer.echo(f"  gate_admitted: {spec.gate_admitted} -> {v.admitted} (promotable: {v.admitted})")


@app.command()
def challenger(family: str = typer.Option("", help="Restrict to one rule family"), n_trials: int = 15) -> None:
    """Weekly Optuna search → new challengers for rule families."""
    from arena.challenger import optimize

    settings, conn, universe = _ctx()
    end = _now()
    history, fees, start, thr = _history_and_null(conn, universe, end)
    fams = [family] if family else list(optimize.SPACES)
    for fam in fams:
        spec = optimize.run(
            conn,
            fam,
            history,
            universe.symbols,
            start,
            end,
            fees,
            thr,
            n_trials=n_trials,
            seed=int(end.timestamp()) % 10_000,
            bar_hours=universe.bar_hours,
            universe=universe.name,
        )
        typer.echo(f"{fam}: {'challenger ' + spec.name if spec else 'no new challenger'}")
        if spec:
            bstore.add_alert(
                conn,
                Alert(
                    kind="info",
                    competitor_id=spec.id,
                    payload={"detail": f"new challenger {spec.name}: {spec.rationale}"},
                ),
            )
            conn.commit()


@app.command()
def retrain(window_days: int = 365) -> None:
    """Retrain the meta-labeling model → new meta_label challenger."""
    from arena.challenger import retrain as rt
    from arena.runner.history import load_history

    _, conn, universe = _ctx()
    end = _now()
    history = load_history(conn, universe, universe.history_start, end)
    spec = rt.run(conn, history, universe.symbols, end, window_days=window_days)
    typer.echo(f"meta_label: {'challenger ' + spec.name if spec else 'no new challenger'}")


@app.command()
def bootstrap(since: str = typer.Option("2024-01-01"), skip_backfill: bool = False) -> None:
    """migrate → backfill → register null models → gate founders → announce."""
    settings, conn, universe = _ctx()
    run_migrations(conn)
    if not skip_backfill:
        backfill(since)
    existing = {s.name for s in registry.list_competitors(conn)}
    for spec in nulls_for(universe):
        if spec.name not in existing:
            registry.insert_competitor(conn, spec)
    conn.commit()
    end = _now()
    history, fees, start, thr = _history_and_null(conn, universe, end)
    typer.echo(f"null 95th pct Sharpe: {thr:.3f}")
    names_present = {
        s.name for s in registry.list_competitors(conn, statuses=["champion", "challenger"], universe=universe.name)
    }
    lines = []
    for fam, name, extra in founders_for(universe):
        if _uname(universe, name) in names_present:
            lines.append(f"{name}: already present")
            continue
        params = {**REGISTRY[fam].default_params, **extra}
        params.pop("model_str", None)
        adm = admission.admit(
            conn,
            fam,
            params,
            history,
            universe.symbols,
            start,
            end,
            fees,
            thr,
            notes="bootstrap founder",
            bar_hours=universe.bar_hours,
        )
        status = "champion" if adm.verdict.admitted else "challenger"
        cid = registry.insert_competitor(
            conn,
            CompetitorSpec(
                None,
                _uname(universe, name),
                fam,
                1,
                params,
                status=status,
                universe=universe.name,
                gate_admitted=adm.verdict.admitted,
                rationale=(
                    "founder; gate "
                    f"{'admitted' if adm.verdict.admitted else 'rejected: ' + ', '.join(adm.verdict.failed)}; "
                    f"trial {adm.trial_id}"
                ),
            ),
        )
        if not adm.verdict.admitted:
            bstore.add_alert(
                conn,
                Alert(
                    kind="rejected",
                    competitor_id=cid,
                    payload={
                        "detail": (
                            f"{name} rejected at gate ({', '.join(adm.verdict.failed)}); observed forward as a "
                            "challenger but not promotable until it clears the gate"
                        )
                    },
                ),
            )
        conn.commit()
        m = adm.verdict.metrics
        lines.append(
            f"{name}: {status} sharpe={m.get('sharpe', 0):.2f} dsr={m.get('dsr', 0):.2f} "
            f"p={m.get('bootstrap_p', 1):.2f} mdd={m.get('max_drawdown', 0):.1%}"
        )
    if "news_v1" not in names_present and universe.exchange == "binance":  # the news lexicon is crypto-specific
        registry.insert_competitor(
            conn,
            CompetitorSpec(
                None,
                "news_v1",
                "news",
                1,
                {},
                status="challenger",
                universe=universe.name,
                gate_admitted=True,  # cannot be backtested: its scores only exist forward (promotion.py)
                rationale="founder; forward-only family, not backtest-gated",
            ),
        )
        conn.commit()
        lines.append("news: challenger (forward-only)")
    summary = f"Arena bootstrapped ({universe.name})\n" + "\n".join(lines) + f"\nnull 95th pct Sharpe {thr:.2f}"
    typer.echo(summary)
    sender.send(settings, templates.event_alert("info", summary))


@app.command("universe-build")
def universe_build(
    since: str = typer.Option("", help="Override the universe's history_start"),
    workers: int = typer.Option(12, help="Parallel archive downloads"),
    backfill: bool = typer.Option(True, help="Also fetch hourly history for members lacking it"),
) -> None:
    """Rebuild the point-in-time universe from Binance's public archive.

    Probes every archived perpetual on daily bars -- including the delisted
    ones, which is the whole point -- ranks them by trailing dollar volume at
    every weekly date, and stores the membership. A universe built from today's
    survivors is a survivorship bias with a config file around it.
    """
    from arena.runner import wide_backfill as wb
    from arena.store import membership as mstore

    _, conn, universe = _ctx()
    if not universe.membership.enabled:
        typer.echo(f"{universe.name}: membership is not enabled for this arena")
        raise typer.Exit(code=1)
    start = pd.Timestamp(since) if since else pd.Timestamp(universe.history_start)
    end = _now()
    typer.echo(f"probing archived perpetuals from {start.date()} to {end.date()}...")
    build = wb.probe_and_select(start.to_pydatetime(), end, universe.membership_rule(), workers=workers)
    written = 0
    for ts, members in build.members_by_date.items():
        written += mstore.write_members(conn, universe.name, ts, members)
    conn.commit()
    typer.echo(
        f"{universe.name}: {build.candidates} candidates, {build.probed} with history, "
        f"{len(build.members_by_date)} dates, {written} rows"
    )
    typer.echo(f"  {len(build.symbols)} symbols were members at least once, weekly churn {build.churn:.1%}")
    if backfill:
        _backfill_members(conn, universe, build.symbols, start.to_pydatetime(), end, workers)


def _backfill_members(conn, universe, symbols: list[str], start, end, workers: int) -> None:
    """Hourly candles and funding from the archive for every member lacking history.

    The incremental REST ingest only walks forward from the last stored bar, so
    a symbol that joins the universe with no history would start life with a
    single bar and never satisfy a warm-up. The archive fills the past once.
    """
    from arena.data import binance_archive as archive
    from arena.data.http import make_client
    from arena.runner.wide_backfill import fetch_many
    from arena.store import candles as cstore

    missing = [s for s in symbols if cstore.last_candle_ts(conn, "binance", s) is None]
    typer.echo(f"  backfilling {len(missing)} symbols without stored history...")
    frames = fetch_many(missing, start, end, "1h", workers)
    rows = 0
    for sym, frame in frames.items():
        frame["symbol"] = sym
        rows += cstore.upsert_candles(conn, "binance", frame)
        conn.commit()
    funded = 0
    with make_client(timeout=60.0) as client:
        for sym in missing:
            f = archive.funding_history(client, sym, start, end)
            if not f.empty:
                f["symbol"] = sym
                funded += cstore.upsert_funding(conn, "binance", f)
                conn.commit()
    typer.echo(f"  wrote {rows} candles and {funded} funding stamps")


@app.command("train-xs")
def train_xs_cmd(
    family: str = typer.Option("xs_complex", help="Family to train"),
    n_features: int = typer.Option(4000, help="Random Fourier feature width"),
    gamma: float = typer.Option(0.02, help="RBF bandwidth"),
    capacity: float = typer.Option(1_000_000.0, help="Deployment size the labels are costed at"),
) -> None:
    """Train the cross-sectional model and insert it as a challenger.

    Labels are triple barriers on the excess return over the universe, net of
    the per-symbol cost at ``capacity``; the shrinkage is chosen by purged
    combinatorial cross-validation and the probability of backtest overfitting
    is recorded alongside it. A run whose PBO says the winner does not
    generalise inserts nothing.
    """
    from arena.challenger import train_xs as tx
    from arena.runner.history import load_history
    from arena.runner.tick import fees_of
    from arena.store import membership as mstore

    _, conn, universe = _ctx()
    end = _now()
    history = load_history(conn, universe, universe.history_start, end)
    frame = mstore.membership_frame(conn, universe.name)
    if frame.empty:
        typer.echo("no stored membership: run `arena universe-build` first")
        raise typer.Exit(code=1)
    symbols_at = {ts: list(g["symbol"]) for ts, g in frame.groupby("ts")}
    fees = fees_of(universe)
    round_trip = 2 * fees.cost("perp", 0.05, None) if fees.impact else 2 * fees.perp_cost

    data = tx.build_dataset(history.candles, history.funding, symbols_at, costs=round_trip)
    typer.echo(f"dataset: {len(data)} rows, {len(data.columns)} features")
    selection = tx.select(data, n_features=n_features, gamma=gamma)
    typer.echo(
        f"lambda {selection.lam}  cv spread {selection.score:+.5f}  "
        f"pbo {selection.pbo.get('pbo', 1):.2f}  leak {selection.leakage['max_overlap_ns']}"
    )
    if selection.pbo.get("pbo", 1.0) > DEFAULT_GATE.max_pbo:
        typer.echo("PBO says picking the best shrinkage does not generalise; nothing inserted")
        raise typer.Exit(code=0)
    model = tx.train(data, selection)
    version = 1 + max(
        [c.version for c in registry.list_competitors(conn, universe=universe.name) if c.family == family], default=0
    )
    cid = registry.insert_competitor(
        conn,
        CompetitorSpec(
            None,
            _uname(universe, f"{family}_v{version}"),
            family,
            version,
            {},
            status="challenger",
            universe=universe.name,
            gate_admitted=False,
            rationale=f"trained: P={n_features} lambda={selection.lam} cv={selection.score:+.5f}",
        ),
    )
    registry.save_model(conn, cid, model.to_json().encode(), model.metrics)
    conn.commit()
    typer.echo(f"inserted {family}_v{version} (id {cid}) with a {len(model.to_json()) // 1024} KB artefact")
    typer.echo("not promotable until `arena judge` admits it")


@app.command()
def nulls() -> None:
    """Register any missing null models and benchmarks for this universe (idempotent, cheap).

    The forward promotion test compares a challenger to a quantile of the null
    models running beside it, and refuses to promote on fewer than
    ``promotion.MIN_NULL_SAMPLES`` of them. An arena bootstrapped when the
    target was five needs the rest added without re-running the whole
    bootstrap, which would re-gate every founder. Nothing existing is touched;
    new draws simply start their book at the next tick, so the threshold
    becomes usable once they cover a challenger's window.
    """
    _, conn, universe = _ctx()
    existing = {s.name for s in registry.list_competitors(conn, universe=universe.name)}
    added = [spec.name for spec in nulls_for(universe) if spec.name not in existing]
    for spec in nulls_for(universe):
        if spec.name not in existing:
            registry.insert_competitor(conn, spec)
    conn.commit()
    typer.echo(f"{universe.name}: {len(added)} added, {len(existing & {s.name for s in nulls_for(universe)})} present")
    for name in added:
        typer.echo(f"  + {name}")


@app.command()
def audit(q: float = typer.Option(0.10, help="Benjamini-Hochberg false discovery rate")) -> None:
    """Arena-wide multiple testing: how many admissions survive the number of tests run.

    The gate deflates a candidate by its own family's trial count. This asks
    the question that sank the six freqtrade bots instead: across every family
    and every weekly search, how much of what was admitted is just the best of
    many tries?
    """
    from arena.judge import audit as audit_mod

    _, conn, universe = _ctx()
    report = audit_mod.run(conn, universe.name, q)
    typer.echo(report.headline)
    if report.bh_threshold is not None:
        typer.echo(f"Benjamini-Hochberg threshold: p <= {report.bh_threshold:.4f}")
    for row in report.rows:
        if row.verdict != "admitted":
            continue
        mark = "survives" if row.survives_bh else "NOT SIGNIFICANT arena-wide"
        dsr = f"{row.dsr_arena:.2f}" if row.dsr_arena is not None else "n/a"
        typer.echo(
            f"  {row.competitor or row.family} (trial {row.trial_id}): p={row.bootstrap_p:.3f} "
            f"dsr_family={row.dsr_family:.2f} dsr_arena={dsr} -> {mark}"
        )


_UNIVERSE_OPTION = typer.Option(None, "--universe", help="Universe file(s) to watch; default: UNIVERSE_PATH")


@app.command()
def live(
    interval: float = typer.Option(5.0, help="Seconds between two price passes"),
    refresh: float = typer.Option(60.0, help="Seconds between two re-reads of the open legs"),
    universe: list[str] = _UNIVERSE_OPTION,
    tick_minutes: str = typer.Option("5,35", help="Minutes of the hour the ticks run at (quiet windows)"),
) -> None:
    """Watch perp prices and execute stops / ROI exits between ticks (see runner/live.py)."""
    from arena.data.http import make_client
    from arena.runner.live import run_forever

    settings, conn, default_universe = _ctx()
    universes = [load_universe(p) for p in universe] if universe else [default_universe]
    minutes = tuple(int(m) for m in tick_minutes.split(",") if m.strip())
    log.info("live watcher on %s every %.0fs", [u.name for u in universes], interval)
    with make_client(timeout=15.0) as client:
        run_forever(conn, settings, universes, client, interval=interval, refresh=refresh, tick_minutes=minutes)


@app.command()
def web(
    host: str = typer.Option("0.0.0.0", help="Bind address"), port: int = typer.Option(8080, help="TCP port")
) -> None:
    """Serve the read-only dashboard (requires the ``web`` extra).

    Migrates first, like every other sub-command. The dashboard is the one
    process that runs continuously, so on a deploy it is the first to meet the
    new schema: without this it would serve 500s against the old one until the
    next tick happened to migrate.
    """
    import uvicorn

    from arena.web.app import create_app

    settings = Settings.from_env()
    conn = connect(settings.database_url)
    try:
        run_migrations(conn)
    finally:
        conn.close()
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()

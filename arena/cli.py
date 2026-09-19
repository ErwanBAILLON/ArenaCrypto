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
from arena.settings import Settings
from arena.store import books as bstore
from arena.store import registry
from arena.store.db import connect, run_migrations
from arena.telegram import sender, templates

app = typer.Typer(add_completion=False, help="Signal-only crypto research arena.")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("arena")
logging.getLogger("httpx").setLevel(logging.WARNING)

FOUNDERS = ["carry", "trend_ts", "xs_momentum", "regime", "meta_label"]
FUNDING_FAMILIES = {"carry", "bench_carry_equal"}  # need perpetual funding: crypto only


def _uname(universe, name: str) -> str:
    """Competitor names are global: suffix them outside the historical crypto universe."""
    return name if universe.name == "crypto" else f"{name}_{universe.name}"


def founders_for(universe) -> list[str]:
    return [f for f in FOUNDERS if universe.exchange == "binance" or f not in FUNDING_FAMILIES]


def nulls_for(universe) -> list[CompetitorSpec]:
    """Null models and benchmarks of a universe (the classic one has no funding, hence no carry benchmark)."""
    u = universe.name
    mk = lambda name, fam, params, role: CompetitorSpec(  # noqa: E731
        None, _uname(universe, name), fam, 1, params, role=role, status="champion", universe=u
    )
    specs = [mk("null_cash", "null_cash", {}, "null")]
    specs += [mk(f"null_random_{i}", "null_random", {"seed": i}, "null") for i in range(5)]
    if universe.exchange == "binance":
        specs.append(mk("bench_btc_hold", "bench_btc_hold", {}, "benchmark"))
        specs.append(mk("bench_carry_equal", "bench_carry_equal", {}, "benchmark"))
    else:
        specs.append(mk("bench_hold", "bench_hold", {"symbol": universe.reference}, "benchmark"))
    return specs


def _ctx():
    settings = Settings.from_env()
    if not settings.database_url:
        raise typer.BadParameter("DATABASE_URL is required")
    conn = connect(settings.database_url)
    run_migrations(conn)  # idempotent and cheap: every command runs on a current schema
    return settings, conn, load_universe(settings.universe_path)


def _now() -> datetime:
    return datetime.now(UTC)


@app.command()
def migrate() -> None:
    """Apply pending SQL migrations."""
    _, conn, _ = _ctx()
    applied = run_migrations(conn)
    typer.echo(f"applied: {applied or 'nothing'}")


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
    """Run the entry gate and print the verdict (records a trial)."""
    _, conn, universe = _ctx()
    spec = registry.get_competitor(conn, family)
    fam, params = (spec.family, spec.params) if spec else (family, {})
    end = _now()
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
    )
    v = adm.verdict
    typer.echo(f"{fam}: {'ADMITTED' if v.admitted else 'REJECTED'} failed={v.failed}")
    for k, val in v.metrics.items():
        typer.echo(f"  {k}: {val:.4f}" if isinstance(val, float) else f"  {k}: {val}")


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
    families_present = {
        s.family for s in registry.list_competitors(conn, statuses=["champion", "challenger"], universe=universe.name)
    }
    lines = []
    for fam in founders_for(universe):
        if fam in families_present:
            lines.append(f"{fam}: already present")
            continue
        params = dict(REGISTRY[fam].default_params)
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
                _uname(universe, f"{fam}_v1"),
                fam,
                1,
                params,
                status=status,
                universe=universe.name,
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
                        "detail": f"{fam}_v1 rejected at gate ({', '.join(adm.verdict.failed)}); enters as challenger"
                    },
                ),
            )
        conn.commit()
        m = adm.verdict.metrics
        lines.append(
            f"{fam}: {status} sharpe={m.get('sharpe', 0):.2f} dsr={m.get('dsr', 0):.2f} "
            f"p={m.get('bootstrap_p', 1):.2f} mdd={m.get('max_drawdown', 0):.1%}"
        )
    if "news" not in families_present and universe.exchange == "binance":  # the news lexicon is crypto-specific
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
                rationale="founder; forward-only family, not backtest-gated",
            ),
        )
        conn.commit()
        lines.append("news: challenger (forward-only)")
    summary = f"Arena bootstrapped ({universe.name})\n" + "\n".join(lines) + f"\nnull 95th pct Sharpe {thr:.2f}"
    typer.echo(summary)
    sender.send(settings, templates.event_alert("info", summary))


@app.command()
def web(
    host: str = typer.Option("0.0.0.0", help="Bind address"), port: int = typer.Option(8080, help="TCP port")
) -> None:
    """Serve the read-only dashboard (requires the ``web`` extra)."""
    import uvicorn

    from arena.web.app import create_app

    uvicorn.run(create_app(Settings.from_env()), host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()

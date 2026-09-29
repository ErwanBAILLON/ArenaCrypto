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
    ("majors_tsmom", "majors_tsmom_v1", {}),  # the one weekly rule the 2026-09-28 fast harness left standing
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


# Currencies: two rules for one premium (trend), the second a control on the first (see competitors/fx.py).
FX_FOUNDERS: list[tuple[str, str, dict]] = [
    ("fx_tsmom", "fx_tsmom_v1", {}),
    ("fx_breakout", "fx_breakout_v1", {}),
]


def founders_for(universe) -> list[tuple[str, str, dict]]:
    """The founders a universe seeds: only families that declare its market."""
    if universe.membership.enabled:
        pool = list(WIDE_FOUNDERS)
    elif universe.market == "fx":
        pool = list(FX_FOUNDERS)
    else:
        pool = list(FOUNDERS)
    return [f for f in pool if universe.market in REGISTRY[f[0]].markets]


def misfits(conn, universe) -> list[CompetitorSpec]:
    """Active competitors running in a universe their family was not built for."""
    return [
        s
        for s in registry.list_competitors(conn, statuses=["champion", "challenger"], universe=universe.name)
        if s.role == "competitor" and s.family in REGISTRY and universe.market not in REGISTRY[s.family].markets
    ]


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
def prune(apply: bool = typer.Option(False, help="Retire them (default: list only)")) -> None:
    """Retire competitors running in a universe their family does not fit (books kept).

    Every family declares the markets it was built for; the founders of a
    universe are filtered on it, but the arena predates that rule and carried
    perp-born rules on ETFs. This is the sweep: list, then ``--apply``.
    """
    settings, conn, universe = _ctx()
    gone = misfits(conn, universe)
    for s in gone:
        fits = sorted(REGISTRY[s.family].markets)
        typer.echo(f"{s.name}: family {s.family} is built for {fits}, universe is {universe.market}")
        if apply:
            registry.set_status(conn, s.id, "retired")
            bstore.add_alert(
                conn,
                Alert(
                    kind="retired",
                    competitor_id=s.id,
                    payload={
                        "detail": f"{s.name} retired: family {s.family} is not built for the {universe.market} market"
                    },
                ),
            )
    conn.commit()
    typer.echo(f"{len(gone)} misfit(s){' retired' if apply else ''}")


@app.command("watch-ingest")
def watch_ingest(
    intraday: bool = typer.Option(False, help="5-minute bars (60 days on first run) instead of daily"),
) -> None:
    """Refresh the ETF lab's watchlist (config/watchlist.yaml) from Yahoo: daily bars, or 5-minute with --intraday."""
    from arena.data.http import make_client
    from arena.runner.ingest import ingest_watchlist
    from arena.web.etf import watchlist

    _, conn, _ = _ctx()
    tickers = [t["symbol"] for t in watchlist()]
    with make_client(timeout=30.0) as client:
        counts = ingest_watchlist(conn, client, tickers, _now(), intraday=intraday)
    label = "watchlist 5m" if intraday else "watchlist"
    typer.echo(f"{label}: {len(tickers)} lines, {counts['candles']} new candles, {counts['failed']} failed")


@app.command()
def digest() -> None:
    """Send the daily leaderboard digest."""
    from arena.runner.digest import build

    settings, conn, _ = _ctx()
    text = build(conn, _now())
    ok = sender.send(settings, text)
    typer.echo(text if settings.dry_run else f"digest sent: {ok}")


def _symbols_for(conn, universe) -> list[str]:
    """The symbols a gate or a search may read: the static list, or every stored member of a point-in-time universe."""
    if universe.symbols:
        return list(universe.symbols)
    from arena.store import membership as mstore

    frame = mstore.membership_frame(conn, universe.name)
    return sorted(frame["symbol"].unique()) if not frame.empty else []


def _history_and_null(conn, universe, end: datetime):
    from arena.runner.history import load_history
    from arena.runner.tick import fees_of

    history = load_history(conn, universe, universe.history_start, end, symbols=_symbols_for(conn, universe))
    if history.candles.empty:
        raise typer.Exit(
            code=typer.echo(f"no candles stored for universe {universe.name}: run `arena backfill` first") or 2
        )
    fees = fees_of(universe)
    start = pd.Timestamp(universe.history_start)
    pit = _point_in_time(conn, universe)
    thr = admission.cached_null_threshold(
        conn,
        history,
        _symbols_for(conn, universe),
        start,
        end,
        fees,
        bar_hours=universe.bar_hours,
        members_at=pit[0],
        liquidity_at=pit[1],
    )
    return history, fees, start, thr


def _point_in_time(conn, universe) -> tuple[dict | None, dict | None]:
    """The stored membership schedules of a point-in-time universe; ``(None, None)`` for the historical ones.

    On a membership universe the gate must see who was tradable on each bar and
    at what liquidity, exactly as the tick does; judging founders on the
    superset of everything that ever listed would admit a survivorship bias
    with a config file around it. Nothing stored means the gate does not run.
    """
    from arena.store import membership as mstore

    if not universe.membership.enabled:
        return None, None
    members_at, liquidity_at = mstore.schedules(conn, universe.name)
    if not members_at:
        raise typer.Exit(
            code=typer.echo(f"no stored membership for universe {universe.name}: run `arena universe-build` first") or 2
        )
    return members_at, liquidity_at


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
        _symbols_for(conn, universe),
        start,
        end,
        fees,
        thr,
        notes="cli judge",
        bar_hours=universe.bar_hours,
        universe=universe.name,
        members_at=_point_in_time(conn, universe)[0],
        liquidity_at=_point_in_time(conn, universe)[1],
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
            _symbols_for(conn, universe),
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
    spec = rt.run(conn, history, _symbols_for(conn, universe), end, window_days=window_days)
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
    pit = _point_in_time(conn, universe)
    # every name ever registered here, retired ones included: a retired founder stays retired, it is not re-seeded
    names_present = {s.name for s in registry.list_competitors(conn, universe=universe.name)}
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
            _symbols_for(conn, universe),
            start,
            end,
            fees,
            thr,
            notes="bootstrap founder",
            bar_hours=universe.bar_hours,
            universe=universe.name,
            members_at=pit[0],
            liquidity_at=pit[1],
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
    if _uname(universe, "news_v1") not in names_present and universe.exchange == "binance":  # crypto-specific lexicon
        registry.insert_competitor(
            conn,
            CompetitorSpec(
                None,
                _uname(universe, "news_v1"),
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
    archive: bool = typer.Option(
        True, help="Try the monthly archive first (off: REST klines only, when the archive throttles)"
    ),
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
        _backfill_members(conn, universe, build.symbols, start.to_pydatetime(), end, workers, use_archive=archive)


def _backfill_members(conn, universe, symbols: list[str], start, end, workers: int, use_archive: bool = True) -> None:
    """Hourly candles and funding from the archive for every member lacking history.

    The incremental REST ingest only walks forward from the last stored bar, so
    a symbol that joins the universe with no history would start life with a
    single bar and never satisfy a warm-up. The archive fills the past once.
    """
    from arena.data import binance_archive as archive
    from arena.data.http import make_client
    from arena.runner.wide_backfill import fetch_many
    from arena.store import candles as cstore

    # coverage, not presence: the tick's incremental ingest leaves every current member with
    # a few recent bars, which is not a history. A symbol whose earliest stored bar is more
    # than a week after the universe's start (or its own listing) still needs the archive.
    grace = pd.Timedelta(days=7)
    missing = []
    for s in symbols:
        first = cstore.first_candle_ts(conn, "binance", s)
        if first is None or pd.Timestamp(first) > pd.Timestamp(start) + grace:
            missing.append(s)
    typer.echo(f"  backfilling {len(missing)} of {len(symbols)} symbols lacking history before {start.date()}...")
    rows = 0
    batch = max(workers, 8)
    still: list[str] = []
    if not use_archive:
        still, missing_archive = list(missing), []
    else:
        missing_archive = missing
    for i in range(0, len(missing_archive), batch):  # commit per batch: a killed job keeps what it fetched
        chunk = missing_archive[i : i + batch]
        frames = fetch_many(chunk, start, end, "1h", workers)
        for sym in chunk:
            frame = frames.get(sym)
            if frame is None or frame.empty:
                still.append(sym)
                continue
            frame["symbol"] = sym
            rows += cstore.upsert_candles(conn, "binance", frame)
        conn.commit()
        typer.echo(f"    {min(i + batch, len(missing_archive))}/{len(missing_archive)} symbols, {rows} candles so far")
    if still:
        # the archive throttled or lacks the month: the REST klines walk the same history page by page
        from arena.data import binance as rest

        typer.echo(f"  archive gave nothing for {len(still)} symbols; walking REST klines instead...")
        with make_client(timeout=30.0) as client:
            for n, sym in enumerate(still, 1):
                try:
                    frame = rest.klines(client, sym, int(pd.Timestamp(start).timestamp() * 1000), interval="1h")
                except Exception as exc:  # one symbol must not end the run
                    log.warning("REST klines failed for %s: %s", sym, exc)
                    continue
                if not frame.empty:
                    frame["symbol"] = sym
                    rows += cstore.upsert_candles(conn, "binance", frame)
                    conn.commit()
                if n % 25 == 0:
                    typer.echo(f"    REST {n}/{len(still)}, {rows} candles so far")
    funded = 0
    with make_client(timeout=60.0) as client:
        for sym in missing:
            try:
                if use_archive:
                    f = archive.funding_history(client, sym, start, end)
                else:
                    from arena.data import binance as rest

                    f = rest.funding(client, sym, int(pd.Timestamp(start).timestamp() * 1000))
            except Exception as exc:
                log.warning("funding history failed for %s: %s", sym, exc)
                continue
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
    label: str = typer.Option(
        "barrier", help="Training target: 'barrier' (ROI ladder, net of costs) or 'rank' (forward excess-return rank)"
    ),
    horizon: int = typer.Option(72, help="Forward horizon in hours for label='rank'"),
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
    # every symbol that was ever a member: a point-in-time universe has no static list, and the
    # default (universe.symbols) is empty there -- the dataset came out with zero rows
    history = load_history(conn, universe, universe.history_start, end, symbols=_symbols_for(conn, universe))
    frame = mstore.membership_frame(conn, universe.name)
    if frame.empty:
        typer.echo("no stored membership: run `arena universe-build` first")
        raise typer.Exit(code=1)
    symbols_at = {ts: list(g["symbol"]) for ts, g in frame.groupby("ts")}
    fees = fees_of(universe)
    round_trip = 2 * fees.cost("perp", 0.05, None) if fees.impact else 2 * fees.perp_cost

    data = tx.build_dataset(
        history.candles, history.funding, symbols_at, costs=round_trip, label=label, horizon_hours=horizon
    )
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
def research_store(
    out: str = typer.Argument(..., help="Where to write the pickle (weekly panel, hourly closes, funding, liquidity)"),
) -> None:
    """Build the fast-harness store of this universe: the feature panel at each rebalance date, once.

    Needs a stored membership schedule (`arena universe-build`). Minutes on
    five years; every `arena research-run` afterwards takes seconds.
    """
    from arena.core.membership import Member
    from arena.research import fast
    from arena.runner.history import load_history
    from arena.store import membership as mstore

    _, conn, universe = _ctx()
    frame = mstore.membership_frame(conn, universe.name)
    if frame.empty:
        typer.echo("no stored membership: run `arena universe-build` first")
        raise typer.Exit(code=1)
    members = {
        pd.Timestamp(ts): [Member(r.symbol, int(r.rank), float(r.adv_usd), float(r.daily_vol)) for r in g.itertuples()]
        for ts, g in frame.groupby("ts")
    }
    history = load_history(conn, universe, universe.history_start, _now(), symbols=_symbols_for(conn, universe))
    store = fast.build_store(history.candles, history.funding, members, log=typer.echo)
    pd.to_pickle(store, out)
    typer.echo(f"stored panel {store.panel.shape}, closes {store.closes.shape}, {len(store.dates)} dates -> {out}")


@app.command()
def research_run(
    store_path: str = typer.Argument(..., help="Pickle written by `arena research-store`"),
    rules: str = typer.Option("", help="Comma-separated rule names from arena.research.fast.RULES (default: all)"),
    capacity: float = typer.Option(1_000_000.0, help="Deployment size for impact costs; 0 = flat fees"),
    every: int = typer.Option(1, help="Rebalance every N rebalance dates (1 = weekly)"),
    warmup_days: int = typer.Option(100, help="Days after the first rebalance before the first scored quarter"),
    out: str = typer.Option("", help="Optional JSON of the per-quarter tables"),
) -> None:
    """Price weekly weight rules on the store and print one line per rule: Sharpe by quarter, decomposition."""
    import json as _json

    from arena.research import fast

    store = pd.read_pickle(store_path)  # written by research-store on this machine
    wanted = [r.strip() for r in rules.split(",") if r.strip()] or list(fast.RULES)
    unknown = [r for r in wanted if r not in fast.RULES]
    if unknown:
        typer.echo(f"unknown rules {unknown}; known: {', '.join(fast.RULES)}")
        raise typer.Exit(code=2)
    fees = fast.fee_model(capacity or None)
    qs = fast.quarters(store, warmup_days=warmup_days)
    tables = {}
    for name in wanted:
        rule, kw = fast.RULES[name]
        f = fast.summarise(fast.run(store, rule, fees, every=every, **kw), qs)
        tables[name] = f.to_dict(orient="records")
        typer.echo(fast.headline(name, f))
    if out:
        with open(out, "w") as fh:
            _json.dump(tables, fh, default=str)


@app.command()
def seed(
    family: str = typer.Argument(..., help="Registered family"),
    name: str = typer.Argument(..., help="Competitor name (suffixed with the universe outside crypto)"),
    params: str = typer.Option("{}", help="JSON of parameter overrides on the family defaults"),
    rationale: str = typer.Option("seeded by hand", help="Why this challenger exists (shown on its page)"),
) -> None:
    """Insert a challenger without the entry gate: judged forward only, promotable once `arena judge` admits it.

    The bootstrap gates founders against five years of history, which on the
    wide arena is an hour of nulls and an hour per founder at six gigabytes.
    A variant of a family already in the arena (a stop in sigmas instead of
    a percentage, a slower cadence) does not need that to start being scored:
    the forward test is the one that counts, and `arena judge NAME` runs the
    gate later, on the same machine, when the quota allows.
    """
    import json as _json

    _, conn, universe = _ctx()
    if family not in REGISTRY:
        typer.echo(f"unknown family {family!r}; known: {', '.join(sorted(REGISTRY))}")
        raise typer.Exit(code=2)
    if universe.market not in REGISTRY[family].markets:
        typer.echo(f"{family} is not built for the {universe.market} market")
        raise typer.Exit(code=2)
    full = _uname(universe, name)
    if registry.get_competitor(conn, full) is not None:
        typer.echo(f"{full}: already present")
        raise typer.Exit(code=0)
    overrides = _json.loads(params)
    merged = {**REGISTRY[family].default_params, **overrides}
    merged.pop("model_str", None)
    cid = registry.insert_competitor(
        conn,
        CompetitorSpec(
            None,
            full,
            family,
            1,
            merged,
            status="challenger",
            universe=universe.name,
            gate_admitted=False,
            rationale=f"seeded, not gated: {rationale}; overrides {overrides}",
        ),
    )
    conn.commit()
    typer.echo(f"{full}: challenger (id {cid}), overrides {overrides}; run `arena judge {full}` to gate it")


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

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-09-20

### Added
- `price_action` family: mechanical support/resistance levels, breakouts with volume and candle confirmation, Fibonacci pullbacks with reversal candles, trailing stop on recent extremes; two founders, `swing` (daily re-decision) and `position` (weekly), in both arenas.
- Arena badge (crypto / classic markets) on the dashboard.


## [0.3.0] - 2026-09-19

### Added
- French explanation layer (`arena.explain`): what each family does, what it looks at, why it holds each position, gate verdicts in words.
- Dashboard rewritten for a non-technical reader: live page (who speaks, what changed in 24h, data sources and freshness, activity feed), model cards, family overview, euros against "hold BTC" and "do nothing".
- Telegram in French with the why behind every message; digest in four sections.
- Random-window robustness test by market regime (bull / bear / range) as a gate criterion, with French rendering.
- Second arena for classic markets (ETFs, FX) on daily Yahoo Finance bars: universe config, ingestion, CronJobs, bootstrap job; bar size is now a property of the universe.
- `tick_runs` activity log; explicit zero-weight rows on full exits.

### Changed
- Every CLI command applies pending migrations before running.


## [0.2.1] - 2026-09-19

### Changed
- Weekly challenger defaults to 15 Optuna trials per family (40 could exceed the 6h job deadline on 2 cores).


## [0.2.0] - 2026-09-19

### Added
- Read-only web dashboard (`arena web`, FastAPI + server-rendered HTML, inline SVG equity curves): leaderboard, competitor detail, alerts, trials, `/healthz`, `/api/leaderboard.json`. Helm Deployment + IngressRoute with optional Traefik BasicAuth.
- `HoldingCompetitor` tests, MIT licence, CONTRIBUTING, docker-compose, GitHub Actions CI, ruff lint and format in both CIs.

### Changed
- Null distribution runs in forked workers (`ARENA_WORKERS`) and is cached per ISO week in `trials`.


### Added
- MIT license, contributor guide, this changelog, `.editorconfig`.
- `docker-compose.yml` for local development (Postgres 16 + the Arena image).
- GitHub Actions CI (`.github/workflows/ci.yml`): ruff + pytest against a Postgres 16 service.
- `ruff` configuration in `pyproject.toml` and a lint step in the Gitea workflow.
- `tests/competitors/test_holding.py`: behaviour of `HoldingCompetitor` (rebalance slots, dead band,
  exits, weekly variant, state round trip).
- Package docstrings for every sub-package.

### Changed
- Code base formatted and linted with ruff (imports sorted, `datetime.UTC`, explicit `zip(strict=)`,
  unused imports removed). No runtime behaviour change intended.

## [0.1.3] - 2026-09-19

### Changed
- Null distribution runs in forked worker processes (`ARENA_WORKERS`, default 1); the history frames
  are shared copy-on-write and nothing but the seed and the result is pickled.
- The null 95th-percentile Sharpe is cached for a week in `trials`, so `bootstrap`, `judge` and the
  challenger loop share one computation instead of re-running 50 backtests each.

## [0.1.2] - 2026-09-19

### Fixed
- `carry`: weekly rebalance (Monday 00:00 UTC) instead of daily, rank tolerance (`rank_band`) so symbols
  near the funding threshold do not swap every slot, thinner-funding defaults (`min_rate`,
  `lookback_days` 14, `k` 5) and a wider `max_weight`.
- Tests aligned with the new carry defaults; carry warm-up covers the 14-day lookback.

## [0.1.1] - 2026-09-19

### Added
- `HoldingCompetitor`: decide once per rebalance slot (daily 00:00 UTC, or a weekday), hold in between,
  ignore weight changes below a dead band. `trend_ts`, `regime` and `carry` now inherit from it.
- `null_random` redraws its coin flips once a week instead of hourly.

### Changed
- Null distribution reduced to 50 seeded runs; quieter logs; `heavy` resource requests fit under the
  namespace quota.
- Numpy tail helpers for trend indicators (about 3.5x faster backtests); consistent UTC handling.
- CI installs uv standalone (the runner image has no pip) and pins the installer version.

### Fixed
- Real-data diagnosis: hourly re-decision paid about 52 % of NAV per year in fees for `trend_ts`, the
  hourly coin-flip null lost about 700 %/year to fees, and `carry` never entered at 0.01 %/8h. Turnover,
  not the signals, was the problem; holding fixes it.

## [0.1.0] - 2026-09-19

### Added
- Core value types, point-in-time `Snapshot`, virtual `Book` accounting (fees, funding, carry legs),
  SQL schema and migration runner.
- Public data adapters: Binance USD-M futures (klines, funding, open interest), Hyperliquid info
  endpoint, HTTPS RSS feeds, ForexFactory macro calendar. Idempotent ingestion.
- Postgres repositories: candles, news, registry (competitors, trials, models), books, competitor state.
- Rule-based news scorer (VADER + crypto lexicon, `scorer_version` 1).
- Founding competitor families: `carry`, `trend_ts`, `xs_momentum`, `regime` (rule-based v1),
  `meta_label` (LightGBM), `news`; null models `null_cash`, `null_random`; benchmarks `bench_btc_hold`,
  `bench_carry_equal`.
- Judge: backtest with the live book, anchored walk-forward, metrics (Sharpe, Sortino, MDD, profit
  factor, deflated Sharpe, block-bootstrap p-value), null distribution, entry gate.
- Runner: hourly tick (ingest, decide, book, allocate, drift, promote, alert), allocator opinion, drift
  checks, champion/challenger promotion, daily digest.
- Challenger loop: Optuna search over declared spaces for rule families, meta-label retraining.
- Telegram templates and dry-run-capable sender.
- Typer CLI (`migrate`, `backfill`, `news`, `tick`, `digest`, `judge`, `challenger`, `retrain`,
  `bootstrap`), Dockerfile, Helm chart (CronJobs, ExternalSecret, NetworkPolicy, bootstrap Job),
  Gitea Actions build workflow.

[Unreleased]: https://git.ebaillon.fr/bots/arena/compare/v0.1.3...HEAD
[0.1.3]: https://git.ebaillon.fr/bots/arena/compare/v0.1.2...v0.1.3
[0.1.2]: https://git.ebaillon.fr/bots/arena/compare/v0.1.1...v0.1.2
[0.1.1]: https://git.ebaillon.fr/bots/arena/compare/v0.1.0...v0.1.1
[0.1.0]: https://git.ebaillon.fr/bots/arena/releases/tag/v0.1.0

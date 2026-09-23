# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-09-23

A round on the judge, and two families that test one hypothesis about funding.
Design note: `docs/specs/2026-09-23-crowding-and-the-judge.md`.

### Fixed
- **A model the gate refused could become champion by default.** `bootstrap` inserted a refused founder as a challenger, and `should_promote` only compared against a champion "if champion is not None" — so each of the six families without a champion would have crowned a gate-refused model around day 42. Competitors now carry `gate_admitted` (migration 0005, backfilled from the rationale); promotion refuses `never_admitted`; `arena judge <name>` re-runs the gate and flips the flag. Forward-only families (`news`) are exempt by name.
- **The forward null threshold was the 95th percentile of five numbers** while the entry gate used fifty. The arena carries 30 `null_random` competitors, counts only those covering the challenger's own window, and refuses `null_underpowered` below 20.
- **The weekly search ran in the wrong arena.** `optimize.run` took no bar size and no universe: on classic markets it built hourly competitors on daily bars (warm-ups 24× too long, so nothing ever traded) and annualised by 8760. Every Sunday `arena-classic-challenger` produced eighteen folds of exactly 0.00 Sharpe and still charged every evaluation against the *crypto* deflated Sharpe. `count_trials` and the version counter are now scoped by universe, and an admitted classic challenger no longer lands in the crypto arena.
- **Promotion and champion-bleeding annualised daily bars as hourly.** Both now read the bar size from the universe; the dashboard infers it from the bar spacing.
- **The macro calendar read as three days dead** while being polled every thirty minutes: it is upserted with `ON CONFLICT DO NOTHING`, so a week already stored writes no row and `max(fetched_at)` never moved.
- **Trials were left open forever.** ~170 optimisation rows had shown as "running" since 20 September. Evaluations are written closed; `abandon_stale_trials` closes what a killed process left behind.

### Added
- `metrics.effective_n`: autocorrelation-adjusted sample size. A competitor holding one weekly bet for 168 bars provides one observation, not 168 — every statistic below uses it.
- `metrics.sharpe_se`, `probabilistic_sharpe`, `min_track_record_length`, `track_record_verdict`: the standard error of a Sharpe, P(true Sharpe > benchmark), and how many bars are still missing before a claim is allowed.
- `metrics.pbo_cscv`: probability of backtest overfitting by combinatorially symmetric cross-validation, measured over every evaluation of a search and enforced as the gate's `pbo` criterion (≤ 0.50). DSR asks whether one Sharpe is too good for the number of tries; PBO asks whether picking the best generalises.
- `arena audit`: Benjamini-Hochberg across every gate decision in an arena, each admission's deflated Sharpe recomputed against the arena-wide trial count, and the number of admissions chance alone predicts.
- `funding_skew` family: cross-sectional funding crowding as a signal — short the highest z-scores, buy the lowest, gross balanced, stands aside on a flat cross-section.
- `crowded_trend` family: the same crowding measure as a filter on the `trend_ts` signal — no long where the crowd already pays a premium, no short where it is already paid.
- `arena nulls`: register the null models and benchmarks an already-bootstrapped arena is missing, without re-gating its founders.
- `feed_health` table (migration 0006): ingestion records every attempt, and `drift.stale_feeds` watches all six sources with a tolerance per source, where only candles were watched before.

### Changed
- `image.tag` is sha-pinned. The bare version tag is re-pushed on every commit, so with `pullPolicy: IfNotPresent` ArgoCD reported Synced while the node went on serving the build it had already cached — which is how a fix can be "deployed" and absent at the same time.
- **The leaderboard ranks on evidence.** Its sort key was an annualised Sharpe computed on as few as 24 hourly bars, which live put a coin flip in third place and showed the random 95th percentile at 10.36. Every row now carries `Sharpe ± standard error`, the probability it beats the null models, and the time still needed to reach the 95 % promotion bar. The thirty random models collapse into one row.
- The front page opens with the age of the experiment and what nothing-yet-proven means, before a single euro.
- The Telegram digest replaced "en avance sur le champion" with a sentence containing a probability, and says how many days a challenger still needs.
- The trials page hides the parameter-search sampling behind a link and shows the PBO criterion beside the deflated Sharpe.

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
- MIT licence, contributor guide, this changelog, `.editorconfig`.
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

[Unreleased]: https://github.com/ErwanBAILLON/ArenaCrypto/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/ErwanBAILLON/ArenaCrypto/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/ErwanBAILLON/ArenaCrypto/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ErwanBAILLON/ArenaCrypto/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ErwanBAILLON/ArenaCrypto/releases/tag/v0.1.0

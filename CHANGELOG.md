# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Interactive dashboard: uPlot charts (crosshair, tooltip, legend toggles, drag-zoom) with a marker for every real weight change in a competitor's book (entry, exit, reversal, resize, coloured by side) and a journal of who bought what, when, at which price and why. `/api/live`, `/api/series/{id}`, `/api/board`. The header shows the age of the last tick and the countdown to the next, refreshed every second; data refetches only when a tick has written. Selected dark mode, phone-width layout.
- The panel grows to 117 columns: the lottery block (largest recent daily gains, skew, kurtosis, share of extreme days), trend quality (Kaufman efficiency ratio, Hurst proxy, ADX), betas to BTC and ETH with the ETH/BTC ratio's momentum, seasonality.
- `xs_adaptive`: each week keeps only the candidates whose trailing-52-week rank IC is significant and weights them by it. Signals change sign with the regime; this changes signals with them, using nothing from the period it trades.
- `positioning` (migration 0008): long/short account ratios, top-trader position ratios and taker buy/sell volume, stored every tick. Binance keeps thirty days; a history starts today.
- `arena.models.mlp`: a one-hidden-layer network with `hidden=0` as its own linear control, GKX-style, numpy only.
- `tranches=N` (staggered sub-books averaged), `vol_mode="riskparity"` and `long_only` on the shared cross-sectional sizing path.

## [0.6.0] - 2026-09-24

The wide arena. Design: `docs/specs/2026-09-23-xs-panel-design.md`; results: `docs/specs/2026-09-24-xs-panel-results.md`.

### Added
- **A survivorship-free universe.** `arena.data.binance_archive` reads Binance's public archive, which keeps delisted symbols, and `arena universe-build` ranks every archived perpetual — dead ones included — by trailing dollar volume at each weekly date. Measured on the real data: 402 distinct symbols passed through a 50-name universe over 151 weeks, weekly churn is 9.7 %, and only 18 of the first week's members were still in it at the end. Membership is stored (migration 0007) and read back, never recomputed.
- **Per-symbol execution cost.** `arena.core.costs.ImpactModel` implements the square-root impact law, priced at a stated `capacity_nav` rather than at the 10 000 € the book holds — at that size impact is negligible everywhere, which silently flatters illiquid names. `k` is swept, not fitted; missing liquidity data costs 100 bps, never zero.
- **Labels that describe a trade rather than a price move.** `arena.labeling` puts triple barriers on the excess return over the universe, net of costs, with the upper barrier as a ROI ladder in freqtrade's `minimal_roi` shape. Plus sample-uniqueness and trend-scanning weights, so N overlapping labels stop pretending to be N independent opinions.
- **Purged combinatorial cross-validation** (`arena.judge.cpcv`), with a leakage report that must read zero.
- **A ~100-column point-in-time feature panel** (`arena.features`), each raw column shipping with its cross-sectional rank.
- **Two new families, differing only in the predictor**: `xs_sparse` (an unfitted rank composite, Nagel's control) and `xs_complex` (random Fourier features and ridge, the Kelly-Malamud-Zhou machinery applied to the panel). Both dollar-neutral, both closing on the ROI ladder that labelled them. `arena train-xs` trains the second and refuses to insert anything when PBO says the winner does not generalise.
- **An attribution page** per competitor: P&L by symbol, side, conviction and holding time, the worst twelve trades, and a reliability diagram answering whether a stated 70 % confidence happens 70 % of the time. Derived from stored targets, so it works retroactively for every family.
- `hedge` (`picks` | `index` | `anchor`), `vol_mode` (`names` | `portfolio`), `max_gross`, `max_leverage` and `rebalance_every_weeks` on both cross-sectional families, through one shared sizing path. The picks-hedged short leg had negative alpha out of sample and the book realised 2.3 % volatility against a 20 % target; these are the levers that address both.
- `TWO_SIGNALS`, a pre-registered two-signal preset (low volatility + Donchian position). Not the default: choosing it on the year it was read from would be fitting to the test set.
- `train_xs.search`: sweeps width and bandwidth as well as shrinkage and computes PBO across the whole grid. PBO over a shrinkage path alone read 0.00 and meant little.
- `null_neutral`: a random dollar-neutral book at the families' own cadence and size, so "beats the null" measures selection rather than turnover.
- The wide arena in the chart (`wide.*`): hourly tick and weekly `universe-build`, which also backfills hourly history and funding from the archive for members without any.
- `config/universe-wide.yaml`, a third arena using both of the above. The crypto and classic arenas are unchanged and no past verdict is revised.

### Fixed
- A position closed on the ROI ladder stayed in the parent's held book and re-entered after the cooldown at its old weight without re-selection: close, wait a day, re-enter, close again.
- `Snapshot.at` can narrow the visible symbols, and the backtest takes `members_at` / `liquidity_at`, so a point-in-time universe can be scored at all. The warm-up reference was hardcoded to BTC and otherwise fell back to the alphabetically first symbol, which in a wide universe can be one that listed last month.
- A ROI rung of 0 % is freqtrade's force-exit, not a profit target; reading it as one labelled a symbol that exactly matched the market as a winner.
- Purging against the hull of all test rows emptied every combinatorial split whose blocks sat at opposite ends of the sample, and `DatetimeIndex.asi8` (microseconds here) mixed with `Timedelta.value` (always nanoseconds) turned a four-hour embargo into 198 days.
- Winsorising after demeaning left a residual per-date mean in the training target — a market-timing component a dollar-neutral book cannot express.
- Attribution measured excess against the symbols a competitor held rather than the universe, making a single-name position its own benchmark.
- Six test directories lacked `__init__.py`, which only surfaced when two test modules finally shared a basename.

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

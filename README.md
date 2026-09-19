# Arena

Signal-only crypto research arena. Several competing models decide every hour on
the same public market feed (Binance and Hyperliquid public endpoints, RSS news,
a macro calendar), are scored forward on virtual books next to null models, are
improved in the background by a champion/challenger loop, and report their
convictions to Telegram. Nothing places an order. No private API key of any
kind is used for data.

The design choice that matters: **the judge, not the model, is the product.**
Backtests are only an entry gate. What counts is how a competitor does on data
that did not exist when it decided, compared with random, cash and buy-and-hold
running under exactly the same accounting. The judge is deterministic Python
(walk-forward, real fees and funding, deflated Sharpe over the trial count,
block-bootstrap p-value) and rejects by default. Nothing that is running is ever
modified: an improvement is a new version that enters as a challenger and has to
beat its champion forward. Full design: `docs/specs/2026-09-19-arena-design.md`.

## Why this exists

Six freqtrade bots ran in dry-run from March to May 2026 and lost 14 to 20 % of
their virtual capital while their backtests claimed +5 000 % to +145 000 %. The
backtests were structurally biased: no stoploss, optimistic OHLC fills, 103
strategies selected on the same in-sample data, unlimited compounding. The
infrastructure was fine; the judge was the problem. Arena replaces the judge.

Two things this project learned on real data since, both now baked in:

- Hourly re-decision turned sound rules into fee machines (`trend_ts` paid about
  52 % of NAV per year in fees; an hourly coin flip lost about 700 %/year to
  fees). Competitors now decide once per rebalance slot and hold
  (`HoldingCompetitor`).
- A carry rule that "never enters" is a rule whose threshold was set from
  memory, not from the funding distribution. Defaults are now data-driven and
  the weekly search can move them.

None of the founding families has yet proven anything forward. That is the
expected state of an honest arena.

## Architecture

```
          Binance USD-M futures         Hyperliquid            RSS feeds          ForexFactory
      klines / funding / open interest   metaAndAssetCtxs    (HTTPS only)         calendar json
                 |                           |                    |                    |
                 +-------------+-------------+--------------------+--------------------+
                               v            arena.data  (public endpoints, retries, no keys)
                        arena.runner.ingest   (idempotent, closed bars only, point-in-time articles)
                               |
                               v
                        Postgres  (arena.store: candles, funding, oi, hl_snapshots, articles,
                                   article_scores, macro_events, competitors, trials, targets,
                                   books, allocations, alerts, models, competitor_state)
                               |
                               v
                        arena.core.Snapshot   (per bar: everything with ts <= decision_ts, nothing else)
                               |
            +------------------+------------------+----------------------+
            v                  v                  v                      v
      null_cash / null_random   bench_btc_hold /   carry  trend_ts       regime  meta_label  news
      (always in the arena)     bench_carry_equal  xs_momentum           (arena.competitors)
            |                  |                  |                      |
            +------------------+--------+---------+----------------------+
                                        v
                             arena.book.Book  (one accounting code path: mark, funding, fees, carry legs)
                                        |
                  +---------------------+---------------------+
                  v                                           v
        arena.judge  (entry gate)                    arena.runner.tick  (hourly, forward)
        backtest -> walk-forward -> metrics          decide -> book -> allocator -> drift
        null distribution -> deflated Sharpe         -> promotion (arena.challenger) -> alerts
        -> bootstrap p -> admitted / rejected                     |
                                                                  v
                                                   arena.telegram  (templates only, rate-limited)
```

Every box is a sub-package with one job; `arena/cli.py` wires them into
sub-commands, and each Kubernetes CronJob runs one sub-command.

## Quickstart (local)

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), a Postgres 16.

```bash
uv sync --extra dev                       # runtime deps + pytest + ruff
docker compose up -d postgres             # Postgres 16 on localhost:5433, user/db/password "arena"

export DATABASE_URL=postgresql://arena:arena@localhost:5433/arena
export DRY_RUN=true                       # Telegram messages are printed, never sent

uv run arena migrate                      # apply SQL migrations (arena/store/migrations)
uv run arena backfill --since 2025-01-01  # candles, funding, OI, news, macro from public endpoints
uv run arena bootstrap --skip-backfill    # register null models, gate the founders, announce
uv run arena tick --no-ingest             # one forward tick on stored data: decide, book, alert
uv run arena digest                       # the daily leaderboard, printed because DRY_RUN=true
uv run arena judge carry                  # run the entry gate on one family and print the verdict
```

`backfill` from 2024-01-01 (the default) pulls about 15 000 1h candles per
symbol and takes a few minutes against the public rate limits. Open interest
history is only available for the last 30 days (Binance API limit).

Everything also runs through `docker-compose.yml`, which builds the image and
sets `DATABASE_URL` and `DRY_RUN=true` for you:

```bash
docker compose run --rm arena migrate
docker compose run --rm arena tick --no-ingest
```

Tests:

```bash
docker compose exec postgres createdb -U arena arena_test    # once; the fixture DROPS the public schema
PG_TEST_URL=postgresql://arena:arena@localhost:5433/arena_test uv run pytest
uv run ruff check arena tests && uv run ruff format --check arena tests
```

## Configuration

Environment variables (`arena/settings.py`, plus two read directly by the judge and the Snapshot):

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | required | psycopg connection URL |
| `TELEGRAM_BOT_TOKEN` | empty | bot token; empty means every send is a dry run |
| `TELEGRAM_CHAT_ID` | empty | destination chat |
| `DRY_RUN` | `false` | `true`/`1`/`yes`: print Telegram messages instead of sending |
| `UNIVERSE_PATH` | `config/universe.yaml` | symbols, fee model, starting NAV, history start |
| `MAX_ALERTS_PER_DAY` | `20` | cap on non-digest Telegram messages; surplus is folded into the digest |
| `ARENA_WORKERS` | `1` | forked processes for the null distribution (`arena/judge/gate.py`) |

`config/universe.yaml`:

```yaml
symbols: [BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, DOT, LTC, NEAR, SUI, ARB, OP]
binance_suffix: USDT
fees:
  perp_taker: 0.0005   # Hyperliquid taker
  slippage: 0.0002
  spot_taker: 0.0010   # spot leg of a carry position
nav0: 10000.0
history_start: "2024-01-01T00:00:00Z"
```

Symbols are USDT perpetuals on Binance Futures that also trade on Hyperliquid.
Change the list and the fee assumptions here; nothing else hard-codes them.

## The competitor contract

A competitor is a pure function of `(params, Snapshot) -> Decision`. Same input,
same output; no database, no clock, no network. The `Snapshot` exposes, per
symbol, pandas frames strictly limited to `ts <= decision_ts` (1h/4h/1d candles,
funding, open interest, Hyperliquid funding, news features, macro flags).

```python
from arena.competitors.base import HoldingCompetitor, register
from arena.core.snapshot import Snapshot
from arena.core.types import Decision, Target


@register
class FundingSkew(HoldingCompetitor):
    """Short the perps whose funding is far above the cross-section (levered longs overpay)..."""

    family = "funding_skew"
    default_params = {"lookback_days": 7, "k": 3, "max_weight": 0.3}
    rebalance_weekday = 0  # Mondays 00:00 UTC; omit for daily at 00:00 UTC

    def warmup_bars(self) -> int:
        return self.params["lookback_days"] * 24 + 1

    def compute(self, snap: Snapshot) -> Decision:
        ...
        return {"ETH": Target(weight=-0.3, conviction=0.6, reason={"z": -2.1})}
```

- `weight` in `[-1, 1]` is the target fraction of NAV in that perp (negative is
  short). The book caps `sum(|weight|)` at 1. A carry position (long spot,
  short perp, earns funding) is `Target(weight=w, kind="carry")`.
- `conviction` in `[0, 1]` is only for Telegram wording and allocator tie-breaks.
- `reason` is a small dict that ends up verbatim in the Telegram signal line.
- `HoldingCompetitor.compute()` is called at `rebalance_hour` (00:00 UTC) daily,
  or on `rebalance_weekday` only; the held decision is returned in between and
  weight changes below `band` (0.10) are ignored. Subclass `Competitor` and
  implement `decide()` directly only if you have a reason to pay hourly fees.
- State is allowed for hysteresis only and must round-trip through `state()` /
  `restore_state()`; the tick persists it between processes.

To add a family: write the module, import it in `arena/competitors/__init__.py`
(that is what populates `REGISTRY`), add it to `FOUNDERS` in `arena/cli.py` if
it should be gated at bootstrap, declare a search space in
`arena/challenger/spaces.py` if the weekly loop should tune it, ship tests
(including the no-look-ahead test), then `arena judge <family>`. See
`CONTRIBUTING.md` for the checklist.

## The entry gate

`arena judge <family|competitor-name>` backtests with the live book code,
runs an anchored walk-forward (expanding train window, 90-day test folds,
parameters fixed inside a fold), and applies every criterion below. All must
hold; otherwise the verdict is `rejected` with the failing criteria listed, and
a `trials` row is written either way.

| Criterion | Threshold | Why |
|---|---|---|
| Folds with net return > 0 | at least 2/3 of test folds | one lucky year must not carry the verdict |
| Annualised Sharpe | above the 95th percentile of 50 seeded `null_random` backtests on the same period | "better than a coin flip paying the same fees" |
| Deflated Sharpe ratio (Bailey and Lopez de Prado) | > 0.90, using the number of `trials` recorded for the family | every parameter search inflates the best Sharpe; the count deflates it back |
| Stationary block bootstrap p-value of Sharpe <= 0 | < 0.10 (blocks of 24 bars, 1000 draws) | autocorrelated hourly returns need a block bootstrap, not an i.i.d. one |
| Maximum drawdown | < 30 % | a rule nobody would hold through is not a rule |
| Decisions | at least 30 bars with a non-flat target | enough events to say anything |

Null models and benchmarks skip the gate. The `news` family is not backtested
(its scores only exist forward) and enters directly as a challenger.

## Champion / challenger

- Rule families (`carry`, `trend_ts`, `xs_momentum`, `regime`): weekly Optuna
  TPE search over the declared space, at most 40 evaluations per family, every
  evaluation stored as a `trials` row (that is the trial counter behind the
  deflated Sharpe). The best candidate that passes the gate is inserted as a
  `challenger` with `parent_id` pointing at the current champion.
- ML families (`meta_label`): retrained every 14 days on a rolling 12-month
  window, purged walk-forward validation, artefact stored in `models`.
- Promotion, checked every tick: a challenger becomes champion when it has at
  least 6 weeks or 100 decisions in the arena, its forward Sharpe beats the
  champion's over their common window, and it is above the null 95th percentile
  over that window. The old champion becomes `retired`; its books are kept.
- Budget: at most 3 live challengers per family; the oldest surplus is retired.
- The allocator (`weight_i` proportional to `max(0, Sharpe_60d_i)` over
  champions, EMA-smoothed at about 10 %/day, residual to cash) is an opinion
  published in the digest, never an order.

## Deployment

The container image is `python:3.12-slim`, built from `Dockerfile` (`uv sync
--frozen`, non-root uid 1000, read-only root filesystem). The Gitea workflow in
`.gitea/workflows/build.yaml` tests, lints, builds and pushes it tagged with the
`pyproject.toml` version and the short SHA; `.github/workflows/ci.yml` runs the
same tests on GitHub.

The Helm chart in `helm/` deploys one CronJob per sub-command plus an
ExternalSecret and a NetworkPolicy. Values you must change for your cluster:

| Value | Default (homelab) | Change to |
|---|---|---|
| `image.repository` | `git.ebaillon.fr/bots/arena` | your registry |
| `image.tag` | current version | the tag your CI pushed |
| `namespace` | `crypto-research` | an existing namespace; the chart never creates it |
| `secrets.clusterSecretStore` | `vault-backend` | your ExternalSecrets `ClusterSecretStore` |
| `secrets.vaultPath` | `secret/arena` | the key in that store |
| `networkPolicy.postgresNamespace` | `database` | the namespace of your Postgres |
| `timeZone` | `Europe/Paris` | the time zone of the digest schedule |
| `web.host` | `arena.ebaillon.fr` | the hostname of the optional dashboard IngressRoute (set `web.enabled: false` to skip it) |
| `web.traefikNamespace` | `network` | the namespace of your Traefik, if you keep the IngressRoute |

The chart assumes a namespace with default-deny egress plus DNS and HTTPS
allowed, and adds egress to Postgres on 5432. If you do not run External
Secrets Operator, delete `helm/templates/externalsecret.yaml` and create the
Secret by hand.

Secret expected by every CronJob (`envFrom` on `secrets.targetName`, default `arena-secrets`):

| Key | Content |
|---|---|
| `DATABASE_URL` | `postgresql://arena:<password>@<host>:5432/arena` |
| `TELEGRAM_BOT_TOKEN` | bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | destination chat id |

CronJobs (`helm/values.yaml`, `schedules` and `resources`):

| CronJob | Schedule (UTC unless noted) | Command | Resources (requests to limits) |
|---|---|---|---|
| `arena-tick` | `5 * * * *` | `arena tick` | 200m/512Mi to 1 CPU/1Gi |
| `arena-news` | `*/30 * * * *` | `arena news` | 50m/128Mi to 200m/256Mi |
| `arena-digest` | `0 8 * * *` in `timeZone` | `arena digest` | 50m/128Mi to 200m/256Mi |
| `arena-challenger` | `0 3 * * 0` | `arena challenger` | 400m/640Mi to 2 CPU/2Gi |
| `arena-retrain` | `0 4 1,15 * *` | `arena retrain` | 400m/640Mi to 2 CPU/2Gi |

`bootstrap.enabled: true` renders a one-shot Job (`arena bootstrap --since
<bootstrap.since>`) that migrates, backfills, registers the null models and
gates the founders; flip it back to `false` afterwards. `env.ARENA_WORKERS`
should equal the heavy CPU limit.

A read-only web dashboard is available as the optional `web` extra
(`uv sync --extra web`, `arena web`) and, in the chart, under `web.*`; it is
not required for any of the above and has no authentication of its own.

### Dashboard authentication

The dashboard is read-only but exposes your research. With `web.basicAuth.enabled`
the chart expects a Secret holding a single `users` key (htpasswd lines, e.g.
`openssl passwd -apr1`), projected from `web.basicAuth.vaultPath`; Traefik rejects
multi-key secrets for BasicAuth.

## Two arenas: crypto and classic markets

`config/universe.yaml` is the crypto arena (Binance USDT perpetuals, 1h bars,
funding). `config/universe-classic.yaml` is the classic-markets arena (ETFs and
FX from Yahoo Finance, daily bars, no funding so no carry family). Point any
command at a universe with `UNIVERSE_PATH`; competitors, null models and books
are kept per universe, the dashboard and the digest show both.

## Known limitations

- **Carry is idealised.** A carry position is modelled as delta-neutral long
  spot / short perp that earns the perp funding and pays taker fees on both
  legs. There is no basis risk, no spot borrow or custody cost, no liquidation
  mechanics and no funding-rate slippage. Treat carry Sharpe as an upper bound.
- **The null distribution is expensive.** 50 seeded random backtests over the
  full history per gate call, cached for a week in `trials`. `ARENA_WORKERS`
  parallelises it; on a laptop expect minutes, not seconds. The spec asks for
  200 runs; 50 is the current compromise.
- **Public endpoints have rate limits.** Binance klines are fetched 1000 per
  call and Hyperliquid `info` is polled once per tick. A full backfill of 15
  symbols from 2024 is a few hundred calls; running it repeatedly from the same
  IP will get you throttled. Open interest history is capped at 30 days by the
  API.
- **Fees are a single taker model.** No maker rebates, no tiering, no spread
  model beyond a flat slippage number.
- **News scoring is rule-based** (VADER plus a crypto lexicon). It is
  deterministic and versioned so a better scorer can be added as a new
  `scorer_version`, but v1 should not be expected to carry much signal.
- **No forward track record yet.** The arena has run for days, not months.
  Nothing has been promoted on forward data.

## Layout

```
arena/
  core/         Target, Decision, CompetitorSpec, Snapshot, Universe
  data/         Binance, Hyperliquid, RSS, macro adapters (HTTPS, no keys)
  store/        Postgres repositories and SQL migrations
  nlp/          VADER-based news scorer and lexicon
  competitors/  Competitor / HoldingCompetitor contract, REGISTRY, the families
  book/         virtual book accounting (single code path)
  judge/        backtest, walk-forward, metrics, null distribution, gate, admission
  runner/       tick, ingest, allocator, drift, promotion, alerts, digest, history
  challenger/   Optuna spaces and search, meta-label retraining
  telegram/     templates and sender
  cli.py        typer entry point
config/universe.yaml
helm/           chart (CronJobs, ExternalSecret, NetworkPolicy, bootstrap Job)
docs/           design spec and implementation plan
tests/          pytest; Postgres-backed tests need PG_TEST_URL
```

## License

MIT, see `LICENSE`.

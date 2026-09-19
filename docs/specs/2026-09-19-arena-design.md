# Arena — design (2026-09-19)

Signal-only crypto research system. Several competing models run forward on the
same public market feed, are scored on virtual books, improved in the background
by a champion/challenger loop, and report their convictions to Telegram. Nothing
ever places an order. No private API key of any kind is used for data.

## 1. Why

Six freqtrade bots ran in dry-run from March to May 2026 and lost 14–20 % of
their virtual capital while their backtests claimed +5 000 % to +145 000 %.
The backtests were structurally biased (no stoploss, optimistic OHLC fills,
103 strategies selected on the same in-sample data, unlimited compounding).
The judge was the problem, not the infrastructure. Arena replaces the judge.

Principles, in order:

1. **Forward scoring.** The arena judges competitors on data that did not exist
   when they decided. Backtests are only an entry gate.
2. **Null models inside the arena.** Random, cash, BTC buy-and-hold and
   equal-weight carry always run. A competitor that does not leave the null
   distribution has found nothing, and the system says so.
3. **Deterministic judge.** Walk-forward, real fees and funding, deflated Sharpe
   using the trial count, block-bootstrap p-value. Python, no LLM. Reject by
   default.
4. **Nothing running is ever modified.** Improvements are new versions that
   enter as challengers and must beat their champion forward.
5. **Open data, no credentials.** Binance and Hyperliquid public endpoints,
   public RSS feeds, ForexFactory calendar. The only tokens in the system are
   the shared homelab Telegram bot token and the Postgres password.
6. **Signal-only.** Output is Telegram text. The human decides and acts.

## 2. Scope

In scope (this spec): data ingestion and storage, virtual books, arena runner,
null models, six founding competitor families, entry gate, allocator, drift
detection, champion/challenger loop, Telegram reporting, Helm chart, CI,
ArgoCD registration in namespace `crypto-research`.

Out of scope: order execution, exchange credentials, LLM in the decision or
scoring path, deep networks on price series, X/Twitter (paid API), a web UI.

## 3. Universe and clock

- **Assets**: USDT perpetuals on Binance Futures that also trade on Hyperliquid:
  BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, DOT, LTC, NEAR, SUI, ARB, OP.
  Configurable list in `config/universe.yaml`.
- **Bar**: 1h. Everything decides on the close of the hour, using only data
  with `ts <= decision_ts`. 4h and 1d bars are derived in SQL.
- **Clock**: one `tick` per hour at HH:05 UTC. Late data (Binance publishes the
  closed candle a few seconds after the hour) is handled by ingesting up to the
  last *closed* candle only.

## 4. Data (module `arena.data`)

| Source | Endpoint (public, no key) | What | Backfill |
|---|---|---|---|
| Binance Futures | `GET /fapi/v1/klines` | 1h OHLCV | 2024-01-01 → now, 1000/call |
| Binance Futures | `GET /fapi/v1/fundingRate` | 8h funding | same |
| Binance Futures | `GET /futures/data/openInterestHist` | OI 1h | last 30 days only (API limit) |
| Hyperliquid | `POST https://api.hyperliquid.xyz/info` `metaAndAssetCtxs` | current funding, OI, mark per coin | none (snapshot each tick) |
| Hyperliquid | `POST .../info` `fundingHistory` | funding history per coin | as far as API allows |
| News | RSS: CoinDesk, CoinTelegraph, The Block, Decrypt, Bitcoin Magazine, The Defiant | title, summary, link, published | none (forward only) |
| Macro | `https://nfs.faireconomy.media/ff_calendar_thisweek.json` | high-impact events | none |

Rules:

- Every row carries `fetched_at`. Articles are **point-in-time**: never updated
  after insert; a re-fetch with a different title is a new row.
- Ingestion is idempotent (`INSERT … ON CONFLICT DO NOTHING` on natural keys).
- Egress: HTTPS 443 only (matches the namespace NetworkPolicy). No HTTP feeds.

## 5. Storage (Postgres, database `arena` on the shared CNPG cluster)

Migrations are plain SQL files applied by `arena migrate` (table
`schema_migrations`). Core tables:

```
candles(exchange, symbol, tf, ts, open, high, low, close, volume, fetched_at)  PK(exchange,symbol,tf,ts)
funding(exchange, symbol, ts, rate, fetched_at)                                 PK(exchange,symbol,ts)
open_interest(exchange, symbol, ts, oi, fetched_at)                             PK(exchange,symbol,ts)
hl_snapshots(ts, coin, funding, oi, mark, fetched_at)                           PK(ts,coin)
articles(id, source, url, title, summary, published_at, fetched_at)             UNIQUE(source,url,title)
article_scores(article_id, asset, sentiment, event_type, intensity, scorer_version) PK(article_id,asset,scorer_version)
macro_events(id, ts, currency, title, impact, fetched_at)
competitors(id, name, family, version, parent_id, params jsonb, role, status, rationale, created_at)
   role   ∈ {null, benchmark, competitor}
   status ∈ {candidate, challenger, champion, retired}
trials(id, competitor_id, kind, started_at, finished_at, params jsonb, metrics jsonb, verdict, notes)
   kind ∈ {backtest, walkforward, optimize, retrain}
targets(competitor_id, ts, symbol, weight, conviction, reason jsonb)            PK(competitor_id,ts,symbol)
books(competitor_id, ts, nav, ret, gross, turnover, fees, funding_pnl)         PK(competitor_id,ts)
allocations(ts, competitor_id, weight)                                          PK(ts,competitor_id)
alerts(id, ts, kind, competitor_id, symbol, payload jsonb, sent_at)
models(id, competitor_id, trained_at, artifact bytea, metrics jsonb)
```

## 6. Competitor contract (module `arena.competitors`)

```python
class Competitor(Protocol):
    family: str
    version: int
    params: dict
    def warmup_bars(self) -> int
    def decide(self, snap: Snapshot) -> dict[str, Target]   # symbol -> Target(weight, conviction, reason)
```

- `Snapshot` exposes, for every symbol, pandas frames strictly limited to
  `ts <= decision_ts`: 1h/4h/1d candles, funding, OI, Hyperliquid funding,
  news features, macro flags. Competitors cannot reach the database.
- `weight ∈ [-1, 1]` is the target fraction of NAV in that perp (negative =
  short). Constraint `Σ|weight| ≤ 1` (gross leverage 1). A **carry** position
  is expressed with `kind="carry"`: long spot / short perp, delta-neutral,
  earns funding, pays fees on both legs.
- `conviction ∈ [0, 1]` is only used for Telegram wording and allocator tie-break.
- Competitors are pure functions of `(params, Snapshot)`. Same input, same
  output. This is what makes trials countable.

### Founding families

| Family | Economic reason | Sketch |
|---|---|---|
| `carry` | Perp funding is paid by leveraged longs to shorts; harvesting it is a documented premium. | Rank symbols by 3-day mean funding; carry the top-k above a threshold; exit when funding turns negative. |
| `trend_ts` | Time-series momentum (Moskowitz et al.) is robust across assets and decades. | Sign of 30d and 90d returns + EMA(50/200) alignment; position scaled to target 20 % annualised vol per leg; ATR-based exit. |
| `xs_momentum` | Cross-sectional momentum (12-1 style) survives in crypto. | Rank by 30d return skipping the last 2d; long top 3, short bottom 3; weekly rebalance with a rank-hysteresis band. |
| `regime` | Different rules pay in different states; conditioning is where adaptation belongs. | Realised vol tercile × trend sign → {bull_calm, bull_vol, bear, range}; maps each state to a fixed allocation (trend in bull, carry in range, cash in bear-vol). v1 is rule-based; v2 is a LightGBM classifier trained walk-forward. |
| `meta_label` | López de Prado: let a model decide *whether* to take a base signal and *how much*, not *where*. | LightGBM on base-signal events from `trend_ts`/`xs_momentum` with context features (vol, regime, funding, OI change, news 24h/7d, hour, macro-day). Label = sign of forward 24h return net of fees. Output weight = base weight × P(win) above 0.55. |
| `news` | Slow narrative sentiment carries a small, multi-day signal. | Per-asset sentiment 24h and 7d from `article_scores`; long assets whose 7d sentiment is positive and rising, sized by event density; standalone so its marginal value is measurable. |

Null models and benchmarks (always in the arena, never retired):
`null_cash`, `null_random` (seeded, 5 instances), `bench_btc_hold`,
`bench_carry_equal` (carry on all symbols equally).

### News scoring (module `arena.nlp`)

Deterministic, local, no model download at runtime by default:
`vader` scorer = VADER compound score + a crypto lexicon overlay + asset
detection by ticker/name aliases + rule-based `event_type`
(regulation, hack, listing, etf, macro, protocol, other). `scorer_version` is
stored with every score so a better scorer (e.g. FinBERT, optional extra
`arena[finbert]`) can be added as a new version without rewriting history.

## 7. Virtual books (module `arena.arena.book`)

For every competitor, each tick:

1. Mark existing positions to the 1h close.
2. Apply funding to perp positions at funding timestamps (8h), sign-aware.
3. Move to the new targets: turnover = Σ|Δweight|; fee = turnover × 0.05 %
   (Hyperliquid taker) + slippage 0.02 %. Carry legs pay spot 0.10 % + perp fee.
4. Record `books` row: nav, ret, gross, turnover, fees, funding_pnl.

Starting NAV 10 000 (virtual). No leverage beyond gross 1. No compounding
trick: returns are simple period returns on NAV.

## 8. Entry gate (module `arena.judge`)

`arena judge <competitor>` runs on the stored history and writes a `trials`
row with the verdict. Steps:

1. **Backtest** with the same `book` code as live (one code path).
2. **Anchored walk-forward**: expanding train window, 90-day test folds,
   parameters fixed inside a fold.
3. **Metrics** per fold and overall: net return, Sharpe, Sortino, max drawdown,
   profit factor, turnover, exposure, number of decisions.
4. **Null distribution**: 200 seeded `null_random` runs on the same period →
   empirical 95th percentile of Sharpe.
5. **Deflated Sharpe** (Bailey & López de Prado) using the total number of
   trials recorded for this family in `trials`.
6. **Stationary block bootstrap** of the return series (block 24 bars,
   1000 draws) → p-value of Sharpe > 0.

Verdict `admitted` requires all of: net return > 0 on ≥ 2/3 of folds;
Sharpe > null 95th percentile; DSR > 0.90; bootstrap p < 0.10; max drawdown
< 30 %; ≥ 30 decisions. Otherwise `rejected`, with the failing criteria listed.
Null models and benchmarks skip the gate.

**Not gated by backtest**: the `news` family (its scores can only exist
forward). It enters directly as `challenger` and is judged only in the arena.

## 9. Arena runner and allocator (module `arena.arena`)

`arena tick` (hourly CronJob):

1. Ingest closed 1h candles, funding, OI, Hyperliquid snapshot (idempotent).
2. Build the `Snapshot` at the last closed bar.
3. For every competitor with status ≠ retired: `decide`, store `targets`,
   update `book`.
4. **Allocator**: weight_i ∝ max(0, Sharpe_60d_i)^1 over competitors with
   status champion, EMA-smoothed (α = 0.1 per day) to avoid chasing; residual
   to cash. Stored in `allocations`. It is an *opinion* published to Telegram,
   not an order.
5. **Drift checks**: data staleness > 2 bars; a champion's live 30-day Sharpe
   below its walk-forward 5th percentile; a competitor with zero decisions in
   7 days; a book with NaN. Each raises an `alert`.
6. **Champion/challenger evaluation** (see §10).
7. Send Telegram alerts (see §11).

## 10. Champion / challenger (module `arena.challenger`)

Background, weekly (Sunday 03:00 UTC) under a CPU-limited CronJob:

- **Rule families** (`carry`, `trend_ts`, `xs_momentum`, `regime` v1): Optuna
  TPE over the declared parameter space, objective = mean out-of-fold Sharpe in
  the nested walk-forward, ≤ 15 evaluations per family per week (measured: ~3 min per trend_ts evaluation on 2 cores), every
  evaluation stored as a `trials` row (this is the trial counter that deflates
  every Sharpe). The best candidate that passes the gate is inserted as
  `challenger` with `parent_id` = current champion.
- **ML families** (`regime` v2, `meta_label`): retrain every 14 days on a
  rolling 12-month window, purged walk-forward validation, artefact saved in
  `models`. The new version enters as `challenger`.
- **Promotion** (checked every tick): a challenger becomes champion when it has
  ≥ 6 weeks or ≥ 100 decisions in the arena, its forward Sharpe exceeds the
  champion's over the common window, and it is above the null 95th
  percentile over that window. The old champion becomes `retired` (books kept).
  Promotions raise an `alert`.
- **Budget**: at most 3 live challengers per family. Oldest surplus challenger
  is retired.

## 11. Telegram (module `arena.telegram`)

Templates only, no LLM. Bot token and chat id from the ESO-projected secret.

- **Signal alert** (at most once per competitor per symbol per 4h): a champion
  flips direction or changes weight by ≥ 0.25.
  `⚔️ trend_ts v3 → LONG ETH 0.35 (conv 0.71) | regime bull_vol | fund +0.012%/8h | why: ema50>ema200, r30 +18%`
- **Daily digest** 08:00 Europe/Paris: leaderboard (30d Sharpe, 30d return,
  MDD, status), allocator opinion, null 95th percentile line, open challengers
  and their progress to promotion, drift alerts of the day.
- **Event alert**: promotion, rejection at the gate, data stale, exception.

Rate limit: ≤ 20 messages/day besides the digest; surplus is folded into the
digest.

## 12. Deployment

- **Repo** `git.ebaillon.fr/bots/arena`, mirrored at `orgs/bots/arena`.
- **Image** built by Gitea Actions (server is deploy-only): `python:3.12-slim`,
  `uv sync --frozen`, non-root uid 1000. Tagged with the version in
  `pyproject.toml` and the short SHA.
- **Namespace** `crypto-research` (exists: ResourceQuota 2 CPU / 2 Gi requests,
  4 CPU / 4 Gi limits, PSA restricted, default-deny egress with allow-dns and
  allow-internet-https). Chart adds `allow-shared-postgres` (egress 5432 to
  namespace `database`).
- **Secrets**: Vault `secret/arena` = {TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
  DATABASE_URL}; ExternalSecret in the chart projects it to `arena-secrets`.
- **Database**: role `arena` and database `arena` on `shared-postgres`;
  credentials only in Vault.
- **Workloads** (all the same image, different `arena` sub-commands):

| CronJob | Schedule (UTC) | Command | Resources |
|---|---|---|---|
| `arena-tick` | `5 * * * *` | `arena tick` | 200m/512Mi → 1/1Gi |
| `arena-news` | `*/30 * * * *` | `arena ingest news` | 50m/128Mi → 200m/256Mi |
| `arena-digest` | `0 6 * * *` (08:00 Paris in summer, 07:00 in winter → uses `timeZone: Europe/Paris` at `0 8`) | `arena digest` | 100m/256Mi |
| `arena-challenger` | `0 3 * * 0` | `arena challenger` | 500m/1Gi → 2/2Gi, `concurrencyPolicy: Forbid` |
| `arena-retrain` | `0 4 1,15 * *` | `arena retrain` | same as challenger |

  Plus a manual Job `arena-bootstrap` (migrate → backfill → judge founders →
  register). ArgoCD app `arena`, `selfHeal: false` (manual sync, as the rest
  of `crypto-research`).

## 13. Testing

- Unit: book accounting (fees, funding sign, carry legs), metrics (Sharpe,
  DSR, MDD against hand-computed cases), gate verdicts on synthetic series
  (pure noise must be rejected; a planted drift must be admitted), each
  competitor on synthetic snapshots (no look-ahead: shifting future data must
  not change today's decision), NLP scorer on fixed sentences.
- Integration: migrations + ingest + tick against a local Postgres
  (`~/.local/pg16`) with recorded HTTP fixtures; Telegram in dry mode.
- Acceptance before hand-over: chart renders and dry-run applies; image built by
  CI; bootstrap job completes on the cluster; at least one real `tick` with
  books written for every competitor; one real digest received on Telegram.

## 14. What is decommissioned

Freqtrade (6 deployments at 0, 6 PVCs), trading-brain, neuronalpha remain
untouched by this spec but are marked for removal in a separate clean-up;
credentials found in clear in the retired repositories are to be rotated. Tickforge stays as the tick-level
confirmation tool for finalists (future work).

## 15. Roadmap after this spec

- Tick-level confirmation of admitted competitors in tickforge.
- FinBERT/CryptoBERT scorer as `scorer_version` 2.
- Fine-tuning the news scorer once ≥ 3 months of forward labels exist.
- On-chain flows (public explorers) as a data source.

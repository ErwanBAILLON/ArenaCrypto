# Arena Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the signal-only crypto Arena described in `docs/specs/2026-09-19-arena-design.md`: public-data ingestion, virtual books, competing models scored forward, deterministic entry gate, champion/challenger loop, Telegram reporting, deployed as CronJobs in namespace `crypto-research`.

**Architecture:** One Python package `arena` with strictly layered modules: `data` (HTTP → rows), `store` (Postgres), `core` (shared types, Snapshot), `competitors` (pure decide functions), `book` (virtual accounting), `judge` (backtest/walk-forward/metrics/gate), `runner` (tick/allocator/drift/promotion), `challenger` (optimize/retrain), `nlp` (news scoring), `telegram` (templates + sender), `cli`. Competitors never touch the DB; the same `book` code scores backtests and live.

**Tech Stack:** Python 3.12, uv, pandas, numpy, psycopg 3, httpx, feedparser, vaderSentiment, lightgbm, optuna, typer, pytest. Postgres 16 (shared CNPG). Helm + ArgoCD. Gitea Actions builds the image.

---

## File structure

```
orgs/bots/arena/
├── pyproject.toml                 uv project, deps, scripts: arena = "arena.cli:app"
├── uv.lock
├── Dockerfile                     python:3.12-slim, uv sync --frozen, uid 1000
├── .gitea/workflows/build.yaml    build + push git.ebaillon.fr/bots/arena:<version>-<sha>
├── config/universe.yaml           symbols, fees, universe metadata
├── arena/
│   ├── __init__.py
│   ├── settings.py                env → Settings (DATABASE_URL, TELEGRAM_*, DRY_RUN, UNIVERSE_PATH)
│   ├── core/
│   │   ├── types.py               Target, Decision, CompetitorSpec, Verdict, dataclasses
│   │   ├── snapshot.py            Snapshot (frames limited to ts<=decision_ts) + builder from DataFrames
│   │   └── universe.py            load universe.yaml → Universe(symbols, fees)
│   ├── store/
│   │   ├── db.py                  connect(), run_migrations(), helpers
│   │   ├── migrations/0001_core.sql ... 
│   │   ├── candles.py             upsert/read candles, funding, oi, hl_snapshots
│   │   ├── news.py                upsert articles, scores, read news features
│   │   ├── registry.py            competitors, trials, models CRUD
│   │   └── books.py               targets, books, allocations, alerts CRUD
│   ├── data/
│   │   ├── http.py                httpx client with retries/backoff, UA
│   │   ├── binance.py             klines, funding, open_interest_hist
│   │   ├── hyperliquid.py         meta_and_asset_ctxs, funding_history
│   │   ├── rss.py                 fetch feeds → Article rows
│   │   └── macro.py               forexfactory calendar
│   ├── nlp/
│   │   ├── lexicon.py             crypto lexicon overlay + asset aliases
│   │   └── scorer.py              VaderScorer.score(article) -> list[ArticleScore]
│   ├── competitors/
│   │   ├── base.py                Competitor ABC, registry (family → class), build(spec)
│   │   ├── nulls.py               NullCash, NullRandom, BenchBtcHold, BenchCarryEqual
│   │   ├── carry.py
│   │   ├── trend_ts.py
│   │   ├── xs_momentum.py
│   │   ├── regime.py              rule-based v1 + regime_label() helper reused by others
│   │   ├── meta_label.py          LightGBM on base-signal events
│   │   ├── news.py
│   │   └── features.py            shared indicator helpers (ema, atr, realised_vol, returns)
│   ├── book/
│   │   └── book.py                Book.step(prices, funding, targets) → BookRow; fees model
│   ├── judge/
│   │   ├── backtest.py            run(competitor, history, universe) → returns Series + decisions
│   │   ├── walkforward.py         anchored folds
│   │   ├── metrics.py             sharpe, sortino, mdd, profit_factor, dsr, block_bootstrap_p
│   │   └── gate.py                evaluate(...) → Verdict
│   ├── runner/
│   │   ├── tick.py                the hourly tick orchestration
│   │   ├── allocator.py
│   │   ├── drift.py
│   │   └── promotion.py
│   ├── challenger/
│   │   ├── spaces.py              Optuna search spaces per family
│   │   ├── optimize.py
│   │   └── retrain.py
│   ├── telegram/
│   │   ├── templates.py           pure functions → str
│   │   └── sender.py              send(text) with DRY_RUN + rate limit
│   └── cli.py                     migrate, backfill, ingest, tick, digest, judge, challenger, retrain, bootstrap
├── tests/                         mirrors package; conftest with synthetic history + pg fixture
├── helm/                          Chart.yaml, values.yaml, templates/{externalsecret,netpol,cronjobs,bootstrap-job,configmap}.yaml
└── docs/{specs,plans}
```

## Shared contracts (implemented first, in Task 1; every later task imports these verbatim)

```python
# arena/core/types.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Any

Kind = Literal["perp", "carry"]

@dataclass(frozen=True)
class Target:
    weight: float                 # [-1, 1]; for kind="carry" weight>0 = size of the delta-neutral carry
    conviction: float = 0.5       # [0, 1]
    kind: Kind = "perp"
    reason: dict[str, Any] = field(default_factory=dict)

Decision = dict[str, Target]      # symbol -> Target ; missing symbol == flat

@dataclass(frozen=True)
class CompetitorSpec:
    id: int | None
    name: str
    family: str
    version: int
    params: dict[str, Any]
    role: Literal["null", "benchmark", "competitor"] = "competitor"
    status: Literal["candidate", "challenger", "champion", "retired"] = "candidate"
    parent_id: int | None = None
    rationale: str = ""

@dataclass
class BookRow:
    ts: datetime
    nav: float
    ret: float
    gross: float
    turnover: float
    fees: float
    funding_pnl: float

@dataclass
class Verdict:
    admitted: bool
    metrics: dict[str, float]
    failed: list[str]
```

```python
# arena/core/snapshot.py  (public surface)
class Snapshot:
    ts: datetime                                  # decision timestamp (last closed 1h bar)
    symbols: list[str]
    def candles(self, symbol: str, tf: str = "1h") -> pd.DataFrame   # index ts (UTC), cols open high low close volume, ts <= self.ts, tf in {"1h","4h","1d"}
    def funding(self, symbol: str) -> pd.Series                      # index ts, rate per 8h, Binance
    def hl_funding(self, symbol: str) -> float | None                # latest Hyperliquid hourly funding rate
    def open_interest(self, symbol: str) -> pd.Series
    def news(self, symbol: str) -> pd.DataFrame                      # cols sent_24h, sent_7d, n_24h, n_7d, shock (0/1); index ts hourly
    def macro_today(self) -> bool                                    # high-impact event within ±12h
    def close(self, symbol: str) -> float
    def closes(self) -> pd.DataFrame                                 # wide 1h closes, all symbols
```

```python
# arena/competitors/base.py (public surface)
class Competitor(ABC):
    family: ClassVar[str]
    default_params: ClassVar[dict]
    def __init__(self, params: dict | None = None, seed: int = 0): ...
    def warmup_bars(self) -> int: ...
    @abstractmethod
    def decide(self, snap: Snapshot) -> Decision: ...

REGISTRY: dict[str, type[Competitor]]
def register(cls): ...
def build(spec: CompetitorSpec) -> Competitor: ...
```

```python
# arena/book/book.py (public surface)
@dataclass
class FeeModel:
    perp_taker: float = 0.0005
    slippage: float = 0.0002
    spot_taker: float = 0.0010

class Book:
    def __init__(self, nav0: float = 10_000.0, fees: FeeModel = FeeModel()): ...
    positions: dict[str, tuple[Kind, float]]          # symbol -> (kind, weight currently held)
    def step(self, ts, prices: dict[str, float], prev_prices: dict[str, float],
             funding: dict[str, float], targets: Decision) -> BookRow
    # funding: rate accrued during (prev_ts, ts] per symbol (0 if none); short perp receives +rate*|w|, long pays; carry receives +rate*w
```

Database schema is the one in the spec §5, file `arena/store/migrations/0001_core.sql`.

---

## Tasks

### Task 1: Scaffold, settings, core types, universe, migrations
Create `pyproject.toml` (deps: pandas numpy psycopg[binary] httpx feedparser vaderSentiment lightgbm optuna typer pyyaml scipy; dev: pytest pytest-cov), `arena/settings.py`, `arena/core/types.py`, `arena/core/universe.py`, `config/universe.yaml` (15 symbols, fees), `arena/store/db.py` (psycopg connect from `DATABASE_URL`, `run_migrations(conn)` applying `migrations/*.sql` in order, tracked in `schema_migrations`), `arena/store/migrations/0001_core.sql` (spec §5 exactly, timestamptz everywhere, indexes on `(competitor_id, ts)` for books/targets, `(symbol, ts)` for candles). Tests: `tests/core/test_universe.py` (loads 15 symbols, fee values), `tests/store/test_migrations.py` (against `PG_TEST_URL`, skipped if unset; applies twice idempotently; tables exist).

### Task 2: Snapshot
`arena/core/snapshot.py`: `Snapshot.from_frames(ts, candles_1h: DataFrame[long: exchange,symbol,ts,o,h,l,c,v], funding, oi, hl, news, macro)`; resamples 4h/1d from 1h with `label="right", closed="right"` and drops the last partial bar; **all accessors filter `ts <= self.ts`**. Tests: look-ahead guard (appending future rows changes nothing for accessors at same `ts`), resampling alignment (4h bar at 04:00 contains 01:00–04:00 closes), missing symbol returns empty frame not exception.

### Task 3: Book
`arena/book/book.py` per contract. Semantics: `ret = Σ_pos w * (p/p_prev - 1) [perp, signed] + funding_pnl - fees`; carry: price PnL = 0, funding_pnl += rate*w; fees on turnover Σ|Δw| × (perp_taker + slippage), carry turnover additionally × spot_taker; `gross = Σ|w|` after rebalance; if `Σ|w_target| > 1` scale targets down proportionally. Tests with hand-computed numbers: long +10% move → ret 0.10*w minus fees; short funding sign; carry earns funding only; gross cap; zero-turnover → zero fees.

### Task 4: Data adapters
`arena/data/http.py` (httpx.Client, 3 retries exponential, 10s timeout, UA "arena/0.1"); `binance.py` (`klines(symbol, start_ms, end_ms) -> DataFrame` paging 1000 with closed-bar filter `close_time < now`; `funding(symbol, start_ms)`; `open_interest_hist(symbol)`); `hyperliquid.py` (`meta_and_asset_ctxs() -> DataFrame[coin, funding, oi, mark]`, `funding_history(coin, start_ms)`); `rss.py` (feed list constant, `fetch(feeds) -> list[Article]`, parse published with tz, fallback fetched_at); `macro.py`. Symbol mapping: universe symbol `BTC` → Binance `BTCUSDT`, Hyperliquid `BTC`. Tests use recorded JSON fixtures under `tests/fixtures/` with `httpx.MockTransport`; no network in tests.

### Task 5: Store repositories
`arena/store/candles.py` (`upsert_candles(conn, df)`, `read_candles(conn, symbols, tf, start, end)`, same for funding/oi/hl), `news.py` (`upsert_articles`, `upsert_scores`, `news_features(conn, symbols, start, end) -> DataFrame` computing sent_24h/sent_7d/n_24h/n_7d/shock hourly in SQL with window functions on `published_at`), `registry.py` (competitor CRUD, `count_trials(family)`, `add_trial`), `books.py` (`write_targets`, `write_book_rows`, `read_returns(competitor_id, start, end)`, `write_allocations`, `add_alert`, `unsent_alerts`, `mark_sent`). Integration tests on `PG_TEST_URL`.

### Task 6: NLP scorer
`arena/nlp/lexicon.py` (asset aliases: BTC ← bitcoin, btc; ETH ← ethereum, ether…; crypto overlay words: "hack" -3, "exploit" -3, "rug" -3, "delist" -2.5, "ban" -2, "lawsuit" -2, "sec sues" -3, "etf approved" +3, "listing" +1.5, "upgrade" +1, "partnership" +1, "all-time high" +2, "liquidation" -1.5 …; event rules by keywords). `scorer.py`: `VaderScorer(version=1).score(article) -> list[ArticleScore(article_id, asset, sentiment[-1,1], event_type, intensity[0,1])]`, one row per detected asset, `asset="MARKET"` when none detected. Tests on fixed sentences (hack → negative, etf approved → positive, asset detection, MARKET fallback).

### Task 7: Competitors — nulls, carry, trend_ts, xs_momentum, regime v1, features
`features.py`: `ema`, `atr`, `realised_vol(closes, window, bars_per_year=8760)`, `pct_return(closes, bars)`. Each competitor: `default_params`, `warmup_bars`, `decide`. Sketches per spec §6. `regime.py` exposes `regime_label(snap, symbol) -> str` used by meta_label. All decisions respect `Σ|w| ≤ 1`. Tests per competitor: synthetic trending series → trend_ts long; synthetic positive funding → carry picks it; ranking correctness for xs_momentum with hysteresis; regime labels on constructed vol/trend; determinism (same snap → same decision); no look-ahead.

### Task 8: Judge — backtest, walk-forward, metrics, gate
`backtest.run(competitor, history: HistoryFrames, universe, start, end, step="1h") -> BacktestResult(returns: Series, rows: list[BookRow], decisions: int)` building a `Snapshot` per bar (efficiently: pre-sliced frames, `Snapshot.at(ts)` view) and stepping one `Book`. `walkforward.folds(start, end, test_days=90, min_train_days=180)`. `metrics`: `sharpe(r, ppy=8760)`, `sortino`, `max_drawdown`, `profit_factor`, `deflated_sharpe(sr, n_trials, T, skew, kurt)`, `block_bootstrap_p(r, block=24, n=1000, seed)`. `gate.evaluate(result_by_fold, null_sharpes, n_trials) -> Verdict` with the spec §8 thresholds. Tests: metrics on hand cases; gate rejects pure noise competitor and admits a planted drift (returns = 0.0005 + noise); folds non-overlapping and anchored.

### Task 9: Runner — tick, allocator, drift, promotion
`tick.run(conn, settings, now)`: ingest (binance klines/funding/oi for universe since last stored ts, hl snapshot), load history window (max warmup + 400 bars), build Snapshot, for each non-retired competitor build → decide → book step (book state reconstructed from last `books` row + last `targets`), write targets/books, allocator, drift, promotion, queue alerts. `allocator.compute(returns_by_champion: DataFrame, prev_weights, alpha=0.1/24) -> dict`. `drift.check(...) -> list[Alert]`. `promotion.evaluate(conn) -> list[Alert]` per spec §10 rules. Tests: allocator math (negative Sharpe → 0; EMA smoothing), drift detection on constructed series, promotion rule on constructed books.

### Task 10: Challenger — optimize and retrain
`spaces.py` per family (carry: k∈[1,5], threshold∈[0,0.0005], lookback∈[3,21]d; trend_ts: fast∈[20,80], slow∈[100,300], target_vol∈[0.10,0.30], lookbacks; xs_momentum: k∈[2,5], lookback∈[14,60]d, skip∈[0,3]d, band∈[0,3]; regime: vol windows and thresholds). `optimize.run(conn, family, n_trials=40, seed)` → Optuna study, objective mean out-of-fold Sharpe from `walkforward` + `backtest`, every evaluation → `trials` row; best passes `gate` → insert challenger with `parent_id`. `retrain.run(conn, family)` for `meta_label`/`regime` v2: build event dataset from history, purged WF, LightGBM, save artefact to `models`, insert challenger. Tests: optimize on tiny synthetic history with n_trials=3 writes 3 trials; retrain produces a model with AUC > 0.5 on planted signal.

### Task 11: meta_label and news competitors
`meta_label.py`: features per event (vol_30d, regime one-hot, funding_3d, oi_change_24h, sent_24h, sent_7d, hour, macro_today, base_weight); `decide`: get base decisions from `trend_ts` and `xs_momentum` champions' params (passed in `params["bases"]`), score with loaded model (from `models` artefact injected via `params["model_bytes"]`), output base weight × P if P>0.55 else flat. `news.py`: per spec. Tests: meta_label without model = pass-through of base; with a model that always predicts 0.9 → weights scaled; news competitor longs positive rising sentiment only.

### Task 12: Telegram
`templates.py`: `signal_alert(...)`, `daily_digest(leaderboard_df, allocation, null95, challengers, drift_alerts) -> str` (≤ 4000 chars, Markdown-safe escaping), `event_alert`. `sender.py`: `send(settings, text)` POST `https://api.telegram.org/bot<token>/sendMessage`, DRY_RUN prints; rate limit 20/day via `alerts.sent_at` count. Tests: templates on fixtures, escaping, length; sender dry mode.

### Task 13: CLI and bootstrap
`cli.py` (typer): `migrate`, `backfill --since 2024-01-01`, `ingest {market,news,macro}`, `score-news`, `tick`, `digest`, `judge <name>`, `challenger [--family]`, `retrain [--family]`, `bootstrap` (migrate → backfill → register nulls/benchmarks → for each founding family judge default params → insert as champion if admitted, else as challenger with alert "rejected at gate" → news family as challenger). Smoke test: `arena --help`; `bootstrap --dry-run` on local PG with fixtures.

### Task 14: Dockerfile, CI, Helm chart, ArgoCD
Dockerfile multi-stage (uv), non-root 1000, `ENTRYPOINT ["arena"]`. `.gitea/workflows/build.yaml` modelled on `orgs/bots/openalice/.gitea/workflows/build.yaml` (tag = version-sha; secrets REGISTRY_USERNAME/PASSWORD set on the repo). Helm: `values.yaml` (image, schedules, resources, universe), templates: `externalsecret.yaml` (ClusterSecretStore `vault-backend`, `secret/arena` → `arena-secrets`), `netpol-postgres.yaml` (egress 5432 to ns `database`), `cronjobs.yaml` (5 CronJobs, PSA-restricted securityContext, `timeZone`), `bootstrap-job.yaml` (Job, `helm.sh/hook: post-install`? no: manual Job with `argocd.argoproj.io/hook: Skip`; keep as plain Job gated by `bootstrap.enabled`), `configmap-universe.yaml`. Register the app in `orgs/infra/argocd/values.yaml` (`selfHeal: false`) and add repo credential entry. Validate: `helm template | kubectl apply --dry-run=client`.

### Task 15: Provisioning and go-live
Create Gitea repo `bots/arena`, push, set Actions secrets, wait for image. Create role+database `arena` on `shared-postgres` (psql via kubectl exec, password generated locally), write Vault `secret/arena` with the file+wget method, verify ExternalSecret synced. Sync ArgoCD app. Run bootstrap Job; check logs; force one `tick` (`kubectl create job --from=cronjob/arena-tick`); verify `books` rows for every competitor; run `digest` once; confirm Telegram message received. Update homelab `CLAUDE.md` repo layout and memory notes.

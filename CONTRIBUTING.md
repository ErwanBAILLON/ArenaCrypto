# Contributing to Arena

Arena is a research tool with one job: tell the truth about whether a model has
found anything. Contributions are welcome as long as they keep that property.

## Development setup

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), a Postgres 16 (Docker is fine).

```bash
uv sync --extra dev                # venv with pytest, ruff, and the runtime deps
docker compose up -d postgres      # Postgres 16 on localhost:5433 (user/db/password: arena)
```

Create a **dedicated test database**. The `conn` fixture drops and recreates the
`public` schema of whatever `PG_TEST_URL` points to:

```bash
docker compose exec postgres createdb -U arena arena_test
```

## Running the tests

```bash
PG_TEST_URL=postgresql://arena:arena@localhost:5433/arena_test uv run pytest
```

Without `PG_TEST_URL` the store, tick-integration and migration tests are
skipped; everything else (competitors, judge, book, nlp, telegram) runs on
synthetic data and must pass on its own.

CI runs exactly `ruff check`, `ruff format --check` and `pytest` (see
`.github/workflows/ci.yml` and `.gitea/workflows/build.yaml`).

## Style

- `ruff check arena tests` and `ruff format arena tests` must be clean. Line length 120, rules
  `E, F, I, B, UP, W`. No `# noqa` without a reason on the same line.
- One responsibility per module. A module name says what the module does; if you need "and" to
  describe it, split it.
- Every module starts with a docstring. For competitors the docstring states the **economic
  rationale**: who pays the premium you are harvesting and why they keep paying. A rule without a
  reason is curve fitting waiting to be found out.
- Competitors are pure functions of `(params, Snapshot)`. No database access, no clock, no network,
  no randomness except from a seed in `params`. State is allowed only for hysteresis and must be fully
  captured by `state()` / `restore_state()`.
- Types where they help (`Decision`, `Target`, `CompetitorSpec`), `from __future__ import annotations`
  at the top of every module.
- Comments and commit messages in English. Conventional-commit prefixes (`feat:`, `fix:`, `perf:`,
  `test:`, `ci:`, `docs:`, `chore:`).

## Adding a competitor family

1. Create `arena/competitors/<family>.py`, subclass `Competitor` (hourly re-decision) or
   `HoldingCompetitor` (decide once per rebalance slot, hold in between; this is what you want unless
   you have a reason to pay hourly fees). Decorate with `@register`, set `family` and `default_params`,
   implement `warmup_bars()` and `decide()` (or `compute()` for holding competitors).
2. Import the module in `arena/competitors/__init__.py` so it lands in `REGISTRY`.
3. If the family is meant to be a founder, add it to `FOUNDERS` in `arena/cli.py`; if it should be
   searchable by the weekly challenger loop, declare its space in `arena/challenger/spaces.py`.
4. Ship tests in `tests/competitors/test_<family>.py`. At minimum:
   - a **no-look-ahead test**: the decision at bar `t` from a snapshot truncated at `t` must equal the
     decision from a full-history snapshot viewed `.at(t)` (add your factory to `FACTORIES` in
     `tests/competitors/test_no_lookahead.py`);
   - determinism: same snapshot, same decision, twice and across fresh instances;
   - the gross constraint `sum(|w|) <= 1` on a snapshot where the raw signal would exceed it;
   - if the competitor keeps state, a `state()` / `restore_state()` round trip.
5. Run the gate locally on stored history: `arena judge <family>`. Expect a rejection the first time;
   that is the gate working.

## What a pull request must not do

- **Weaken the gate.** Thresholds in `arena/judge/gate.py` (`GateConfig`), the null distribution, the
  deflated Sharpe trial count and the bootstrap p-value are the whole point. A PR that relaxes them
  needs a written argument, not a failing competitor.
- Touch the book accounting to make a competitor look better. `arena/book/book.py` is the single code
  path for backtests and the live arena; changes there need hand-computed test cases.
- Modify a running competitor's parameters in place. Improvements are new versions that enter as
  challengers (`arena/challenger/`).
- Add an LLM, a paid data source or an exchange credential to the decision or scoring path.
- Commit secrets. Configuration comes from environment variables; the Helm chart projects them from an
  external secret store.

## Reporting problems

Open an issue with the command you ran, the version (`pyproject.toml`), and, for judge/gate
questions, the `trials` row (`metrics` and `verdict`). For data adapter issues include the endpoint
and the raw response shape; every adapter is tested against a mock HTTP transport in `tests/data/`.

## Where the code lives

The public repository is <https://github.com/ErwanBAILLON/ArenaCrypto> (issues and
pull requests go there). The author also deploys from a private Gitea mirror whose
CI builds the container image; that is an implementation detail of one
deployment, not a requirement to run Arena.

## Secrets policy

Nothing that can be used to act on a market or reach a private system is ever
committed: no exchange keys (Arena needs none), no bot tokens, no database URLs
with passwords, no chat ids. Configuration that must be private comes from the
environment (`DATABASE_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`). Before
opening a pull request, run a secret scan on your diff (for example
`gitleaks detect --no-git` or `git diff | grep -iE "token|password|secret"`).

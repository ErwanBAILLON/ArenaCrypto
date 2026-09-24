# Results: the cross-sectional panel, and what the cost model did to it

2026-09-24. Outcome of the design in `2026-09-23-xs-panel-design.md`. Read the
pre-registered predictions there before this, because one of them was right and
it was not the interesting one.

## The setup that produced these numbers

- **Universe**: point-in-time top-50 USDT perpetuals by trailing 30-day dollar
  volume, rebuilt weekly from Binance's public archive. 682 candidates probed,
  670 with usable history, 151 weekly dates. **402 distinct symbols** passed
  through those 50 places; weekly churn is **9.7 %**; only **18 of the first
  week's members** were still in the universe at the end. Delisted symbols are
  included, which is the entire point.
- **Data**: 6 677 544 hourly bars and 1 343 632 funding stamps, 2023-11 to
  2026-09.
- **Labels**: triple barriers on the excess return over the universe, net of
  costs, upper barrier a ROI ladder (6 % → 3 % at 24 h → 1.5 % at 72 h → exit at
  168 h), stop at 8 %. 4 898 labelled symbol-weeks, 60.9 % of them winners.
- **Split**: train 2023-11-06 → 2025-09-22 (99 weeks), then a **7-day embargo**
  because a label stamped in the last training week resolves inside the test
  period, then test 2025-09-29 → 2026-09-21 — a full year neither arm saw.
- **Selection**: purged combinatorial cross-validation inside the training set,
  15 splits, 2 % embargo, scored on the long-short spread. Leakage report read
  zero overlap on every configuration.

## What training said

| width | best λ | CV long-short spread | PBO |
|---|---|---|---|
| P = 500 | 10 | +0.00320 | 0.00 |
| P = 2 000 | 10 | +0.00444 | 0.00 |
| P = 8 000 | 10 | +0.00665 | 0.00 |

The cross-validated spread rises monotonically with width. That is the virtue-
of-complexity signature exactly as Kelly, Malamud and Zhou describe it, obtained
under purging and embargo, with zero measured leakage. On this evidence the
complex model was the better one and P = 8 000 was chosen.

## What the held-out year said

Out of sample, 2025-09-29 → 2026-09-21, impact model at k = 1.0 and $1M capacity:

| arm | return | Sharpe ± se | max DD | turnover | PSR |
|---|---|---|---|---|---|
| **xs_sparse** (unfitted rank composite) | **+4.0 %** | **1.72 ± 1.11** | 1.3 % | 30 | 0.94 |
| xs_complex (8 000 random features) | +0.5 % | 0.23 ± 1.07 | 3.2 % | 32 | 0.59 |
| null_random_0 | −85.6 % | −3.50 ± 1.20 | 88.3 % | 158 | 0.00 |
| null_random_1 | −95.9 % | −5.72 ± 1.18 | 96.2 % | 159 | 0.00 |

**The control won, and it was not close.** Pre-registered row three: the panel
is too small for `P ≫ T` at this breadth. 4 898 labelled observations against
8 000 parameters is the regime the paper describes, but the paper's `T` is
decades of a liquid, well-behaved series, and this one is two years of crypto.

The more useful reading is Nagel's. The complexity gain was visible in
cross-validation — monotone in width, PBO zero, no leakage — and did not
survive to a period genuinely outside the selection. Everything we built to
catch that failure said the model was fine. **Only the held-out year caught it.**

A note against myself on the PBO: measured over a *shrinkage path*, it is a
weak test. Nine λ values produce nine nearly identical models, so the in-sample
winner is almost bound to rank well out of sample. PBO over a real parameter
search, where the configurations differ, is a much stronger statement than the
0.00 reported here. That number should not have reassured me as much as it did.

## The table that matters most

The same held-out year, under different cost assumptions:

| cost assumption | xs_sparse | xs_complex |
|---|---|---|
| flat 7 bps (**what the arena uses today**) | +7.1 % (Sharpe **3.03**) | +3.7 % (1.53) |
| impact k = 0.5, $1M | +5.3 % (2.29) | +1.9 % (0.77) |
| impact k = 1.0, $1M (base) | +4.0 % (1.72) | +0.5 % (0.23) |
| impact k = 2.0, $1M | +1.3 % (0.58) | −2.1 % (−0.85) |
| impact k = 1.0, **$5M** | +0.7 % (0.31) | −2.7 % (−1.10) |

Under the arena's existing flat fee model this looks like a Sharpe-3 strategy
and a Sharpe-1.5 one. Under an honest per-symbol impact model at a plausible
deployment size, one of them is a Sharpe-1.7 strategy and the other is nothing.
At five million dollars, neither is anything.

So the capacity limit is roughly **one million dollars**, and that is the real
answer to "a model that beats everyone": it beats everyone at small size and
dies somewhere between one and five million. Without the cost model this round
would have reported Sharpe 3.03 and been wrong in the most expensive direction.

Note also what the nulls say. Losing 86–96 % is not a statement about luck, it
is a statement about fees: `null_random` rebalances every bar and runs five
times the turnover. With realistic costs the "beat a coin flip" test collapses
into a "trade less than a coin flip" test, which is far easier to pass. A
turnover-matched null is the honest comparison and is reported below.

## Where the result actually comes from

**The held-out year was a bear market**: the equal-weight universe returned
−28.4 % with a 54.7 % drawdown, BTCUSDT −29.9 %. A neutral book returning +4.0 %
through that is not beta, but it does raise two suspicions worth settling.

**Is it shorting coins that died?** No. Splitting the legs:

| leg | return | Sharpe | max DD | turnover |
|---|---|---|---|---|
| both | +4.0 % | 1.72 | 1.3 % | 30 |
| long only | +3.9 % | 1.91 | 2.2 % | 13 |
| short only | +0.8 % | 0.33 | 2.2 % | 17 |

The long leg carries essentially all of it. The short leg made +0.8 % in a
market down 28 %, which is *less* than a naive short of the same gross would
have earned — the short selection has negative alpha. Whatever this is, it is
not delisting harvesting, which is good news for tradability: the least
tradable version of the result is the one that is not there.

**Is it one signal wearing five hats?** Yes. Rank correlation between each
signal and the next week's excess return, per rebalance date, signed as the
rule uses it, over 147 dates:

| signal | train IC | test IC | test t |
|---|---|---|---|
| `rank_ret_30d_skip_7d` (momentum) | −0.004 | −0.042 | −1.38 |
| `rank_ret_7d` (one-week reversal) | −0.012 | −0.054 | −2.33 |
| `rank_funding_crowding_z` (crowding) | −0.008 | −0.072 | −2.53 |
| **`rank_vol_30d` (low volatility)** | **+0.142** | **+0.145** | **+5.20** |
| `rank_donchian_position` | +0.025 | +0.049 | +2.34 |

One signal does the work. **Low realised volatility predicts higher excess
return in the cross-section of the fifty most liquid crypto perpetuals at a
weekly horizon, with IC 0.145 and t = 5.2** — the crypto analogue of the
low-volatility anomaly. Three of the other four are actively *wrong* out of
sample, two of them significantly. The rule works despite them, not because of
them, and the equal-weight composite's own test IC is only +0.005 because
averaging one good signal with three bad ones destroys most of it.

The important part is the first column. Low volatility scores +0.142 in a
training era that contained the 2024 bull run and +0.145 in a bear year. It is
not a regime artefact, which is a far stronger statement than the backtest
alone supports.

**What must not happen next.** The obvious move is to drop the three signals
with negative test IC and report the improved number. That would be fitting to
the held-out set, and the improved number would mean nothing. The correct move
is to pre-register "a two-signal rule on low volatility and Donchian position
beats the five-signal one" and test it on data that does not exist yet. Written
down here so it is on the record before the temptation.

## Standing caveats

- **PSR 0.94 against a zero benchmark**, below the 0.95 the promotion rule
  requires. By the arena's own standard this is not yet promotable, on a full
  year of held-out data. That is the system working, not a technicality.
- `k` is an assumption, not a calibration. The arena has no order-book data.
  The sweep is the honest substitute and it shows the result is fragile to it.
- Winsorisation of the training target uses quantiles over the whole training
  set, including the rows that land in CV test folds. A small leak in the
  scaling of the target, not in its direction; worth closing.
- The result rests on one signal with `t = 5.2` over 51 test dates. That is a
  real t-statistic, but 51 weekly observations of a cross-sectional IC are not
  51 independent observations of anything: volatility regimes persist for
  months.
- One held-out year is one path. The combinatorial machinery gives a
  distribution in training but the final comparison is a single window, and
  2025-09 to 2026-09 had its own character.

## Round two: the structural levers, walked forward

The held-out year was spent, so the second round was scored the only honest way
left for an *unfitted* rule: every quarter of the history is out of sample, so
each variant is run on all ten quarters (2024-Q2 → 2026-Q3) and the
distribution is reported, not one number. Same universe, same impact model at
$1M.

| variant | median Sharpe | mean | worst | quarters > 0 | ann. return | turnover / q |
|---|---|---|---|---|---|---|
| **baseline: 5 signals, picks-hedged, weekly** | **1.52** | 1.02 | −2.01 | 7/10 | +2.4 % | 3.3 |
| biweekly | 1.29 | 0.57 | −3.76 | 6/10 | +1.3 % | 1.7 |
| portfolio-vol sizing, gross ≤ 0.6 | 0.90 | 0.63 | −2.66 | 6/10 | **+5.3 %** | — |
| index hedge | 0.66 | 0.44 | −3.06 | 7/10 | +1.0 % | 3.7 |
| monthly | 0.50 | 0.16 | −3.64 | 5/10 | +0.6 % | 0.9 |
| index + monthly | 0.23 | 0.06 | −3.53 | 6/10 | +0.3 % | — |
| BTC/ETH anchor hedge | −1.08 | −0.33 | −4.12 | 4/10 | −1.6 % | 4.3 |
| **two signals (pre-registered)** | **−0.92** | −0.36 | −3.36 | 4/10 | −0.2 % | — |

Nearly every structural hypothesis from the analysis of the single held-out
year is refuted across ten quarters:

- **Hedging with the index or with BTC/ETH is worse than hedging with picks.**
  The short leg's alpha was small, but the picks it shorted were the
  right-tailed junk whose absence from the hedge adds basis risk. Shorting BTC
  against a low-volatility long book is a bet that BTC falls with the alts; in
  quarters where it did not, it lost.
- **Slowing down does not help.** Monthly halves turnover and halves the
  Sharpe; the signal decays faster than a month. Capacity is still the binding
  constraint, but it cannot be bought with cadence.
- **The two-signal rule fails.** This is the important one. The per-signal ICs
  on the bear year said low volatility carried everything and three signals
  were negative. Across ten quarters the five-signal composite is the best rule
  and the two-signal one loses money. The "bad" signals were bad *in that
  year*. Pre-registering the hypothesis instead of shipping it is the only
  reason this is a finding and not a deployed mistake.
- **Portfolio-level sizing is the one lever that does what it claims**, and it
  trades Sharpe for money: 0.90 against 1.52, +5.3 % against +2.4 %. That is
  the correct trade for the stated objective — most money at tolerable
  drawdown — and the correct thing to say about it is that it buys return with
  gross, and gross is what the impact model charges for.

Even the winner is a factor, not a machine: three quarters in ten are negative
and the worst is a Sharpe of −2. A rule that loses a quarter in three is what a
real cross-sectional premium looks like.

## Round two: a fair trial for the complex model, and the honest null

`xs_complex` was retried the way its critics would demand: history from
2021-01 (299 weekly dates, 8 741 597 bars), retrained before **every** quarter
on everything before it, embargoed by the ladder horizon, with width,
bandwidth and shrinkage all selected inside the training set by purged CPCV.
Seventeen out-of-sample quarters, 2022-Q3 → 2026-Q3, both arms on the
baseline construction.

| arm | median Sharpe | mean | quarters > 0 | ann. return |
|---|---|---|---|---|
| xs_complex (rolling retrain, grid-searched) | −0.72 | −1.05 | 5/17 | −1.9 % |
| xs_sparse (unfitted) | 0.18 | 0.47 | 10/17 | +1.1 % |

The complex model loses money in twelve quarters of seventeen. In every single
fold its cross-validated spread was positive (+0.0075 to +0.0133) and its PBO
at or near zero. Cross-validation, purged and embargoed, with a real grid PBO
this time, approved a model that then lost money for four years running. That
is not a bug in the validation; it is what overfitting looks like when the
signal-to-noise ratio is this low. The selected bandwidth was γ = 0.005 in
sixteen folds of seventeen — the *most linear* option offered. The search kept
asking for a linear model and the random features gave it noise to fit instead.

`xs_sparse` over seventeen quarters is weaker than over the last ten (median
0.18 against 1.52): 2022–2023 were poor for it. Ten of seventeen positive is a
factor with a real but thin premium, not a machine. Written down so the last
ten quarters are not mistaken for the whole story.

**Against a turnover-matched null.** `null_neutral` — k random longs against k
random shorts, weekly, same per-name cap — over the same ten quarters as the
variant table, five seeds: median Sharpes −3.01, +1.02, −0.33, −1.90, −1.76.
The baseline's 1.52 beats every seed. Selection is doing something; the earlier
"beat null_random" comparison mostly measured fees.

## What ships, and what does not

- **Ships**: `xs_sparse` with its original five signals, picks-hedged, weekly,
  as the wide arena's founder — the configuration nothing in two rounds could
  beat. Plus a second founder with `vol_mode="portfolio", max_gross=0.6`, which
  trades Sharpe for money and is the right trade for the stated objective.
  Both enter as challengers; both must clear the gate; nothing is promotable
  under PSR 0.95 yet.
- **Does not ship**: `xs_complex`. The code stays — it is the control's
  control — but it is not a founder. Four years of rolling out-of-sample losses
  is a verdict.
- **Does not ship**: the two-signal preset, index and anchor hedging, monthly
  cadence. Each was a hypothesis this round and each was refuted.

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

## Is the edge just shorting coins that died?

A survivorship-free universe contains symbols that went to zero, and a neutral
book shorting the worst-ranked names could be harvesting delistings. That is a
real effect but the least tradable one there is: borrow, liquidity and the
exchange halting the market all bite hardest exactly there. The decomposition is
in `§ side and survival` below.

## Standing caveats

- **PSR 0.94 against a zero benchmark**, below the 0.95 the promotion rule
  requires. By the arena's own standard this is not yet promotable, on a full
  year of held-out data. That is the system working, not a technicality.
- `k` is an assumption, not a calibration. The arena has no order-book data.
  The sweep is the honest substitute and it shows the result is fragile to it.
- Winsorisation of the training target uses quantiles over the whole training
  set, including the rows that land in CV test folds. A small leak in the
  scaling of the target, not in its direction; worth closing.
- One held-out year is one path. The combinatorial machinery gives a
  distribution in training but the final comparison is a single window, and
  2025-09 to 2026-09 had its own character.

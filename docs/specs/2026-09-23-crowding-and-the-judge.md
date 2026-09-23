# Funding crowding, and what the judge was still missing

2026-09-23. Companion to `2026-09-19-arena-design.md`; that document defines the
arena, this one records a round of work on its judge and two families that test
one hypothesis. Written before any forward result exists, on purpose.

## 1. What was wrong with the judge

The arena was built on the premise that the judge is the product. Four things
in it were not living up to that.

**The gate was advisory where it mattered.** `bootstrap` inserted a refused
founder as a challenger, and `should_promote` only compared against a champion
"if champion is not None". Six of the nine families have no champion. Every one
of them would have crowned a gate-refused model around day 42 on the forward
null test alone. The judge rejected, and the arena shrugged.

Fixed by recording the gate verdict on the competitor (`gate_admitted`) and
refusing promotion without it. A refused model keeps its forward book, because
watching a refused rule lose is the cheapest evidence the arena produces; it
simply cannot be crowned. `arena judge <name>` re-runs the gate on fresh data
and flips the flag, which is the intended path back.

**Forward significance was tested against five numbers.** The entry gate draws
50 null backtests. Forward promotion took the 95th percentile of the null
models present in the arena — there were five. The threshold protecting the
most consequential decision was the noisiest one in the system. The arena now
carries 30, counts only those covering the challenger's own window, and refuses
to promote below 20 of them rather than guessing.

**Nothing accounted for overlap.** This is the subtle one and it changes every
number on the dashboard. A `HoldingCompetitor` decides once a week and holds.
Its hourly book returns are the same bet repeated 168 times. Treating them as
168 independent observations is precisely how a Sharpe ratio becomes fiction:

| series | bars | effective observations | PSR |
|---|---|---|---|
| i.i.d. hourly | 2000 | 2000 | as computed |
| one weekly bet held through | 2000 | ~250 | 0.92, not 0.98 |

`metrics.effective_n` applies a Bartlett-weighted autocorrelation discount over
a Newey-West bandwidth, and `sharpe_se`, `probabilistic_sharpe` and
`min_track_record_length` all use it. The last of these is the one worth having
on the wall: given the Sharpe observed so far, *how many more bars before the
claim is allowed?* Promotion now needs `P(true Sharpe > null 95th) >= 0.95`,
and against an incumbent, `P(true Sharpe of the paired difference > 0) >= 0.95`
— paired, because two books riding the same market are not two independent
opinions about it.

**The deflated Sharpe answers only half the overfitting question.** DSR asks
"is this Sharpe too good for the number of tries?". It does not ask "does
picking the best try generalise?". Those fail differently: a search can produce
a winner whose Sharpe is modest enough to pass the DSR and whose ranking is
pure noise out of sample. `metrics.pbo_cscv` (Bailey, Borwein, López de Prado,
Zhu 2017) measures the second directly, over the out-of-fold series of every
evaluation in a search, and the gate now requires `pbo <= 0.50`. A search with
fewer than eight usable configurations scores 1.0 — too small to have validated
anything.

**And none of it was corrected across families.** The DSR deflates by the
family's own trial count. The freqtrade failure this project exists to avoid
was not one over-tuned strategy; it was 103 strategies selected on the same
data. Nine families searched weekly is the same machine, slower. `arena audit`
applies Benjamini-Hochberg across every gate decision in an arena, recomputes
each admission's DSR against the whole arena's trial count, and prints the
number that belongs next to any leaderboard: how many admissions chance alone
predicts at the level we test at.

None of this changes a single past verdict. It changes what the arena is
allowed to say about them.

## 2. Funding crowding: one hypothesis, two families

Perpetual funding is the price of leverage. The arena harvests its **level**
(`carry`, delta-neutral). Nothing asked what its **cross-section** means.

The claim: a symbol whose funding sits far above the rest of the universe is
where the leveraged crowd is. Crowded positioning is fragile in a specific,
mechanical way — it is the side a liquidation cascade clears first — and that
fragility is not in the price, it is in the positioning.

Two readings, deliberately opposed, so that the experiment can fail informatively:

- **`funding_skew`** treats crowding as a *signal*: short the highest funding
  z-scores, buy the lowest, gross balanced. If the claim holds, the spread pays.
- **`crowded_trend`** treats crowding as a *filter*: the `trend_ts` signal minus
  every leg the crowd has already taken. If the claim holds, the same trend
  signal survives its drawdowns better.

Against `trend_ts` as the unfiltered control, the three read as one experiment:
is funding a signal, a filter, or noise? They share the judge, the fee model
and the book, so the answer is comparable whichever way it lands.

Three design choices that keep it a claim rather than a mood:

1. **Z-score, never the level.** In a bull run every perp pays. Ranking on the
   level ranks beta and calls it crowding.
2. **Market neutral by construction** in `funding_skew`. A family that can win
   by being long crypto in a bull run has demonstrated nothing the
   buy-and-hold benchmark does not already.
3. **A dispersion floor**, `1e-6` per 8h (~0.1 %/year across the cross-section).
   Below it there is no crowding a taker fee would let you trade. This project
   has already shipped a carry threshold set from memory rather than from the
   funding distribution, and the rule never entered; the floor here is
   scale-free and was the guard that caught a real bug — on a perfectly flat
   cross-section the sample standard deviation is ~1e-20, not 0, and dividing
   by it produced confident z-scores out of floating-point dust.

### Pre-registered predictions

Stated now, before any forward data exists, because no statistic substitutes
for having committed in advance:

| if | then |
|---|---|
| `funding_skew` clears the gate and beats the null forward, `crowded_trend` does not | crowding is a tradable signal in its own right; the arena should search its horizon, not its threshold |
| `crowded_trend` beats `trend_ts` forward, `funding_skew` does not | crowding is a risk filter, not an alpha; the right move is to apply the filter to `xs_momentum` too |
| both beat their controls | the effect is real and the two are capturing correlated pieces of it; expect their books to correlate above 0.5, and treat that as one discovery, not two |
| neither does | funding carries no cross-sectional information at a weekly horizon on 15 symbols. Retire both. Do not "try a different lookback" — that is the search that the deflated Sharpe exists to punish |

The honest prior: **the fourth row.** Crowding and momentum are correlated — an
extreme-funding perp is usually one that has been rising — so part of
`funding_skew` is a short-term reversal bet wearing a positioning costume, and
the gate cannot separate them. Recording that expectation is the point.

## 3. What this arena cannot currently answer

Worth writing down, because the absence of a family is otherwise indistinguishable
from a family that failed.

- **Open interest.** Binance serves 30 days of OI history. Every positioning
  hypothesis built on OI (rising OI into a stalling price, liquidation-cascade
  timing) is untestable on the arena's 2024-onward window: such a family could
  only enter forward-only, ungated, which is the door that was just closed.
  Fixing this means storing OI forward from now and waiting a year, or paying
  for history.
- **Hyperliquid vs Binance basis.** HL snapshots exist only from the day the
  arena started. Same problem, same fix.
- **News beyond sentiment.** The interesting news variable is not polarity but
  *attention* — article-volume shocks per symbol, which in retail-driven assets
  precede short-horizon reversal far more reliably than sentiment does. The
  article table also only starts when the arena did. `news` is grandfathered as
  forward-only because it cannot be backtested in principle; a second such
  family should wait until there is enough forward history to gate it.
- **Basis risk in `carry`.** Still modelled as a perfect hedge: no spot borrow,
  no custody, no liquidation mechanics, no funding slippage. Its Sharpe remains
  an upper bound, and it is the arena's only champion. That combination deserves
  more discomfort than it currently gets.

## 4. First read on real data (entry gate, reduced power)

Run 2026-09-23 on 21 months of Binance USD-M history (358 560 hourly candles,
15 symbols, 2024-01-01 → 2026-09-23), anchored walk-forward with 90-day test
folds, live fee model. **Reduced power, and it must be read as such**: 12 null
draws instead of 50, so the null threshold is noisy; the `robust_regimes`
criterion is skipped entirely, and it is the one that rejected most of the
existing founders; `n_trials = 1`, which is honest for a first look at a new
family and will not stay true.

| family | verdict | Sharpe ± se | folds > 0 | DSR | bootstrap p | max DD | return | turnover |
|---|---|---|---|---|---|---|---|---|
| `carry` (incumbent) | admitted | 4.54 ± 1.00 | 67 % | 1.000 | 0.000 | 0.5 % | +5.3 % | 27 |
| `funding_skew` | **admitted** | 1.04 ± 0.67 | 78 % | 0.939 | 0.053 | 12.5 % | +46.4 % | 103 |
| `crowded_trend` | rejected | −0.04 ± 0.67 | 44 % | 0.479 | 0.520 | 58.1 % | −31.5 % | 346 |
| `trend_ts` (control) | rejected | −0.28 ± 0.67 | 11 % | 0.340 | 0.661 | 68.3 % | −51.3 % | 333 |

Null 95th percentile over 12 draws: 0.97.

Four things worth saying out loud.

**Crowding carries information, and both families show it.** `funding_skew`
clears every criterion, and the filter reading improves `trend_ts` on *every*
single metric: Sharpe −0.28 → −0.04, drawdown 68.3 % → 58.1 %, return −51.3 %
→ −31.5 %, positive folds 11 % → 44 %. That is not noise, and it is not churn
either: turnover barely moves (333 → 346), so the improvement comes from which
legs are dropped, not from trading less. The pre-registered predictions offered
signal *or* filter; the data says both, and only one of them clears the bar.

**`crowded_trend` is rejected anyway, and should be.** A filter that turns a
disastrous rule into a merely bad one has demonstrated the filter, not a
competitor. It enters the arena as a challenger to be watched, not promoted,
which is exactly what the gate is for.

**`funding_skew` is admitted marginally, and the margin is the story.** Sharpe
1.04 with a standard error of 0.67 puts the 95 % interval at roughly
[−0.3, 2.4]; DSR 0.939 sits just above the 0.90 bar and will fall the moment
the weekly search starts recording trials against this family; bootstrap p is
0.053 against a 0.10 gate. This is the profile of something real and small, or
of something lucky — and the full gate's regime robustness, skipped here, is
precisely the test that separates those. Treat it as *admitted pending the full
judge*, not as a discovery.

**The incumbent's numbers are a warning, not a reassurance.** `carry` posts
Sharpe 4.54 and a 0.5 % drawdown for +5.3 % over 21 months: about 3 %/year,
near-riskless — *under a hedge assumed to be perfect*. No basis risk, no spot
borrow, no liquidation mechanics. The arena's only champion is the family whose
cost model is the most idealised, and its Sharpe is the one most inflated by
that idealisation. `funding_skew` earned nine times the return at a quarter of
the Sharpe, and it is the one whose costs are modelled honestly. That
comparison is worth more attention than either number alone.

**What to run before believing any of it**: `arena judge funding_skew` at full
power (50 nulls, 120 robustness windows) on the production database, and
`arena audit` afterwards to see what the admission is worth against every test
the arena has run.

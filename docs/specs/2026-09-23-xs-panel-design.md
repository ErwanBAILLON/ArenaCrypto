# A cross-sectional panel competitor, and the infrastructure that makes it believable

2026-09-23. Design for two new families (`xs_sparse`, `xs_complex`), the data
and cost work they need, and an attribution dashboard. Companion to
`2026-09-23-crowding-and-the-judge.md`.

## 1. What is being optimised, and why that choice shapes everything

The objective chosen is **the most money at a tolerable drawdown**, not the
highest Sharpe. That single decision determines the architecture, because of
Grinold's fundamental law:

```
IR ≈ IC × √breadth
```

A directional rule rebalanced weekly makes ~52 bets a year on what is really
*one* bet, because crypto beta dominates every position. A market-neutral
cross-sectional rule on N symbols makes up to N near-independent bets per
period. Breadth, not predictive cleverness, is the binding constraint — and the
arena's own numbers already show it:

| | Sharpe | turnover | `effective_n` / bars |
|---|---|---|---|
| `carry` (near-deterministic collection) | 4.54 | 27 | 18 958 / 19 440 |
| `funding_skew` (cross-sectional, neutral) | 1.04 | 103 | **19 196 / 19 440** |
| `trend_ts` (one directional bet, repeated) | −0.28 | 333 | 19 440 / 19 440 |

Market neutrality buys statistical power: `funding_skew`'s hourly book returns
are almost unautocorrelated because the P&L is driven by idiosyncratic spreads
rather than a common wave. That shortens `min_track_record_length` without
cheating, which is the only honest way to make a Sharpe-1 strategy provable in
a human timeframe.

So: **cross-sectional, market-neutral, as wide a universe as the cost model can
honestly support.**

## 2. The four ways this could be a lie

The design is organised around defeating these, in this order. Every one of
them has produced a spectacular fake backtest in this project's own history.

1. **Survivorship.** Taking today's top-50 perps by volume selects the ones
   that survived. Measured: Binance's public archive holds **864 USDT perpetual
   symbols**, of which **525 still trade**. A universe built from survivors
   discards 339 symbols, many of which went to zero. This is the single largest
   threat and it is the reason for §3.
2. **Cost fiction.** A flat 0.05 % taker + 0.02 % slippage is conservative on
   BTC and fantasy on the 60th-ranked perp. Widening the universe without
   widening the cost model manufactures alpha out of thin liquidity (§4).
3. **Label leakage.** Labels computed with hindsight-optimal exits are
   unlearnable and produce models that look perfect in sample. §5 uses
   pre-specified barriers instead.
4. **Multiple testing.** Two more families, each with a search space, on top of
   nine. `arena audit` already measures the damage; §8 states what result would
   make us stop rather than tune.

## 3. Point-in-time universe

New adapter `arena/data/binance_archive.py` over `data.binance.vision`, which
**does archive delisted symbols** (verified: `SRMUSDT`, `FTTUSDT`, `TOMOUSDT`,
`HNTUSDT`, `BTCSTUSDT` all resolve to monthly kline zips for 2022-09).

- Enumerate every symbol ever archived via the S3 listing API.
- Classify against `fapi/v1/exchangeInfo`: keep `contractType == PERPETUAL` and
  `underlyingType == COIN`. That excludes the 201 `TRADIFI_PERPETUAL` equity
  and commodity perps Binance now lists. Symbols present only in the archive
  (i.e. delisted) predate the TradFi launch and are kept.
- Download monthly 1h klines per symbol, store in `candles` as today.

New table `universe_members(ts, symbol, rank, dollar_volume, universe)`: at each
weekly rebalance date, rank every symbol **trading at that date** by trailing
30-day dollar volume and keep the top `N` above a liquidity floor. Membership
is stored, not recomputed, so a backtest and the live tick read the same
universe.

**Delisting is an exit, not a disappearance.** A symbol that leaves the universe
because it stopped trading is closed at its last observed price on its last
bar, and the book takes the loss. Silently dropping it is how a backtest earns
returns that no one could have collected.

## 4. Cost model per symbol

`arena/core/costs.py`. Replaces the flat slippage with the square-root impact
law, which is [empirically confirmed on BTC/USD futures](https://arxiv.org/pdf/2305.07559)
with an exponent near 0.5 and is [remarkably universal across asset classes](https://bouchaud.substack.com/p/the-square-root-law-of-market-impact):

```
slippage_bps(symbol) = half_spread + k · σ_daily · (notional / ADV)^δ
```

with `δ = 0.5`, `ADV` the trailing 30-day dollar volume and `σ_daily` the
realised daily volatility, both point-in-time. `k` is **an assumption, not a
calibration** — the arena has no order-book data to fit it — so it is a
parameter with a deliberately pessimistic default, and the gate is run at
several values of `k` to see where the edge dies. A strategy whose alpha exists
only at optimistic `k` is a strategy that does not exist.

A `liquidity_floor` on trailing dollar volume excludes symbols outright.

## 5. Labels: the triple barrier, with a ROI ladder, relative to the market

The target is not the next period's return. It is: **was there a trade here
that paid, after costs, better than the market?**

For each (symbol, event) the label is set by whichever barrier is touched first
on the *excess* return over an equal-weight benchmark of the universe:

- **upper barrier — a ROI ladder.** Instead of one profit target, a
  time-decaying table in the spirit of freqtrade's `minimal_roi`: take profit
  at +6 % within 12 h, +3 % within 2 days, +1.5 % within 5 days, 0 % after 10
  days. This is the "ROI table that can close" made rigorous: a ROI ladder is
  exactly a profit barrier that decays with holding time, so it drops straight
  into the triple-barrier framework and is used identically for labelling and
  for live exits.
- **lower barrier** — a volatility-scaled stop, `m × σ`, so the barrier means
  the same thing on BTC and on a 200 %-vol altcoin.
- **vertical barrier** — the ladder's final horizon.

All barriers are **net of the §4 cost model**, so the label literally encodes
tradability rather than price movement.

Two weighting schemes on top, both from López de Prado (*Advances in Financial
Machine Learning*, 2018; *Machine Learning for Asset Managers*, 2020):

- **Trend-scanning weights**: the t-statistic of the strongest trend beginning
  at each event, used as a sample weight so the model spends its capacity on
  unambiguous episodes. Recent work confirms [trend scanning is valuable when
  properly filtered, and that not every trend signal is worth trading](https://www.mql5.com/en/articles/19253).
- **Sample uniqueness**: overlapping label windows mean overlapping information;
  concurrent events are down-weighted so a single market move cannot vote
  fifty times.

Honest scope: the [empirical comparisons](https://www.mql5.com/en/articles/18864)
show triple-barrier clearly beating fixed-horizon labelling, with modest
absolute gains. This is a better-posed problem, not an alpha generator.

## 6. Validation

Combinatorial purged cross-validation with embargo, in `arena/judge/cpcv.py`,
reusing the PBO machinery already built. Purging removes training samples whose
label window overlaps the test window; the embargo drops a further margin after
each test block to kill serial leakage. Without both, event-based labels leak
by construction and every number downstream is decoration.

## 7. The two models

Identical universe, labels, barriers, costs, portfolio construction and judge.
They differ **only** in the predictor, so the comparison measures complexity and
nothing else.

**Features** (~100 raw, all point-in-time, all cross-sectionally ranked as well
as raw — the rank is often the only stationary version):

- *price / trend*: returns over 1, 3, 7, 14, 30, 60, 90 d; 30d-skip-7d
  momentum; EMA ratios 10/20/50/100/200; distance to rolling high and low;
  drawdown from rolling max; MACD; RSI; Bollinger %b; Donchian position; ATR
  over price.
- *volatility*: realised vol 7/30/90 d; Parkinson and Garman-Klass estimators;
  downside semi-deviation; vol term structure (7d ÷ 30d); vol of vol; beta to
  BTC; idiosyncratic and residual vol.
- *liquidity*: dollar volume 1/7/30 d; volume z-score and trend; Amihud
  illiquidity; turnover.
- *funding and positioning*: mean funding over 1/7/14/30 d; cross-sectional
  funding z-score; funding slope and volatility; cumulative funding;
  Hyperliquid−Binance funding spread; open-interest change and OI/volume.
- *cross-sectional state*: rank of every feature above; dispersion of the
  cross-section; market breadth; aggregate funding; BTC regime label.
- *calendar and news*: days since listing; day of week; macro-event proximity;
  news sentiment 24 h / 7 d; article-count z-score (attention).

`xs_sparse` — the Nagel control. Five to eight of these, standardised and
combined by equal-risk rank averaging with no fitting. Cheap, robust, and it
consumes almost no trials, so the deflated Sharpe favours it.

`xs_complex` — Random Fourier Features over the raw block, `P` from 1 000 to
20 000, ridge over a wide shrinkage grid. This is the Kelly-Malamud-Zhou
machinery ([*Journal of Finance* 79(1), 2024](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13298)),
applied where this arena actually has observations: the **panel**. KMZ time the
market, where `T` is a few hundred months of one series; here `T` is
50 symbols × ~140 weeks. The panel supplies the observations and the
cross-sectional demeaning makes the residuals usable.

What complexity is *for* here is interactions a linear composite cannot state:
reversal pays when funding is extreme, momentum pays when dispersion is wide,
both die when volatility regime-shifts. Not "more parameters" as a virtue.

**The standing objection**: [Nagel (2025)](https://voices.uchicago.edu/stefannagel/files/2025/07/Complexity_2.pdf)
shows the RFF gain largely reduces to volatility-timed momentum, and KMZ's
results exclude transaction costs. `xs_sparse` exists precisely so that this
arena answers the question on crypto data instead of taking a side.

**Portfolio construction**, shared: rank predicted excess outcomes, go
dollar-neutral long-short on the extremes, size to a target volatility, cap per
symbol, band the weights to avoid churn, and exit on the ROI ladder or the stop
— the same barriers used for training, so live behaviour and labels agree.

## 8. Attribution dashboard

A parallel view answering "where does it win, where does it lose". New table
`predictions(competitor_id, ts, symbol, prediction, confidence, action, barrier_hit,
holding_bars, pnl_net, features_digest)` plus pages that slice it:

- **journal**: what it predicted, how confident, what it did, which barrier was
  hit, the net P&L.
- **P&L decomposition** by symbol, regime, funding decile, volatility decile,
  holding duration, side, conviction decile. A model that earns everything on
  three symbols is not a model.
- **reliability diagram**: when it says 70 %, does it happen 70 % of the time?
  This is the diagnostic that separates a model that knows what it does not
  know from one that bluffs, and it is what makes conviction usable for sizing.
- **worst decile**, isolated, with what those trades have in common.

It serves the existing families too, which is most of its value on day one.

## 9. Pre-registered predictions

| if | then |
|---|---|
| `xs_complex` beats `xs_sparse` out of sample, after costs | complexity earns its keep on crypto panels; Nagel's critique does not carry over from equity timing |
| both work, indistinguishably | Nagel is right here: ship `xs_sparse`, retire the complex one, and say so |
| `xs_sparse` works, `xs_complex` does not | the panel is too small for `P ≫ T` at this breadth; revisit only with a wider universe |
| neither works after honest costs | the cross-section of large crypto perps is efficient at a weekly horizon. Stop. Do not widen the search space |

The honest prior is **row two**. Nagel's mechanism is general, crypto data is
noisier than equities, and the cost model is being made deliberately harsher
than the one under which `funding_skew` was admitted.

## 10. What would make this a failure worth publishing anyway

If the point-in-time universe alone moves the existing families' backtest
results materially — that is, if `carry` and `funding_skew` look meaningfully
worse once survivorship is removed and costs scale with illiquidity — then the
infrastructure was the finding and the models are a footnote. That result would
be more valuable than a new competitor, and it is a realistic outcome.

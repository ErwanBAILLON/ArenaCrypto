-- Positioning data Binance only serves for the last 30 days: long/short account
-- ratios, top-trader position ratios, taker buy/sell volume. The alpha the
-- crypto ML literature finds lives in positioning, and nobody can backtest it
-- because nobody stored it. Storing it from today is the only way to have a
-- history a year from now.
CREATE TABLE IF NOT EXISTS positioning (
  exchange        text        NOT NULL,
  symbol          text        NOT NULL,
  ts              timestamptz NOT NULL,
  global_ls_ratio double precision,   -- all accounts, long/short by count
  top_ls_ratio    double precision,   -- top traders, long/short by position
  taker_bs_ratio  double precision,   -- taker buy volume / sell volume
  fetched_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (exchange, symbol, ts)
);

-- Forced liquidations, one row per filled liquidation order, from the venue
-- that serves them over REST (OKX; see arena/data/okx.py for why not Binance).
-- Stored raw: the hourly aggregates a signal wants (count, notional, long/short
-- imbalance) are one GROUP BY away and the grain decides what can be asked later.
CREATE TABLE IF NOT EXISTS liquidations (
  exchange    text        NOT NULL,
  symbol      text        NOT NULL,   -- the arena's symbol (BTC, or BTCUSDT on the wide arena)
  ts          timestamptz NOT NULL,
  side        text        NOT NULL,   -- the liquidated position: long | short
  price       double precision NOT NULL,
  qty         double precision NOT NULL,
  notional    double precision NOT NULL,
  fetched_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (exchange, symbol, ts, side, price, qty)
);
CREATE INDEX IF NOT EXISTS liquidations_symbol_ts ON liquidations (symbol, ts);

-- Point-in-time universe membership. The arena's universe was a fixed list of
-- fifteen symbols chosen today; backtesting that list over two years selects the
-- survivors. Binance's archive holds 864 USDT perpetual symbols and 525 still
-- trade, so the fixed list quietly discards ~40 % of the cross-section, and the
-- discarded part is the part that went to zero.
--
-- Membership is computed once per rebalance date and stored, never recomputed,
-- so a backtest and the live tick read exactly the same universe.
CREATE TABLE IF NOT EXISTS universe_members (
  universe   text        NOT NULL,
  ts         timestamptz NOT NULL,   -- rebalance date the membership applies from
  symbol     text        NOT NULL,
  rank       integer     NOT NULL,   -- 0 = most liquid
  adv_usd    double precision NOT NULL,
  daily_vol  double precision NOT NULL,
  PRIMARY KEY (universe, ts, symbol)
);
CREATE INDEX IF NOT EXISTS universe_members_ts ON universe_members (universe, ts DESC);

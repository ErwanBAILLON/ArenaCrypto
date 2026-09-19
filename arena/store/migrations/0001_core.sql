-- Arena core schema. All timestamps are UTC (timestamptz).

CREATE TABLE IF NOT EXISTS candles (
  exchange   text NOT NULL,
  symbol     text NOT NULL,
  tf         text NOT NULL,
  ts         timestamptz NOT NULL,
  open       double precision NOT NULL,
  high       double precision NOT NULL,
  low        double precision NOT NULL,
  close      double precision NOT NULL,
  volume     double precision NOT NULL,
  fetched_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (exchange, symbol, tf, ts)
);
CREATE INDEX IF NOT EXISTS candles_symbol_ts ON candles (symbol, ts);

CREATE TABLE IF NOT EXISTS funding (
  exchange   text NOT NULL,
  symbol     text NOT NULL,
  ts         timestamptz NOT NULL,
  rate       double precision NOT NULL,
  fetched_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (exchange, symbol, ts)
);

CREATE TABLE IF NOT EXISTS open_interest (
  exchange   text NOT NULL,
  symbol     text NOT NULL,
  ts         timestamptz NOT NULL,
  oi         double precision NOT NULL,
  fetched_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (exchange, symbol, ts)
);

CREATE TABLE IF NOT EXISTS hl_snapshots (
  ts         timestamptz NOT NULL,
  coin       text NOT NULL,
  funding    double precision,
  oi         double precision,
  mark       double precision,
  fetched_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (ts, coin)
);

CREATE TABLE IF NOT EXISTS articles (
  id           bigserial PRIMARY KEY,
  source       text NOT NULL,
  url          text NOT NULL,
  title        text NOT NULL,
  summary      text NOT NULL DEFAULT '',
  published_at timestamptz NOT NULL,
  fetched_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE (source, url, title)
);
CREATE INDEX IF NOT EXISTS articles_published ON articles (published_at);

CREATE TABLE IF NOT EXISTS article_scores (
  article_id     bigint NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
  asset          text NOT NULL,
  sentiment      double precision NOT NULL,
  event_type     text NOT NULL,
  intensity      double precision NOT NULL,
  scorer_version integer NOT NULL,
  PRIMARY KEY (article_id, asset, scorer_version)
);
CREATE INDEX IF NOT EXISTS article_scores_asset ON article_scores (asset, scorer_version);

CREATE TABLE IF NOT EXISTS macro_events (
  id         bigserial PRIMARY KEY,
  ts         timestamptz NOT NULL,
  currency   text NOT NULL,
  title      text NOT NULL,
  impact     text NOT NULL,
  fetched_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (ts, currency, title)
);

CREATE TABLE IF NOT EXISTS competitors (
  id         bigserial PRIMARY KEY,
  name       text NOT NULL UNIQUE,
  family     text NOT NULL,
  version    integer NOT NULL,
  parent_id  bigint REFERENCES competitors(id),
  params     jsonb NOT NULL DEFAULT '{}'::jsonb,
  role       text NOT NULL CHECK (role IN ('null','benchmark','competitor')),
  status     text NOT NULL CHECK (status IN ('candidate','challenger','champion','retired')),
  rationale  text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS trials (
  id            bigserial PRIMARY KEY,
  competitor_id bigint REFERENCES competitors(id),
  family        text NOT NULL,
  kind          text NOT NULL CHECK (kind IN ('backtest','walkforward','optimize','retrain')),
  started_at    timestamptz NOT NULL DEFAULT now(),
  finished_at   timestamptz,
  params        jsonb NOT NULL DEFAULT '{}'::jsonb,
  metrics       jsonb NOT NULL DEFAULT '{}'::jsonb,
  verdict       text,
  notes         text NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS trials_family ON trials (family);

CREATE TABLE IF NOT EXISTS targets (
  competitor_id bigint NOT NULL REFERENCES competitors(id),
  ts            timestamptz NOT NULL,
  symbol        text NOT NULL,
  weight        double precision NOT NULL,
  conviction    double precision NOT NULL,
  kind          text NOT NULL DEFAULT 'perp',
  reason        jsonb NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (competitor_id, ts, symbol)
);

CREATE TABLE IF NOT EXISTS books (
  competitor_id bigint NOT NULL REFERENCES competitors(id),
  ts            timestamptz NOT NULL,
  nav           double precision NOT NULL,
  ret           double precision NOT NULL,
  gross         double precision NOT NULL,
  turnover      double precision NOT NULL,
  fees          double precision NOT NULL,
  funding_pnl   double precision NOT NULL,
  PRIMARY KEY (competitor_id, ts)
);

CREATE TABLE IF NOT EXISTS allocations (
  ts            timestamptz NOT NULL,
  competitor_id bigint NOT NULL REFERENCES competitors(id),
  weight        double precision NOT NULL,
  PRIMARY KEY (ts, competitor_id)
);

CREATE TABLE IF NOT EXISTS alerts (
  id            bigserial PRIMARY KEY,
  ts            timestamptz NOT NULL DEFAULT now(),
  kind          text NOT NULL,
  competitor_id bigint REFERENCES competitors(id),
  symbol        text,
  payload       jsonb NOT NULL DEFAULT '{}'::jsonb,
  sent_at       timestamptz
);
CREATE INDEX IF NOT EXISTS alerts_unsent ON alerts (sent_at) WHERE sent_at IS NULL;

CREATE TABLE IF NOT EXISTS models (
  id            bigserial PRIMARY KEY,
  competitor_id bigint NOT NULL REFERENCES competitors(id),
  trained_at    timestamptz NOT NULL DEFAULT now(),
  artifact      bytea NOT NULL,
  metrics       jsonb NOT NULL DEFAULT '{}'::jsonb
);

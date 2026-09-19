-- One row per hourly tick: what happened, so humans can see the system act.
CREATE TABLE IF NOT EXISTS tick_runs (
  id          bigserial PRIMARY KEY,
  bar_ts      timestamptz,
  started_at  timestamptz NOT NULL,
  finished_at timestamptz NOT NULL,
  booked      integer NOT NULL DEFAULT 0,
  skipped     integer NOT NULL DEFAULT 0,
  failed      text[] NOT NULL DEFAULT '{}',
  changes     integer NOT NULL DEFAULT 0,   -- competitors whose targets changed vs the previous bar
  alerts      integer NOT NULL DEFAULT 0,
  ingested    jsonb NOT NULL DEFAULT '{}'::jsonb,
  ok          boolean NOT NULL DEFAULT true
);
CREATE INDEX IF NOT EXISTS tick_runs_started ON tick_runs (started_at DESC);

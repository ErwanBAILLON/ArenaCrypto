-- Per-competitor serialisable state (hysteresis) carried between hourly ticks.
CREATE TABLE IF NOT EXISTS competitor_state (
  competitor_id bigint PRIMARY KEY REFERENCES competitors(id),
  ts            timestamptz NOT NULL,
  state         jsonb NOT NULL DEFAULT '{}'::jsonb
);

-- Intra-hour exits executed by the live watcher on streamed prices, between two ticks.
-- The tick reads them to account the closed leg from the previous close to the fill.
CREATE TABLE IF NOT EXISTS fills (
  id            bigserial PRIMARY KEY,
  competitor_id bigint NOT NULL REFERENCES competitors(id),
  ts            timestamptz NOT NULL,
  symbol        text NOT NULL,
  kind          text NOT NULL DEFAULT 'perp',
  weight_before double precision NOT NULL,
  weight_after  double precision NOT NULL DEFAULT 0,
  price         double precision NOT NULL,
  reason        text NOT NULL,                 -- stop | roi
  excess        double precision,              -- the rule's measure at the fill (excess or raw return)
  booked_ts     timestamptz                    -- the bar whose book row accounted this fill (NULL until then)
);
CREATE INDEX IF NOT EXISTS fills_competitor_ts ON fills (competitor_id, ts);

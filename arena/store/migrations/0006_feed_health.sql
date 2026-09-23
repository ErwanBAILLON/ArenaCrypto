-- Freshness used to be inferred from max(fetched_at) of each data table. That
-- works only for tables that grow: the macro calendar is upserted with ON
-- CONFLICT DO NOTHING, so a week whose events are all already known writes
-- nothing and the dashboard reported "very late" while ingestion ran fine every
-- thirty minutes. A health panel that cries wolf is a health panel nobody reads.
-- Ingestion now records the fetch itself, whether or not it wrote a row.
CREATE TABLE IF NOT EXISTS feed_health (
  source        text        NOT NULL,
  universe      text        NOT NULL DEFAULT 'crypto',
  last_fetch_at timestamptz NOT NULL DEFAULT now(),
  last_ok_at    timestamptz,
  last_data_ts  timestamptz,           -- newest datum the source carries, not when we asked
  rows_written  integer     NOT NULL DEFAULT 0,
  ok            boolean     NOT NULL DEFAULT true,
  detail        text        NOT NULL DEFAULT '',
  PRIMARY KEY (source, universe)
);

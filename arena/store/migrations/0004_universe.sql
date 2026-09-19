-- Several arenas (crypto 1h perps, classic 1d markets) share the schema; a competitor belongs to one.
ALTER TABLE competitors ADD COLUMN IF NOT EXISTS universe text NOT NULL DEFAULT 'crypto';
CREATE INDEX IF NOT EXISTS competitors_universe ON competitors (universe, status);

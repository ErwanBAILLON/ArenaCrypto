-- The quoted spread at the moment of a live fill: the first real-world cost measurement the arena has.
-- Everything else about costs is a model; this is what the book saw. Calibration data, not an input.
ALTER TABLE fills ADD COLUMN IF NOT EXISTS bid double precision;
ALTER TABLE fills ADD COLUMN IF NOT EXISTS ask double precision;
ALTER TABLE fills ADD COLUMN IF NOT EXISTS spread_bps double precision;

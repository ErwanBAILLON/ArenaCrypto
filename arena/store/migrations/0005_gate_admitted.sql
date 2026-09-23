-- A competitor refused by the entry gate used to enter the arena as a challenger
-- anyway, and a family with no champion promoted its challenger on the forward
-- null test alone. Recording the gate verdict lets promotion refuse a model the
-- judge never admitted.
ALTER TABLE competitors ADD COLUMN IF NOT EXISTS gate_admitted boolean NOT NULL DEFAULT false;

-- Backfill from the rationale written at insert time ("gate admitted" / "gate rejected: ...").
-- Null models and benchmarks never face the gate and are not promotable, so they stay false.
UPDATE competitors
   SET gate_admitted = true
 WHERE role = 'competitor'
   AND rationale NOT ILIKE '%reject%'
   AND gate_admitted = false;

-- Trials belong to one arena: a daily-bar search must not inflate the trial
-- counter that deflates an hourly-bar Sharpe.
ALTER TABLE trials ADD COLUMN IF NOT EXISTS universe text NOT NULL DEFAULT 'crypto';
CREATE INDEX IF NOT EXISTS trials_family_universe ON trials (family, universe);

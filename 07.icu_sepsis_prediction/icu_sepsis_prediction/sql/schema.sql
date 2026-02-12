CREATE SCHEMA IF NOT EXISTS icu;

CREATE TABLE IF NOT EXISTS icu.windows (
  patient_id TEXT NOT NULL,
  t_hour INT NOT NULL,
  y INT NOT NULL,
  features JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (patient_id, t_hour)
);

CREATE INDEX IF NOT EXISTS idx_icu_windows_y ON icu.windows(y);

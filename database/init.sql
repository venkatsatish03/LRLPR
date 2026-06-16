CREATE TABLE IF NOT EXISTS recognition_jobs (
  id UUID PRIMARY KEY,
  image_url TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS detection_results (
  id UUID PRIMARY KEY,
  job_id UUID NOT NULL REFERENCES recognition_jobs(id),
  plate_text TEXT,
  detection_confidence NUMERIC(5, 4),
  ocr_confidence NUMERIC(5, 4),
  final_confidence NUMERIC(5, 4),
  plate_image_url TEXT,
  bounding_box JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

export type DetectionResult = {
  plate: string;
  confidence: number;
  candidates: PlateCandidate[];
};

export type PlateCandidate = {
  plate: string;
  confidence: number;
};

export type BoundingBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type PlateDetection = {
  confidence: number;
  final_confidence: number;
  coordinates: BoundingBox;
  cropped_plate_path: string;
  cropped_plate_url: string;
  text: string;
  ocr_confidence: number;
  candidates: PlateCandidate[];
};

export type PlateDetectionResponse = {
  filename: string;
  path: string;
  detector: string;
  model_configured: boolean;
  annotated_image_path: string | null;
  annotated_image_url: string | null;
  plates_detected: number;
  detections: PlateDetection[];
};

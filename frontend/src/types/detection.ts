export type DetectionResult = {
  plate: string;
  confidence: number;
  candidates: PlateCandidate[];
};

export type PlateCandidate = {
  plate: string;
  confidence: number;
};

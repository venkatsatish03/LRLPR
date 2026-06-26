from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class PlateCandidate(BaseModel):
    plate: str
    confidence: float


class PlateDetection(BaseModel):
    confidence: float
    final_confidence: float
    coordinates: BoundingBox
    cropped_plate_path: str
    cropped_plate_url: str
    text: str
    ocr_confidence: float
    candidates: list[PlateCandidate] = []


class PlateDetectionResponse(BaseModel):
    filename: str
    path: str
    detector: str
    model_configured: bool
    annotated_image_path: str | None
    annotated_image_url: str | None
    plates_detected: int
    detections: list[PlateDetection]


class SimpleDetectionResponse(BaseModel):
    plate: str
    confidence: float
    candidates: list[PlateCandidate] = []


class OCRComparisonItem(BaseModel):
    plate: str
    confidence: float
    candidates: list[PlateCandidate] = []
    detector: str
    model_configured: bool
    annotated_image_path: str | None
    cropped_plate_path: str | None


class EnhancementInfo(BaseModel):
    configured: bool
    method: str
    enhanced_image_path: str
    enhanced_image_url: str


class OCRComparisonResponse(BaseModel):
    filename: str
    original_image_path: str
    enhancement: EnhancementInfo
    before_enhancement: OCRComparisonItem
    after_enhancement: OCRComparisonItem
    improved: bool

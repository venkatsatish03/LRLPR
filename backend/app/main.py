from fastapi import FastAPI, File, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.upload import router as upload_router
from app.core.config import settings
from app.schemas.detection import OCRComparisonResponse, PlateDetectionResponse, SimpleDetectionResponse
from app.services.image_enhancement import ImageEnhancementService
from app.services.image_storage import ImageStorageService
from app.services.plate_detector import PlateDetectorService

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

app.include_router(upload_router, prefix=settings.API_PREFIX)
storage_service = ImageStorageService()
enhancement_service = ImageEnhancementService()
plate_detector_service = PlateDetectorService()


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>License Plate Recognition Upload</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            max-width: 720px;
            margin: 56px auto;
            padding: 0 20px;
            line-height: 1.5;
          }
          form {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 24px;
          }
          input {
            margin-top: 8px;
          }
          button {
            display: block;
            margin-top: 18px;
            padding: 10px 16px;
            cursor: pointer;
          }
          code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
          }
        </style>
      </head>
      <body>
        <h1>License Plate Recognition</h1>
        <p>Upload an image. The backend will detect the plate, crop it, run OCR, and return JSON with the plate text and confidence.</p>
        <form action="/detect" method="post" enctype="multipart/form-data">
          <label for="file">Image file</label><br>
          <input id="file" name="file" type="file" accept="image/*" required>
          <button type="submit">Upload Image</button>
        </form>
        <p>API docs: <a href="/docs">/docs</a></p>
      </body>
    </html>
    """


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    saved_image = await storage_service.save_upload(file)
    return {
        "filename": saved_image["filename"],
        "path": saved_image["image_path"],
    }


@app.post("/detect", response_model=SimpleDetectionResponse)
async def detect_license_plate(file: UploadFile = File(...)) -> SimpleDetectionResponse:
    detection_response = await detect_license_plate_details(file)
    best_detection = _select_best_detection(detection_response.detections)

    return SimpleDetectionResponse(
        plate=best_detection["text"],
        confidence=best_detection["confidence"],
        candidates=best_detection["candidates"],
    )


@app.post("/detect/details", response_model=PlateDetectionResponse)
async def detect_license_plate_details(file: UploadFile = File(...)) -> PlateDetectionResponse:
    saved_image = await storage_service.save_upload(file)
    detection_result = plate_detector_service.detect(saved_image["image_path"])

    return PlateDetectionResponse(
        filename=saved_image["filename"],
        path=saved_image["image_path"],
        **detection_result,
    )


@app.post("/detect/compare", response_model=OCRComparisonResponse)
async def compare_detection_before_after_enhancement(file: UploadFile = File(...)) -> OCRComparisonResponse:
    saved_image = await storage_service.save_upload(file)

    original_result = PlateDetectionResponse(
        filename=saved_image["filename"],
        path=saved_image["image_path"],
        **plate_detector_service.detect(saved_image["image_path"]),
    )

    enhancement = enhancement_service.enhance(saved_image["image_path"])
    enhanced_result = PlateDetectionResponse(
        filename=saved_image["filename"],
        path=enhancement["enhanced_image_path"],
        **plate_detector_service.detect(enhancement["enhanced_image_path"]),
    )

    before = _summarize_detection(original_result)
    after = _summarize_detection(enhanced_result)

    return OCRComparisonResponse(
        filename=saved_image["filename"],
        original_image_path=saved_image["image_path"],
        enhancement=enhancement,
        before_enhancement=before,
        after_enhancement=after,
        improved=bool(after["plate"]) and after["plate"] == before["plate"] and after["confidence"] > before["confidence"],
    )


def _select_best_detection(detections):
    if not detections:
        return {"text": "", "confidence": 0.0, "candidates": []}

    best_detection = max(
        detections,
        key=lambda detection: (
            len(detection.text),
            detection.ocr_confidence,
            detection.confidence,
        ),
    )

    final_confidence = (best_detection.confidence * 0.4) + (best_detection.ocr_confidence * 0.6)
    return {
        "text": best_detection.text,
        "confidence": round(final_confidence, 4),
        "candidates": best_detection.candidates if best_detection.ocr_confidence < 0.8 else [],
    }


def _summarize_detection(response: PlateDetectionResponse):
    best = _select_best_detection(response.detections)
    if not response.detections:
        return {
            "plate": best["text"],
            "confidence": best["confidence"],
            "candidates": best["candidates"],
            "detector": response.detector,
            "model_configured": response.model_configured,
            "annotated_image_path": response.annotated_image_path,
            "cropped_plate_path": None,
        }

    best_detection = max(
        response.detections,
        key=lambda detection: (
            len(detection.text),
            detection.ocr_confidence,
            detection.confidence,
        ),
    )
    return {
        "plate": best["text"],
        "confidence": best["confidence"],
        "candidates": best["candidates"],
        "detector": response.detector,
        "model_configured": response.model_configured,
        "annotated_image_path": response.annotated_image_path,
        "cropped_plate_path": best_detection.cropped_plate_path,
    }


@app.get("/model/status")
def model_status():
    return plate_detector_service.model_status()


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=status.HTTP_204_NO_CONTENT)

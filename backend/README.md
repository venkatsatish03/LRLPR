# FastAPI Backend

Backend API for the License Plate Recognition System.

## Requirements

- Python 3.11

## Setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Or use the included run script:

```powershell
.\run.ps1
```

Open the upload page in your browser with:

```text
http://localhost:8000
```

Avoid using `http://0.0.0.0:8000` in the browser. If you run Uvicorn with `--host 0.0.0.0`, that address only means "listen on all network interfaces"; it is not the URL you should open on Windows.

## Endpoints

- `GET /`
- `GET /health`
- `GET /docs`
- `GET /api/v1/images/upload`
- `POST /api/v1/images/upload`
- `POST /upload`
- `POST /detect`
- `POST /detect/details`
- `POST /detect/compare`
- `GET /model/status`

Open the upload form in your browser:

```text
http://localhost:8000
```

## YOLOv8 Plate Detection

Place your trained YOLOv8 license plate weights here:

```text
backend/models/license_plate_detector.pt
```

Or set a custom path in `.env`:

```text
YOLO_MODEL_PATH=models/license_plate_detector.pt
YOLO_CONFIDENCE_THRESHOLD=0.25
```

The default YOLO object-detection weights do not reliably detect license plates. Use a model trained for the `license_plate` class.

If no YOLO weights are installed, `/detect` falls back to a basic OpenCV plate-candidate detector so the application can still return coordinates, save a crop, and draw a bounding box during development. Add trained YOLOv8 weights for reliable results.

## EasyOCR

The detection pipeline now runs OCR on each cropped plate:

```text
Vehicle Image
  -> YOLO/OpenCV plate detection
  -> Plate crop
  -> Real-ESRGAN/OpenCV crop enhancement
  -> EasyOCR on enhanced color + adaptive-threshold binary crop
  -> Text + confidence score
```

OCR settings can be configured in `backend/.env`:

```text
OCR_LANGUAGES=en
OCR_GPU=false
```

## Real-ESRGAN Enhancement

The comparison endpoint runs OCR before and after enhancement:

```text
Uploaded Image
  -> OCR pipeline on original image
  -> Real-ESRGAN/OpenCV enhancement
  -> Plate detection
  -> Plate crop
  -> EasyOCR
  -> Compare before vs after OCR
```

Place Real-ESRGAN weights here:

```text
backend/weights/RealESRGAN_x4plus.pth
```

If the weights are missing, the app logs a warning and uses an OpenCV enhancement fallback so the workflow still runs. For actual Real-ESRGAN output, install dependencies and download the `.pth` weights file:

```powershell
python .\scripts\download_weights.py
```

Configuration:

```text
REAL_ESRGAN_MODEL_PATH=weights/RealESRGAN_x4plus.pth
REAL_ESRGAN_SCALE=4
REAL_ESRGAN_TILE=128
REAL_ESRGAN_HALF=false
```

Check whether the model is installed:

```powershell
.\scripts\check-model.ps1
```

You can also check in the browser:

```text
http://localhost:8000/model/status
```

For quick development only, you can set this in `backend/.env`:

```text
YOLO_ALLOW_GENERIC_FALLBACK=true
```

That lets Ultralytics load `yolov8n.pt`, but it is not a real license plate detector and should not be used for production LPR.

## Upload Example

Simple endpoint:

```powershell
curl.exe -X POST "http://localhost:8000/upload" -F "file=@C:\path\to\car.jpg"
```

Simple response:

```json
{
  "filename": "generated-file-name.jpg",
  "path": "uploads/generated-file-name.jpg"
}
```

Detection endpoint:

```powershell
curl.exe -X POST "http://localhost:8000/detect" -F "file=@C:\path\to\car.jpg"
```

Detection response:

```json
{
  "plate": "TS09AB1234",
  "confidence": 0.93,
  "candidates": []
}
```

If OCR confidence is below `0.80`, the API returns up to five possible plate candidates sorted by confidence:

```json
{
  "plate": "TS09AB1234",
  "confidence": 0.67,
  "candidates": [
    { "plate": "TS09AB1234", "confidence": 0.72 },
    { "plate": "TS09A81234", "confidence": 0.61 }
  ]
}
```

Detailed detection endpoint:

```powershell
curl.exe -X POST "http://localhost:8000/detect/details" -F "file=@C:\path\to\car.jpg"
```

Detailed response:

```json
{
  "filename": "generated-file-name.jpg",
  "path": "uploads/generated-file-name.jpg",
  "detector": "yolov8",
  "model_configured": true,
  "annotated_image_path": "uploads/annotated/generated-file-name_annotated.jpg",
  "annotated_image_url": "/uploads/annotated/generated-file-name_annotated.jpg",
  "plates_detected": 1,
  "detections": [
    {
      "confidence": 0.9123,
      "coordinates": {
        "x1": 120,
        "y1": 220,
        "x2": 360,
        "y2": 285
      },
      "cropped_plate_path": "uploads/plates/generated-file-name_plate_1_ab12cd34.jpg",
      "cropped_plate_url": "/uploads/plates/generated-file-name_plate_1_ab12cd34.jpg",
      "text": "ABC1234",
      "ocr_confidence": 0.8742,
      "candidates": []
    }
  ]
}
```

Enhancement comparison endpoint:

```powershell
curl.exe -X POST "http://localhost:8000/detect/compare" -F "file=@C:\path\to\car.jpg"
```

Comparison response:

```json
{
  "filename": "generated-file-name.jpg",
  "original_image_path": "uploads/generated-file-name.jpg",
  "enhancement": {
    "configured": true,
    "method": "real_esrgan",
    "enhanced_image_path": "uploads/enhanced/generated-file-name_enhanced.jpg",
    "enhanced_image_url": "/uploads/enhanced/generated-file-name_enhanced.jpg"
  },
  "before_enhancement": {
    "plate": "TS09AB1234",
    "confidence": 0.72,
    "detector": "yolov8",
    "model_configured": true,
    "annotated_image_path": "uploads/annotated/generated-file-name_annotated.jpg",
    "cropped_plate_path": "uploads/plates/generated-file-name_plate_1.jpg"
  },
  "after_enhancement": {
    "plate": "TS09AB1234",
    "confidence": 0.93,
    "detector": "yolov8",
    "model_configured": true,
    "annotated_image_path": "uploads/annotated/generated-file-name_enhanced_annotated.jpg",
    "cropped_plate_path": "uploads/plates/generated-file-name_enhanced_plate_1.jpg"
  },
  "improved": true
}
```

Versioned API endpoint:

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/images/upload" -F "file=@C:\path\to\car.jpg"
```

Example response:

```json
{
  "filename": "generated-file-name.jpg",
  "original_filename": "car.jpg",
  "content_type": "image/jpeg",
  "size_bytes": 123456,
  "image_path": "uploads/generated-file-name.jpg",
  "image_url": "/uploads/generated-file-name.jpg"
}
```

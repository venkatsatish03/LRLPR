# License Plate Recognition

A web application for analyzing vehicle images and reading license plates. Upload an image to detect plate regions, run OCR, inspect confidence and candidate readings, and export the results for review.

> **Project status:** Active development. The current recognition flow uses local filesystem storage and runs detection/OCR inside the FastAPI backend. PostgreSQL, Redis, MinIO, and the separate AI worker are scaffolded for future integration; they are not currently used by the upload and recognition flow.

## Features

- Upload an image from the web interface and preview it before analysis.
- Detect multiple plate regions with a trained YOLO model, or use the OpenCV candidate-detection fallback.
- Run OCR on detected plate crops with image preprocessing and confidence-ranked candidates.
- Select a plate region manually when automatic detection needs help.
- View annotated images, plate crops, coordinates, and confidence details.
- Export an analysis as JSON, CSV, or PDF.
- Try image enhancement and compare recognition results before and after enhancement.
- Use the FastAPI interactive API documentation for direct endpoint testing.

## Technology

- **Frontend:** Next.js, React, TypeScript
- **API and recognition pipeline:** FastAPI, Python, OpenCV, Ultralytics YOLO, EasyOCR
- **Optional image enhancement:** Real-ESRGAN, with an OpenCV fallback
- **Infrastructure scaffold:** Docker Compose, PostgreSQL, Redis, MinIO

## Get Started (Local)

### Prerequisites

- Python 3.11
- Node.js and npm (Node.js 22 is used by the frontend Docker image)
- Git

The first EasyOCR run may download OCR model data. For YOLO-based detection, you also need trained license-plate model weights; see [Model files](#model-files).

### Configure

From the repository root, create the local environment files:

```powershell
Copy-Item .env.example backend/.env
Copy-Item frontend/.env.example frontend/.env.local
```

These are development defaults. Do not use the sample credentials or commit real secrets in a public repository.

### Start the backend

In a PowerShell terminal:

```powershell
Set-Location backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\run.ps1
```

The API listens at <http://localhost:8000>. If PowerShell blocks virtual-environment activation, use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000` instead.

### Start the frontend

In a second terminal:

```powershell
Set-Location frontend
npm ci
npm run dev
```

Open <http://localhost:3000>. The frontend calls the backend at `http://localhost:8000` by default; change `NEXT_PUBLIC_API_BASE_URL` in `frontend/.env.local` if your API uses another address.

## API

Interactive documentation is available at <http://localhost:8000/docs>. Useful endpoints include:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check API availability |
| `GET` | `/model/status` | Inspect detector/model status |
| `POST` | `/detect` | Return the best plate text and confidence |
| `POST` | `/detect/details` | Return detections, coordinates, OCR candidates, and image paths |
| `POST` | `/detect/manual-crop` | Run OCR on a selected image region |
| `POST` | `/detect/compare` | Compare detection before and after enhancement |
| `POST` | `/upload` | Save an uploaded image |

Uploads use the multipart form field `file`. Example:

```powershell
curl.exe -X POST "http://localhost:8000/detect/details" -F "file=@C:\path\to\car.jpg"
```

The API also serves a basic upload form at <http://localhost:8000>. More implementation notes and response examples are in [backend/README.md](backend/README.md).

## Model Files

For reliable plate detection, provide YOLO weights trained to detect license plates at:

```text
backend/models/license_plate_detector.pt
```

Alternatively set `YOLO_MODEL_PATH` in `backend/.env`. Without trained weights, the backend uses its OpenCV candidate-detection fallback; detection quality may be lower. A generic YOLO model is not a substitute for a license-plate detector.

Real-ESRGAN weights are optional. To download them using the repository script, run from `backend`:

```powershell
python .\scripts\download_weights.py
```

Without those weights, enhancement falls back to OpenCV. Check model availability at <http://localhost:8000/model/status> or with `backend/scripts/check-model.ps1`.

Model weights may be large. Before publishing them to GitHub, check their size, license, and whether distribution is permitted; document any separately hosted weights and their expected paths.

## Tests

Backend unit tests use Python's built-in `unittest` runner. From the `backend` directory, with backend dependencies installed:

```powershell
python -m unittest discover -s tests
```

## Repository Layout

```text
ai-service/     Separate worker scaffold (currently a placeholder)
backend/        FastAPI API, detection/OCR services, scripts, and tests
database/       SQL initialization and migration scaffolding
docs/           Environment and configuration reference
frontend/       Next.js user interface
infrastructure/ Docker/Kubernetes and monitoring scaffolding
shared/         Shared project resources
```

For environment variable details, see [docs/environment.md](docs/environment.md). The Docker Compose file starts the broader infrastructure scaffold; not all listed services are connected to the current recognition request path.

## Responsible Use

Use this project only with images and data you are authorized to process. Recognition results can be incorrect; verify plate readings before relying on them.

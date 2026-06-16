# License Plate Recognition System

Production-oriented scaffold for an LPR application with image upload, license plate detection, OCR extraction, confidence scoring, and result persistence.

## Stack

- Frontend: Next.js / React
- Backend: FastAPI
- Database: PostgreSQL
- Storage: S3 or MinIO
- Queue: Redis + Celery
- AI model: YOLO-based plate detector
- OCR engine: PaddleOCR or EasyOCR

## Main Flow

1. User uploads an image from the frontend.
2. Backend validates and stores the original image.
3. Backend creates an async recognition job.
4. AI service detects and crops the plate.
5. OCR extracts the plate text.
6. Confidence score is calculated.
7. Backend stores the result in PostgreSQL.
8. Frontend displays status and result.

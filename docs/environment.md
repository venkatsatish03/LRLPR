# Environment Variables

This project has two environment scopes:

- Root/backend runtime variables are loaded by Docker Compose and by the FastAPI backend settings layer.
- Frontend variables prefixed with `NEXT_PUBLIC_` are compiled into browser-visible JavaScript and must not contain secrets.

The root `.env.example` is the canonical sample for the full project. `frontend/.env.example` contains only frontend-specific public variables.

## Startup Validation

The backend validates configuration when the FastAPI app starts.

Always required for backend startup:

- `APP_ENV`
- `PROJECT_NAME`
- `API_VERSION`
- `API_PREFIX`
- `OCR_LANGUAGES`
- `CORS_ORIGINS`

Validation also checks port ranges, upload size, YOLO confidence range, and Real-ESRGAN numeric settings.

When `APP_ENV=production`, these integration variables are also required:

- `DATABASE_URL`
- `REDIS_URL`
- `S3_ENDPOINT`
- `S3_BUCKET`
- `S3_ACCESS_KEY`
- `S3_SECRET_KEY`
- `AI_SERVICE_URL`

## Variable Reference

| Variable | Required | Default / Example | Used by | Purpose |
| --- | --- | --- | --- | --- |
| `APP_ENV` | Yes | `development` | Backend | Runtime mode. Allowed values: `development`, `test`, `production`. |
| `PROJECT_NAME` | Yes | `License Plate Recognition API` | Backend | FastAPI application title. |
| `API_VERSION` | Yes | `1.0.0` | Backend | FastAPI application version. |
| `API_PREFIX` | Yes | `/api/v1` | Backend | Prefix for versioned API routes. Must start with `/`. |
| `CORS_ORIGINS` | Yes | `http://localhost:3000,http://127.0.0.1:3000` | Backend | Comma-separated allowed browser origins. |
| `API_PORT` | Yes | `8000` | Docker Compose, Backend validation | Host port for the backend service. Must be 1-65535. |
| `FRONTEND_PORT` | Yes | `3000` | Docker Compose, Backend validation | Host port for the frontend service. Must be 1-65535. |
| `NEXT_PUBLIC_API_BASE_URL` | Frontend | `http://localhost:8000` | Frontend | Browser-visible base URL for FastAPI requests. |
| `UPLOAD_DIR` | Backend | `uploads` | Backend | Directory for original uploaded files. |
| `ANNOTATED_DIR` | Backend | `uploads/annotated` | Backend | Directory for annotated detection images. |
| `PLATE_CROP_DIR` | Backend | `uploads/plates` | Backend | Directory for cropped plate images. |
| `ENHANCED_DIR` | Backend | `uploads/enhanced` | Backend | Directory for enhanced images. |
| `MAX_UPLOAD_SIZE_BYTES` | Backend | `10485760` | Backend | Maximum accepted upload size in bytes. Must be greater than 0. |
| `POSTGRES_DB` | Compose | `lpr` | Docker Compose | PostgreSQL database name. |
| `POSTGRES_USER` | Compose | `lpr_user` | Docker Compose | PostgreSQL user. |
| `POSTGRES_PASSWORD` | Compose | `lpr_password` | Docker Compose | PostgreSQL password. |
| `DATABASE_URL` | Production | `postgresql://lpr_user:lpr_password@postgres:5432/lpr` | Backend config, planned persistence | PostgreSQL connection string. Required in production. |
| `REDIS_URL` | Production | `redis://redis:6379/0` | Backend config, planned queue | Redis connection string. Required in production. |
| `S3_ENDPOINT` | Production | `http://minio:9000` | Backend config, planned object storage | S3 or MinIO endpoint. Required in production. |
| `S3_BUCKET` | Production | `lpr-images` | Backend config, planned object storage | Bucket for image storage. Required in production. |
| `S3_ACCESS_KEY` | Production | `minioadmin` | Docker Compose, Backend config | S3 or MinIO access key. Required in production. |
| `S3_SECRET_KEY` | Production | `minioadmin` | Docker Compose, Backend config | S3 or MinIO secret key. Required in production. |
| `AI_SERVICE_URL` | Production | `http://ai-service:9000` | Backend config, planned AI service | URL for the separate AI service. Required in production. |
| `YOLO_MODEL_PATH` | Backend | `models/license_plate_detector.pt` | Backend | Path to trained YOLO license plate weights. |
| `YOLO_CONFIDENCE_THRESHOLD` | Backend | `0.25` | Backend | Minimum YOLO detection confidence. Must be between 0 and 1. |
| `YOLO_NMS_IOU_THRESHOLD` | Backend | `0.45` | Backend | YOLO non-maximum suppression IoU threshold. Lower values suppress more overlapping boxes. |
| `YOLO_IMAGE_SIZE` | Backend | `960` | Backend | Inference image size passed to YOLO. Larger values can improve small plates but reduce speed. |
| `YOLO_MAX_DETECTIONS` | Backend | `20` | Backend | Maximum number of YOLO boxes to keep before backend post-filtering. |
| `YOLO_CLASSES` | Backend | empty | Backend | Optional comma-separated class IDs to pass to YOLO, for example `0`. Overrides class-name matching. |
| `YOLO_CLASS_NAMES` | Backend | `license_plate,plate,number_plate,licence_plate` | Backend | Class names treated as plate classes when model metadata contains names. |
| `YOLO_AGNOSTIC_NMS` | Backend | `false` | Backend | Enables class-agnostic NMS in YOLO. Usually false for single-class plate models. |
| `YOLO_MIN_BOX_WIDTH` | Backend | `12` | Backend | Minimum accepted YOLO box width in pixels after inference. |
| `YOLO_MIN_BOX_HEIGHT` | Backend | `6` | Backend | Minimum accepted YOLO box height in pixels after inference. |
| `YOLO_MIN_BOX_AREA_RATIO` | Backend | `0.00005` | Backend | Minimum accepted box area divided by image area. |
| `YOLO_MAX_BOX_AREA_RATIO` | Backend | `0.20` | Backend | Maximum accepted box area divided by image area. |
| `YOLO_MIN_ASPECT_RATIO` | Backend | `1.4` | Backend | Minimum accepted box width/height ratio. |
| `YOLO_MAX_ASPECT_RATIO` | Backend | `9.5` | Backend | Maximum accepted box width/height ratio. |
| `YOLO_CROP_PADDING_X` | Backend | `0.06` | Backend | Horizontal padding ratio added to YOLO boxes for OCR crop only. |
| `YOLO_CROP_PADDING_Y` | Backend | `0.16` | Backend | Vertical padding ratio added to YOLO boxes for OCR crop only. |
| `YOLO_ALLOW_GENERIC_FALLBACK` | Backend | `false` | Backend | Allows loading generic `yolov8n.pt` for development only. |
| `FALLBACK_PREPROCESSING_VARIANTS` | Backend | `original,contrast_clahe,low_light_gamma,sharpened` | Backend | Comma-separated OpenCV fallback variants. Use more variants for recall, fewer for speed. |
| `FALLBACK_MAX_IMAGE_SIDE` | Backend | `1280` | Backend | Maximum side length used during OpenCV candidate search. Larger values may find smaller plates but are slower. |
| `FALLBACK_MAX_CONTOURS` | Backend | `40` | Backend | Maximum contours examined per fallback variant. |
| `FALLBACK_MAX_OCR_CANDIDATES` | Backend | `4` | Backend | Maximum fallback candidate crops sent to OCR. This is the most important speed cap when YOLO weights are missing. |
| `FALLBACK_MAX_DETECTIONS` | Backend | `2` | Backend | Maximum fallback detections returned after OCR. |
| `FALLBACK_ENABLE_UPSCALE` | Backend | `false` | Backend | Enables the expensive upscaled fallback search variant. Prefer false for local speed. |
| `FALLBACK_ENABLE_WHOLE_IMAGE_PLATE_CANDIDATE` | Backend | `true` | Backend | OCRs the whole upload when it looks like a direct plate image or fallback candidates appear fragmented. |
| `REAL_ESRGAN_MODEL_PATH` | Backend | `models/RealESRGAN_x4plus.pth` | Backend | Path to Real-ESRGAN enhancement weights. |
| `REAL_ESRGAN_SCALE` | Backend | `4` | Backend | Real-ESRGAN scale factor. Must be greater than 0. |
| `REAL_ESRGAN_TILE` | Backend | `0` | Backend | Real-ESRGAN tile size. `0` disables tiling. |
| `REAL_ESRGAN_HALF` | Backend | `false` | Backend | Enables half precision when CUDA is available. |
| `OCR_LANGUAGES` | Yes | `en` | Backend | Comma-separated EasyOCR languages. |
| `OCR_GPU` | Backend | `false` | Backend | Enables EasyOCR GPU mode. |
| `OCR_PREPROCESSING_STRATEGIES` | Backend | `original_resized,clahe,adaptive_threshold` | Backend | Comma-separated OCR variants to run. Set to `all` for maximum recall and slower processing. |
| `OCR_MAX_VARIANTS` | Backend | `3` | Backend | Maximum OCR variants run for each crop. |
| `OCR_MIN_VARIANTS_BEFORE_EARLY_EXIT` | Backend | `1` | Backend | Minimum OCR variants to run before allowing confidence-based early exit. |
| `OCR_EARLY_EXIT_CONFIDENCE` | Backend | `0.82` | Backend | Stops OCR variants early once the best candidate reaches this confidence. |
| `OCR_TARGET_HEIGHT` | Backend | `96` | Backend | Target crop text height for OCR upscaling. Higher values are slower. |
| `OCR_MAX_UPSCALE` | Backend | `4` | Backend | Maximum crop upscaling factor before OCR. |
| `OCR_MAX_IMAGE_WIDTH` | Backend | `768` | Backend | Maximum OCR variant width to keep EasyOCR inference bounded. |

## Notes

- `DATABASE_URL`, `REDIS_URL`, S3 variables, and `AI_SERVICE_URL` are documented and validated for production because they are part of the scaffolded architecture, but current request handling still uses local filesystem storage and in-process AI/OCR.
- `NEXT_PUBLIC_API_BASE_URL` is public. Do not put tokens, passwords, or private hostnames in it.
- If the backend fails to start with an environment error, copy the missing or invalid variable names from the error message and update `.env`.

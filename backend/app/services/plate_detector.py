from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import HTTPException, status
from ultralytics import YOLO

from app.core.config import settings
from app.services.ocr_service import OCRService


class PlateDetectorService:
    def __init__(
        self,
        model_path: Path = settings.YOLO_MODEL_PATH,
        annotated_dir: Path = settings.ANNOTATED_DIR,
        crop_dir: Path = settings.PLATE_CROP_DIR,
        confidence_threshold: float = settings.YOLO_CONFIDENCE_THRESHOLD,
        ocr_service: OCRService | None = None,
    ) -> None:
        self.model_path = model_path
        self.annotated_dir = annotated_dir
        self.crop_dir = crop_dir
        self.confidence_threshold = confidence_threshold
        self.ocr_service = ocr_service or OCRService()
        self._model: YOLO | None = None
        self._resolved_model_path: Path | str | None = None

        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.crop_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> YOLO:
        if self._model is None:
            model_path = self._resolve_model_path()
            if model_path is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=self.model_status(),
                )
            self._resolved_model_path = model_path
            self._model = YOLO(str(model_path))
        return self._model

    def model_status(self) -> dict:
        model_path = self._resolve_model_path()
        searched_paths = [str(path) for path in self._candidate_model_paths()]

        if model_path is not None:
            return {
                "configured": True,
                "model_path": str(model_path),
                "message": "YOLOv8 license plate model is configured.",
            }

        return {
            "configured": False,
            "model_path": None,
            "message": "YOLOv8 license plate model not found.",
            "searched_paths": searched_paths,
            "fix": (
                "Place your trained license plate weights at "
                "backend/models/license_plate_detector.pt, or set YOLO_MODEL_PATH "
                "in backend/.env to the correct .pt file."
            ),
            "note": (
                "A normal YOLOv8 COCO model does not detect license plates reliably. "
                "Use weights trained for license plates."
            ),
        }

    def detect(self, image_path: str | Path) -> dict:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to read uploaded image.",
            )

        if self._resolve_model_path() is None:
            return self._detect_with_opencv_fallback(image_path, image)

        return self._detect_with_yolo(image_path, image)

    def _detect_with_yolo(self, image_path: Path, image) -> dict:
        result = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            verbose=False,
        )[0]

        detections = []
        annotated_image = image.copy()

        for index, box in enumerate(result.boxes):
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            crop_filename = f"{image_path.stem}_plate_{index + 1}_{uuid4().hex[:8]}.jpg"
            crop_path = self.crop_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)
            ocr_result = self.ocr_service.extract_text(crop_path)

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_image,
                f"plate {confidence:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            detections.append(
                {
                    "confidence": round(confidence, 4),
                    "coordinates": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                    "cropped_plate_path": str(crop_path),
                    "cropped_plate_url": f"/uploads/plates/{crop_filename}",
                    "text": ocr_result["text"],
                    "ocr_confidence": ocr_result["ocr_confidence"],
                    "candidates": ocr_result["candidates"],
                }
            )

        annotated_path = None
        annotated_url = None
        if detections:
            annotated_filename = f"{image_path.stem}_annotated.jpg"
            annotated_path = self.annotated_dir / annotated_filename
            cv2.imwrite(str(annotated_path), annotated_image)
            annotated_url = f"/uploads/annotated/{annotated_filename}"

        return {
            "detector": "yolov8",
            "model_configured": True,
            "annotated_image_path": str(annotated_path) if annotated_path else None,
            "annotated_image_url": annotated_url,
            "plates_detected": len(detections),
            "detections": detections,
        }

    def _detect_with_opencv_fallback(self, image_path: Path, image) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(filtered, 30, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        candidates = []
        image_area = image.shape[0] * image.shape[1]
        image_height, image_width = image.shape[:2]

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if height == 0:
                continue

            area = width * height
            aspect_ratio = width / height
            area_ratio = area / image_area
            y_center_ratio = (y + height / 2) / image_height
            height_ratio = height / image_height

            if 2.0 <= aspect_ratio <= 7.5 and 0.004 <= area_ratio <= 0.25 and height_ratio >= 0.05:
                ratio_score = 1.0 - min(abs(aspect_ratio - 4.2) / 4.2, 1.0)
                area_score = min(area_ratio / 0.08, 1.0)
                lower_half_score = 1.0 if y_center_ratio >= 0.45 else 0.45
                score = (ratio_score * 0.35) + (area_score * 0.35) + (lower_half_score * 0.30)
                candidates.append((score, x, y, x + width, y + height))

        candidates.extend(self._find_bright_plate_candidates(image))

        detections = []
        annotated_image = image.copy()

        best_candidate = None
        if candidates:
            candidates.sort(reverse=True)
            for score, x1, y1, x2, y2 in candidates[:20]:
                expanded_x1, expanded_y1, expanded_x2, expanded_y2 = self._expand_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    image_width,
                    image_height,
                )
                crop = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
                ocr_result = self.ocr_service.extract_text_from_image(crop)
                ocr_score = self._ocr_candidate_score(ocr_result)
                candidate = (
                    ocr_score,
                    score,
                    expanded_x1,
                    expanded_y1,
                    expanded_x2,
                    expanded_y2,
                    crop,
                    ocr_result,
                )

                if best_candidate is None or candidate[:2] > best_candidate[:2]:
                    best_candidate = candidate

                if len(str(ocr_result["text"])) >= 6 and float(ocr_result["ocr_confidence"]) >= 0.2:
                    break

        if best_candidate:
            _, fallback_score, x1, y1, x2, y2, crop, ocr_result = best_candidate
            crop_filename = f"{image_path.stem}_plate_1_{uuid4().hex[:8]}.jpg"
            crop_path = self.crop_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 170, 0), 2)
            cv2.putText(
                annotated_image,
                f"plate candidate {fallback_score:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 170, 0),
                2,
            )

            detections.append(
                {
                    "confidence": 0.5,
                    "coordinates": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    },
                    "cropped_plate_path": str(crop_path),
                    "cropped_plate_url": f"/uploads/plates/{crop_filename}",
                    "text": ocr_result["text"],
                    "ocr_confidence": ocr_result["ocr_confidence"],
                    "candidates": ocr_result["candidates"],
                }
            )

        annotated_path = None
        annotated_url = None
        if detections:
            annotated_filename = f"{image_path.stem}_annotated.jpg"
            annotated_path = self.annotated_dir / annotated_filename
            cv2.imwrite(str(annotated_path), annotated_image)
            annotated_url = f"/uploads/annotated/{annotated_filename}"

        return {
            "detector": "opencv_fallback",
            "model_configured": False,
            "annotated_image_path": str(annotated_path) if annotated_path else None,
            "annotated_image_url": annotated_url,
            "plates_detected": len(detections),
            "detections": detections,
        }

    @staticmethod
    def _expand_box(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * 0.08)
        pad_y = int(height * 0.20)

        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(image_width, x2 + pad_x),
            min(image_height, y2 + pad_y),
        )

    @staticmethod
    def _find_bright_plate_candidates(image) -> list[tuple[float, int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

        image_height, image_width = image.shape[:2]
        image_area = image_height * image_width
        candidates = []

        for mask in (threshold, closed):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                if height == 0:
                    continue

                aspect_ratio = width / height
                area_ratio = (width * height) / image_area
                y_center_ratio = (y + height / 2) / image_height

                if 1.8 <= aspect_ratio <= 7.5 and 0.01 <= area_ratio <= 0.35 and y_center_ratio >= 0.35:
                    ratio_score = 1.0 - min(abs(aspect_ratio - 4.0) / 4.0, 1.0)
                    area_score = min(area_ratio / 0.12, 1.0)
                    lower_half_score = min(y_center_ratio / 0.65, 1.0)
                    score = 0.25 + (ratio_score * 0.25) + (area_score * 0.30) + (lower_half_score * 0.20)
                    candidates.append((score, x, y, x + width, y + height))

        return candidates

    @staticmethod
    def _ocr_candidate_score(ocr_result: dict[str, str | float]) -> float:
        text = str(ocr_result["text"])
        confidence = float(ocr_result["ocr_confidence"])
        has_digit = any(char.isdigit() for char in text)
        has_alpha = any(char.isalpha() for char in text)
        length_score = min(len(text), 10) / 10
        content_bonus = 0.25 if has_digit and has_alpha else 0.0
        return confidence + length_score + content_bonus

    def _resolve_model_path(self) -> Path | str | None:
        for path in self._candidate_model_paths():
            if path.exists():
                return path

        if settings.YOLO_ALLOW_GENERIC_FALLBACK:
            return "yolov8n.pt"

        return None

    def _candidate_model_paths(self) -> list[Path]:
        project_root = Path(__file__).resolve().parents[3]
        backend_root = Path(__file__).resolve().parents[2]

        configured_path = self.model_path
        candidates = [
            configured_path,
            backend_root / configured_path,
            backend_root / "models" / "license_plate_detector.pt",
            backend_root / "models" / "best.pt",
            project_root / "ai-service" / "detector" / "weights" / "license_plate_detector.pt",
            project_root / "ai-service" / "detector" / "weights" / "best.pt",
        ]

        unique_candidates = []
        for candidate in candidates:
            candidate = candidate.resolve() if not candidate.is_absolute() else candidate
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

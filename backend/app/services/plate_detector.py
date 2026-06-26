from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import HTTPException, status
from ultralytics import YOLO

from app.core.config import settings
from app.services.ocr_service import OCRService


class PlateDetectorService:
    FALLBACK_NMS_IOU_THRESHOLD = 0.35

    def __init__(
        self,
        model_path: Path = settings.YOLO_MODEL_PATH,
        annotated_dir: Path = settings.ANNOTATED_DIR,
        crop_dir: Path = settings.PLATE_CROP_DIR,
        confidence_threshold: float = settings.YOLO_CONFIDENCE_THRESHOLD,
        nms_iou_threshold: float = settings.YOLO_NMS_IOU_THRESHOLD,
        image_size: int = settings.YOLO_IMAGE_SIZE,
        max_detections: int = settings.YOLO_MAX_DETECTIONS,
        ocr_service: OCRService | None = None,
    ) -> None:
        self.model_path = model_path
        self.annotated_dir = annotated_dir
        self.crop_dir = crop_dir
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.image_size = image_size
        self.max_detections = max_detections
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

        direct_plate_result = self._detect_direct_plate_upload(image_path, image)
        if direct_plate_result is not None:
            return direct_plate_result

        if self._resolve_model_path() is None:
            return self._detect_with_opencv_fallback(image_path, image)

        return self._detect_with_yolo(image_path, image)

    def _detect_direct_plate_upload(self, image_path: Path, image) -> dict | None:
        candidate = self._whole_image_plate_candidate(image, [])
        if candidate is None:
            return None

        x1, y1, x2, y2 = candidate["box"]
        detection = self._build_detection(
            image_path=image_path,
            crop=image,
            index=0,
            confidence=float(candidate["score"]),
            coordinates=(x1, y1, x2, y2),
        )
        if not detection["text"] or float(detection["final_confidence"]) < 0.45:
            return None

        detections = [detection]
        annotated_image = image.copy()
        self._annotate_detections(annotated_image, detections, color=(0, 220, 120), label="plate")

        annotated_filename = f"{image_path.stem}_annotated.jpg"
        annotated_path = self.annotated_dir / annotated_filename
        cv2.imwrite(str(annotated_path), annotated_image)

        return {
            "detector": "direct_plate_image",
            "model_configured": self._resolve_model_path() is not None,
            "annotated_image_path": str(annotated_path),
            "annotated_image_url": f"/uploads/annotated/{annotated_filename}",
            "plates_detected": len(detections),
            "detections": detections,
        }

    def _detect_with_yolo(self, image_path: Path, image) -> dict:
        result = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            imgsz=self.image_size,
            max_det=self.max_detections,
            classes=self._resolve_yolo_classes(),
            agnostic_nms=settings.YOLO_AGNOSTIC_NMS,
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

            if not self._is_plausible_yolo_box(x1, y1, x2, y2, image.shape[1], image.shape[0]):
                continue

            crop_box = self._expand_box_with_padding(
                x1,
                y1,
                x2,
                y2,
                image.shape[1],
                image.shape[0],
                settings.YOLO_CROP_PADDING_X,
                settings.YOLO_CROP_PADDING_Y,
            )
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            detections.append(
                self._build_detection(
                    image_path=image_path,
                    crop=crop,
                    index=index,
                    confidence=confidence,
                    coordinates=(x1, y1, x2, y2),
                )
            )

        detections = self._filter_low_value_detections(self._sort_detections(detections))
        if not detections:
            return self._detect_with_opencv_fallback(image_path, image)

        self._annotate_detections(annotated_image, detections, color=(0, 255, 0), label="plate")

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
        candidates = []
        image_height, image_width = image.shape[:2]
        scan_image, scan_scale = self._resize_for_fallback_scan(image)

        for variant in self._build_fallback_image_variants(scan_image):
            total_scale = scan_scale * float(variant["scale"])
            candidates.extend(
                self._find_edge_plate_candidates(
                    variant["image"],
                    scale=total_scale,
                    source=str(variant["source"]),
                )
            )
            candidates.extend(
                self._find_bright_plate_candidates(
                    variant["image"],
                    scale=total_scale,
                    source=str(variant["source"]),
                )
            )

        candidates = self._deduplicate_candidates(candidates)
        if settings.FALLBACK_ENABLE_WHOLE_IMAGE_PLATE_CANDIDATE:
            whole_image_candidate = self._whole_image_plate_candidate(image, candidates)
            if whole_image_candidate is not None:
                candidates = [whole_image_candidate, *candidates]

        detections = []
        annotated_image = image.copy()

        for index, candidate in enumerate(candidates[: settings.FALLBACK_MAX_OCR_CANDIDATES]):
            score = float(candidate["score"])
            x1, y1, x2, y2 = candidate["box"]
            expanded_box = self._expand_box(
                x1,
                y1,
                x2,
                y2,
                image_width,
                image_height,
            )
            crop = self._crop_candidate(image, candidate, expanded_box)
            if crop is None or crop.size == 0:
                continue

            detection = self._build_detection(
                image_path=image_path,
                crop=crop,
                index=index,
                confidence=score,
                coordinates=expanded_box,
            )

            if detection["text"] or detection["final_confidence"] >= 0.25:
                detections.append(detection)

            if len(detections) >= settings.FALLBACK_MAX_DETECTIONS:
                break

        detections = self._filter_low_value_detections(self._sort_detections(detections))
        self._annotate_detections(annotated_image, detections, color=(255, 170, 0), label="candidate")

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

    def _build_fallback_image_variants(self, image) -> list[dict]:
        contrast_image = None
        denoised_image = None

        def contrast():
            nonlocal contrast_image
            if contrast_image is None:
                contrast_image = self._enhance_local_contrast(image)
            return contrast_image

        def denoised():
            nonlocal denoised_image
            if denoised_image is None:
                denoised_image = cv2.bilateralFilter(contrast(), 7, 55, 55)
            return denoised_image

        def upscaled_contrast():
            height, width = image.shape[:2]
            if not settings.FALLBACK_ENABLE_UPSCALE:
                return None
            return cv2.resize(contrast(), (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

        builders = {
            "original": lambda: {"source": "original", "image": image, "scale": 1.0},
            "contrast_clahe": lambda: {"source": "contrast_clahe", "image": contrast(), "scale": 1.0},
            "fog_rain_contrast": lambda: {
                "source": "fog_rain_contrast",
                "image": self._contrast_stretch(contrast()),
                "scale": 1.0,
            },
            "low_light_gamma": lambda: {
                "source": "low_light_gamma",
                "image": self._gamma_correct(contrast(), gamma=0.55),
                "scale": 1.0,
            },
            "highlight_compressed": lambda: {
                "source": "highlight_compressed",
                "image": self._compress_highlights(image),
                "scale": 1.0,
            },
            "glare_reduced": lambda: {"source": "glare_reduced", "image": self._reduce_glare(image), "scale": 1.0},
            "denoised": lambda: {"source": "denoised", "image": denoised(), "scale": 1.0},
            "sharpened": lambda: {"source": "sharpened", "image": self._sharpen_bgr(denoised()), "scale": 1.0},
            "upscaled_contrast": lambda: {
                "source": "upscaled_contrast",
                "image": upscaled_contrast(),
                "scale": 2.0,
            },
        }

        variants = []
        for variant_name in self._selected_fallback_variant_names():
            builder = builders.get(variant_name)
            if builder is None:
                continue

            variant = builder()
            if variant["image"] is not None:
                variants.append(variant)

        return variants or [{"source": "original", "image": image, "scale": 1.0}]

    @staticmethod
    def _selected_fallback_variant_names() -> list[str]:
        return [
            name.strip()
            for name in settings.FALLBACK_PREPROCESSING_VARIANTS.split(",")
            if name.strip()
        ]

    @staticmethod
    def _resize_for_fallback_scan(image):
        height, width = image.shape[:2]
        max_side = max(height, width)
        if max_side <= settings.FALLBACK_MAX_IMAGE_SIDE:
            return image, 1.0

        scale = settings.FALLBACK_MAX_IMAGE_SIDE / max_side
        resized = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    @staticmethod
    def _whole_image_plate_candidate(image, candidates: list[dict]) -> dict | None:
        image_height, image_width = image.shape[:2]
        if image_width <= 0 or image_height <= 0:
            return None

        aspect_ratio = image_width / image_height
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_ratio = cv2.countNonZero(cv2.inRange(gray, 165, 255)) / max(image_width * image_height, 1)
        dark_ratio = cv2.countNonZero(cv2.inRange(gray, 0, 85)) / max(image_width * image_height, 1)

        looks_like_single_line_plate = 2.0 <= aspect_ratio <= 6.8 and bright_ratio >= 0.32 and dark_ratio >= 0.04
        looks_like_two_line_plate_photo = (
            0.75 <= aspect_ratio <= 1.7
            and max(image_width, image_height) <= 1200
            and bright_ratio >= 0.38
            and dark_ratio >= 0.04
        )
        fallback_is_fragmented = bool(candidates) and max(
            (candidate["box"][2] - candidate["box"][0]) / max(image_width, 1)
            for candidate in candidates[: min(len(candidates), 3)]
        ) < 0.45

        if not (looks_like_single_line_plate or looks_like_two_line_plate_photo or fallback_is_fragmented):
            return None

        score = 0.62
        if looks_like_single_line_plate:
            score = 0.74
        elif looks_like_two_line_plate_photo:
            score = 0.68

        return PlateDetectorService._candidate(
            score,
            0,
            0,
            image_width,
            image_height,
            "whole_image_plate_candidate",
        )

    @staticmethod
    def _enhance_local_contrast(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced_lightness = clahe.apply(lightness)
        enhanced = cv2.merge((enhanced_lightness, channel_a, channel_b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _contrast_stretch(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        low, high = np.percentile(lightness, (2, 98))
        if high <= low:
            return image.copy()

        stretched = np.clip((lightness.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        enhanced = cv2.merge((stretched, channel_a, channel_b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _gamma_correct(image, gamma: float):
        gamma = max(gamma, 0.01)
        table = np.array([((value / 255.0) ** gamma) * 255 for value in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def _compress_highlights(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        value_float = value.astype(np.float32)
        highlight_mask = value_float > 210
        value_float[highlight_mask] = 210 + ((value_float[highlight_mask] - 210) * 0.35)
        compressed = cv2.merge((hue, saturation, np.clip(value_float, 0, 255).astype(np.uint8)))
        return cv2.cvtColor(compressed, cv2.COLOR_HSV2BGR)

    @staticmethod
    def _reduce_glare(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, saturation, value = cv2.split(hsv)
        glare_mask = cv2.inRange(value, 235, 255)
        low_saturation_mask = cv2.inRange(saturation, 0, 75)
        glare_mask = cv2.bitwise_and(glare_mask, low_saturation_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)
        if cv2.countNonZero(glare_mask) == 0:
            return image.copy()

        return cv2.inpaint(image, glare_mask, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def _sharpen_bgr(image):
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
        return cv2.addWeighted(image, 1.55, blurred, -0.55, 0)

    def _find_edge_plate_candidates(self, image, scale: float, source: str) -> list[dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(filtered, 25, 190)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[: settings.FALLBACK_MAX_CONTOURS]

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

            if 1.45 <= aspect_ratio <= 9.0 and 0.00035 <= area_ratio <= 0.30 and height_ratio >= 0.012:
                ratio_score = 1.0 - min(abs(aspect_ratio - 4.2) / 4.2, 1.0)
                area_score = min(area_ratio / 0.06, 1.0)
                lower_half_score = 1.0 if y_center_ratio >= 0.30 else 0.55
                partial_plate_bonus = (
                    0.08
                    if self._touches_image_edge(x, y, x + width, y + height, image_width, image_height)
                    else 0.0
                )
                score = min(
                    1.0,
                    (ratio_score * 0.36)
                    + (area_score * 0.34)
                    + (lower_half_score * 0.22)
                    + partial_plate_bonus,
                )
                candidates.append(
                    self._candidate_from_scaled(
                        score,
                        x,
                        y,
                        x + width,
                        y + height,
                        f"{source}:edge_contour",
                        scale,
                    )
                )

            rotated_candidate = self._rotated_candidate_from_contour(
                contour,
                image_width,
                image_height,
                image_area,
                source=f"{source}:rotated_contour",
                scale=scale,
            )
            if rotated_candidate is not None:
                candidates.append(rotated_candidate)

        return candidates

    def _build_detection(
        self,
        image_path: Path,
        crop,
        index: int,
        confidence: float,
        coordinates: tuple[int, int, int, int],
    ) -> dict:
        crop_filename = f"{image_path.stem}_plate_{index + 1}_{uuid4().hex[:8]}.jpg"
        crop_path = self.crop_dir / crop_filename
        cv2.imwrite(str(crop_path), crop)
        ocr_result = self.ocr_service.extract_text_from_image(crop)
        ocr_confidence = float(ocr_result["ocr_confidence"])
        final_confidence = self._combined_confidence(confidence, ocr_confidence, str(ocr_result["text"]))
        x1, y1, x2, y2 = coordinates

        return {
            "confidence": round(float(confidence), 4),
            "final_confidence": final_confidence,
            "coordinates": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            },
            "cropped_plate_path": str(crop_path),
            "cropped_plate_url": f"/uploads/plates/{crop_filename}",
            "text": ocr_result["text"],
            "ocr_confidence": round(ocr_confidence, 4),
            "candidates": ocr_result["candidates"],
        }

    def _resolve_yolo_classes(self) -> list[int] | None:
        configured_classes = self._parse_class_ids(settings.YOLO_CLASSES)
        if configured_classes:
            return configured_classes

        model_names = getattr(self.model, "names", None)
        if not isinstance(model_names, dict):
            return None

        allowed_names = {
            name.strip().lower()
            for name in settings.YOLO_CLASS_NAMES.split(",")
            if name.strip()
        }
        if not allowed_names:
            return None

        matched_classes = [
            int(class_id)
            for class_id, name in model_names.items()
            if str(name).strip().lower() in allowed_names
        ]
        return matched_classes or None

    @staticmethod
    def _parse_class_ids(raw_classes: str) -> list[int] | None:
        class_ids = []
        for raw_value in raw_classes.split(","):
            raw_value = raw_value.strip()
            if not raw_value:
                continue

            try:
                class_ids.append(int(raw_value))
            except ValueError:
                continue

        return class_ids or None

    @staticmethod
    def _expand_box_with_padding(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_width: int,
        image_height: int,
        padding_x_ratio: float,
        padding_y_ratio: float,
    ) -> tuple[int, int, int, int]:
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * padding_x_ratio)
        pad_y = int(height * padding_y_ratio)

        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(image_width, x2 + pad_x),
            min(image_height, y2 + pad_y),
        )

    @staticmethod
    def _is_plausible_yolo_box(x1: int, y1: int, x2: int, y2: int, image_width: int, image_height: int) -> bool:
        width = x2 - x1
        height = y2 - y1
        if width < settings.YOLO_MIN_BOX_WIDTH or height < settings.YOLO_MIN_BOX_HEIGHT:
            return False

        aspect_ratio = width / max(height, 1)
        if aspect_ratio < settings.YOLO_MIN_ASPECT_RATIO or aspect_ratio > settings.YOLO_MAX_ASPECT_RATIO:
            return False

        area_ratio = (width * height) / max(image_width * image_height, 1)
        if area_ratio < settings.YOLO_MIN_BOX_AREA_RATIO or area_ratio > settings.YOLO_MAX_BOX_AREA_RATIO:
            return False

        return True

    @staticmethod
    def _combined_confidence(detector_confidence: float, ocr_confidence: float, text: str) -> float:
        if text:
            return round(min(1.0, (detector_confidence * 0.45) + (ocr_confidence * 0.55)), 4)

        return round(min(1.0, detector_confidence * 0.35), 4)

    @staticmethod
    def _sort_detections(detections: list[dict]) -> list[dict]:
        return sorted(
            detections,
            key=lambda detection: (
                float(detection["final_confidence"]),
                float(detection["ocr_confidence"]),
                float(detection["confidence"]),
                len(str(detection["text"])),
            ),
            reverse=True,
        )

    @staticmethod
    def _filter_low_value_detections(detections: list[dict]) -> list[dict]:
        if not any(str(detection["text"]) for detection in detections):
            return detections

        return [
            detection
            for detection in detections
            if str(detection["text"]) or float(detection["final_confidence"]) >= 0.45
        ]

    @staticmethod
    def _annotate_detections(annotated_image, detections: list[dict], color: tuple[int, int, int], label: str) -> None:
        for index, detection in enumerate(detections, start=1):
            coordinates = detection["coordinates"]
            x1 = coordinates["x1"]
            y1 = coordinates["y1"]
            x2 = coordinates["x2"]
            y2 = coordinates["y2"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_image,
                f"{label} {index} {detection['final_confidence']:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    @staticmethod
    def _candidate(score: float, x1: int, y1: int, x2: int, y2: int, source: str, points=None) -> dict:
        return {
            "score": round(float(score), 4),
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "source": source,
            "points": points,
        }

    def _candidate_from_scaled(
        self,
        score: float,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        source: str,
        scale: float,
        points=None,
    ) -> dict:
        mapped_points = points / scale if points is not None and scale != 1.0 else points
        return self._candidate(
            score,
            round(x1 / scale),
            round(y1 / scale),
            round(x2 / scale),
            round(y2 / scale),
            source,
            points=mapped_points,
        )

    def _rotated_candidate_from_contour(
        self,
        contour,
        image_width: int,
        image_height: int,
        image_area: int,
        source: str = "rotated_contour",
        scale: float = 1.0,
    ) -> dict | None:
        rect = cv2.minAreaRect(contour)
        (_, _), (rect_width, rect_height), angle = rect
        if rect_width <= 0 or rect_height <= 0:
            return None

        plate_width = max(rect_width, rect_height)
        plate_height = min(rect_width, rect_height)
        aspect_ratio = plate_width / max(plate_height, 1)
        area_ratio = (plate_width * plate_height) / image_area
        if not (1.8 <= aspect_ratio <= 8.5 and 0.001 <= area_ratio <= 0.22):
            return None

        points = cv2.boxPoints(rect)
        x, y, width, height = cv2.boundingRect(points.astype(int))
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + width)
        y2 = min(image_height, y + height)
        if x2 <= x1 or y2 <= y1:
            return None

        ratio_score = 1.0 - min(abs(aspect_ratio - 4.2) / 4.2, 1.0)
        area_score = min(area_ratio / 0.06, 1.0)
        angle_score = 1.0 if abs(angle) <= 20 else 0.78
        score = (ratio_score * 0.45) + (area_score * 0.35) + (angle_score * 0.20)
        return self._candidate_from_scaled(score, x1, y1, x2, y2, source, scale, points=points)

    @staticmethod
    def _crop_candidate(image, candidate: dict, expanded_box: tuple[int, int, int, int]):
        points = candidate.get("points")
        if points is not None:
            crop = PlateDetectorService._crop_rotated_region(image, points)
            if crop is not None and crop.size > 0:
                return crop

        x1, y1, x2, y2 = expanded_box
        return image[y1:y2, x1:x2]

    @staticmethod
    def _crop_rotated_region(image, points):
        ordered_points = PlateDetectorService._order_points(points.astype("float32"))
        top_width = np.linalg.norm(ordered_points[1] - ordered_points[0])
        bottom_width = np.linalg.norm(ordered_points[2] - ordered_points[3])
        left_height = np.linalg.norm(ordered_points[3] - ordered_points[0])
        right_height = np.linalg.norm(ordered_points[2] - ordered_points[1])
        width = int(max(top_width, bottom_width))
        height = int(max(left_height, right_height))

        if width <= 0 or height <= 0:
            return None

        destination = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype="float32",
        )
        transform = cv2.getPerspectiveTransform(ordered_points, destination)
        crop = cv2.warpPerspective(image, transform, (width, height))
        if crop.shape[0] > crop.shape[1]:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        return crop

    @staticmethod
    def _order_points(points):
        ordered = np.zeros((4, 2), dtype="float32")
        point_sum = points.sum(axis=1)
        point_diff = np.diff(points, axis=1)
        ordered[0] = points[np.argmin(point_sum)]
        ordered[2] = points[np.argmax(point_sum)]
        ordered[1] = points[np.argmin(point_diff)]
        ordered[3] = points[np.argmax(point_diff)]
        return ordered

    def _deduplicate_candidates(self, candidates: list[dict]) -> list[dict]:
        sorted_candidates = sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)
        selected = []

        for candidate in sorted_candidates:
            if all(
                self._box_iou(candidate["box"], existing["box"]) < self.FALLBACK_NMS_IOU_THRESHOLD
                for existing in selected
            ):
                selected.append(candidate)
                if len(selected) >= settings.FALLBACK_MAX_OCR_CANDIDATES:
                    break

        return selected

    @staticmethod
    def _box_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
        left_x1, left_y1, left_x2, left_y2 = left
        right_x1, right_y1, right_x2, right_y2 = right
        intersection_x1 = max(left_x1, right_x1)
        intersection_y1 = max(left_y1, right_y1)
        intersection_x2 = min(left_x2, right_x2)
        intersection_y2 = min(left_y2, right_y2)
        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height
        if intersection_area == 0:
            return 0.0

        left_area = max(0, left_x2 - left_x1) * max(0, left_y2 - left_y1)
        right_area = max(0, right_x2 - right_x1) * max(0, right_y2 - right_y1)
        return intersection_area / max(left_area + right_area - intersection_area, 1)

    @staticmethod
    def _touches_image_edge(x1: int, y1: int, x2: int, y2: int, image_width: int, image_height: int) -> bool:
        edge_margin = 3
        return (
            x1 <= edge_margin
            or y1 <= edge_margin
            or x2 >= image_width - edge_margin
            or y2 >= image_height - edge_margin
        )

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

    def _find_bright_plate_candidates(self, image, scale: float = 1.0, source: str = "bright_region") -> list[dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

        image_height, image_width = image.shape[:2]
        image_area = image_height * image_width
        candidates = []

        for mask in (threshold, closed):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[: settings.FALLBACK_MAX_CONTOURS]
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
                    candidates.append(
                        self._candidate_from_scaled(
                            score,
                            x,
                            y,
                            x + width,
                            y + height,
                            f"{source}:bright_region",
                            scale,
                        )
                    )

        return candidates

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

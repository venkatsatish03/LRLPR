import re
from pathlib import Path

import cv2
import easyocr
import numpy as np
from fastapi import HTTPException, status

from app.core.config import settings
from app.services.indian_plate_validator import IndianPlateValidationEngine


class OCRService:
    LOW_CONFIDENCE_THRESHOLD = 0.8
    MAX_VARIANT_CANDIDATES = 8
    MAX_ALTERNATE_CANDIDATES = 5
    ALL_VARIANT_NAMES = [
        "original_resized",
        "grayscale",
        "clahe",
        "denoised",
        "contrast_enhanced",
        "gamma_brightened",
        "glare_compressed",
        "sharpened",
        "adaptive_threshold",
        "adaptive_threshold_inverted",
        "sharpened_adaptive_threshold",
        "otsu_threshold",
        "closed_threshold",
    ]

    def __init__(
        self,
        languages: str = settings.OCR_LANGUAGES,
        gpu: bool = settings.OCR_GPU,
    ) -> None:
        self.languages = [language.strip() for language in languages.split(",") if language.strip()]
        self.gpu = gpu
        self._reader: easyocr.Reader | None = None
        self.plate_engine = IndianPlateValidationEngine()

    @property
    def reader(self) -> easyocr.Reader:
        if self._reader is None:
            try:
                self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"EasyOCR failed to initialize: {exc}",
                ) from exc
        return self._reader

    def extract_text(self, image_path: str | Path) -> dict:
        image_path = Path(image_path)
        if not image_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cropped plate image not found: {image_path}",
            )

        image = cv2.imread(str(image_path))
        if image is None:
            return {"text": "", "ocr_confidence": 0.0, "candidates": []}

        return self.extract_text_from_image(image)

    def extract_text_from_image(self, image: np.ndarray) -> dict:
        candidate_votes = []
        variants_processed = 0

        for variant_name, variant in self._iter_selected_ocr_variants(image):
            results = self.reader.readtext(
                variant,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1,
                paragraph=False,
            )
            variants_processed += 1
            parsed_result = self._parse_results(results)
            for rank, candidate in enumerate(parsed_result["candidates"][: self.MAX_VARIANT_CANDIDATES]):
                candidate_votes.append(
                    {
                        "plate": str(candidate["plate"]),
                        "confidence": float(candidate["confidence"]),
                        "variant": variant_name,
                        "rank": rank,
                    }
                )

            ranked_candidates = self._vote_candidates(candidate_votes)
            if self._should_stop_early(ranked_candidates, variants_processed):
                return self._format_response(ranked_candidates)

        ranked_candidates = self._vote_candidates(candidate_votes)
        return self._format_response(ranked_candidates)

    def _should_stop_early(self, ranked_candidates: list[dict[str, str | float]], variants_processed: int) -> bool:
        if variants_processed < settings.OCR_MIN_VARIANTS_BEFORE_EARLY_EXIT:
            return False

        if not ranked_candidates:
            return False

        return float(ranked_candidates[0]["confidence"]) >= settings.OCR_EARLY_EXIT_CONFIDENCE

    def _format_response(self, ranked_candidates: list[dict[str, str | float]]) -> dict:
        if not ranked_candidates:
            return {"text": "", "ocr_confidence": 0.0, "candidates": []}

        best_candidate = ranked_candidates[0]
        alternates = [
            candidate
            for candidate in ranked_candidates[1 : self.MAX_ALTERNATE_CANDIDATES + 1]
            if candidate["plate"] != best_candidate["plate"]
        ]

        return {
            "text": best_candidate["plate"],
            "ocr_confidence": best_candidate["confidence"],
            "candidates": alternates,
        }

    def _parse_results(self, results) -> dict:
        text_parts = []
        confidences = []
        for text, confidence in self._ordered_plate_tokens(results):
            cleaned_text = self._normalize_plate_text(text)
            if cleaned_text:
                text_parts.append(cleaned_text)
                confidences.append(float(confidence))

        if not text_parts:
            return {
                "text": "",
                "ocr_confidence": 0.0,
                "candidates": [],
            }

        joined_text = "".join(text_parts)
        average_confidence = round(sum(confidences) / len(confidences), 4)
        candidates = self._generate_plate_candidates(joined_text, average_confidence)
        plate_text = candidates[0]["plate"] if candidates else ""
        if not plate_text:
            return {
                "text": "",
                "ocr_confidence": 0.0,
                "candidates": candidates,
            }

        return {
            "text": plate_text,
            "ocr_confidence": candidates[0]["confidence"],
            "candidates": candidates,
        }

    @classmethod
    def _ordered_plate_tokens(cls, results) -> list[tuple[str, float]]:
        tokens = []
        for box, text, confidence in results:
            cleaned_text = cls._normalize_plate_text(text)
            if not cleaned_text or cls._is_country_marker(cleaned_text):
                continue

            points = np.array(box, dtype=np.float32)
            center_x = float(points[:, 0].mean())
            center_y = float(points[:, 1].mean())
            tokens.append(
                {
                    "text": cleaned_text,
                    "confidence": float(confidence),
                    "center_x": center_x,
                    "center_y": center_y,
                }
            )

        tokens.sort(key=lambda token: (token["center_y"], token["center_x"]))
        return [(str(token["text"]), float(token["confidence"])) for token in tokens]

    @staticmethod
    def _is_country_marker(text: str) -> bool:
        return text in {"IND", "IIND", "IN", "ND"}

    @classmethod
    def _build_ocr_variants(cls, image: np.ndarray) -> list[tuple[str, np.ndarray]]:
        return list(cls._iter_ocr_variants(image, cls.ALL_VARIANT_NAMES))

    @classmethod
    def _iter_selected_ocr_variants(cls, image: np.ndarray):
        yield from cls._iter_ocr_variants(image, cls._selected_variant_names())

    @classmethod
    def _iter_ocr_variants(cls, image: np.ndarray, variant_names: list[str]):
        image = cls._ensure_bgr(image)
        resized = cls._resize_for_ocr(image)
        gray_image = None
        clahe_image = None
        denoised_image = None
        contrast_image = None
        sharpened_image = None
        adaptive_image = None
        otsu_image = None

        def gray():
            nonlocal gray_image
            if gray_image is None:
                gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            return gray_image

        def clahe():
            nonlocal clahe_image
            if clahe_image is None:
                clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                clahe_image = clahe_filter.apply(gray())
            return clahe_image

        def denoised():
            nonlocal denoised_image
            if denoised_image is None:
                denoised_image = cv2.bilateralFilter(clahe(), 9, 75, 75)
            return denoised_image

        def contrast_enhanced():
            nonlocal contrast_image
            if contrast_image is None:
                contrast_image = cv2.convertScaleAbs(gray(), alpha=1.35, beta=12)
            return contrast_image

        def sharpened():
            nonlocal sharpened_image
            if sharpened_image is None:
                sharpened_image = cls._sharpen(contrast_enhanced())
            return sharpened_image

        def adaptive_threshold():
            nonlocal adaptive_image
            if adaptive_image is None:
                adaptive_image = cv2.adaptiveThreshold(
                    denoised(),
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31,
                    7,
                )
            return adaptive_image

        def otsu_threshold():
            nonlocal otsu_image
            if otsu_image is None:
                _, otsu_image = cv2.threshold(denoised(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return otsu_image

        builders = {
            "original_resized": lambda: resized,
            "grayscale": lambda: cls._gray_to_bgr(gray()),
            "clahe": lambda: cls._gray_to_bgr(clahe()),
            "denoised": lambda: cls._gray_to_bgr(denoised()),
            "contrast_enhanced": lambda: cls._gray_to_bgr(contrast_enhanced()),
            "gamma_brightened": lambda: cls._gray_to_bgr(cls._gamma_correct(gray(), gamma=0.65)),
            "glare_compressed": lambda: cls._gray_to_bgr(cls._compress_gray_highlights(gray())),
            "sharpened": lambda: cls._gray_to_bgr(sharpened()),
            "adaptive_threshold": lambda: cls._gray_to_bgr(adaptive_threshold()),
            "adaptive_threshold_inverted": lambda: cls._gray_to_bgr(cv2.bitwise_not(adaptive_threshold())),
            "sharpened_adaptive_threshold": lambda: cls._gray_to_bgr(
                cv2.adaptiveThreshold(
                    sharpened(),
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31,
                    5,
                )
            ),
            "otsu_threshold": lambda: cls._gray_to_bgr(otsu_threshold()),
            "closed_threshold": lambda: cls._gray_to_bgr(
                cv2.morphologyEx(
                    otsu_threshold(),
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                )
            ),
        }

        for variant_name in variant_names:
            builder = builders.get(variant_name)
            if builder is not None:
                yield variant_name, builder()

    @classmethod
    def _selected_variant_names(cls) -> list[str]:
        configured = settings.OCR_PREPROCESSING_STRATEGIES.strip()
        if configured.lower() == "all":
            return cls.ALL_VARIANT_NAMES[: settings.OCR_MAX_VARIANTS]

        selected_names = [
            name.strip()
            for name in configured.split(",")
            if name.strip() in cls.ALL_VARIANT_NAMES
        ]
        return (selected_names or ["original_resized", "grayscale", "clahe"])[: settings.OCR_MAX_VARIANTS]

    @staticmethod
    def _resize_for_ocr(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = 1
        if height < settings.OCR_TARGET_HEIGHT:
            scale = min(settings.OCR_MAX_UPSCALE, max(1, round(settings.OCR_TARGET_HEIGHT / max(height, 1))))

        target_width = max(1, int(width * scale))
        target_height = max(1, int(height * scale))
        if target_width > settings.OCR_MAX_IMAGE_WIDTH:
            shrink_ratio = settings.OCR_MAX_IMAGE_WIDTH / target_width
            target_width = settings.OCR_MAX_IMAGE_WIDTH
            target_height = max(1, int(target_height * shrink_ratio))

        if target_width == width and target_height == height:
            return image

        interpolation = cv2.INTER_CUBIC if target_width >= width else cv2.INTER_AREA
        return cv2.resize(image, (target_width, target_height), interpolation=interpolation)

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image

    @staticmethod
    def _gray_to_bgr(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _sharpen(image: np.ndarray) -> np.ndarray:
        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ]
        )
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def _gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
        gamma = max(gamma, 0.01)
        table = np.array([((value / 255.0) ** gamma) * 255 for value in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def _compress_gray_highlights(image: np.ndarray) -> np.ndarray:
        compressed = image.astype(np.float32)
        highlight_mask = compressed > 220
        compressed[highlight_mask] = 220 + ((compressed[highlight_mask] - 220) * 0.35)
        return np.clip(compressed, 0, 255).astype(np.uint8)

    @staticmethod
    def _vote_candidates(candidate_votes: list[dict[str, str | float | int]]) -> list[dict[str, str | float]]:
        candidate_stats: dict[str, dict] = {}

        for vote in candidate_votes:
            plate = str(vote["plate"])
            confidence = float(vote["confidence"])
            rank = int(vote["rank"])
            variant = str(vote["variant"])

            if not plate:
                continue

            rank_weight = max(0.35, 1.0 - (rank * 0.12))
            weighted_confidence = confidence * rank_weight
            stats = candidate_stats.setdefault(
                plate,
                {
                    "weighted_sum": 0.0,
                    "weight": 0.0,
                    "max_confidence": 0.0,
                    "variants": set(),
                },
            )
            stats["weighted_sum"] += weighted_confidence
            stats["weight"] += rank_weight
            stats["max_confidence"] = max(stats["max_confidence"], confidence)
            stats["variants"].add(variant)

        ranked_candidates = []
        for plate, stats in candidate_stats.items():
            variant_count = len(stats["variants"])
            average_confidence = stats["weighted_sum"] / max(stats["weight"], 0.01)
            agreement_bonus = min(max(variant_count - 1, 0), 4) * 0.04
            confidence = min(
                1.0,
                (average_confidence * 0.75) + (stats["max_confidence"] * 0.25) + agreement_bonus,
            )
            ranking_score = confidence + min(max(variant_count - 1, 0), 4) * 0.08
            ranked_candidates.append(
                {
                    "plate": plate,
                    "confidence": round(confidence, 4),
                    "_score": ranking_score,
                    "_variant_count": variant_count,
                }
            )

        ranked_candidates.sort(
            key=lambda candidate: (
                candidate["_score"],
                candidate["_variant_count"],
                candidate["confidence"],
                len(candidate["plate"]),
            ),
            reverse=True,
        )

        return [
            {
                "plate": str(candidate["plate"]),
                "confidence": float(candidate["confidence"]),
            }
            for candidate in ranked_candidates
        ]

    @staticmethod
    def _normalize_plate_text(text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def _generate_plate_candidates(self, text: str, base_confidence: float) -> list[dict[str, str | float]]:
        result = self.plate_engine.correct(text, base_confidence)
        best_candidate = str(result["best_candidate"])
        if not best_candidate:
            return []

        return [
            {"plate": best_candidate, "confidence": float(result["confidence"])},
            *result["alternatives"],
        ]

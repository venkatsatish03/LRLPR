import re
from pathlib import Path
import threading
import time

import cv2
import easyocr
import numpy as np
from fastapi import HTTPException, status

from app.core.config import settings
from app.services.candidate_generator import PlateCandidateGenerator
from app.services.image_enhancement import ImageEnhancementService, get_image_enhancement_service


_reader_cache: dict[tuple[tuple[str, ...], bool], easyocr.Reader] = {}
_reader_lock = threading.Lock()


class OCRService:
    LOW_CONFIDENCE_THRESHOLD = 0.8
    SHARP_VARIANCE_THRESHOLD = 200.0
    MILD_BLUR_VARIANCE_THRESHOLD = 80.0
    GRAYSCALE_RETRY_CONFIDENCE = 0.5
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
    INDIAN_STATE_CODES = {
        "AN",
        "AP",
        "AR",
        "AS",
        "BR",
        "CG",
        "CH",
        "DD",
        "DL",
        "DN",
        "GA",
        "GJ",
        "HP",
        "HR",
        "JH",
        "JK",
        "KA",
        "KL",
        "LA",
        "LD",
        "MH",
        "ML",
        "MN",
        "MP",
        "MZ",
        "NL",
        "OD",
        "OR",
        "PB",
        "PY",
        "RJ",
        "SK",
        "TN",
        "TR",
        "TS",
        "UK",
        "UP",
        "WB",
    }

    def __init__(
        self,
        languages: str = settings.OCR_LANGUAGES,
        gpu: bool = settings.OCR_GPU,
        enhancement_service: ImageEnhancementService | None = None,
    ) -> None:
        self.languages = [language.strip() for language in languages.split(",") if language.strip()]
        self.gpu = bool(gpu)
        self._reader: easyocr.Reader | None = None
        self.enhancement_service = enhancement_service or get_image_enhancement_service()
        self.candidate_generator = PlateCandidateGenerator()

    @property
    def reader(self) -> easyocr.Reader:
        if self._reader is None:
            self.initialize_reader()
        return self._reader

    def initialize_reader(self) -> None:
        cache_key = (tuple(self.languages), self.gpu)
        if cache_key in _reader_cache:
            self._reader = _reader_cache[cache_key]
            return

        with _reader_lock:
            if cache_key not in _reader_cache:
                try:
                    _reader_cache[cache_key] = easyocr.Reader(self.languages, gpu=self.gpu)
                except Exception as exc:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"EasyOCR failed to initialize: {exc}",
                    ) from exc

            self._reader = _reader_cache[cache_key]

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

    def extract_text_from_image(self, image: np.ndarray, include_debug: bool = False) -> dict:
        analysis = self.analyze_plate_crop(image)
        response = {
            "text": analysis["text"],
            "ocr_confidence": analysis["ocr_confidence"],
            "candidates": analysis["candidates"],
        }
        if include_debug:
            response["debug"] = analysis

        return response

    def analyze_plate_crop(self, image: np.ndarray) -> dict:
        started_at = time.perf_counter()
        crop_bgr = self._trim_crop_edges(self._ensure_bgr(image))
        quality_score = self.compute_quality_score(crop_bgr)
        prepared_bgr, enhancement_path = self._prepare_crop_for_ocr(crop_bgr, quality_score)

        # Stage 1: EasyOCR reads color RGB first; clear plates should not be thresholded.
        color_result = self._read_plate_variant(prepared_bgr, "color", input_is_bgr=True)
        attempts = [color_result]
        best_result = color_result

        # Stage 2: only retry on grayscale if the color read is weak; this avoids doubling OCR cost.
        if float(color_result["ocr_confidence"]) < self.GRAYSCALE_RETRY_CONFIDENCE:
            gray_image = cv2.cvtColor(prepared_bgr, cv2.COLOR_BGR2GRAY)
            gray_result = self._read_plate_variant(gray_image, "grayscale", input_is_bgr=False)
            attempts.append(gray_result)
            best_result = self._best_ocr_result(attempts)

        return {
            "text": best_result["text"],
            "ocr_confidence": best_result["ocr_confidence"],
            "candidates": best_result["candidates"],
            "quality_score": round(float(quality_score), 2),
            "enhancement_path": enhancement_path,
            "raw_ocr": [attempt["raw_ocr"] for attempt in attempts],
            "variant": best_result["variant"],
            "processing_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }

    def _prepare_crop_for_ocr(self, crop_bgr: np.ndarray, quality_score: float) -> tuple[np.ndarray, str]:
        height, width = crop_bgr.shape[:2]

        if quality_score > self.SHARP_VARIANCE_THRESHOLD:
            # Sharp crops preserve the original color signal; preprocessing can damage readable glyphs.
            return crop_bgr, "skipped_sharp_color"

        if quality_score >= self.MILD_BLUR_VARIANCE_THRESHOLD:
            # Mild blur gets local contrast and edge boost, but no destructive binarization.
            return self._apply_mild_preprocessing(crop_bgr), "mild_clahe_sharpen"

        if self.enhancement_service.should_use_super_resolution(crop_bgr):
            # Heavy blur on tiny crops is the only path that pays the Real-ESRGAN cost.
            enhanced, method, _ = self.enhancement_service.enhance_plate_crop_with_metadata(
                crop_bgr,
                allow_realesrgan=True,
                allow_opencv_fallback=True,
            )
            return enhanced, method

        # Large but blurry crops skip super-resolution and use cheap local enhancement.
        return self._apply_mild_preprocessing(crop_bgr), f"large_low_quality_preprocess_{width}x{height}"

    def _read_plate_variant(self, image: np.ndarray, variant: str, input_is_bgr: bool) -> dict:
        easyocr_image = self._to_easyocr_image(image, input_is_bgr=input_is_bgr)
        results = self.reader.readtext(
            easyocr_image,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
            paragraph=False,
            min_size=10,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=0.6,
            low_text=0.3,
        )
        parsed = self._parse_results(results)
        parsed["variant"] = variant
        parsed["raw_ocr"] = {
            "variant": variant,
            "items": self._format_raw_ocr(results),
        }
        return parsed

    @staticmethod
    def _best_ocr_result(results: list[dict]) -> dict:
        return max(
            results,
            key=lambda result: (
                float(result["ocr_confidence"]),
                len(str(result["text"])),
            ),
        )

    @staticmethod
    def compute_quality_score(image: np.ndarray) -> float:
        bgr = OCRService._ensure_bgr(image)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _apply_mild_preprocessing(image: np.ndarray) -> np.ndarray:
        # CLAHE on LAB lightness improves weak plate/text contrast without discarding color detail.
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lightness)
        contrast_bgr = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

        # A light unsharp mask restores glyph edges for mildly blurred crops.
        blurred = cv2.GaussianBlur(contrast_bgr, (0, 0), 1.2)
        return cv2.addWeighted(contrast_bgr, 1.35, blurred, -0.35, 0)

    @staticmethod
    def _to_easyocr_image(image: np.ndarray, input_is_bgr: bool) -> np.ndarray:
        if image.ndim == 2:
            return image

        if input_is_bgr:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    @staticmethod
    def _format_raw_ocr(results) -> list[dict]:
        raw_results = []
        for box, text, confidence in results:
            raw_results.append(
                {
                    "box": [[round(float(x), 2), round(float(y), 2)] for x, y in box],
                    "text": str(text),
                    "confidence": round(float(confidence), 4),
                }
            )

        return raw_results

    @staticmethod
    def _build_binary_ocr_image(image: np.ndarray) -> np.ndarray:
        # Stage 1: convert enhanced color crop to grayscale for thresholding.
        gray = cv2.cvtColor(OCRService._ensure_bgr(image), cv2.COLOR_BGR2GRAY)

        # Stage 2: adaptive threshold separates dark glyphs from uneven plate backgrounds.
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )

        # Stage 3: opening removes isolated threshold noise before EasyOCR.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def _trim_crop_edges(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if height < 24 or width < 60:
            return image

        trim_x = max(1, int(width * 0.02))
        trim_y = max(1, int(height * 0.04))
        if width - (trim_x * 2) < 40 or height - (trim_y * 2) < 16:
            return image

        # YOLO crops often include a thin border of bumper/background that pulls OCR away from glyphs.
        return image[trim_y : height - trim_y, trim_x : width - trim_x]

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
        char_confidences = []
        for text, confidence in self._ordered_plate_tokens(results):
            cleaned_text = self._normalize_plate_text(text)
            if cleaned_text:
                text_parts.append(cleaned_text)
                token_confidence = float(confidence)
                confidences.append(token_confidence)
                char_confidences.extend([token_confidence] * len(cleaned_text))

        if not text_parts:
            return {
                "text": "",
                "ocr_confidence": 0.0,
                "candidates": [],
            }

        joined_text = "".join(text_parts)
        country_prefix_length = self._embedded_country_prefix_length(joined_text)
        if country_prefix_length:
            joined_text = joined_text[country_prefix_length:]
            char_confidences = char_confidences[country_prefix_length:]

        average_confidence = round(sum(confidences) / len(confidences), 4)
        candidates = self._generate_plate_candidates(joined_text, average_confidence, char_confidences)
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
            height = float(points[:, 1].max() - points[:, 1].min())
            tokens.append(
                {
                    "text": cleaned_text,
                    "confidence": float(confidence),
                    "center_x": center_x,
                    "center_y": center_y,
                    "height": height,
                }
            )

        rows = cls._group_tokens_into_rows(tokens)
        ordered_tokens = []
        for row in rows:
            ordered_tokens.extend(cls._clean_ordered_row_tokens(sorted(row, key=lambda token: token["center_x"])))

        return [(str(token["text"]), float(token["confidence"])) for token in ordered_tokens]

    @staticmethod
    def _is_country_marker(text: str) -> bool:
        return text in {"IND", "IIND", "IN", "ND"}

    @staticmethod
    def _group_tokens_into_rows(tokens: list[dict]) -> list[list[dict]]:
        rows: list[list[dict]] = []
        for token in sorted(tokens, key=lambda item: item["center_y"]):
            if rows:
                row_center = sum(float(item["center_y"]) for item in rows[-1]) / len(rows[-1])
                row_height = max(float(item["height"]) for item in rows[-1])
                tolerance = max(12.0, max(row_height, float(token["height"])) * 0.45)
                if abs(float(token["center_y"]) - row_center) <= tolerance:
                    rows[-1].append(token)
                    continue

            rows.append([token])

        return rows

    @staticmethod
    def _clean_ordered_row_tokens(tokens: list[dict]) -> list[dict]:
        for index, token in enumerate(tokens[:-1]):
            text = str(token["text"])
            next_text = str(tokens[index + 1]["text"])
            if (
                len(text) >= 2
                and text[-1] in {"I", "1"}
                and text[:-1].isalpha()
                and next_text.isdigit()
                and len(next_text) >= 3
            ):
                token = dict(token)
                token["text"] = text[:-1]
                tokens[index] = token

        return tokens

    @classmethod
    def _embedded_country_prefix_length(cls, text: str) -> int:
        cleaned_text = cls._normalize_plate_text(text)
        if len(cleaned_text) < 8:
            return 0

        if cleaned_text.startswith("IND") and cleaned_text[3:5] in cls.INDIAN_STATE_CODES:
            return 3

        if cleaned_text[0] in {"I", "1"} and cleaned_text[1:3] in cls.INDIAN_STATE_CODES:
            return 1

        return 0

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

    def _generate_plate_candidates(
        self,
        text: str,
        base_confidence: float,
        char_confidences: list[float] | None = None,
    ) -> list[dict[str, str | float]]:
        cleaned_text = self._normalize_plate_text(text)
        if not cleaned_text:
            return []

        # Candidates now come from one-character OCR-confusion substitutions, never trailing deletion.
        return self.candidate_generator.generate(
            cleaned_text,
            base_confidence,
            char_confidences=char_confidences,
            max_candidates=self.MAX_ALTERNATE_CANDIDATES,
            regex_boost_enabled=True,
        )

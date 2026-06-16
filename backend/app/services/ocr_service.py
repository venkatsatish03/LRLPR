import re
from itertools import product
from pathlib import Path

import cv2
import easyocr
import numpy as np
from fastapi import HTTPException, status

from app.core.config import settings


class OCRService:
    LOW_CONFIDENCE_THRESHOLD = 0.8
    INDIAN_PLATE_PATTERN = re.compile(r"[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}")

    def __init__(
        self,
        languages: str = settings.OCR_LANGUAGES,
        gpu: bool = settings.OCR_GPU,
    ) -> None:
        self.languages = [language.strip() for language in languages.split(",") if language.strip()]
        self.gpu = gpu
        self._reader: easyocr.Reader | None = None

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
        best_result = {"text": "", "ocr_confidence": 0.0, "candidates": []}
        candidate_scores: dict[str, float] = {}

        for variant in self._build_ocr_variants(image):
            results = self.reader.readtext(
                variant,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1,
                paragraph=False,
            )
            parsed_result = self._parse_results(results)
            for candidate in parsed_result["candidates"]:
                plate = str(candidate["plate"])
                confidence = float(candidate["confidence"])
                candidate_scores[plate] = max(candidate_scores.get(plate, 0.0), confidence)

            if self._score_result(parsed_result) > self._score_result(best_result):
                best_result = parsed_result

        sorted_candidates = self._sort_candidates(candidate_scores)
        if best_result["text"] and best_result["ocr_confidence"] < self.LOW_CONFIDENCE_THRESHOLD:
            best_result["candidates"] = sorted_candidates[:5]
        elif not best_result["text"]:
            best_result["candidates"] = sorted_candidates[:5]
        else:
            best_result["candidates"] = []

        return best_result

    def _parse_results(self, results) -> dict:
        text_parts = []
        confidences = []
        for _, text, confidence in results:
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

    @staticmethod
    def _build_ocr_variants(image: np.ndarray) -> list[np.ndarray]:
        height, width = image.shape[:2]
        scale = max(2, int(160 / max(height, 1)))
        resized = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        threshold = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )
        inverted_threshold = cv2.bitwise_not(threshold)

        return [
            resized,
            cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(inverted_threshold, cv2.COLOR_GRAY2BGR),
        ]

    @staticmethod
    def _score_result(result: dict) -> float:
        text = str(result["text"])
        confidence = float(result["ocr_confidence"])
        length_score = min(len(text), 10) / 10
        return confidence + length_score

    @staticmethod
    def _normalize_plate_text(text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def _generate_plate_candidates(self, text: str, base_confidence: float) -> list[dict[str, str | float]]:
        normalized = self._normalize_plate_text(text)
        candidate_scores: dict[str, float] = {}

        direct_matches = self.INDIAN_PLATE_PATTERN.findall(normalized)
        for match in direct_matches:
            candidate_scores[match] = max(candidate_scores.get(match, 0.0), base_confidence)

        for start in range(0, max(len(normalized) - 10 + 1, 0)):
            window = normalized[start : start + 10]
            for corrected in self._expand_indian_plate_window(window):
                if self.INDIAN_PLATE_PATTERN.fullmatch(corrected):
                    correction_penalty = self._correction_penalty(window, corrected)
                    confidence = round(max(base_confidence - correction_penalty, 0.01), 4)
                    candidate_scores[corrected] = max(candidate_scores.get(corrected, 0.0), confidence)

        return self._sort_candidates(candidate_scores)

    def _expand_indian_plate_window(self, window: str) -> list[str]:
        expected_layout = "LLDDLLDDDD"
        options = []
        for char, expected in zip(window, expected_layout):
            options.append(self._character_options(char, expected))

        candidates = []
        for chars in product(*options):
            candidate = "".join(chars)
            if candidate not in candidates:
                candidates.append(candidate)
            if len(candidates) >= 40:
                break
        return candidates

    @staticmethod
    def _character_options(char: str, expected: str) -> list[str]:
        digit_map = {
            "O": ["0"],
            "Q": ["0"],
            "D": ["0"],
            "I": ["1"],
            "L": ["1"],
            "Z": ["2"],
            "S": ["5"],
            "B": ["8"],
            "G": ["6"],
            "T": ["7"],
        }
        letter_map = {
            "0": ["O", "D", "Q"],
            "1": ["I", "L"],
            "2": ["Z"],
            "5": ["S"],
            "6": ["G"],
            "8": ["B"],
            "O": ["O", "Q", "D"],
            "Q": ["Q", "O", "D"],
            "D": ["D", "O", "Q"],
        }

        if expected == "D":
            return digit_map.get(char, [char]) if char.isalpha() else [char]
        return letter_map.get(char, [char]) if char.isdigit() or char in letter_map else [char]

    @staticmethod
    def _correction_penalty(original: str, corrected: str) -> float:
        changes = sum(1 for left, right in zip(original, corrected) if left != right)
        return changes * 0.035

    @staticmethod
    def _sort_candidates(candidate_scores: dict[str, float]) -> list[dict[str, str | float]]:
        return [
            {"plate": plate, "confidence": round(confidence, 4)}
            for plate, confidence in sorted(
                candidate_scores.items(),
                key=lambda item: (item[1], len(item[0])),
                reverse=True,
            )
        ]

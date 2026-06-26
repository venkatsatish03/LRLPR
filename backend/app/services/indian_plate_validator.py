import re
from dataclasses import dataclass
from itertools import product
from typing import Callable


@dataclass(frozen=True)
class PlateTemplate:
    name: str
    layout: str
    prior: float
    validator: Callable[[str], bool]


@dataclass(frozen=True)
class CharacterOption:
    value: str
    penalty: float
    corrected: bool


class IndianPlateValidationEngine:
    DIGIT_SLOT = "#"
    LETTER_SLOT = "L"
    MAX_CANDIDATES = 10
    MAX_TEMPLATE_EXPANSIONS = 80
    OCR_CONFUSIONS = {
        "O": "0",
        "0": "O",
        "I": "1",
        "1": "I",
        "B": "8",
        "8": "B",
        "S": "5",
        "5": "S",
        "Z": "2",
        "2": "Z",
        "G": "6",
        "6": "G",
    }
    DIGIT_SLOT_CONFUSIONS = {
        "L": [("4", 0.06)],
        "A": [("4", 0.075)],
        "T": [("7", 0.06)],
        "Z": [("2", 0.055), ("7", 0.08)],
        "G": [("6", 0.055), ("4", 0.09)],
    }
    STATE_LETTER_CONFUSIONS = {
        0: {
            "C": [("M", 0.09)],
            "H": [("M", 0.10)],
            "N": [("M", 0.08)],
        },
        1: {
            "N": [("H", 0.08)],
        },
    }
    STATE_CODES = {
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
    STATE_RTO_LIMITS = {
        "CH": (1, 4),
    }
    STANDARD_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$")
    BH_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
    DIPLOMATIC_PATTERN = re.compile(r"^[0-9]{1,3}(CD|CC|UN|IOD)[0-9]{1,4}$")

    def __init__(self) -> None:
        self.templates = self._build_templates()

    def correct(self, text: str, base_confidence: float = 1.0) -> dict[str, str | float | list[dict[str, str | float]]]:
        candidates = self.generate_candidates(text, base_confidence)
        if not candidates:
            return {
                "best_candidate": "",
                "confidence": 0.0,
                "alternatives": [],
            }

        best_candidate = candidates[0]
        return {
            "best_candidate": best_candidate["plate"],
            "confidence": best_candidate["confidence"],
            "alternatives": candidates[1 : self.MAX_CANDIDATES],
        }

    def generate_candidates(self, text: str, base_confidence: float = 1.0) -> list[dict[str, str | float]]:
        normalized = self.normalize(text)
        if not normalized:
            return []

        base_confidence = self._clamp(base_confidence)
        candidate_scores: dict[str, float] = {}

        for template in self.templates:
            if len(normalized) < len(template.layout):
                continue

            for start in range(0, len(normalized) - len(template.layout) + 1):
                window = normalized[start : start + len(template.layout)]
                ignored_chars = len(normalized) - len(window)
                for plate, correction_penalty in self._expand_window(window, template.layout):
                    if not template.validator(plate):
                        continue

                    confidence = self._score_candidate(
                        plate=plate,
                        template=template,
                        base_confidence=base_confidence,
                        correction_penalty=correction_penalty,
                        ignored_chars=ignored_chars,
                    )
                    candidate_scores[plate] = max(candidate_scores.get(plate, 0.0), confidence)

        return self._sort_candidates(candidate_scores)[: self.MAX_CANDIDATES]

    def validate(self, plate: str) -> bool:
        normalized = self.normalize(plate)
        return any(len(normalized) == len(template.layout) and template.validator(normalized) for template in self.templates)

    @classmethod
    def normalize(cls, text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def _build_templates(self) -> list[PlateTemplate]:
        templates = []

        for rto_length in (2, 1):
            for series_length in (2, 1, 3, 0):
                for serial_length in (4, 3, 2, 1):
                    layout = (
                        self.LETTER_SLOT * 2
                        + (self.DIGIT_SLOT * rto_length)
                        + (self.LETTER_SLOT * series_length)
                        + (self.DIGIT_SLOT * serial_length)
                    )
                    prior = self._standard_prior(rto_length, series_length, serial_length)
                    templates.append(
                        PlateTemplate(
                            name=f"standard_rto{rto_length}_series{series_length}_serial{serial_length}",
                            layout=layout,
                            prior=prior,
                            validator=self._validate_standard,
                        )
                    )

        for suffix_length in (2, 1):
            templates.append(
                PlateTemplate(
                    name=f"bharat_bh_suffix{suffix_length}",
                    layout=(self.DIGIT_SLOT * 2) + "BH" + (self.DIGIT_SLOT * 4) + (self.LETTER_SLOT * suffix_length),
                    prior=0.92 if suffix_length == 2 else 0.86,
                    validator=self._validate_bh,
                )
            )

        for mission_length in (3, 2, 1):
            for mission_type in ("CD", "CC", "UN", "IOD"):
                for serial_length in (4, 3, 2, 1):
                    templates.append(
                        PlateTemplate(
                            name=f"diplomatic_{mission_type.lower()}",
                            layout=(self.DIGIT_SLOT * mission_length) + mission_type + (self.DIGIT_SLOT * serial_length),
                            prior=0.78,
                            validator=self._validate_diplomatic,
                        )
                    )

        return templates

    @staticmethod
    def _standard_prior(rto_length: int, series_length: int, serial_length: int) -> float:
        prior = 0.95
        if rto_length == 1:
            prior -= 0.04
        if series_length == 0:
            prior -= 0.08
        elif series_length == 3:
            prior -= 0.03
        if serial_length < 4:
            prior -= (4 - serial_length) * 0.06
        return max(prior, 0.65)

    def _validate_standard(self, plate: str) -> bool:
        if not self.STANDARD_PATTERN.fullmatch(plate):
            return False

        state_code = plate[:2]
        if state_code not in self.STATE_CODES:
            return False

        rest = plate[2:]
        rto_match = re.match(r"[0-9]{1,2}", rest)
        if not rto_match:
            return False

        rto_number = int(rto_match.group())
        if rto_number <= 0:
            return False

        rto_limits = self.STATE_RTO_LIMITS.get(state_code)
        if rto_limits is not None and not (rto_limits[0] <= rto_number <= rto_limits[1]):
            return False

        remaining = rest[len(rto_match.group()) :]
        serial_match = re.search(r"[0-9]{1,4}$", remaining)
        if not serial_match:
            return False

        serial_number = int(serial_match.group())
        if serial_number <= 0:
            return False

        series = remaining[: serial_match.start()]
        if any(char in {"I", "O"} for char in series):
            return False

        return True

    def _validate_bh(self, plate: str) -> bool:
        if not self.BH_PATTERN.fullmatch(plate):
            return False

        registration_year = int(plate[:2])
        if registration_year < 21:
            return False

        serial_number = int(plate[4:8])
        if serial_number <= 0:
            return False

        suffix = plate[8:]
        return not any(char in {"I", "O"} for char in suffix)

    def _validate_diplomatic(self, plate: str) -> bool:
        match = self.DIPLOMATIC_PATTERN.fullmatch(plate)
        if not match:
            return False

        mission_code = int(plate[: match.start(1)])
        serial_number = int(plate[match.end(1) :])
        return mission_code > 0 and serial_number > 0

    def _expand_window(self, window: str, layout: str) -> list[tuple[str, float]]:
        options = []
        for index, (char, expected) in enumerate(zip(window, layout)):
            character_options = self._character_options(char, expected, index)
            if not character_options:
                return []
            options.append(character_options)

        candidates = []
        for expanded in product(*options):
            candidate = "".join(option.value for option in expanded)
            correction_penalty = sum(option.penalty for option in expanded)
            correction_penalty = self._adjust_rto_pair_penalty(window, layout, candidate, correction_penalty)
            candidates.append((candidate, correction_penalty))
            if len(candidates) >= self.MAX_TEMPLATE_EXPANSIONS:
                break

        return candidates

    @classmethod
    def _adjust_rto_pair_penalty(cls, window: str, layout: str, candidate: str, correction_penalty: float) -> float:
        if not layout.startswith(cls.LETTER_SLOT * 2 + cls.DIGIT_SLOT * 2):
            return correction_penalty

        raw_rto_pair = window[2:4]
        corrected_rto_pair = candidate[2:4]
        if raw_rto_pair in {"L7", "LZ", "G7", "GZ"} and corrected_rto_pair == "47":
            return max(0.0, correction_penalty - 0.12)

        return correction_penalty

    def _character_options(self, char: str, expected: str, index: int = -1) -> list[CharacterOption]:
        if expected == self.LETTER_SLOT:
            options = []
            if char.isalpha():
                options.append(CharacterOption(char, 0.0, False))
                for value, penalty in self.STATE_LETTER_CONFUSIONS.get(index, {}).get(char, []):
                    options.append(CharacterOption(value, penalty, True))
                return self._unique_options(options)

            corrected = self.OCR_CONFUSIONS.get(char)
            if corrected and corrected.isalpha():
                return [CharacterOption(corrected, 0.055, True)]

            return []

        if expected == self.DIGIT_SLOT:
            if char.isdigit():
                return [CharacterOption(char, 0.0, False)]

            options = []
            corrected = self.OCR_CONFUSIONS.get(char)
            if corrected and corrected.isdigit():
                options.append(CharacterOption(corrected, 0.055, True))

            for value, penalty in self.DIGIT_SLOT_CONFUSIONS.get(char, []):
                options.append(CharacterOption(value, penalty, True))

            if options:
                return self._unique_options(options)

            return []

        if char == expected:
            return [CharacterOption(char, 0.0, False)]

        corrected = self.OCR_CONFUSIONS.get(char)
        if corrected == expected:
            return [CharacterOption(expected, 0.065, True)]

        return []

    @staticmethod
    def _unique_options(options: list[CharacterOption]) -> list[CharacterOption]:
        by_value: dict[str, CharacterOption] = {}
        for option in options:
            existing = by_value.get(option.value)
            if existing is None or option.penalty < existing.penalty:
                by_value[option.value] = option
        return sorted(by_value.values(), key=lambda option: option.penalty)

    @staticmethod
    def _score_candidate(
        plate: str,
        template: PlateTemplate,
        base_confidence: float,
        correction_penalty: float,
        ignored_chars: int,
    ) -> float:
        ignored_penalty = min(ignored_chars * 0.07, 0.45)
        length_bonus = min(len(plate), 10) * 0.008
        full_coverage_bonus = 0.07 if ignored_chars == 0 and len(plate) >= 9 else 0.0
        confidence = (
            (base_confidence * template.prior)
            - correction_penalty
            - ignored_penalty
            + length_bonus
            + full_coverage_bonus
        )
        return round(max(0.01, min(confidence, 1.0)), 4)

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

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

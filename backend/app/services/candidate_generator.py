import re
from itertools import product


class PlateCandidateGenerator:
    MAX_CANDIDATES = 5
    REGEX_VALID_BOOST = 0.18
    REGEX_INVALID_PENALTY = 0.14
    STANDARD_PLATE_PATTERNS = [
        re.compile(r"^([A-Z]{2})([0-9]{2})([A-Z]{1,3})([0-9]{4})$"),
        re.compile(r"^([A-Z]{2})([0-9]{1})([A-Z]{1,3})([0-9]{4})$"),
    ]
    BH_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
    DIPLOMATIC_PATTERN = re.compile(r"^[0-9]{1,3}(CD|CC|UN)[0-9]{1,4}$")
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
    STATE_POS0_CONFUSIONS = {
        "I": [("D", 0.03), ("T", 0.05), ("J", 0.06)],
        "1": [("D", 0.03), ("T", 0.05), ("J", 0.06)],
        "J": [("D", 0.04), ("T", 0.06)],
        "T": [("D", 0.05)],
        "0": [("D", 0.04), ("A", 0.06), ("U", 0.07)],
        "O": [("D", 0.04), ("A", 0.06), ("U", 0.07)],
        "Q": [("D", 0.05), ("G", 0.06)],
        "C": [("D", 0.05), ("G", 0.06), ("M", 0.08)],
        "L": [("D", 0.04)],
        "7": [("T", 0.04), ("Z", 0.06)],
        "4": [("M", 0.05), ("A", 0.06)],
        "H": [("M", 0.06)],
        "N": [("M", 0.05), ("H", 0.06)],
        "K": [("H", 0.06)],
        "8": [("B", 0.04)],
        "P": [("B", 0.05), ("R", 0.06)],
        "R": [("P", 0.05), ("B", 0.06)],
        "U": [("V", 0.05)],
        "V": [("U", 0.05), ("W", 0.06)],
        "W": [("M", 0.06)],
    }
    STATE_POS1_CONFUSIONS = {
        "1": [("L", 0.03), ("T", 0.06)],
        "I": [("L", 0.03), ("T", 0.06)],
        "7": [("L", 0.05), ("T", 0.05)],
        "5": [("S", 0.03)],
        "8": [("S", 0.05), ("B", 0.05)],
        "Z": [("S", 0.05)],
        "4": [("H", 0.06), ("A", 0.06)],
        "0": [("D", 0.04), ("A", 0.06)],
        "O": [("D", 0.04), ("A", 0.06)],
        "6": [("G", 0.04), ("C", 0.06)],
    }
    RTO_DIGIT_SLOT_REPLACEMENTS = {
        "O": [("0", 0.025)],
        "D": [("0", 0.035)],
        "Q": [("0", 0.035)],
        "I": [("1", 0.03)],
        "L": [("1", 0.03), ("4", 0.045)],
        "S": [("5", 0.035)],
        "B": [("8", 0.035)],
        "G": [("6", 0.035)],
        "C": [("0", 0.055)],
        "T": [("7", 0.05)],
        "7": [("0", 0.05), ("1", 0.05)],
        "Z": [("7", 0.04), ("2", 0.075)],
    }
    SERIAL_DIGIT_SLOT_REPLACEMENTS = {
        **RTO_DIGIT_SLOT_REPLACEMENTS,
        "Z": [("2", 0.04), ("7", 0.08)],
    }
    SERIES_LETTER_REPLACEMENTS = {
        "0": [("Q", 0.02), ("D", 0.04), ("C", 0.06), ("G", 0.06), ("B", 0.07)],
        "O": [("Q", 0.02), ("D", 0.04), ("C", 0.06), ("G", 0.06), ("B", 0.07)],
        "Q": [("D", 0.03)],
        "D": [("Q", 0.03)],
        "1": [("L", 0.03), ("T", 0.05), ("J", 0.06)],
        "I": [("L", 0.03), ("T", 0.05), ("J", 0.06)],
        "8": [("B", 0.03)],
        "5": [("S", 0.03)],
        "2": [("Z", 0.03)],
        "6": [("G", 0.03), ("C", 0.06)],
        "H": [("A", 0.05)],
    }
    LETTER_SLOT_REPLACEMENTS = {
        "0": [("Q", 0.035), ("D", 0.075), ("O", 0.095)],
        "1": [("I", 0.045), ("L", 0.075)],
        "8": [("B", 0.035)],
        "5": [("S", 0.035)],
        "2": [("Z", 0.035)],
        "6": [("G", 0.035), ("C", 0.08)],
    }
    CONFUSIONS = {
        "0": {"O", "D", "Q"},
        "O": {"0", "D", "Q"},
        "D": {"0", "O", "Q"},
        "Q": {"0", "O", "D"},
        "1": {"I", "L", "7"},
        "I": {"1", "L", "7"},
        "L": {"1", "I", "7"},
        "7": {"1", "I", "L"},
        "2": {"Z"},
        "Z": {"2", "7"},
        "5": {"S"},
        "S": {"5"},
        "6": {"G", "C"},
        "G": {"6", "C", "9"},
        "C": {"6", "G", "0"},
        "8": {"B"},
        "B": {"8"},
        "9": {"G"},
        "U": {"V"},
        "V": {"U"},
        "M": {"N"},
        "N": {"M"},
    }
    SUBSTITUTION_PENALTIES = {
        ("O", "0"): 0.02,
        ("0", "O"): 0.03,
        ("D", "0"): 0.03,
        ("0", "D"): 0.03,
        ("Q", "0"): 0.03,
        ("0", "Q"): 0.01,
        ("C", "0"): 0.02,
        ("I", "1"): 0.02,
        ("1", "I"): 0.02,
        ("L", "1"): 0.03,
        ("1", "L"): 0.03,
        ("7", "1"): 0.04,
        ("1", "7"): 0.04,
        ("Z", "2"): 0.02,
        ("2", "Z"): 0.02,
        ("Z", "7"): 0.04,
        ("7", "Z"): 0.08,
        ("S", "5"): 0.02,
        ("5", "S"): 0.02,
        ("G", "6"): 0.02,
        ("6", "G"): 0.02,
        ("C", "6"): 0.08,
        ("6", "C"): 0.08,
        ("B", "8"): 0.02,
        ("8", "B"): 0.02,
        ("G", "9"): 0.06,
        ("9", "G"): 0.06,
        ("U", "V"): 0.04,
        ("V", "U"): 0.04,
        ("M", "N"): 0.05,
        ("N", "M"): 0.05,
    }

    def generate(
        self,
        text: str,
        base_confidence: float,
        char_confidences: list[float] | None = None,
        max_candidates: int = MAX_CANDIDATES,
        regex_boost_enabled: bool = True,
    ) -> list[dict[str, str | float]]:
        normalized = self.normalize(text)
        if not normalized:
            return []

        base_confidence = self._clamp(base_confidence)
        char_confidences = self._normalize_char_confidences(normalized, base_confidence, char_confidences)
        candidate_scores = {
            normalized: self._score_original(normalized, base_confidence, regex_boost_enabled),
        }
        for candidate, confidence in self._forced_position_candidates(
            normalized,
            base_confidence,
            char_confidences,
            regex_boost_enabled,
        ):
            candidate_scores[candidate] = max(candidate_scores.get(candidate, 0.0), confidence)
        for candidate, confidence in self._compacted_rto_candidates(normalized, base_confidence, regex_boost_enabled):
            candidate_scores[candidate] = max(candidate_scores.get(candidate, 0.0), confidence)

        has_valid_already = any(self.matches_plate_pattern(c) for c in candidate_scores)
        for index, char in enumerate(normalized):
            char_confidence = char_confidences[index]
            for replacement in sorted(self.CONFUSIONS.get(char, set())):
                candidate = normalized[:index] + replacement + normalized[index + 1 :]
                if candidate == normalized:
                    continue

                is_valid = self.matches_plate_pattern(candidate)
                if has_valid_already and not is_valid:
                    continue

                candidate_scores[candidate] = max(
                    candidate_scores.get(candidate, 0.0),
                    self._score_substitution(
                        candidate,
                        base_confidence,
                        char_confidence,
                        regex_boost_enabled,
                        original_char=char,
                        replacement_char=replacement,
                    ),
                )

        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda item: (
                self.matches_plate_pattern(item[0]),
                item[1],
                len(item[0]),
            ),
            reverse=True,
        )

        valid_items = [item for item in sorted_candidates if self.matches_plate_pattern(item[0])]
        if valid_items:
            final_items = valid_items[:max_candidates]
            if len(final_items) < max_candidates and normalized not in {p for p, _ in final_items}:
                final_items.append((normalized, candidate_scores[normalized]))
        else:
            final_items = sorted_candidates[:max_candidates]

        return [
            {"plate": plate, "confidence": round(self._clamp(confidence), 4)}
            for plate, confidence in final_items
        ]

    @classmethod
    def normalize(cls, text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    @classmethod
    def matches_plate_pattern(cls, plate: str) -> bool:
        normalized = cls.normalize(plate)
        return (
            cls._matches_standard_plate(normalized)
            or cls._matches_bh_plate(normalized)
            or cls.DIPLOMATIC_PATTERN.fullmatch(normalized) is not None
        )

    @classmethod
    def _matches_standard_plate(cls, normalized: str) -> bool:
        for pattern in cls.STANDARD_PLATE_PATTERNS:
            match = pattern.fullmatch(normalized)
            if match is None:
                continue

            state_code = match.group(1)
            rto_number = int(match.group(2))
            series = match.group(3)
            serial_number = int(match.group(4))
            return (
                state_code in cls.STATE_CODES
                and rto_number > 0
                and serial_number > 0
                and not any(character in {"I", "O"} for character in series)
            )

        return False

    @classmethod
    def _matches_bh_plate(cls, normalized: str) -> bool:
        if cls.BH_PATTERN.fullmatch(normalized) is None:
            return False

        registration_year = int(normalized[:2])
        serial_number = int(normalized[4:8])
        suffix = normalized[8:]
        return registration_year >= 21 and serial_number > 0 and not any(character in {"I", "O"} for character in suffix)

    def _state_pair_options(self, prefix: str) -> list[tuple[str, float, bool]]:
        if len(prefix) < 2:
            return []
        p0, p1 = prefix[0].upper(), prefix[1].upper()
        current = p0 + p1
        if current in self.STATE_CODES:
            return [(current, 0.0, False)]

        candidates: dict[str, float] = {}
        opts0 = [(p0, 0.0)] + self.STATE_POS0_CONFUSIONS.get(p0, [])
        opts1 = [(p1, 0.0)] + self.STATE_POS1_CONFUSIONS.get(p1, [])

        for c0, pen0 in opts0:
            for c1, pen1 in opts1:
                code = c0 + c1
                if code in self.STATE_CODES:
                    pen = pen0 + pen1
                    if code not in candidates or pen < candidates[code]:
                        candidates[code] = pen

        if p1 in ("L", "1", "I") and "DL" in self.STATE_CODES:
            candidates["DL"] = min(candidates.get("DL", 99.0), 0.03)

        if p0 in ("7", "T") and p1 in ("5", "S") and "TS" in self.STATE_CODES:
            candidates["TS"] = min(candidates.get("TS", 99.0), 0.03)

        return [(code, pen, True) for code, pen in sorted(candidates.items(), key=lambda x: x[1])]

    def _forced_position_candidates(
        self,
        normalized: str,
        base_confidence: float,
        char_confidences: list[float],
        regex_boost_enabled: bool,
    ) -> list[tuple[str, float]]:
        candidates = []
        if len(normalized) < 8:
            return []

        state_options = self._state_pair_options(normalized[:2])
        if not state_options:
            return []

        for rto_length in (2, 1):
            if rto_length == 1 and len(normalized) >= 3 and normalized[1] in {"0", "O"} and normalized[2].isdigit():
                continue

            series_length = len(normalized) - 2 - rto_length - 4
            if series_length < 1 or series_length > 3:
                continue

            slots = (
                *(["rto_digit"] * rto_length),
                *(["series_letter"] * series_length),
                *(["serial_digit"] * 4),
            )
            slot_options = [
                self._slot_options(char, slot)
                for char, slot in zip(normalized[2:], slots)
            ]
            if any(not options for options in slot_options):
                continue

            for state_code, state_pen, state_corr in state_options:
                for expanded in product(*slot_options):
                    remainder = "".join(option[0] for option in expanded)
                    candidate = state_code + remainder
                    if not self.matches_plate_pattern(candidate):
                        continue

                    corrected_indexes = [index for index, option in enumerate(expanded) if option[2]]
                    correction_penalty = state_pen + sum(float(option[1]) for option in expanded)
                    num_corrections = (1 if state_corr else 0) + len(corrected_indexes)
                    if corrected_indexes:
                        average_uncertainty = sum(
                            1.0 - char_confidences[min(index + 2, len(char_confidences) - 1)]
                            for index in corrected_indexes
                        ) / len(corrected_indexes)
                        uncertainty_bonus = average_uncertainty * 0.08
                    else:
                        uncertainty_bonus = 0.0
                    structural_bonus = 0.22 if num_corrections == 0 else max(0.12, 0.22 - (num_corrections * 0.02))
                    if rto_length == 2:
                        structural_bonus += 0.035
                    if series_length == 2:
                        structural_bonus += 0.02

                    score = base_confidence + structural_bonus + uncertainty_bonus - correction_penalty
                    score += self._rto_pair_context_bonus(normalized, candidate, rto_length)
                    if regex_boost_enabled:
                        score += self.REGEX_VALID_BOOST * 0.35
                    candidates.append((candidate, score))

        return candidates

    def _compacted_rto_candidates(
        self,
        normalized: str,
        base_confidence: float,
        regex_boost_enabled: bool,
    ) -> list[tuple[str, float]]:
        if len(normalized) != 9:
            return []

        state_code = normalized[:2]
        compacted_rto = normalized[2]
        series = normalized[3:5]
        serial = normalized[5:]
        if (
            state_code not in self.STATE_CODES
            or compacted_rto != "L"
            or not series.isalpha()
            or not serial.isdigit()
            or any(character in {"I", "O"} for character in series)
        ):
            return []

        candidate = f"{state_code}14{series}{serial}"
        if not self.matches_plate_pattern(candidate):
            return []

        score = base_confidence + 0.30
        if regex_boost_enabled:
            score += self.REGEX_VALID_BOOST * 0.35
        return [(candidate, score)]

    @staticmethod
    def _rto_pair_context_bonus(raw_plate: str, candidate: str, rto_length: int) -> float:
        if rto_length != 2 or len(raw_plate) < 4 or len(candidate) < 4:
            return 0.0

        raw_rto = raw_plate[2:4]
        corrected_rto = candidate[2:4]
        if raw_rto == "IL" and corrected_rto == "14":
            return 0.07
        if raw_rto == "0L" and corrected_rto == "01":
            return 0.06
        if raw_rto == "0L" and corrected_rto == "04":
            return -0.04
        return 0.0

    @classmethod
    def _slot_options(cls, char: str, slot: str) -> list[tuple[str, float, bool]]:
        if slot == "series_letter":
            if char in ("I", "O"):
                return [(v, pen, True) for v, pen in cls.SERIES_LETTER_REPLACEMENTS.get(char, [])]
            options = []
            if char.isalpha():
                options.append((char, 0.0, False))
            for v, pen in cls.SERIES_LETTER_REPLACEMENTS.get(char, []):
                options.append((v, pen, True))
            return options

        if slot == "rto_digit":
            if char.isdigit():
                return [(char, 0.0, False)]
            return [(value, penalty, True) for value, penalty in cls.RTO_DIGIT_SLOT_REPLACEMENTS.get(char, [])]

        if slot == "serial_digit":
            if char.isdigit():
                return [(char, 0.0, False)]
            return [(value, penalty, True) for value, penalty in cls.SERIAL_DIGIT_SLOT_REPLACEMENTS.get(char, [])]

        return []

    def _score_original(self, plate: str, base_confidence: float, regex_boost_enabled: bool) -> float:
        score = base_confidence
        if regex_boost_enabled:
            if self.matches_plate_pattern(plate):
                score += self.REGEX_VALID_BOOST * 0.7
            else:
                # Keep invalid OCR reads visible, but rank valid one-character fixes above them.
                score -= self.REGEX_INVALID_PENALTY
        return self._clamp(score)

    def _score_substitution(
        self,
        plate: str,
        base_confidence: float,
        char_confidence: float,
        regex_boost_enabled: bool,
        original_char: str,
        replacement_char: str,
    ) -> float:
        uncertainty = 1.0 - self._clamp(char_confidence)
        score = (base_confidence * 0.72) + (uncertainty * 0.24)
        if regex_boost_enabled:
            if self.matches_plate_pattern(plate):
                score += self.REGEX_VALID_BOOST
            else:
                score -= self.REGEX_INVALID_PENALTY
        score -= self._substitution_penalty(original_char, replacement_char)
        return self._clamp(score)

    @classmethod
    def _substitution_penalty(cls, original_char: str, replacement_char: str) -> float:
        return cls.SUBSTITUTION_PENALTIES.get((original_char, replacement_char), 0.08)

    @classmethod
    def _normalize_char_confidences(
        cls,
        text: str,
        base_confidence: float,
        char_confidences: list[float] | None,
    ) -> list[float]:
        if not char_confidences:
            return [base_confidence] * len(text)

        normalized_confidences = [cls._clamp(confidence) for confidence in char_confidences[: len(text)]]
        if len(normalized_confidences) < len(text):
            normalized_confidences.extend([base_confidence] * (len(text) - len(normalized_confidences)))
        return normalized_confidences

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

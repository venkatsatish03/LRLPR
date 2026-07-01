import re


class PlateCandidateGenerator:
    MAX_CANDIDATES = 5
    REGEX_VALID_BOOST = 0.18
    REGEX_INVALID_PENALTY = 0.14
    PLATE_PATTERNS = [
        re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$"),
        re.compile(r"^[A-Z]{2}[0-9]{1}[A-Z]{1,3}[0-9]{4}$"),
        re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"),
        re.compile(r"^[0-9]{1,3}(CD|CC|UN)[0-9]{1,4}$"),
    ]
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
        "Z": {"2"},
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

        for index, char in enumerate(normalized):
            char_confidence = char_confidences[index]
            for replacement in sorted(self.CONFUSIONS.get(char, set())):
                candidate = normalized[:index] + replacement + normalized[index + 1 :]
                if candidate == normalized:
                    continue

                # Replacing one weak character is plausible; deleting trailing chars was not.
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

        return [
            {"plate": plate, "confidence": round(confidence, 4)}
            for plate, confidence in sorted(
                candidate_scores.items(),
                key=lambda item: (item[1], self.matches_plate_pattern(item[0]), len(item[0])),
                reverse=True,
            )[:max_candidates]
        ]

    @classmethod
    def normalize(cls, text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    @classmethod
    def matches_plate_pattern(cls, plate: str) -> bool:
        normalized = cls.normalize(plate)
        return any(pattern.fullmatch(normalized) for pattern in cls.PLATE_PATTERNS)

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

import unittest

from app.services.candidate_generator import PlateCandidateGenerator


class PlateCandidateGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = PlateCandidateGenerator()

    def test_includes_original_read(self) -> None:
        candidates = self.generator.generate("MH47BP8265", 0.9)

        self.assertIn("MH47BP8265", {candidate["plate"] for candidate in candidates})

    def test_corrects_single_invalid_digit_position_without_truncating(self) -> None:
        candidates = self.generator.generate(
            "APC9CH1116",
            0.56,
            char_confidences=[0.9, 0.9, 0.2, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        )

        self.assertEqual(candidates[0]["plate"], "AP09CH1116")
        scores = {candidate["plate"]: candidate["confidence"] for candidate in candidates}
        self.assertGreater(scores["AP09CH1116"], scores["AP69CH1116"])
        self.assertNotIn("APC9CH111", {candidate["plate"] for candidate in candidates})

    def test_prioritizes_low_confidence_character_substitution(self) -> None:
        candidates = self.generator.generate(
            "MH12AB123B",
            0.72,
            char_confidences=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.18],
        )

        self.assertEqual(candidates[0]["plate"], "MH12AB1238")

    def test_regex_boost_ranks_valid_candidate_above_invalid_original(self) -> None:
        candidates = self.generator.generate("APC9CH1116", 0.94)

        self.assertEqual(candidates[0]["plate"], "AP09CH1116")
        self.assertIn("APC9CH1116", {candidate["plate"] for candidate in candidates})

    def test_forces_letter_to_digit_in_rto_digit_slot(self) -> None:
        candidates = self.generator.generate("MH0LCD3968", 0.44)

        self.assertEqual(candidates[0]["plate"], "MH01CD3968")

    def test_contextually_corrects_il_rto_pair_as_fourteen(self) -> None:
        candidates = self.generator.generate("MHILGN9239", 1.0)

        self.assertEqual(candidates[0]["plate"], "MH14GN9239")

    def test_recovers_compacted_rto_pair_for_two_row_plate(self) -> None:
        candidates = self.generator.generate("MHLGN9239", 1.0)

        self.assertEqual(candidates[0]["plate"], "MH14GN9239")

    def test_forces_digit_and_letter_slots_for_single_digit_rto(self) -> None:
        candidates = self.generator.generate("DLZC01939", 0.45)

        self.assertEqual(candidates[0]["plate"], "DL7CQ1939")


if __name__ == "__main__":
    unittest.main()

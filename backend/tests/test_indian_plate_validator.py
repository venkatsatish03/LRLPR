import unittest

from app.services.indian_plate_validator import IndianPlateValidationEngine


class IndianPlateValidationEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = IndianPlateValidationEngine()

    def test_standard_plate_is_valid(self) -> None:
        result = self.engine.correct("MH12AB1234", 0.9)

        self.assertEqual(result["best_candidate"], "MH12AB1234")
        self.assertGreater(result["confidence"], 0.8)

    def test_corrects_digit_positions(self) -> None:
        result = self.engine.correct("MHIZAB1234", 0.9)

        self.assertEqual(result["best_candidate"], "MH12AB1234")

    def test_corrects_letter_positions(self) -> None:
        result = self.engine.correct("KA038M1234", 0.9)

        self.assertEqual(result["best_candidate"], "KA03BM1234")

    def test_prefers_complete_plate_over_short_substring(self) -> None:
        result = self.engine.correct("CHL7BP8265", 0.94)

        self.assertEqual(result["best_candidate"], "MH47BP8265")
        self.assertNotEqual(result["best_candidate"], "PB265")

    def test_corrects_stylized_rto_digit_pair(self) -> None:
        result = self.engine.correct("NHGZBP8265", 0.77)

        self.assertEqual(result["best_candidate"], "MH47BP8265")

    def test_corrects_c_as_zero_in_digit_position(self) -> None:
        result = self.engine.correct("APC9CH1116", 0.56)

        self.assertEqual(result["best_candidate"], "AP09CH1116")

    def test_supports_bharat_series(self) -> None:
        result = self.engine.correct("218H2345AA", 0.9)

        self.assertEqual(result["best_candidate"], "21BH2345AA")

    def test_supports_diplomatic_series(self) -> None:
        result = self.engine.correct("123CD4567", 0.9)

        self.assertEqual(result["best_candidate"], "123CD4567")

    def test_rejects_unfixable_plate(self) -> None:
        result = self.engine.correct("XXYYQQQQ", 0.9)

        self.assertEqual(result["best_candidate"], "")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["alternatives"], [])


if __name__ == "__main__":
    unittest.main()

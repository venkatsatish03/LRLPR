import unittest

import numpy as np

from app.core.config import settings
from app.services.ocr_service import OCRService


class OCRVariantTest(unittest.TestCase):
    def test_builds_variants_for_degraded_plate_crops(self) -> None:
        image = np.full((24, 96, 3), 80, dtype=np.uint8)

        variants = OCRService._build_ocr_variants(image)
        names = {name for name, _ in variants}

        self.assertIn("gamma_brightened", names)
        self.assertIn("glare_compressed", names)
        self.assertIn("closed_threshold", names)
        self.assertIn("otsu_threshold", names)

    def test_selected_variants_use_fast_configuration(self) -> None:
        original_strategies = settings.OCR_PREPROCESSING_STRATEGIES
        original_max_variants = settings.OCR_MAX_VARIANTS
        try:
            settings.OCR_PREPROCESSING_STRATEGIES = "original_resized,clahe,adaptive_threshold,closed_threshold"
            settings.OCR_MAX_VARIANTS = 3
            image = np.full((24, 96, 3), 80, dtype=np.uint8)

            variants = list(OCRService._iter_selected_ocr_variants(image))
            names = [name for name, _ in variants]

            self.assertEqual(names, ["original_resized", "clahe", "adaptive_threshold"])
        finally:
            settings.OCR_PREPROCESSING_STRATEGIES = original_strategies
            settings.OCR_MAX_VARIANTS = original_max_variants

    def test_ignores_ind_marker_on_two_line_motorcycle_plate(self) -> None:
        service = OCRService()
        results = [
            ([[130, 167], [445, 167], [445, 299], [130, 299]], "KA02", 0.99),
            ([[45, 307], [120, 307], [120, 348], [45, 348]], "IND", 0.99),
            ([[136, 296], [467, 296], [467, 401], [136, 401]], "JR1207", 0.99),
        ]

        parsed = service._parse_results(results)

        self.assertEqual(parsed["text"], "KA02JR1207")
        self.assertGreater(parsed["ocr_confidence"], 0.8)

    def test_strips_country_marker_when_easyocr_merges_it_into_plate_token(self) -> None:
        service = OCRService()
        results = [
            ([[0, 0], [300, 0], [300, 60], [0, 60]], "IHR26D05551", 0.53),
        ]

        parsed = service._parse_results(results)

        self.assertEqual(parsed["text"], "HR26DQ5551")

    def test_strips_merged_ind_prefix_from_direct_plate_read(self) -> None:
        service = OCRService()
        results = [
            ([[0, 0], [300, 0], [300, 60], [0, 60]], "IKA19P8488", 0.72),
        ]

        parsed = service._parse_results(results)

        self.assertEqual(parsed["text"], "KA19P8488")

    def test_groups_same_row_tokens_before_sorting_two_line_plate(self) -> None:
        service = OCRService()
        results = [
            ([[124, 12], [436, 12], [436, 137], [124, 137]], "TS09", 0.98),
            ([[52, 124], [200, 124], [200, 246], [52, 246]], "PBI", 0.39),
            ([[251, 123], [511, 123], [511, 245], [251, 245]], "2381", 0.99),
        ]

        parsed = service._parse_results(results)

        self.assertEqual(parsed["text"], "TS09PB2381")


if __name__ == "__main__":
    unittest.main()

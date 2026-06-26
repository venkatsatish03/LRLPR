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


if __name__ == "__main__":
    unittest.main()

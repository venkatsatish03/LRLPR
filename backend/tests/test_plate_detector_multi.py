import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.services.plate_detector import PlateDetectorService


class FakeOCRService:
    def __init__(self, responses):
        self.responses = list(responses)

    def extract_text_from_image(self, crop):
        return self.responses.pop(0)


class PlateDetectorMultiDetectionTest(unittest.TestCase):
    def test_builds_and_sorts_independent_plate_detections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            service = PlateDetectorService(
                annotated_dir=tmp_path / "annotated",
                crop_dir=tmp_path / "plates",
                ocr_service=FakeOCRService(
                    [
                        {"text": "MH12AB1234", "ocr_confidence": 0.72, "candidates": []},
                        {"text": "DL1CAA1111", "ocr_confidence": 0.91, "candidates": []},
                    ]
                ),
            )
            image = np.zeros((30, 120, 3), dtype=np.uint8)

            first_detection = service._build_detection(
                image_path=tmp_path / "group.jpg",
                crop=image,
                index=0,
                confidence=0.9,
                coordinates=(0, 0, 60, 20),
            )
            second_detection = service._build_detection(
                image_path=tmp_path / "group.jpg",
                crop=image,
                index=1,
                confidence=0.86,
                coordinates=(60, 0, 120, 20),
            )

            sorted_detections = service._sort_detections([first_detection, second_detection])

            self.assertEqual(sorted_detections[0]["text"], "DL1CAA1111")
            self.assertIn("final_confidence", sorted_detections[0])
            self.assertGreater(sorted_detections[0]["final_confidence"], sorted_detections[1]["final_confidence"])

    def test_deduplicates_overlapping_candidates(self) -> None:
        service = PlateDetectorService(ocr_service=FakeOCRService([]))
        candidates = [
            service._candidate(0.9, 0, 0, 100, 30, "edge_contour"),
            service._candidate(0.7, 5, 2, 105, 32, "bright_region"),
            service._candidate(0.8, 160, 0, 260, 30, "edge_contour"),
        ]

        deduped = service._deduplicate_candidates(candidates)

        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["score"], 0.9)
        self.assertEqual(deduped[1]["score"], 0.8)

    def test_filters_unread_fragments_when_full_plate_is_read(self) -> None:
        detections = [
            {"text": "KA02JR1207", "final_confidence": 0.85, "ocr_confidence": 0.98, "confidence": 0.68},
            {"text": "", "final_confidence": 0.26, "ocr_confidence": 0.0, "confidence": 0.74},
        ]

        filtered = PlateDetectorService._filter_low_value_detections(detections)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["text"], "KA02JR1207")

    def test_rejects_badge_watermark_and_contaminated_banner_text(self) -> None:
        self.assertFalse(PlateDetectorService._has_realistic_plate_text("ZX"))
        self.assertFalse(PlateDetectorService._has_realistic_plate_text("GMHNC"))
        self.assertFalse(
            PlateDetectorService._has_realistic_plate_text(
                "360021620186713033080822E30TS09PB2381POLICEMEW8"
            )
        )
        self.assertTrue(PlateDetectorService._has_realistic_plate_text("HR26DQ5551"))

    def test_fallback_geometry_gate_uses_plate_aspect_range(self) -> None:
        service = PlateDetectorService(ocr_service=FakeOCRService([]))
        candidates = [
            service._candidate(0.9, 0, 0, 100, 100, "near_square"),
            service._candidate(0.8, 10, 10, 130, 40, "plate_shape"),
            service._candidate(0.7, 0, 0, 200, 20, "banner"),
        ]

        filtered = service._filter_candidates_by_geometry(candidates, 640, 480)

        self.assertEqual([candidate["source"] for candidate in filtered], ["plate_shape"])

    def test_corner_candidates_are_rejected_when_non_corner_candidate_exists(self) -> None:
        service = PlateDetectorService(ocr_service=FakeOCRService([]))
        candidates = [
            service._candidate(0.88, 548, 430, 628, 455, "bottom_right_watermark"),
            service._candidate(0.82, 220, 330, 380, 370, "lower_center_plate"),
        ]

        filtered = service._filter_corner_candidates(candidates, 640, 480)

        self.assertEqual([candidate["source"] for candidate in filtered], ["lower_center_plate"])

    def test_only_corner_candidate_is_kept_for_edge_case_images(self) -> None:
        service = PlateDetectorService(ocr_service=FakeOCRService([]))
        candidates = [
            service._candidate(0.88, 548, 430, 628, 455, "bottom_right_plate"),
        ]

        filtered = service._filter_corner_candidates(candidates, 640, 480)

        self.assertEqual(filtered, candidates)

    def test_low_plausibility_fallback_candidates_are_filtered_before_ocr(self) -> None:
        service = PlateDetectorService(ocr_service=FakeOCRService([]))
        candidates = [
            service._candidate(0.32, 120, 220, 260, 260, "door_panel"),
            service._candidate(0.46, 220, 330, 380, 370, "plate_panel"),
        ]

        filtered = service._filter_candidates_by_plausibility(candidates)

        self.assertEqual([candidate["source"] for candidate in filtered], ["plate_panel"])

    def test_lower_center_rescue_finds_horizontal_plate_like_region(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[330:370, 220:420] = 230
        image[342:358, 250:390] = 20

        candidate = PlateDetectorService._find_lower_center_plate_candidate(image)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["source"], "lower_center_plate_candidate")

    def test_prefers_valid_plate_detection_over_secondary_junk_text(self) -> None:
        detections = [
            {"text": "HR26DQ5551", "final_confidence": 0.79, "ocr_confidence": 0.73, "confidence": 0.87},
            {"text": "6MCHIZ", "final_confidence": 0.55, "ocr_confidence": 0.25, "confidence": 0.91},
        ]

        filtered = PlateDetectorService._filter_low_value_detections(detections)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["text"], "HR26DQ5551")

    def test_builds_condition_variants_for_degraded_images(self) -> None:
        service = PlateDetectorService(ocr_service=FakeOCRService([]))
        image = np.full((80, 160, 3), 64, dtype=np.uint8)

        variants = service._build_fallback_image_variants(image)
        sources = {variant["source"] for variant in variants}

        self.assertIn("contrast_clahe", sources)
        self.assertIn("low_light_gamma", sources)
        self.assertIn("sharpened", sources)
        self.assertNotIn("upscaled_contrast", sources)

    def test_upscaled_fallback_variant_is_opt_in(self) -> None:
        original_variants = settings.FALLBACK_PREPROCESSING_VARIANTS
        original_upscale = settings.FALLBACK_ENABLE_UPSCALE
        try:
            settings.FALLBACK_PREPROCESSING_VARIANTS = "original,upscaled_contrast"
            settings.FALLBACK_ENABLE_UPSCALE = True
            service = PlateDetectorService(ocr_service=FakeOCRService([]))
            image = np.full((80, 160, 3), 64, dtype=np.uint8)

            variants = service._build_fallback_image_variants(image)
            sources = {variant["source"] for variant in variants}

            self.assertIn("upscaled_contrast", sources)
        finally:
            settings.FALLBACK_PREPROCESSING_VARIANTS = original_variants
            settings.FALLBACK_ENABLE_UPSCALE = original_upscale

    def test_scaled_candidates_map_to_original_coordinates(self) -> None:
        service = PlateDetectorService(ocr_service=FakeOCRService([]))

        candidate = service._candidate_from_scaled(
            score=0.8,
            x1=20,
            y1=10,
            x2=120,
            y2=50,
            source="upscaled_contrast:edge_contour",
            scale=2.0,
        )

        self.assertEqual(candidate["box"], (10, 5, 60, 25))

    def test_whole_image_candidate_for_direct_plate_upload(self) -> None:
        image = np.full((120, 480, 3), 235, dtype=np.uint8)
        image[:, :70] = (180, 40, 20)
        image[35:85, 120:430] = 20

        candidate = PlateDetectorService._whole_image_plate_candidate(image, [])

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["box"], (0, 0, 480, 120))
        self.assertEqual(candidate["source"], "whole_image_plate_candidate")

    def test_whole_image_candidate_for_two_line_plate_photo(self) -> None:
        image = np.full((500, 500, 3), 245, dtype=np.uint8)
        image[150:420, 10:490] = 230
        image[185:250, 140:430] = 25
        image[310:380, 140:430] = 25

        candidate = PlateDetectorService._whole_image_plate_candidate(image, [])

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["box"], (0, 0, 500, 500))

    def test_whole_image_candidate_rejects_full_vehicle_photo_shape(self) -> None:
        image = np.zeros((705, 800, 3), dtype=np.uint8)
        image[:420, :] = 235
        image[520:565, 280:520] = 245
        image[532:552, 320:480] = 20

        candidate = PlateDetectorService._whole_image_plate_candidate(image, [])

        self.assertIsNone(candidate)

    def test_direct_plate_upload_uses_whole_image_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            service = PlateDetectorService(
                annotated_dir=tmp_path / "annotated",
                crop_dir=tmp_path / "plates",
                ocr_service=FakeOCRService(
                    [{"text": "KA19P8488", "ocr_confidence": 0.93, "candidates": []}]
                ),
            )
            image = np.full((120, 480, 3), 235, dtype=np.uint8)
            image[:, :70] = (180, 40, 20)
            image[35:85, 120:430] = 20

            result = service._detect_direct_plate_upload(tmp_path / "plate.jpg", image)

            self.assertIsNotNone(result)
            self.assertEqual(result["detector"], "direct_plate_image")
            self.assertEqual(result["plates_detected"], 1)
            self.assertEqual(result["detections"][0]["coordinates"], {"x1": 0, "y1": 0, "x2": 480, "y2": 120})

    def test_parses_yolo_class_filter_ids(self) -> None:
        self.assertEqual(PlateDetectorService._parse_class_ids("0, 2, bad, 5"), [0, 2, 5])
        self.assertIsNone(PlateDetectorService._parse_class_ids(""))

    def test_rejects_implausible_yolo_boxes(self) -> None:
        self.assertFalse(PlateDetectorService._is_plausible_yolo_box(0, 0, 4, 4, 640, 480))
        self.assertFalse(PlateDetectorService._is_plausible_yolo_box(0, 0, 300, 300, 640, 480))
        self.assertTrue(PlateDetectorService._is_plausible_yolo_box(10, 10, 130, 40, 640, 480))


if __name__ == "__main__":
    unittest.main()

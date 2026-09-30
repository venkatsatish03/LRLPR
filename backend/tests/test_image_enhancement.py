from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.services.image_enhancement import ImageEnhancementService
from app.services.ocr_service import OCRService


class FakeReader:
    def __init__(self) -> None:
        self.calls = []

    def readtext(self, image, **kwargs):
        self.calls.append((image.copy(), kwargs))
        return [([[0, 0], [120, 0], [120, 30], [0, 30]], "MH47BP8265", 0.91)]


class ManualRestorationReader:
    def __init__(self) -> None:
        self.calls = 0

    def readtext(self, image, **kwargs):
        self.calls += 1
        if self.calls <= 2:
            return [([[0, 0], [60, 0], [60, 20], [0, 20]], "PB265", 0.32)]

        return [([[0, 0], [160, 0], [160, 40], [0, 40]], "MH47BP8265", 0.88)]


class LargeCropEnhancementGuard:
    def should_use_super_resolution(self, image) -> bool:
        return False

    def enhance_plate_crop_with_metadata(self, *args, **kwargs):
        raise AssertionError("Large manual crops should skip deep super-resolution")


class ImageEnhancementTest(unittest.TestCase):
    def test_opencv_fallback_upscales_tiny_plate_crop_by_four(self) -> None:
        image = np.full((9, 22, 3), 128, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temporary_dir:
            service = ImageEnhancementService(
                enhanced_dir=Path(temporary_dir) / "enhanced",
                model_path=Path("missing/RealESRGAN_x4plus.pth"),
                scale=4,
                tile=128,
                half=False,
            )

            enhanced, method, configured = service.enhance_plate_crop_with_metadata(image)

        self.assertEqual(method, "opencv_fallback")
        self.assertFalse(configured)
        self.assertEqual(enhanced.shape, (36, 88, 3))

    def test_binary_ocr_preprocessing_keeps_enhanced_crop_dimensions(self) -> None:
        enhanced = np.full((36, 88, 3), 190, dtype=np.uint8)

        binary = OCRService._build_binary_ocr_image(enhanced)

        self.assertEqual(binary.shape, (36, 88))
        self.assertEqual(binary.dtype, np.uint8)

    def test_sharp_crop_skips_enhancement_and_uses_rgb_easyocr_input(self) -> None:
        image = np.zeros((50, 160, 3), dtype=np.uint8)
        image[:, 40:120] = (255, 255, 255)
        image[2, 6] = (255, 0, 0)

        with tempfile.TemporaryDirectory() as temporary_dir:
            enhancement_service = ImageEnhancementService(
                enhanced_dir=Path(temporary_dir) / "enhanced",
                model_path=Path("missing/RealESRGAN_x4plus.pth"),
            )
            service = OCRService(enhancement_service=enhancement_service)
            fake_reader = FakeReader()
            service._reader = fake_reader

            result = service.extract_text_from_image(image, include_debug=True)

        easyocr_image, readtext_kwargs = fake_reader.calls[0]
        self.assertEqual(result["debug"]["enhancement_path"], "skipped_sharp_color")
        self.assertEqual(easyocr_image[0, 0].tolist(), [0, 0, 255])
        self.assertEqual(readtext_kwargs["allowlist"], "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        self.assertFalse(readtext_kwargs["paragraph"])
        self.assertEqual(readtext_kwargs["min_size"], 10)
        self.assertEqual(readtext_kwargs["contrast_ths"], 0.1)
        self.assertEqual(readtext_kwargs["adjust_contrast"], 0.5)
        self.assertEqual(readtext_kwargs["text_threshold"], 0.6)
        self.assertEqual(readtext_kwargs["low_text"], 0.3)
        self.assertEqual(result["text"], "MH47BP8265")

    def test_generates_confusion_candidate_instead_of_truncated_substring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            enhancement_service = ImageEnhancementService(
                enhanced_dir=Path(temporary_dir) / "enhanced",
                model_path=Path("missing/RealESRGAN_x4plus.pth"),
            )
            service = OCRService(enhancement_service=enhancement_service)

            candidates = service._generate_plate_candidates("APC9CH1116", 0.56)

        self.assertEqual(candidates[0]["plate"], "AP09CH1116")
        self.assertNotIn("APC9CH111", {candidate["plate"] for candidate in candidates})

    def test_manual_aggressive_path_votes_across_restored_variants(self) -> None:
        crop = np.full((12, 40, 3), 120, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temporary_dir:
            enhancement_service = ImageEnhancementService(
                enhanced_dir=Path(temporary_dir) / "enhanced",
                model_path=Path("missing/RealESRGAN_x4plus.pth"),
            )
            service = OCRService(enhancement_service=enhancement_service)
            fake_reader = ManualRestorationReader()
            service._reader = fake_reader

            result = service.extract_text_from_image(crop, include_debug=True, aggressive_enhancement=True)

        self.assertGreater(fake_reader.calls, 2)
        self.assertEqual(result["text"], "MH47BP8265")
        self.assertEqual(result["debug"]["variant"], "manual_restoration_vote")
        self.assertIn("manual_", result["debug"]["enhancement_path"])

    def test_manual_path_keeps_clear_valid_direct_read(self) -> None:
        image = np.zeros((50, 160, 3), dtype=np.uint8)
        image[:, 40:120] = (255, 255, 255)

        with tempfile.TemporaryDirectory() as temporary_dir:
            enhancement_service = ImageEnhancementService(
                enhanced_dir=Path(temporary_dir) / "enhanced",
                model_path=Path("missing/RealESRGAN_x4plus.pth"),
            )
            service = OCRService(enhancement_service=enhancement_service)
            fake_reader = FakeReader()
            service._reader = fake_reader

            result = service.extract_text_from_image(image, include_debug=True, aggressive_enhancement=True)

        _, readtext_kwargs = fake_reader.calls[0]
        self.assertEqual(len(fake_reader.calls), 1)
        self.assertEqual(readtext_kwargs["decoder"], "beamsearch")
        self.assertEqual(readtext_kwargs["beamWidth"], 8)
        self.assertEqual(readtext_kwargs["min_size"], 6)
        self.assertEqual(result["text"], "MH47BP8265")
        self.assertEqual(result["debug"]["variant"], "color")
        self.assertEqual(result["debug"]["enhancement_path"], "manual_direct_color")

    def test_manual_region_normalizer_crops_loose_selection_to_plate_shape(self) -> None:
        image = np.zeros((100, 260, 3), dtype=np.uint8)
        image[38:68, 55:210] = (235, 235, 235)
        image[45:61, 78:190] = (20, 20, 20)

        normalized = OCRService._normalize_manual_plate_region(image)

        self.assertLess(normalized.shape[0], image.shape[0])
        self.assertLess(normalized.shape[1], image.shape[1])
        self.assertGreaterEqual(normalized.shape[0], 30)
        self.assertGreaterEqual(normalized.shape[1], 150)

    def test_large_manual_restoration_skips_deep_super_resolution(self) -> None:
        crop = np.full((143, 476, 3), 180, dtype=np.uint8)
        service = OCRService(enhancement_service=LargeCropEnhancementGuard())

        variants = service._manual_restoration_variants(crop)
        variant_names = {name for name, _, _ in variants}

        self.assertNotIn("manual_real_esrgan_color", variant_names)
        self.assertNotIn("manual_opencv_fallback_color", variant_names)
        self.assertIn("manual_glare_controlled", variant_names)
        self.assertIn("manual_blackhat_binary", variant_names)


if __name__ == "__main__":
    unittest.main()

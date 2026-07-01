from pathlib import Path
import logging
import sys
import threading
import types

import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)
_default_enhancement_service: "ImageEnhancementService | None" = None
_default_enhancement_lock = threading.Lock()


class ImageEnhancementService:
    def __init__(
        self,
        enhanced_dir: Path = settings.ENHANCED_DIR,
        model_path: Path = settings.REAL_ESRGAN_MODEL_PATH,
        scale: int = settings.REAL_ESRGAN_SCALE,
        tile: int = settings.REAL_ESRGAN_TILE,
        half: bool = settings.REAL_ESRGAN_HALF,
    ) -> None:
        self.enhanced_dir = enhanced_dir
        self.model_path = model_path
        self.scale = scale
        self.tile = tile
        self.half = half
        self._upsampler = None
        self._resolved_model_path = self._resolve_model_path(model_path)
        self._upsampler_lock = threading.Lock()

        self.enhanced_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_upsampler()

    def enhance(self, image_path: str | Path) -> dict[str, str | bool]:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image for enhancement: {image_path}")

        output_filename = f"{image_path.stem}_enhanced.jpg"
        output_path = self.enhanced_dir / output_filename

        enhanced, method, configured = self.enhance_plate_crop_with_metadata(image)

        cv2.imwrite(str(output_path), enhanced)

        return {
            "configured": configured,
            "method": method,
            "enhanced_image_path": str(output_path),
            "enhanced_image_url": f"/uploads/enhanced/{output_filename}",
        }

    def enhance_plate_crop(
        self,
        image: np.ndarray,
        allow_realesrgan: bool = True,
        allow_opencv_fallback: bool = True,
    ) -> np.ndarray:
        enhanced, _, _ = self.enhance_plate_crop_with_metadata(
            image,
            allow_realesrgan=allow_realesrgan,
            allow_opencv_fallback=allow_opencv_fallback,
        )
        return enhanced

    def enhance_plate_crop_with_metadata(
        self,
        image: np.ndarray,
        allow_realesrgan: bool = True,
        allow_opencv_fallback: bool = True,
    ) -> tuple[np.ndarray, str, bool]:
        image = self._ensure_bgr(image)
        if allow_realesrgan and self._upsampler is not None:
            try:
                return self._enhance_with_realesrgan(image), "real_esrgan", True
            except Exception as exc:
                logger.warning("Real-ESRGAN plate enhancement failed; using OpenCV fallback. Error: %s", exc)

        if allow_opencv_fallback:
            return self._enhance_with_opencv_fallback(image), "opencv_fallback", False

        return image, "skipped", self.is_realesrgan_configured()

    def is_realesrgan_configured(self) -> bool:
        return self._upsampler is not None

    def _enhance_with_realesrgan(self, image):
        tile = self._tile_for_crop(image)
        with self._upsampler_lock:
            previous_tile = getattr(self._upsampler, "tile", self.tile)
            self._upsampler.tile = tile
            try:
                enhanced, _ = self._upsampler.enhance(image, outscale=self.scale)
            finally:
                self._upsampler.tile = previous_tile
        return enhanced

    def _initialize_upsampler(self) -> None:
        if self._resolved_model_path is None:
            logger.warning(
                "Real-ESRGAN weights not found at %s; cropped plates will use OpenCV enhancement fallback.",
                self.model_path,
            )
            return

        try:
            self._patch_torchvision_functional_tensor()
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            import torch
        except Exception as exc:
            logger.warning("Real-ESRGAN dependencies are unavailable; using OpenCV fallback. Error: %s", exc)
            return

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.scale,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._upsampler = RealESRGANer(
                scale=self.scale,
                model_path=str(self._resolved_model_path),
                model=model,
                tile=self.tile,
                tile_pad=10,
                pre_pad=0,
                half=False,
                device=device,
            )
        except Exception as exc:
            logger.warning("Real-ESRGAN failed to initialize; using OpenCV fallback. Error: %s", exc)
            self._upsampler = None

    @staticmethod
    def _patch_torchvision_functional_tensor() -> None:
        if "torchvision.transforms.functional_tensor" in sys.modules:
            return

        from torchvision.transforms import functional

        # BasicSR 1.4.2 imports this legacy torchvision module name.
        compatibility_module = types.ModuleType("torchvision.transforms.functional_tensor")
        compatibility_module.rgb_to_grayscale = functional.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = compatibility_module

    @staticmethod
    def should_use_super_resolution(image: np.ndarray) -> bool:
        height, width = image.shape[:2]
        return width < 120 or height < 40

    def _tile_for_crop(self, image: np.ndarray) -> int:
        height, width = image.shape[:2]
        # Tiny crops are faster without Real-ESRGAN tiling; larger crops keep the configured low-VRAM tile.
        if width < 150 and height < 50:
            return 0

        return self.tile

    @staticmethod
    def _enhance_with_opencv_fallback(image):
        height, width = image.shape[:2]

        # Stage 1: upscale tiny YOLO crops by 4x with high-quality Lanczos interpolation.
        upscaled = cv2.resize(image, (width * 4, height * 4), interpolation=cv2.INTER_LANCZOS4)

        # Stage 2: remove color noise introduced by upscaling or low-light compression.
        denoised = cv2.fastNlMeansDenoisingColored(
            upscaled,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        # Stage 3: unsharp mask to bring plate glyph edges back before OCR.
        blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
        sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

        # Stage 4: CLAHE on LAB lightness to improve local text/background contrast.
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        lightness = clahe.apply(lightness)
        enhanced = cv2.merge((lightness, channel_a, channel_b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    @staticmethod
    def _resolve_model_path(model_path: Path) -> Path | None:
        backend_root = Path(__file__).resolve().parents[2]
        project_root = Path(__file__).resolve().parents[3]
        candidates = [
            model_path,
            backend_root / model_path,
            project_root / model_path,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None


def get_image_enhancement_service() -> ImageEnhancementService:
    global _default_enhancement_service
    if _default_enhancement_service is None:
        with _default_enhancement_lock:
            if _default_enhancement_service is None:
                _default_enhancement_service = ImageEnhancementService()

    return _default_enhancement_service

from pathlib import Path

import cv2
import torch

from app.core.config import settings


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

        self.enhanced_dir.mkdir(parents=True, exist_ok=True)

    def enhance(self, image_path: str | Path) -> dict[str, str | bool]:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image for enhancement: {image_path}")

        output_filename = f"{image_path.stem}_enhanced.jpg"
        output_path = self.enhanced_dir / output_filename

        if self.is_realesrgan_configured():
            enhanced = self._enhance_with_realesrgan(image)
            method = "real_esrgan"
            configured = True
        else:
            enhanced = self._enhance_with_opencv(image)
            method = "opencv_fallback"
            configured = False

        cv2.imwrite(str(output_path), enhanced)

        return {
            "configured": configured,
            "method": method,
            "enhanced_image_path": str(output_path),
            "enhanced_image_url": f"/uploads/enhanced/{output_filename}",
        }

    def is_realesrgan_configured(self) -> bool:
        return self.model_path.exists()

    def _enhance_with_realesrgan(self, image):
        upsampler = self._get_upsampler()
        enhanced, _ = upsampler.enhance(image, outscale=self.scale)
        return enhanced

    def _get_upsampler(self):
        if self._upsampler is not None:
            return self._upsampler

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as exc:
            raise RuntimeError(
                "Real-ESRGAN dependencies are not installed. Run: pip install -r requirements.txt"
            ) from exc

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.scale,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._upsampler = RealESRGANer(
            scale=self.scale,
            model_path=str(self.model_path),
            model=model,
            tile=self.tile,
            tile_pad=10,
            pre_pad=0,
            half=self.half and device == "cuda",
            device=device,
        )
        return self._upsampler

    @staticmethod
    def _enhance_with_opencv(image):
        height, width = image.shape[:2]
        scale = 2 if max(height, width) < 1800 else 1
        if scale > 1:
            image = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)
        enhanced = cv2.merge((lightness, channel_a, channel_b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)

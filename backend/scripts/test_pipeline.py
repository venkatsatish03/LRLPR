from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.ocr_service import OCRService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the crop-to-OCR pipeline on one image.")
    parser.add_argument("image_path", type=Path, help="Path to a cropped plate image or direct plate photo.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Unable to read image: {args.image_path}", file=sys.stderr)
        return 1

    started_at = time.perf_counter()
    service = OCRService()
    result = service.extract_text_from_image(image, include_debug=True)
    debug = result["debug"]

    report = {
        "quality_score": debug["quality_score"],
        "enhancement_path": debug["enhancement_path"],
        "raw_ocr_output": debug["raw_ocr"],
        "cleaned_plate": result["text"],
        "confidence": result["ocr_confidence"],
        "total_processing_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

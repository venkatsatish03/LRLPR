from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.plate_detector import PlateDetectorService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full LPR detection on one or more image files.")
    parser.add_argument("image_paths", nargs="+", type=Path, help="Image path(s) to analyze.")
    return parser.parse_args()


def summarize_detection(detection: dict) -> dict:
    return {
        "text": detection.get("text", ""),
        "final_confidence": detection.get("final_confidence", 0.0),
        "ocr_confidence": detection.get("ocr_confidence", 0.0),
        "detection_confidence": detection.get("confidence", 0.0),
        "box": detection.get("coordinates", {}),
        "candidates": detection.get("candidates", []),
    }


def main() -> int:
    args = parse_args()
    service = PlateDetectorService()
    reports = []

    for image_path in args.image_paths:
        started_at = time.perf_counter()
        try:
            result = service.detect(image_path)
            detections = result.get("detections", [])
            reports.append(
                {
                    "image": str(image_path),
                    "detector": result.get("detector"),
                    "plates_detected": result.get("plates_detected", 0),
                    "best": summarize_detection(detections[0]) if detections else None,
                    "detections": [summarize_detection(detection) for detection in detections],
                    "processing_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
                }
            )
        except Exception as exc:
            reports.append(
                {
                    "image": str(image_path),
                    "error": str(exc),
                    "processing_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
                }
            )

    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.candidate_generator import PlateCandidateGenerator  # noqa: E402


def parse_confidences(raw_confidences: str | None) -> list[float] | None:
    if not raw_confidences:
        return None

    return [
        float(value.strip())
        for value in raw_confidences.split(",")
        if value.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank LPR candidates using one-character OCR confusions.")
    parser.add_argument("plate", help="Raw OCR plate string, for example APC9CH1116")
    parser.add_argument("--confidence", type=float, default=0.6, help="Base EasyOCR confidence from 0.0 to 1.0")
    parser.add_argument(
        "--confidences",
        help="Optional comma-separated per-character confidences, same order as the raw plate string.",
    )
    parser.add_argument("--max-candidates", type=int, default=5, help="Number of ranked candidates to print")
    args = parser.parse_args()

    generator = PlateCandidateGenerator()
    candidates = generator.generate(
        args.plate,
        args.confidence,
        char_confidences=parse_confidences(args.confidences),
        max_candidates=args.max_candidates,
    )
    print(json.dumps(candidates, indent=2))


if __name__ == "__main__":
    main()

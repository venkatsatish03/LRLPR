from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REAL_ESRGAN_X4PLUS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
DEFAULT_FILENAME = "RealESRGAN_x4plus.pth"


def download_file(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"Weights already exist: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")

    request = Request(url, headers={"User-Agent": "LPR-RealESRGAN-weights-downloader"})
    try:
        with urlopen(request, timeout=120) as response, temporary_path.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break

                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = (downloaded / total) * 100
                    print(f"\rDownloading {percent:5.1f}%", end="")

        if total:
            print()
        temporary_path.replace(destination)
        print(f"Downloaded Real-ESRGAN weights to: {destination}")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(f"Failed to download Real-ESRGAN weights: {exc}") from exc


def parse_args() -> argparse.Namespace:
    backend_root = Path(__file__).resolve().parents[1]
    default_output_dir = backend_root / "weights"

    parser = argparse.ArgumentParser(description="Download RealESRGAN_x4plus.pth weights.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where RealESRGAN_x4plus.pth should be stored.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing weights file.")
    parser.add_argument("--url", default=REAL_ESRGAN_X4PLUS_URL, help="Weights URL to download.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.output_dir / DEFAULT_FILENAME
    try:
        download_file(args.url, destination, force=args.force)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

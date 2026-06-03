from __future__ import annotations

import argparse
from pathlib import Path
import re

import cv2

from .core import DEFAULT_YOLO_MODEL, LiquidHeightResult, estimate_liquid_height_ratio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate orange/red liquid height ratio in a fixed goblet image.")
    parser.add_argument("image", help="Path to an input image or a directory of images.")
    parser.add_argument("--debug", type=Path, help="Optional directory for debug images.")
    parser.add_argument(
        "--method",
        choices=("yolo", "opencv", "auto"),
        default="yolo",
        help="Estimator backend. yolo localizes the cup first (needs ultralytics); opencv is colour-blob only.",
    )
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL, help="YOLO detection model path/name for --method yolo/auto.")
    args = parser.parse_args(argv)

    input_path = Path(args.image)
    if input_path.is_dir():
        image_paths = _iter_image_paths(input_path)
        if not image_paths:
            print(f"No images found in {input_path}")
            return 2
        for image_path in image_paths:
            debug_dir = args.debug / _safe_stem(image_path) if args.debug is not None else None
            result = _estimate_one(image_path, debug_dir, method=args.method, yolo_model=args.yolo_model)
            _print_result(image_path.name, result)
        return 0

    result = _estimate_one(input_path, args.debug, method=args.method, yolo_model=args.yolo_model)
    _print_result("", result)
    return 0 if result.ratio is not None else 2


def _estimate_one(image_path: Path, debug_dir: Path | None, *, method: str, yolo_model: str) -> LiquidHeightResult:
    result = estimate_liquid_height_ratio(image_path, debug=True, method=method, yolo_model=yolo_model)
    if not isinstance(result, LiquidHeightResult):
        raise RuntimeError("debug=True must return a LiquidHeightResult")

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        for name, debug_image in result.debug_images.items():
            cv2.imwrite(str(debug_dir / f"{name}.png"), debug_image)
    return result


def _print_result(prefix: str, result: LiquidHeightResult) -> None:
    label = f"{prefix} " if prefix else ""
    if result.ratio is None:
        print(f"{label}ratio=None confidence={result.confidence:.3f}")
        return

    print(f"{label}ratio={result.ratio:.4f} confidence={result.confidence:.3f}")


def _iter_image_paths(directory: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _safe_stem(path: Path) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)
    return safe[:80] or "image"


if __name__ == "__main__":
    raise SystemExit(main())

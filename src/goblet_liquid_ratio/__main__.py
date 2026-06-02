from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .core import LiquidHeightResult, estimate_liquid_height_ratio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate opaque liquid height ratio in a fixed goblet image.")
    parser.add_argument("image", help="Path to an input image.")
    parser.add_argument("--debug", type=Path, help="Optional directory for debug images.")
    args = parser.parse_args(argv)

    result = estimate_liquid_height_ratio(args.image, debug=True)
    if not isinstance(result, LiquidHeightResult):
        raise RuntimeError("debug=True must return a LiquidHeightResult")

    if args.debug is not None:
        args.debug.mkdir(parents=True, exist_ok=True)
        for name, debug_image in result.debug_images.items():
            cv2.imwrite(str(args.debug / f"{name}.png"), debug_image)

    if result.ratio is None:
        print(f"ratio=None confidence={result.confidence:.3f}")
        return 2

    print(f"ratio={result.ratio:.4f} confidence={result.confidence:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

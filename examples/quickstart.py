"""Minimal usage example for goblet-liquid-ratio.

    python examples/quickstart.py path/to/photo.jpg
    python examples/quickstart.py path/to/photo.jpg --method opencv

Prints the estimated juice height ratio (0.0-1.0), or a message if no
juice-bearing goblet was found.
"""
from __future__ import annotations

import argparse

from goblet_liquid_ratio import estimate_liquid_height_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to an image containing the goblet.")
    parser.add_argument(
        "--method",
        choices=("yolo", "opencv", "auto"),
        default="yolo",
        help="yolo (default) localizes the cup first and needs the [yolo] extra; "
        "opencv is colour-blob only and needs no torch.",
    )
    args = parser.parse_args()

    try:
        ratio = estimate_liquid_height_ratio(args.image, method=args.method)
    except ImportError as exc:
        # Raised when method='yolo' but ultralytics/torch is not installed.
        print(exc)
        return 2

    if ratio is None:
        print("No juice-bearing goblet detected.")
        return 1

    print(f"liquid height ratio = {ratio:.3f}  ({ratio * 100:.0f}% of the bowl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

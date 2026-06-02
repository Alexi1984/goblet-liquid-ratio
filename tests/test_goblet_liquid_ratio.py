import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from goblet_liquid_ratio import LiquidHeightResult, estimate_liquid_height_ratio


def synthetic_goblet_image(
    ratio: float,
    *,
    offset: tuple[int, int] = (220, 90),
    canvas_size: tuple[int, int] = (480, 640),
    liquid_bgr: tuple[int, int, int] = (20, 80, 220),
) -> np.ndarray:
    image = np.full((canvas_size[0], canvas_size[1], 3), 232, dtype=np.uint8)
    ox, oy = offset
    cup_outer = np.array(
        [
            [ox + 12, oy + 18],
            [ox + 148, oy + 18],
            [ox + 126, oy + 164],
            [ox + 34, oy + 164],
        ],
        dtype=np.int32,
    )
    cup_inner = np.array(
        [
            [ox + 25, oy + 34],
            [ox + 135, oy + 34],
            [ox + 116, oy + 154],
            [ox + 44, oy + 154],
        ],
        dtype=np.int32,
    )
    stem = np.array(
        [
            [ox + 73, oy + 164],
            [ox + 87, oy + 164],
            [ox + 87, oy + 226],
            [ox + 73, oy + 226],
        ],
        dtype=np.int32,
    )
    base = np.array(
        [
            [ox + 35, oy + 226],
            [ox + 125, oy + 226],
            [ox + 145, oy + 240],
            [ox + 15, oy + 240],
        ],
        dtype=np.int32,
    )

    cv2.rectangle(image, (18, 30), (74, 420), (205, 210, 218), 4)
    cv2.circle(image, (520, 310), 45, (210, 220, 225), 3)

    cup_top = oy + 34
    cup_bottom = oy + 154
    surface_y = int(round(cup_bottom - ratio * (cup_bottom - cup_top)))
    cup_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(cup_mask, [cup_inner], 255)
    fill_mask = np.zeros_like(cup_mask)
    cv2.rectangle(fill_mask, (ox, surface_y), (ox + 160, cup_bottom + 4), 255, -1)
    liquid_mask = cv2.bitwise_and(cup_mask, fill_mask)
    image[liquid_mask > 0] = liquid_bgr

    cv2.polylines(image, [cup_outer], True, (80, 80, 80), 3, cv2.LINE_AA)
    cv2.ellipse(image, (ox + 80, oy + 20), (69, 12), 0, 0, 360, (95, 95, 95), 2, cv2.LINE_AA)
    cv2.polylines(image, [stem], True, (95, 95, 95), 3, cv2.LINE_AA)
    cv2.polylines(image, [base], True, (95, 95, 95), 3, cv2.LINE_AA)
    return image


@pytest.mark.parametrize(
    ("expected_ratio", "liquid_bgr"),
    [
        (0.25, (30, 180, 40)),
        (0.55, (180, 30, 180)),
        (0.82, (90, 20, 20)),
    ],
)
def test_estimates_liquid_height_ratio_for_multiple_opaque_colors(expected_ratio, liquid_bgr):
    image = synthetic_goblet_image(expected_ratio, liquid_bgr=liquid_bgr)

    ratio = estimate_liquid_height_ratio(image)

    assert ratio is not None
    assert ratio == pytest.approx(expected_ratio, abs=0.10)


def test_estimates_same_ratio_when_goblet_position_changes():
    left_image = synthetic_goblet_image(0.45, offset=(110, 120), liquid_bgr=(20, 120, 210))
    right_image = synthetic_goblet_image(0.45, offset=(360, 80), liquid_bgr=(20, 120, 210))

    left_ratio = estimate_liquid_height_ratio(left_image)
    right_ratio = estimate_liquid_height_ratio(right_image)

    assert left_ratio is not None
    assert right_ratio is not None
    assert left_ratio == pytest.approx(0.45, abs=0.10)
    assert right_ratio == pytest.approx(0.45, abs=0.10)


def test_accepts_image_path(tmp_path: Path):
    image_path = tmp_path / "goblet.png"
    cv2.imwrite(str(image_path), synthetic_goblet_image(0.65, liquid_bgr=(10, 150, 220)))

    ratio = estimate_liquid_height_ratio(image_path)

    assert ratio is not None
    assert ratio == pytest.approx(0.65, abs=0.10)


def test_debug_mode_returns_intermediate_result():
    image = synthetic_goblet_image(0.35, liquid_bgr=(160, 90, 20))

    result = estimate_liquid_height_ratio(image, debug=True)

    assert isinstance(result, LiquidHeightResult)
    assert result.ratio == pytest.approx(0.35, abs=0.10)
    assert result.confidence > 0
    assert result.goblet_bbox is not None
    assert result.liquid_surface_y is not None
    assert result.cup_top_y is not None
    assert result.cup_bottom_y is not None
    assert {"edges", "template_match", "cup_mask", "liquid_mask"}.issubset(result.debug_images)


def test_returns_none_when_no_goblet_is_present():
    image = np.full((360, 520, 3), 230, dtype=np.uint8)
    cv2.rectangle(image, (30, 40), (90, 320), (200, 210, 220), 4)
    cv2.circle(image, (410, 250), 55, (210, 210, 210), 3)

    assert estimate_liquid_height_ratio(image) is None


def test_cli_prints_ratio_for_image_path(tmp_path: Path):
    image_path = tmp_path / "cup.png"
    cv2.imwrite(str(image_path), synthetic_goblet_image(0.50, liquid_bgr=(40, 40, 190)))

    completed = subprocess.run(
        [sys.executable, "-m", "goblet_liquid_ratio", str(image_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ratio=" in completed.stdout

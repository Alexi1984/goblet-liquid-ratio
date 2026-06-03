import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import goblet_liquid_ratio.core as core
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


def synthetic_realistic_desktop_scene(
    ratio: float = 0.58,
    *,
    offset: tuple[int, int] = (140, 190),
    canvas_size: tuple[int, int] = (720, 1080),
    liquid_bgr: tuple[int, int, int] = (20, 95, 225),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = np.full((canvas_size[0], canvas_size[1], 3), 224, dtype=np.uint8)
    ox, oy = offset

    # Distractors seen in the real desktop photos.
    cv2.rectangle(image, (790, 120), (900, 500), (190, 198, 188), 4)
    cv2.rectangle(image, (818, 180), (870, 420), (178, 185, 180), 3)
    cv2.ellipse(image, (840, 112), (32, 8), 0, 0, 360, (85, 85, 85), 3, cv2.LINE_AA)
    cv2.rectangle(image, (430, 250), (560, 410), (215, 200, 175), -1)
    cv2.ellipse(image, (495, 250), (67, 12), 0, 0, 360, (180, 170, 160), 2, cv2.LINE_AA)
    cv2.rectangle(image, (330, 110), (405, 370), (210, 218, 225), 4)

    rim_center = (ox + 190, oy + 35)
    rim_axes = (180, 35)
    cup_top = oy + 35
    cup_bottom = oy + 260
    cup_left_top = ox + 12
    cup_right_top = ox + 368
    cup_left_bottom = ox + 76
    cup_right_bottom = ox + 304
    cup_outer = np.array(
        [
            [cup_left_top, cup_top],
            [cup_right_top, cup_top],
            [cup_right_bottom, cup_bottom],
            [cup_left_bottom, cup_bottom],
        ],
        dtype=np.int32,
    )
    cup_inner = np.array(
        [
            [ox + 28, oy + 54],
            [ox + 352, oy + 54],
            [ox + 292, oy + 246],
            [ox + 88, oy + 246],
        ],
        dtype=np.int32,
    )
    expected_bbox = (cup_left_top, cup_top, cup_right_top - cup_left_top, 405)

    cup_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(cup_mask, [cup_inner], 255)
    surface_y = int(round(cup_bottom - ratio * (cup_bottom - cup_top)))
    fill_mask = np.zeros_like(cup_mask)
    cv2.rectangle(fill_mask, (ox, surface_y), (ox + 390, cup_bottom + 8), 255, -1)
    liquid_mask = cv2.bitwise_and(cup_mask, fill_mask)
    image[liquid_mask > 0] = liquid_bgr

    # Sparse splashes above the liquid should not become the liquid surface.
    for center, radius in [((ox + 165, surface_y - 42), 25), ((ox + 235, surface_y - 25), 20)]:
        cv2.circle(image, center, radius, liquid_bgr, -1, cv2.LINE_AA)

    cv2.polylines(image, [cup_outer], True, (115, 105, 96), 3, cv2.LINE_AA)
    cv2.ellipse(image, rim_center, rim_axes, 0, 0, 360, (116, 105, 95), 3, cv2.LINE_AA)
    cv2.rectangle(image, (ox + 178, oy + 260), (ox + 202, oy + 348), (118, 107, 98), 3)
    cv2.ellipse(image, (ox + 190, oy + 380), (130, 34), 0, 0, 360, (120, 108, 100), 3, cv2.LINE_AA)
    return image, expected_bbox


def synthetic_closeup_goblet_scene(
    ratio: float = 0.52,
    *,
    offset: tuple[int, int] = (170, 60),
    canvas_size: tuple[int, int] = (720, 960),
    liquid_bgr: tuple[int, int, int] = (10, 95, 230),
) -> np.ndarray:
    image = np.full((canvas_size[0], canvas_size[1], 3), 226, dtype=np.uint8)
    ox, oy = offset

    cup_top = oy + 60
    cup_bottom = oy + 430
    cup_inner = np.array(
        [
            [ox + 105, oy + 118],
            [ox + 490, oy + 118],
            [ox + 386, oy + 500],
            [ox + 165, oy + 500],
        ],
        dtype=np.int32,
    )
    surface_y = int(round(cup_bottom - ratio * (cup_bottom - cup_top)))
    cup_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(cup_mask, [cup_inner], 255)
    fill_mask = np.zeros_like(cup_mask)
    cv2.rectangle(fill_mask, (ox, surface_y), (ox + 560, cup_bottom + 80), 255, -1)
    liquid_mask = cv2.bitwise_and(cup_mask, fill_mask)
    image[liquid_mask > 0] = liquid_bgr

    cv2.circle(image, (ox + 245, surface_y - 70), 34, liquid_bgr, -1, cv2.LINE_AA)
    cv2.circle(image, (ox + 320, surface_y - 40), 26, liquid_bgr, -1, cv2.LINE_AA)
    cv2.ellipse(image, (ox + 300, cup_top), (250, 48), 0, 0, 360, (105, 100, 96), 3, cv2.LINE_AA)
    cv2.polylines(
        image,
        [
            np.array(
                [
                    [ox + 48, cup_top],
                    [ox + 550, cup_top],
                    [ox + 420, cup_bottom + 80],
                    [ox + 135, cup_bottom + 80],
                ],
                dtype=np.int32,
            )
        ],
        True,
        (112, 105, 98),
        3,
        cv2.LINE_AA,
    )
    cv2.rectangle(image, (ox + 283, cup_bottom + 80), (ox + 317, cup_bottom + 170), (112, 105, 98), 3)
    cv2.ellipse(image, (ox + 300, cup_bottom + 195), (170, 38), 0, 0, 360, (112, 105, 98), 3, cv2.LINE_AA)
    return image


def synthetic_connected_reflection_scene(ratio: float = 0.52) -> np.ndarray:
    image = synthetic_realistic_desktop_scene(ratio, offset=(250, 90), canvas_size=(720, 960))[0]
    liquid_bgr = (20, 95, 225)
    ox, oy = (250, 90)
    cup_bottom = oy + 260

    cv2.rectangle(image, (ox + 172, cup_bottom - 8), (ox + 208, cup_bottom + 118), liquid_bgr, -1, cv2.LINE_AA)
    cv2.ellipse(image, (ox + 190, cup_bottom + 140), (120, 36), 0, 0, 360, liquid_bgr, -1, cv2.LINE_AA)
    cv2.ellipse(image, (ox + 190, cup_bottom + 140), (130, 42), 0, 0, 360, (120, 108, 100), 3, cv2.LINE_AA)
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


def test_prefers_wide_filled_goblet_over_slender_glass_bottle():
    image, expected_bbox = synthetic_realistic_desktop_scene(0.58)

    result = estimate_liquid_height_ratio(image, debug=True)

    assert isinstance(result, LiquidHeightResult)
    assert result.ratio == pytest.approx(0.58, abs=0.12)
    assert result.goblet_bbox is not None
    expected_x, expected_y, expected_w, _ = expected_bbox
    actual_x, actual_y, actual_w, _ = result.goblet_bbox
    assert actual_x == pytest.approx(expected_x, abs=80)
    assert actual_y == pytest.approx(expected_y, abs=90)
    assert actual_w == pytest.approx(expected_w, abs=140)


def test_estimates_closeup_goblet_when_liquid_component_is_not_wide():
    image = synthetic_closeup_goblet_scene(0.52)

    result = estimate_liquid_height_ratio(image, debug=True)

    assert isinstance(result, LiquidHeightResult)
    assert result.ratio == pytest.approx(0.52, abs=0.16)
    assert result.goblet_bbox is not None
    assert result.goblet_bbox[2] > 250


def test_ignores_connected_base_reflection_below_bowl():
    image = synthetic_connected_reflection_scene(0.52)

    result = estimate_liquid_height_ratio(image, debug=True)

    assert isinstance(result, LiquidHeightResult)
    assert result.ratio == pytest.approx(0.52, abs=0.14)
    assert result.goblet_bbox is not None
    assert result.goblet_bbox[1] < 190
    assert result.goblet_bbox[3] < 430


def test_yolo_method_uses_optional_segmenter(monkeypatch):
    image = synthetic_goblet_image(0.50)
    expected = LiquidHeightResult(
        ratio=0.42,
        confidence=0.91,
        goblet_bbox=(10, 20, 100, 120),
        liquid_surface_y=45,
        cup_top_y=20,
        cup_bottom_y=120,
        debug_images={},
    )

    def fake_segmenter(bgr, edges, config, debug_images, model_path):
        assert bgr.shape == image.shape
        assert model_path == "fake.pt"
        return expected

    monkeypatch.setattr(core, "_estimate_from_yolo_segmenter", fake_segmenter)

    result = estimate_liquid_height_ratio(image, debug=True, method="yolo", yolo_model="fake.pt")

    assert result is expected


def test_opencv_method_keeps_low_confidence_detected_result(monkeypatch):
    image = synthetic_goblet_image(0.50)
    expected = LiquidHeightResult(
        ratio=0.31,
        confidence=0.42,
        goblet_bbox=(20, 30, 110, 150),
        liquid_surface_y=80,
        cup_top_y=30,
        cup_bottom_y=150,
        debug_images={},
    )

    def fake_opencv(bgr, edges, config, debug_images):
        assert bgr.shape == image.shape
        return expected

    monkeypatch.setattr(core, "_estimate_with_opencv", fake_opencv)

    result = estimate_liquid_height_ratio(image, debug=True, method="opencv")

    assert result is expected


def test_rejects_unknown_estimation_method():
    with pytest.raises(ValueError, match="method"):
        estimate_liquid_height_ratio(synthetic_goblet_image(0.50), method="magic")


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


def test_cli_accepts_method_argument(tmp_path: Path):
    image_path = tmp_path / "cup.png"
    cv2.imwrite(str(image_path), synthetic_goblet_image(0.50, liquid_bgr=(40, 40, 190)))

    completed = subprocess.run(
        [sys.executable, "-m", "goblet_liquid_ratio", str(image_path), "--method", "opencv"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ratio=" in completed.stdout


def test_cli_processes_directory_without_quoting_individual_filenames(tmp_path: Path):
    image_path = tmp_path / "image&with&shell&chars.jpeg"
    cv2.imwrite(str(image_path), synthetic_goblet_image(0.50, liquid_bgr=(40, 40, 190)))

    completed = subprocess.run(
        [sys.executable, "-m", "goblet_liquid_ratio", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "image&with&shell&chars.jpeg" in completed.stdout
    assert "ratio=" in completed.stdout

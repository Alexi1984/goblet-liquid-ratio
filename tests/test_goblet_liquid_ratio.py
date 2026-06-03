"""Tests for the YOLO-localized goblet liquid height estimator.

Three layers:
  * Pure-unit tests for the colour/blob math (no detector, always run).
  * Public-API contract tests using method="opencv" (lightweight colour-blob
    localization, no torch needed).
  * Real-photo accuracy regression gated on ultralytics + sample images being
    present (skipped otherwise).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import goblet_liquid_ratio.core as core
from goblet_liquid_ratio import LiquidHeightResult, estimate_liquid_height_ratio

ROOT = Path(__file__).resolve().parent.parent
GT_PATH = ROOT / "eval" / "ground_truth.json"
SAMPLES_DIR = Path("/home/yuyuyu/桌面")
MODEL_PATH = ROOT / "models" / "yolo11n.pt"

try:
    import ultralytics  # noqa: F401

    HAVE_YOLO = True
except ImportError:
    HAVE_YOLO = False


# --------------------------------------------------------------------------- #
# Synthetic scene: an upright bowl of opaque juice on a plain background, with
# an empty transparent rim band above the liquid. Used for the colour-blob
# (method="opencv") path that does not need a detector.
# --------------------------------------------------------------------------- #
def synthetic_bowl_scene(
    ratio: float,
    *,
    offset: tuple[int, int] = (240, 120),
    canvas_size: tuple[int, int] = (640, 720),
    juice_bgr: tuple[int, int, int] = (20, 95, 225),
) -> np.ndarray:
    image = np.full((canvas_size[0], canvas_size[1], 3), 232, dtype=np.uint8)
    ox, oy = offset
    rim_y = oy + 20
    bowl_bottom = oy + 210
    bowl_left_top, bowl_right_top = ox + 10, ox + 250
    bowl_left_bot, bowl_right_bot = ox + 55, ox + 205
    cavity = np.array(
        [[bowl_left_top, rim_y], [bowl_right_top, rim_y],
         [bowl_right_bot, bowl_bottom], [bowl_left_bot, bowl_bottom]],
        dtype=np.int32,
    )
    surface_y = int(round(bowl_bottom - ratio * (bowl_bottom - rim_y)))
    cup_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(cup_mask, [cavity], 255)
    fill = np.zeros_like(cup_mask)
    cv2.rectangle(fill, (ox, surface_y), (ox + 260, bowl_bottom + 4), 255, -1)
    image[cv2.bitwise_and(cup_mask, fill) > 0] = juice_bgr
    # thin glass outline + stem + base
    cv2.polylines(image, [cavity], True, (90, 90, 90), 2, cv2.LINE_AA)
    cv2.rectangle(image, (ox + 120, bowl_bottom), (ox + 140, bowl_bottom + 70), (95, 95, 95), 2)
    cv2.ellipse(image, (ox + 130, bowl_bottom + 95), (90, 22), 0, 0, 360, (95, 95, 95), 2, cv2.LINE_AA)
    return image


# ----------------------------- pure-unit tests ----------------------------- #
def test_juice_confidence_fires_on_orange_not_blue():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[:, :10] = (20, 95, 225)   # orange (BGR)
    img[:, 10:] = (225, 95, 20)   # blue (BGR)
    conf = core._juice_confidence(img)
    assert conf[:, :10].mean() > 0.5
    assert conf[:, 10:].mean() < 0.05


def test_liquid_band_picks_solid_run_and_ignores_thin_top_bleed():
    # 60-row blob: rows 0-3 a thin background bleed, 20-49 the solid bowl band.
    blob = np.zeros((60, 40), dtype=np.float32)
    blob[0:4, 18:22] = 1.0       # narrow top bleed (e.g. background object)
    blob[20:50, 5:35] = 1.0      # solid liquid band
    band = core._liquid_band(blob, core.EstimatorConfig())
    assert band is not None
    top, bottom = band
    assert 18 <= top <= 22
    assert 47 <= bottom <= 50


def test_liquid_band_excludes_detached_base_reflection():
    blob = np.zeros((80, 40), dtype=np.float32)
    blob[10:45, 5:35] = 1.0      # bowl liquid
    blob[60:70, 12:28] = 1.0     # detached, narrower base reflection
    band = core._liquid_band(blob, core.EstimatorConfig())
    assert band is not None
    top, bottom = band
    assert bottom <= 46          # base reflection must not extend the bottom


# -------------------------- public-API contract ---------------------------- #
def test_estimates_height_for_multiple_opaque_colors():
    # The estimator targets orange/red juice; vary brightness/shade within that band.
    for expected, juice in [(0.35, (20, 95, 225)), (0.55, (30, 120, 210)), (0.75, (40, 70, 180))]:
        img = synthetic_bowl_scene(expected, juice_bgr=juice)
        ratio = estimate_liquid_height_ratio(img, method="opencv")
        assert ratio is not None, f"no detection for {juice}"
        assert ratio == pytest.approx(expected, abs=0.15)


def test_same_ratio_when_bowl_shifts():
    left = estimate_liquid_height_ratio(synthetic_bowl_scene(0.5, offset=(120, 150)), method="opencv")
    right = estimate_liquid_height_ratio(synthetic_bowl_scene(0.5, offset=(380, 110)), method="opencv")
    assert left is not None and right is not None
    assert left == pytest.approx(0.5, abs=0.15)
    assert right == pytest.approx(0.5, abs=0.15)


def test_accepts_image_path(tmp_path: Path):
    p = tmp_path / "bowl.png"
    cv2.imwrite(str(p), synthetic_bowl_scene(0.6, juice_bgr=(10, 150, 220)))
    ratio = estimate_liquid_height_ratio(p, method="opencv")
    assert ratio is not None
    assert ratio == pytest.approx(0.6, abs=0.15)


def test_debug_mode_returns_structured_result():
    img = synthetic_bowl_scene(0.45, juice_bgr=(30, 110, 215))
    res = estimate_liquid_height_ratio(img, debug=True, method="opencv")
    assert isinstance(res, LiquidHeightResult)
    assert res.ratio is not None
    assert res.goblet_bbox is not None
    assert res.liquid_surface_y is not None
    assert res.cup_top_y is not None and res.cup_bottom_y is not None
    assert {"juice_confidence", "overlay"}.issubset(res.debug_images)


def test_returns_none_when_no_juice_present():
    image = np.full((360, 520, 3), 230, dtype=np.uint8)
    cv2.rectangle(image, (30, 40), (90, 320), (200, 210, 220), 4)  # clear bottle
    assert estimate_liquid_height_ratio(image, method="opencv") is None


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        estimate_liquid_height_ratio(synthetic_bowl_scene(0.5), method="magic")


def test_yolo_method_without_ultralytics_raises(monkeypatch):
    # Force the optional import to fail and confirm method="yolo" surfaces it.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ImportError("simulated missing ultralytics")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError):
        estimate_liquid_height_ratio(synthetic_bowl_scene(0.5), method="yolo")


# --------------------------- real-photo regression ------------------------- #
@pytest.mark.skipif(not HAVE_YOLO, reason="ultralytics not installed")
@pytest.mark.skipif(not GT_PATH.exists(), reason="ground truth missing")
@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="sample photos missing")
def test_real_photo_accuracy_regression():
    gt = {s["id"]: s for s in json.loads(GT_PATH.read_text())["samples"]}
    paths = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    model = str(MODEL_PATH) if MODEL_PATH.exists() else "yolo11n.pt"

    errs = []
    for i, p in enumerate(paths, 1):
        if i not in gt:
            continue
        ratio = estimate_liquid_height_ratio(p, method="yolo", yolo_model=model)
        assert ratio is not None, f"sample {i} returned None"
        errs.append(abs(ratio - gt[i]["gt_ratio"]))

    assert len(errs) == len(gt)
    mae = sum(errs) / len(errs)
    # Old colour-blob estimator scored MAE 0.125 with 3 hard failures here.
    assert mae <= 0.09, f"MAE regressed to {mae:.3f}"
    assert max(errs) <= 0.20, f"a sample failed hard: max err {max(errs):.3f}"


# --------------------------------- CLI ------------------------------------- #
def test_cli_prints_ratio_for_image_path(tmp_path: Path):
    p = tmp_path / "cup.png"
    cv2.imwrite(str(p), synthetic_bowl_scene(0.5, juice_bgr=(40, 40, 190)))
    out = subprocess.run(
        [sys.executable, "-m", "goblet_liquid_ratio", str(p), "--method", "opencv"],
        check=True, capture_output=True, text=True,
    )
    assert "ratio=" in out.stdout


def test_cli_processes_directory_with_shell_chars(tmp_path: Path):
    p = tmp_path / "image&with&chars.jpeg"
    cv2.imwrite(str(p), synthetic_bowl_scene(0.5, juice_bgr=(40, 40, 190)))
    out = subprocess.run(
        [sys.executable, "-m", "goblet_liquid_ratio", str(tmp_path), "--method", "opencv"],
        check=True, capture_output=True, text=True,
    )
    assert "image&with&chars.jpeg" in out.stdout
    assert "ratio=" in out.stdout

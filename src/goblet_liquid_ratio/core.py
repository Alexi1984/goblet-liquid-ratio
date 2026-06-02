from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CANONICAL_WIDTH = 160
CANONICAL_HEIGHT = 260
CUP_TOP_Y = 34
CUP_BOTTOM_Y = 154


@dataclass(frozen=True)
class LiquidHeightResult:
    ratio: float | None
    confidence: float
    goblet_bbox: tuple[int, int, int, int] | None
    liquid_surface_y: int | None
    cup_top_y: int | None
    cup_bottom_y: int | None
    debug_images: dict[str, np.ndarray]


@dataclass(frozen=True)
class EstimatorConfig:
    min_edge_pixels: int = 350
    match_threshold: float = 0.18
    pixel_liquid_threshold: float = 0.18
    min_row_fill_fraction: float = 0.34
    min_row_strength: float = 0.10
    min_filled_rows: int = 5
    max_gap_rows: int = 3
    scale_min: float = 0.45
    scale_max: float = 2.75
    scale_steps: int = 47


@dataclass(frozen=True)
class _Location:
    bbox: tuple[int, int, int, int]
    score: float


def estimate_liquid_height_ratio(image: str | Path | np.ndarray, *, debug: bool = False) -> float | None | LiquidHeightResult:
    config = EstimatorConfig()
    bgr = _load_image_bgr(image)
    edges = _edge_image(bgr)
    debug_images: dict[str, np.ndarray] = {"edges": edges}

    location = _locate_goblet(edges, config)
    if location is None:
        result = LiquidHeightResult(None, 0.0, None, None, None, None, debug_images)
        return result if debug else None

    x, y, w, h = location.bbox
    crop = bgr[y : y + h, x : x + w]
    if crop.size == 0:
        result = LiquidHeightResult(None, 0.0, location.bbox, None, None, None, debug_images)
        return result if debug else None

    canonical = cv2.resize(crop, (CANONICAL_WIDTH, CANONICAL_HEIGHT), interpolation=cv2.INTER_AREA)
    cup_mask = _make_cup_mask()
    surface = _estimate_surface_y(canonical, cup_mask, config)
    debug_images["template_match"] = _draw_bbox(bgr, location.bbox, location.score)
    debug_images["cup_mask"] = cup_mask.copy()

    if surface is None:
        debug_images["liquid_mask"] = np.zeros_like(cup_mask)
        result = LiquidHeightResult(None, location.score, location.bbox, None, None, None, debug_images)
        return result if debug else None

    surface_y, liquid_mask, evidence_score = surface
    ratio = (CUP_BOTTOM_Y - surface_y) / float(CUP_BOTTOM_Y - CUP_TOP_Y)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    confidence = float(np.clip(location.score * 0.65 + evidence_score * 0.35, 0.0, 1.0))
    debug_images["liquid_mask"] = (liquid_mask.astype(np.uint8) * 255)

    scale_y = h / CANONICAL_HEIGHT
    global_surface_y = int(round(y + surface_y * scale_y))
    global_top_y = int(round(y + CUP_TOP_Y * scale_y))
    global_bottom_y = int(round(y + CUP_BOTTOM_Y * scale_y))
    result = LiquidHeightResult(
        ratio=ratio,
        confidence=confidence,
        goblet_bbox=location.bbox,
        liquid_surface_y=global_surface_y,
        cup_top_y=global_top_y,
        cup_bottom_y=global_bottom_y,
        debug_images=debug_images,
    )
    return result if debug else result.ratio


def _load_image_bgr(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, str | Path):
        loaded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if loaded is None:
            raise ValueError(f"Could not read image: {image}")
        return loaded

    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("image array must have shape HxW, HxWx3, or HxWx4")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _edge_image(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)
    kernel = np.ones((2, 2), dtype=np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)


def _locate_goblet(edges: np.ndarray, config: EstimatorConfig) -> _Location | None:
    if int(np.count_nonzero(edges)) < config.min_edge_pixels:
        return None

    template = _make_template_edges()
    best_score = -1.0
    best_bbox: tuple[int, int, int, int] | None = None
    image_h, image_w = edges.shape[:2]

    for scale in np.linspace(config.scale_min, config.scale_max, config.scale_steps):
        width = int(round(CANONICAL_WIDTH * scale))
        height = int(round(CANONICAL_HEIGHT * scale))
        if width < 40 or height < 65 or width > image_w or height > image_h:
            continue
        scaled_template = cv2.resize(template, (width, height), interpolation=cv2.INTER_NEAREST)
        if int(np.count_nonzero(scaled_template)) < 40:
            continue
        response = cv2.matchTemplate(edges, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, score, _, top_left = cv2.minMaxLoc(response)
        if score > best_score:
            best_score = float(score)
            best_bbox = (int(top_left[0]), int(top_left[1]), width, height)

    if best_bbox is None or best_score < config.match_threshold:
        return None
    return _Location(best_bbox, best_score)


def _make_template_edges() -> np.ndarray:
    template = np.zeros((CANONICAL_HEIGHT, CANONICAL_WIDTH), dtype=np.uint8)
    cup_outer = np.array([[12, 18], [148, 18], [126, 164], [34, 164]], dtype=np.int32)
    stem = np.array([[73, 164], [87, 164], [87, 226], [73, 226]], dtype=np.int32)
    base = np.array([[35, 226], [125, 226], [145, 240], [15, 240]], dtype=np.int32)
    cv2.polylines(template, [cup_outer], True, 255, 3, cv2.LINE_AA)
    cv2.ellipse(template, (80, 20), (69, 12), 0, 0, 360, 255, 2, cv2.LINE_AA)
    cv2.polylines(template, [stem], True, 255, 3, cv2.LINE_AA)
    cv2.polylines(template, [base], True, 255, 3, cv2.LINE_AA)
    return cv2.dilate(template, np.ones((2, 2), dtype=np.uint8), iterations=1)


def _make_cup_mask() -> np.ndarray:
    mask = np.zeros((CANONICAL_HEIGHT, CANONICAL_WIDTH), dtype=np.uint8)
    cup_inner = np.array([[25, CUP_TOP_Y], [135, CUP_TOP_Y], [116, CUP_BOTTOM_Y], [44, CUP_BOTTOM_Y]], dtype=np.int32)
    cv2.fillPoly(mask, [cup_inner], 255)
    return mask


def _estimate_surface_y(
    canonical_bgr: np.ndarray,
    cup_mask: np.ndarray,
    config: EstimatorConfig,
) -> tuple[int, np.ndarray, float] | None:
    cup_pixels = cup_mask > 0
    outside_pixels = ~cup_pixels
    if not np.any(cup_pixels) or not np.any(outside_pixels):
        return None

    hsv = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bgr_float = canonical_bgr.astype(np.float32)
    background_gray = float(np.median(gray[outside_pixels]))

    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    chroma = (bgr_float.max(axis=2) - bgr_float.min(axis=2)) / 160.0
    gray_difference = np.abs(gray - background_gray) / 85.0
    confidence = np.maximum.reduce([saturation, chroma, gray_difference])
    confidence = np.clip(confidence, 0.0, 1.0)
    confidence[~cup_pixels] = 0.0

    liquid_pixels = (confidence >= config.pixel_liquid_threshold) & cup_pixels
    row_fill_fraction = np.zeros(CANONICAL_HEIGHT, dtype=np.float32)
    row_strength = np.zeros(CANONICAL_HEIGHT, dtype=np.float32)
    for row in range(CUP_TOP_Y, CUP_BOTTOM_Y + 1):
        row_mask = cup_pixels[row]
        if not np.any(row_mask):
            continue
        row_liquid = liquid_pixels[row, row_mask]
        row_scores = confidence[row, row_mask]
        row_fill_fraction[row] = float(np.mean(row_liquid))
        row_strength[row] = float(np.mean(row_scores))

    filled_rows = (row_fill_fraction >= config.min_row_fill_fraction) & (row_strength >= config.min_row_strength)
    filled_rows = _close_row_gaps(filled_rows, config.max_gap_rows)
    surface_y = _top_of_bottom_connected_run(filled_rows, config)
    if surface_y is None:
        return None

    filled_count = int(np.count_nonzero(filled_rows[surface_y : CUP_BOTTOM_Y + 1]))
    if filled_count < config.min_filled_rows:
        return None

    liquid_mask = np.zeros_like(cup_mask, dtype=bool)
    liquid_mask[surface_y : CUP_BOTTOM_Y + 1] = cup_pixels[surface_y : CUP_BOTTOM_Y + 1]
    evidence = float(np.clip(np.mean(row_fill_fraction[surface_y : CUP_BOTTOM_Y + 1]), 0.0, 1.0))
    return surface_y, liquid_mask, evidence


def _close_row_gaps(rows: np.ndarray, max_gap_rows: int) -> np.ndarray:
    closed = rows.copy()
    gap_start: int | None = None
    for row in range(CUP_TOP_Y, CUP_BOTTOM_Y + 1):
        if closed[row]:
            if gap_start is not None and row - gap_start <= max_gap_rows:
                closed[gap_start:row] = True
            gap_start = None
        elif gap_start is None:
            gap_start = row
    return closed


def _top_of_bottom_connected_run(rows: np.ndarray, config: EstimatorConfig) -> int | None:
    anchors = np.flatnonzero(rows[max(CUP_TOP_Y, CUP_BOTTOM_Y - 8) : CUP_BOTTOM_Y + 1])
    if anchors.size == 0:
        return None

    row = CUP_BOTTOM_Y
    while row >= CUP_TOP_Y and not rows[row]:
        row -= 1
    if row < CUP_TOP_Y:
        return None

    gaps = 0
    top = row
    while row >= CUP_TOP_Y:
        if rows[row]:
            top = row
            gaps = 0
        else:
            gaps += 1
            if gaps > config.max_gap_rows:
                break
        row -= 1
    return top


def _draw_bbox(bgr: np.ndarray, bbox: tuple[int, int, int, int], score: float) -> np.ndarray:
    image = bgr.copy()
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"score={score:.2f}",
        (x, max(16, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 160, 0),
        1,
        cv2.LINE_AA,
    )
    return image

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Literal

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
    blob_pixel_threshold: float = 0.25
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


EstimationMethod = Literal["opencv", "yolo", "auto"]


def estimate_liquid_height_ratio(
    image: str | Path | np.ndarray,
    *,
    debug: bool = False,
    method: EstimationMethod = "opencv",
    yolo_model: str | Path = "yolov8n-seg.pt",
) -> float | None | LiquidHeightResult:
    if method not in {"opencv", "yolo", "auto"}:
        raise ValueError("method must be one of: opencv, yolo, auto")

    config = EstimatorConfig()
    bgr = _load_image_bgr(image)
    edges = _edge_image(bgr)
    debug_images: dict[str, np.ndarray] = {"edges": edges}

    yolo_result: LiquidHeightResult | None = None
    if method in {"yolo", "auto"}:
        try:
            yolo_result = _estimate_from_yolo_segmenter(bgr, edges, config, debug_images, str(yolo_model))
        except ImportError:
            if method == "yolo":
                raise

    if method == "yolo" and yolo_result is not None:
        return yolo_result if debug else yolo_result.ratio

    opencv_result = None if method == "yolo" else _estimate_with_opencv(bgr, edges, config, debug_images)
    result = _choose_backend_result(yolo_result, opencv_result)
    if result is None:
        empty = LiquidHeightResult(None, 0.0, None, None, None, None, debug_images)
        return empty if debug else None
    return result if debug else result.ratio


def _estimate_with_opencv(
    bgr: np.ndarray,
    edges: np.ndarray,
    config: EstimatorConfig,
    debug_images: dict[str, np.ndarray],
) -> LiquidHeightResult | None:
    blob_result = _estimate_from_liquid_blob(bgr, edges, config, debug_images)
    template_result = _estimate_from_template_match(bgr, edges, config, debug_images)
    return _choose_result(blob_result, template_result)


def _choose_backend_result(
    yolo_result: LiquidHeightResult | None,
    opencv_result: LiquidHeightResult | None,
) -> LiquidHeightResult | None:
    if yolo_result is None:
        return opencv_result
    if opencv_result is None:
        return yolo_result
    if yolo_result.confidence >= opencv_result.confidence + 0.05:
        return yolo_result
    return opencv_result


def _estimate_from_template_match(
    bgr: np.ndarray,
    edges: np.ndarray,
    config: EstimatorConfig,
    base_debug_images: dict[str, np.ndarray],
) -> LiquidHeightResult | None:
    location = _locate_goblet(edges, config)
    if location is None:
        return None

    x, y, w, h = location.bbox
    crop = bgr[y : y + h, x : x + w]
    if crop.size == 0:
        return None

    canonical = cv2.resize(crop, (CANONICAL_WIDTH, CANONICAL_HEIGHT), interpolation=cv2.INTER_AREA)
    cup_mask = _make_cup_mask()
    surface = _estimate_surface_y(canonical, cup_mask, config)
    debug_images = dict(base_debug_images)
    debug_images["template_match"] = _draw_bbox(bgr, location.bbox, location.score)
    debug_images["cup_mask"] = cup_mask.copy()

    if surface is None:
        debug_images["liquid_mask"] = np.zeros_like(cup_mask)
        return None

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
    return result


def _choose_result(blob_result: LiquidHeightResult | None, template_result: LiquidHeightResult | None) -> LiquidHeightResult | None:
    if template_result is not None and template_result.confidence < 0.58:
        template_result = None
    if blob_result is None:
        return template_result
    if template_result is None:
        return blob_result
    if template_result.confidence >= 0.62 and blob_result.confidence < template_result.confidence + 0.18:
        return template_result
    if blob_result.confidence >= max(0.74, template_result.confidence + 0.08):
        return blob_result
    return template_result


def _estimate_from_liquid_blob(
    bgr: np.ndarray,
    edges: np.ndarray,
    config: EstimatorConfig,
    debug_images: dict[str, np.ndarray],
) -> LiquidHeightResult | None:
    confidence = _opaque_confidence_image(bgr)
    mask = ((confidence >= config.blob_pixel_threshold).astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    debug_images["opaque_mask"] = mask.copy()

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best: LiquidHeightResult | None = None
    best_score = 0.0
    image_h, image_w = bgr.shape[:2]
    min_area = max(900, int(image_h * image_w * 0.0012))

    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < min_area or w < 45 or h < 20:
            continue
        if w > image_w * 0.58 or h > image_h * 0.65:
            continue
        aspect = w / float(h)
        if aspect < 0.72 or aspect > 6.0:
            continue

        component = labels == label
        candidate = _evaluate_liquid_component(component, confidence, edges, (x, y, w, h), image_w, image_h, config)
        if candidate is None:
            continue
        score, result = candidate
        if score > best_score:
            best_score = score
            best = result

    if best is None:
        return None

    debug_images.setdefault("template_match", _draw_bbox(bgr, best.goblet_bbox or (0, 0, 0, 0), best.confidence))
    if best.cup_top_y is not None and best.cup_bottom_y is not None and best.goblet_bbox is not None:
        cup_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        x, _, w, _ = best.goblet_bbox
        cv2.rectangle(cup_mask, (x, best.cup_top_y), (x + w, best.cup_bottom_y), 255, -1)
        debug_images["cup_mask"] = cup_mask
    debug_images.setdefault("liquid_mask", np.zeros(bgr.shape[:2], dtype=np.uint8))
    best.debug_images.update(debug_images)
    return best


def _opaque_confidence_image(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bgr_float = bgr.astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    chroma = (bgr_float.max(axis=2) - bgr_float.min(axis=2)) / 160.0
    confidence = np.maximum(saturation, chroma)
    return np.clip(confidence, 0.0, 1.0)


def _evaluate_liquid_component(
    component: np.ndarray,
    confidence: np.ndarray,
    edges: np.ndarray,
    bbox: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    config: EstimatorConfig,
) -> tuple[float, LiquidHeightResult] | None:
    x, y, w, h = bbox
    row_fill = np.zeros(image_h, dtype=np.float32)
    for row in range(y, min(image_h, y + h)):
        row_component = component[row, x : x + w]
        if row_component.size == 0:
            continue
        row_fill[row] = float(np.mean(row_component))

    dense_rows = row_fill >= max(0.55, config.min_row_fill_fraction)
    dense_rows = _close_arbitrary_row_gaps(dense_rows, y, y + h - 1, max_gap_rows=6)
    dense_run = _dominant_dense_row_run(dense_rows, y, y + h - 1, min_rows=config.min_filled_rows)
    if dense_run is None:
        return None
    surface_y, dense_bottom_y, has_multiple_dense_runs = dense_run

    liquid_rows = component[surface_y : dense_bottom_y + 1, x : x + w]
    if not np.any(liquid_rows):
        return None
    ys, xs = np.nonzero(liquid_rows)
    liquid_x1 = int(x + xs.min())
    liquid_x2 = int(x + xs.max())
    liquid_bottom_y = int(surface_y + ys.max())
    original_liquid_bottom_y = liquid_bottom_y
    liquid_bottom_y = _trim_reexpanding_tail(component, (x, y, w, h), surface_y, liquid_bottom_y)
    tail_was_trimmed = liquid_bottom_y < original_liquid_bottom_y - 10
    liquid_rows = component[surface_y : liquid_bottom_y + 1, x : x + w]
    if not np.any(liquid_rows):
        return None
    ys, xs = np.nonzero(liquid_rows)
    liquid_x1 = int(x + xs.min())
    liquid_x2 = int(x + xs.max())
    liquid_w = max(1, liquid_x2 - liquid_x1 + 1)
    liquid_h = max(1, liquid_bottom_y - surface_y + 1)
    if liquid_w / float(liquid_h) < 0.70:
        return None

    if tail_was_trimmed or has_multiple_dense_runs:
        surface_y = _refine_surface_with_horizontal_edge(edges, liquid_x1, liquid_x2, surface_y, liquid_bottom_y)
        liquid_h = max(1, liquid_bottom_y - surface_y + 1)

    cup_top_y, rim_score = _find_rim_top(edges, liquid_x1, liquid_x2, surface_y, liquid_bottom_y)
    if cup_top_y is None:
        cup_top_y = max(0, surface_y - int(round(liquid_w * 0.55)))
        rim_score = 0.05

    cup_bottom_y = _find_cup_bottom(edges, liquid_x1, liquid_x2, liquid_bottom_y, image_h)
    if cup_bottom_y is None:
        cup_bottom_y = liquid_bottom_y

    usable_height = cup_bottom_y - cup_top_y
    if usable_height < 45 or surface_y <= cup_top_y or surface_y >= cup_bottom_y:
        return None

    ratio = float(np.clip((cup_bottom_y - surface_y) / float(usable_height), 0.0, 1.0))
    if ratio <= 0.03:
        return None

    expand_x = int(round(liquid_w * 0.12))
    result_x = max(0, liquid_x1 - expand_x)
    result_w = min(image_w - result_x, liquid_w + 2 * expand_x)
    result_y = max(0, cup_top_y)
    result_h = min(image_h - result_y, max(cup_bottom_y + int(liquid_w * 0.35), liquid_bottom_y) - result_y)
    max_result_height = image_h * (0.86 if rim_score >= 0.35 else 0.62)
    if result_w > image_w * 0.58 or result_h > max_result_height:
        return None
    goblet_bbox = (int(result_x), int(result_y), int(result_w), int(result_h))

    liquid_mask = np.zeros_like(edges, dtype=np.uint8)
    liquid_mask[component] = 255
    overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(overlay, (goblet_bbox[0], goblet_bbox[1]), (goblet_bbox[0] + goblet_bbox[2], goblet_bbox[1] + goblet_bbox[3]), (0, 255, 0), 2)
    cv2.line(overlay, (goblet_bbox[0], surface_y), (goblet_bbox[0] + goblet_bbox[2], surface_y), (255, 0, 255), 2)

    component_score = min(1.0, (liquid_w * liquid_h) / float(image_w * image_h * 0.035))
    aspect_score = float(np.clip((liquid_w / float(liquid_h) - 0.70) / 3.30, 0.0, 1.0))
    confidence_score = float(np.clip(np.mean(confidence[component]), 0.0, 1.0))
    score = 0.40 * component_score + 0.25 * confidence_score + 0.20 * aspect_score + 0.15 * rim_score
    result = LiquidHeightResult(
        ratio=ratio,
        confidence=float(np.clip(score, 0.0, 1.0)),
        goblet_bbox=goblet_bbox,
        liquid_surface_y=int(surface_y),
        cup_top_y=int(cup_top_y),
        cup_bottom_y=int(cup_bottom_y),
        debug_images={
            "template_match": overlay,
            "liquid_mask": liquid_mask,
        },
    )
    return score, result


def _trim_reexpanding_tail(
    component: np.ndarray,
    bbox: tuple[int, int, int, int],
    surface_y: int,
    liquid_bottom_y: int,
) -> int:
    x, _, w, _ = bbox
    if liquid_bottom_y - surface_y < 90:
        return liquid_bottom_y

    widths: list[int] = []
    for row in range(surface_y, liquid_bottom_y + 1):
        cols = np.flatnonzero(component[row, x : x + w])
        widths.append(0 if cols.size == 0 else int(cols.max() - cols.min() + 1))

    smoothed = np.convolve(np.asarray(widths, dtype=np.float32), np.ones(11, dtype=np.float32) / 11.0, mode="same")
    if smoothed.size < 80:
        return liquid_bottom_y

    peak = float(smoothed[: max(1, int(smoothed.size * 0.70))].max())
    if peak < 80:
        return liquid_bottom_y

    start = max(20, int(smoothed.size * 0.25))
    stop = max(start, smoothed.size - 30)
    for index in range(start, stop):
        previous_peak = float(smoothed[max(0, index - 70) : index + 1].max())
        future_peak = float(smoothed[index + 8 : min(smoothed.size, index + 80)].max())
        current = float(smoothed[index])
        if current <= peak * 0.84 and current <= previous_peak * 0.90 and future_peak >= current + peak * 0.12:
            return int(surface_y + index)

    return liquid_bottom_y


def _refine_surface_with_horizontal_edge(edges: np.ndarray, liquid_x1: int, liquid_x2: int, surface_y: int, liquid_bottom_y: int) -> int:
    liquid_w = max(1, liquid_x2 - liquid_x1 + 1)
    search_top = surface_y + 6
    search_bottom = min(liquid_bottom_y - 8, surface_y + int(max(35, liquid_w * 0.32)))
    if search_bottom <= search_top:
        return surface_y

    band = edges[search_top : search_bottom + 1, liquid_x1 : liquid_x2 + 1] > 0
    if band.size == 0:
        return surface_y

    density = band.mean(axis=1)
    smoothed = np.convolve(density, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    best_idx = int(np.argmax(smoothed))
    best_score = float(smoothed[best_idx])
    if best_score < 0.055:
        return surface_y
    return int(search_top + best_idx)


def _estimate_from_yolo_segmenter(
    bgr: np.ndarray,
    edges: np.ndarray,
    config: EstimatorConfig,
    debug_images: dict[str, np.ndarray],
    model_path: str,
) -> LiquidHeightResult | None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            'YOLO mode requires optional dependencies. Install with: python -m pip install -e ".[yolo]"'
        ) from exc

    model = YOLO(model_path)
    predictions = model.predict(bgr, verbose=False)
    if not predictions:
        return None

    confidence = _opaque_confidence_image(bgr)
    image_h, image_w = bgr.shape[:2]
    best: LiquidHeightResult | None = None
    best_score = 0.0
    overlay = bgr.copy()

    for prediction in predictions:
        boxes = getattr(prediction, "boxes", None)
        masks = getattr(prediction, "masks", None)
        if boxes is None:
            continue

        class_names = getattr(prediction, "names", None) or getattr(model, "names", {})
        classes = _to_numpy(getattr(boxes, "cls", []))
        scores = _to_numpy(getattr(boxes, "conf", []))
        xyxy = _to_numpy(getattr(boxes, "xyxy", []))
        mask_data = None if masks is None else _to_numpy(getattr(masks, "data", None))

        for index, class_id_float in enumerate(classes):
            class_id = int(class_id_float)
            class_name = _class_name(class_names, class_id)
            if class_name not in {"wine glass", "cup"}:
                continue
            if index >= len(xyxy):
                continue

            detector_score = float(scores[index]) if index < len(scores) else 0.50
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy[index][:4])
            x1 = max(0, min(image_w - 1, x1))
            x2 = max(0, min(image_w, x2))
            y1 = max(0, min(image_h - 1, y1))
            y2 = max(0, min(image_h, y2))
            if x2 - x1 < 40 or y2 - y1 < 60:
                continue

            goblet_mask = np.zeros((image_h, image_w), dtype=bool)
            if mask_data is not None and index < len(mask_data):
                mask = mask_data[index]
                if mask.shape != (image_h, image_w):
                    mask = cv2.resize(mask.astype(np.float32), (image_w, image_h), interpolation=cv2.INTER_LINEAR)
                goblet_mask = mask > 0.50
            else:
                goblet_mask[y1:y2, x1:x2] = True

            component_mask = (confidence >= config.blob_pixel_threshold) & goblet_mask
            candidate = _best_component_in_mask(component_mask, confidence, edges, image_w, image_h, config)
            if candidate is None:
                continue

            score, result = candidate
            score = float(np.clip(score * 0.75 + detector_score * 0.25, 0.0, 1.0))
            result = replace(result, confidence=score)
            if score > best_score:
                best_score = score
                best = result

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 2)

    if best is None:
        return None

    debug_images["yolo_segment"] = overlay
    best.debug_images.update(debug_images)
    return best


def _best_component_in_mask(
    component_mask: np.ndarray,
    confidence: np.ndarray,
    edges: np.ndarray,
    image_w: int,
    image_h: int,
    config: EstimatorConfig,
) -> tuple[float, LiquidHeightResult] | None:
    mask = (component_mask.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best: tuple[float, LiquidHeightResult] | None = None
    best_score = 0.0
    min_area = max(300, int(image_h * image_w * 0.0005))
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < min_area or w < 25 or h < 12:
            continue
        candidate = _evaluate_liquid_component(labels == label, confidence, edges, (x, y, w, h), image_w, image_h, config)
        if candidate is None:
            continue
        score, _ = candidate
        if score > best_score:
            best_score = score
            best = candidate
    return best


def _to_numpy(value) -> np.ndarray:
    if value is None:
        return np.array([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_name(class_names, class_id: int) -> str:
    if isinstance(class_names, dict):
        return str(class_names.get(class_id, class_id)).lower()
    if isinstance(class_names, (list, tuple)) and 0 <= class_id < len(class_names):
        return str(class_names[class_id]).lower()
    return str(class_id)


def _close_arbitrary_row_gaps(rows: np.ndarray, start: int, end: int, *, max_gap_rows: int) -> np.ndarray:
    closed = rows.copy()
    gap_start: int | None = None
    for row in range(max(0, start), min(len(rows), end + 1)):
        if closed[row]:
            if gap_start is not None and row - gap_start <= max_gap_rows:
                closed[gap_start:row] = True
            gap_start = None
        elif gap_start is None:
            gap_start = row
    return closed


def _dominant_dense_row_run(rows: np.ndarray, start: int, end: int, *, min_rows: int) -> tuple[int, int, bool] | None:
    runs: list[tuple[int, int]] = []
    row = max(0, start)
    stop = min(len(rows) - 1, end)
    while row <= stop:
        while row <= stop and not rows[row]:
            row += 1
        if row > stop:
            break
        run_start = row
        while row <= stop and rows[row]:
            row += 1
        run_end = row - 1
        if run_end - run_start + 1 >= min_rows:
            runs.append((run_start, run_end))

    if not runs:
        return None

    longest = max(end_y - start_y + 1 for start_y, end_y in runs)
    strong_runs = [(start_y, end_y) for start_y, end_y in runs if end_y - start_y + 1 >= longest * 0.75]
    run_start, run_end = min(strong_runs, key=lambda run: run[0])
    return run_start, run_end, len(runs) > 1


def _find_rim_top(edges: np.ndarray, liquid_x1: int, liquid_x2: int, surface_y: int, liquid_bottom_y: int) -> tuple[int | None, float]:
    image_h, image_w = edges.shape[:2]
    liquid_w = max(1, liquid_x2 - liquid_x1 + 1)
    x1 = max(0, liquid_x1 - int(liquid_w * 0.28))
    x2 = min(image_w - 1, liquid_x2 + int(liquid_w * 0.28))
    search_top = max(0, surface_y - int(max(70, liquid_w * 0.85)))
    search_bottom = max(0, surface_y - 8)
    if search_bottom <= search_top:
        return None, 0.0

    band = edges[search_top:search_bottom + 1, x1:x2 + 1] > 0
    if band.size == 0:
        return None, 0.0
    density = band.mean(axis=1)
    smoothed = np.convolve(density, np.ones(5) / 5.0, mode="same")
    best_idx = int(np.argmax(smoothed))
    best_score = float(smoothed[best_idx])
    if best_score < 0.018:
        return None, best_score

    rim_y = search_top + best_idx
    if liquid_bottom_y - rim_y < 45:
        return None, best_score
    return int(rim_y), min(1.0, best_score * 14.0)


def _find_cup_bottom(edges: np.ndarray, liquid_x1: int, liquid_x2: int, liquid_bottom_y: int, image_h: int) -> int | None:
    liquid_w = max(1, liquid_x2 - liquid_x1 + 1)
    x1 = max(0, liquid_x1 - int(liquid_w * 0.12))
    x2 = min(edges.shape[1] - 1, liquid_x2 + int(liquid_w * 0.12))
    search_top = max(0, liquid_bottom_y - 8)
    search_bottom = min(image_h - 1, liquid_bottom_y + int(max(14, liquid_w * 0.20)))
    if search_bottom <= search_top:
        return None
    band = edges[search_top:search_bottom + 1, x1:x2 + 1] > 0
    density = band.mean(axis=1)
    if density.size == 0 or float(density.max()) < 0.012:
        return int(liquid_bottom_y)
    return int(search_top + int(np.argmax(density)))


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
        if width < 95 or height < 140 or width > image_w or height > image_h:
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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


# Bundled detector weights (falls back to ultralytics auto-download by name).
_PKG_DIR = Path(__file__).resolve().parent
_BUNDLED_MODEL = _PKG_DIR.parent.parent / "models" / "yolo11n.pt"
DEFAULT_YOLO_MODEL: str = str(_BUNDLED_MODEL) if _BUNDLED_MODEL.exists() else "yolo11n.pt"

# COCO class ids for transparent drinkware.
_WINE_GLASS = 40
_CUP = 41

EstimationMethod = Literal["yolo", "opencv", "auto"]


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
    # Juice (opaque drink) colour evidence.
    juice_pixel_threshold: float = 0.32
    min_row_fill_fraction: float = 0.30
    band_width_ratio: float = 0.55
    max_gap_rows: int = 4
    min_filled_rows: int = 6
    # Fixed-goblet shape prior: full bowl-cavity height (rim -> bowl bottom) as a
    # multiple of the bowl inner width, calibrated on real photos.
    cavity_over_bowl_width: float = 0.89
    # Candidate gating.
    yolo_conf: float = 0.10
    min_box_w: int = 30
    min_box_h: int = 45
    edge_touch_margin: int = 4


def estimate_liquid_height_ratio(
    image: str | Path | np.ndarray,
    *,
    debug: bool = False,
    method: EstimationMethod = "yolo",
    yolo_model: str | Path = DEFAULT_YOLO_MODEL,
) -> float | None | LiquidHeightResult:
    if method not in {"yolo", "opencv", "auto"}:
        raise ValueError("method must be one of: yolo, opencv, auto")

    bgr = _load_image_bgr(image)
    juice = _juice_confidence(bgr)
    debug_images: dict[str, np.ndarray] = {"juice_confidence": (juice * 255).astype(np.uint8)}

    candidates: list[_Candidate] = []
    if method in {"yolo", "auto"}:
        try:
            candidates = _yolo_candidates(bgr, str(yolo_model), EstimatorConfig())
        except ImportError:
            if method == "yolo":
                raise
    if not candidates and method in {"opencv", "auto"}:
        candidates = _colorblob_candidates(bgr, juice, EstimatorConfig())

    result = _pick_best(bgr, juice, candidates, EstimatorConfig(), debug_images)
    if result is None:
        empty = LiquidHeightResult(None, 0.0, None, None, None, None, debug_images)
        return empty if debug else None
    return result if debug else result.ratio


@dataclass(frozen=True)
class _Candidate:
    bbox: tuple[int, int, int, int]
    det_conf: float
    is_wine_glass: bool
    touches_edge: bool


def _pick_best(
    bgr: np.ndarray,
    juice: np.ndarray,
    candidates: list[_Candidate],
    config: EstimatorConfig,
    debug_images: dict[str, np.ndarray],
) -> LiquidHeightResult | None:
    best: LiquidHeightResult | None = None
    best_score = 0.0
    for cand in candidates:
        measured = _measure_liquid_in_box(bgr, juice, cand, config)
        if measured is None:
            continue
        score, result = measured
        if score > best_score:
            best_score = score
            best = result
    if best is not None:
        overlay = _draw_overlay(bgr, best)
        debug_images["overlay"] = overlay
        best.debug_images.update(debug_images)
    return best


def _measure_liquid_in_box(
    bgr: np.ndarray,
    juice: np.ndarray,
    cand: _Candidate,
    config: EstimatorConfig,
) -> tuple[float, LiquidHeightResult] | None:
    x, y, w, h = cand.bbox
    sub = juice[y : y + h, x : x + w]
    if sub.size == 0:
        return None

    mask = (sub >= config.juice_pixel_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    blob = _dominant_bottom_blob(mask)
    if blob is None:
        return None
    blob_mask, bottomness = blob

    band = _liquid_band(blob_mask, config)
    if band is None:
        return None
    surf_local, bowl_bottom_local = band

    # Reject bottle-like fills: in a cup the rim band (top of the box) is empty
    # glass, so juice must NOT extend to the very top of the detection box.
    if surf_local <= h * 0.06:
        return None
    juice_height = bowl_bottom_local - surf_local
    if juice_height < config.min_filled_rows:
        return None
    row_fill = blob_mask.mean(axis=1)

    # Scale ruler: the bowl inner width (widest liquid row) is a fixed-shape
    # invariant that, unlike the YOLO box height, does not depend on how much
    # stem/base the detector happened to include. The full bowl cavity height
    # (rim -> bowl bottom) is a calibrated multiple of that width.
    bowl_width = float(blob_mask[surf_local : bowl_bottom_local + 1].sum(axis=1).max())
    if bowl_width <= 1.0:
        return None
    cavity_height = config.cavity_over_bowl_width * bowl_width
    if cavity_height <= 1.0:
        return None
    ratio = float(np.clip(juice_height / cavity_height, 0.0, 1.0))

    rim_local = bowl_bottom_local - cavity_height
    surface_y = y + surf_local
    cup_bottom_y = y + bowl_bottom_local
    cup_top_y = int(round(y + rim_local))

    confidence = _confidence(
        cand=cand,
        box_h=h,
        bowl_bottom_local=bowl_bottom_local,
        bottomness=bottomness,
        fill_in_juice=float(row_fill[surf_local : bowl_bottom_local + 1].mean()),
        config=config,
    )

    result = LiquidHeightResult(
        ratio=ratio,
        confidence=confidence,
        goblet_bbox=cand.bbox,
        liquid_surface_y=int(surface_y),
        cup_top_y=int(cup_top_y),
        cup_bottom_y=int(cup_bottom_y),
        debug_images={},
    )
    score = confidence + (0.05 if cand.is_wine_glass else 0.0)
    return score, result


def _confidence(
    *,
    cand: _Candidate,
    box_h: int,
    bowl_bottom_local: int,
    bottomness: float,
    fill_in_juice: float,
    config: EstimatorConfig,
) -> float:
    # Bowl bottom should sit at a shape-consistent fraction of the box.
    expected = 0.67
    bowl_frac = bowl_bottom_local / float(box_h)
    geom = 1.0 - min(1.0, abs(bowl_frac - expected) / 0.30)
    # A real cup has an empty rim band: juice should occupy a middle band, not
    # the whole box (bottomness near 0.5-0.8); penalise full-frame fills.
    band = 1.0 - min(1.0, abs(bottomness - 0.68) / 0.42)
    edge = 0.7 if cand.touches_edge else 1.0
    score = (0.35 * cand.det_conf + 0.30 * geom + 0.20 * band + 0.15 * fill_in_juice) * edge
    return float(np.clip(score, 0.0, 1.0))


def _dominant_bottom_blob(mask: np.ndarray) -> tuple[np.ndarray, float] | None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    h, w = mask.shape
    best_label = -1
    best_area = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        if bw < w * 0.18:
            continue
        if area > best_area:
            best_area = area
            best_label = label
    if best_label < 0:
        return None
    blob = (labels == best_label).astype(np.float32)
    rowfill = blob.mean(axis=1)
    n = len(rowfill)
    top_half = float(rowfill[: n // 2].mean())
    bot_half = float(rowfill[n // 2 :].mean())
    bottomness = bot_half / (top_half + bot_half + 1e-6)
    return blob, bottomness


def _close_gaps(rows: np.ndarray, max_gap: int) -> np.ndarray:
    closed = rows.copy()
    gap_start: int | None = None
    for i, on in enumerate(rows):
        if on:
            if gap_start is not None and i - gap_start <= max_gap:
                closed[gap_start:i] = True
            gap_start = None
        elif gap_start is None:
            gap_start = i
    return closed


def _liquid_band(blob_mask: np.ndarray, config: EstimatorConfig) -> tuple[int, int] | None:
    """Top/bottom rows of the solid liquid body inside the bowl.

    Uses per-row width: the liquid fills the bowl, so its rows are near the peak
    width. Isolated narrow rows above (background bleed at the box top) or a
    detached band below (stem/base reflection) fall outside the dominant
    full-width run and are excluded.
    """
    widths = blob_mask.sum(axis=1).astype(np.float32)
    if widths.max() <= 0:
        return None
    smooth = np.convolve(widths, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    peak = float(smooth.max())
    solid = smooth >= peak * config.band_width_ratio
    solid = _close_gaps(solid, config.max_gap_rows)

    runs: list[tuple[int, int]] = []
    i = 0
    n = len(solid)
    while i < n:
        if not solid[i]:
            i += 1
            continue
        start = i
        while i < n and solid[i]:
            i += 1
        runs.append((start, i - 1))
    runs = [(a, b) for a, b in runs if b - a + 1 >= config.min_filled_rows]
    if not runs:
        return None
    # The bowl liquid is the longest near-peak-width run.
    top, bottom = max(runs, key=lambda r: r[1] - r[0])
    return top, bottom



def _juice_confidence(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    bgr_f = bgr.astype(np.float32)
    chroma = (bgr_f.max(axis=2) - bgr_f.min(axis=2)) / 160.0
    # Orange/red hue band in OpenCV H (0-180): 0-25 or 165-180.
    hue_ok = ((h <= 25.0) | (h >= 165.0)).astype(np.float32)
    colour = np.maximum(s, np.clip(chroma, 0.0, 1.0))
    confidence = hue_ok * colour * (v >= 0.20).astype(np.float32)
    return np.clip(confidence, 0.0, 1.0)


def _yolo_candidates(bgr: np.ndarray, model_path: str, config: EstimatorConfig) -> list[_Candidate]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            'YOLO mode requires optional dependencies. Install with: python -m pip install -e ".[yolo]"'
        ) from exc

    model = YOLO(model_path)
    image_h, image_w = bgr.shape[:2]
    out: list[_Candidate] = []
    for pred in model.predict(bgr, verbose=False, conf=config.yolo_conf):
        boxes = getattr(pred, "boxes", None)
        if boxes is None:
            continue
        cls = _to_numpy(getattr(boxes, "cls", []))
        conf = _to_numpy(getattr(boxes, "conf", []))
        xyxy = _to_numpy(getattr(boxes, "xyxy", []))
        for i, cid_f in enumerate(cls):
            cid = int(cid_f)
            if cid not in {_WINE_GLASS, _CUP}:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy[i][:4])
            x1, x2 = max(0, x1), min(image_w, x2)
            y1, y2 = max(0, y1), min(image_h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < config.min_box_w or bh < config.min_box_h:
                continue
            m = config.edge_touch_margin
            touches = x1 <= m or y1 <= m or x2 >= image_w - m or y2 >= image_h - m
            out.append(
                _Candidate(
                    bbox=(x1, y1, bw, bh),
                    det_conf=float(conf[i]) if i < len(conf) else 0.3,
                    is_wine_glass=cid == _WINE_GLASS,
                    touches_edge=touches,
                )
            )
    return out


def _colorblob_candidates(bgr: np.ndarray, juice: np.ndarray, config: EstimatorConfig) -> list[_Candidate]:
    image_h, image_w = bgr.shape[:2]
    mask = (juice >= config.juice_pixel_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: list[_Candidate] = []
    min_area = max(400, int(image_h * image_w * 0.0008))
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < min_area or w < config.min_box_w:
            continue
        # Expand upward to include the empty rim band above the juice so the
        # detection box approximates the full goblet (rim -> bowl bottom).
        grow_up = int(h * 1.4)
        ny = max(0, y - grow_up)
        nh = (y + h) - ny
        m = config.edge_touch_margin
        touches = x <= m or x + w >= image_w - m
        out.append(
            _Candidate(
                bbox=(x, ny, w, nh),
                det_conf=0.5,
                is_wine_glass=False,
                touches_edge=touches,
            )
        )
    return out


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


def _load_image_bgr(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
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


def _draw_overlay(bgr: np.ndarray, result: LiquidHeightResult) -> np.ndarray:
    overlay = bgr.copy()
    if result.goblet_bbox is not None:
        x, y, w, h = result.goblet_bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if result.cup_top_y is not None:
            cv2.line(overlay, (x, result.cup_top_y), (x + w, result.cup_top_y), (255, 200, 0), 2)
        if result.cup_bottom_y is not None:
            cv2.line(overlay, (x, result.cup_bottom_y), (x + w, result.cup_bottom_y), (255, 200, 0), 2)
        if result.liquid_surface_y is not None:
            cv2.line(overlay, (x, result.liquid_surface_y), (x + w, result.liquid_surface_y), (255, 0, 255), 2)
        if result.ratio is not None:
            cv2.putText(
                overlay,
                f"ratio={result.ratio:.2f} conf={result.confidence:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return overlay

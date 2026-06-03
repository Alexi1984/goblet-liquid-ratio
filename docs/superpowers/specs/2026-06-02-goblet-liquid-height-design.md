# Goblet Liquid Height Ratio Design

## Goal

Build a lightweight computer vision utility that estimates the liquid surface height ratio in images containing the same transparent goblet shape as the reference image. The goblet may appear at different positions in the image, but the camera viewpoint is expected to stay similar. The liquid is opaque and may have many colors.

The primary output is the liquid height ratio:

```text
ratio = liquid_height / usable_inner_cup_height
```

The value is clamped to `[0.0, 1.0]`.

## Non-Goals

- Do not estimate physical liquid volume by cup geometry integration.
- Do not use heavy segmentation or detection models such as SAM or full YOLO models.
- Do not integrate with the existing RL repository.
- Do not assume the liquid is orange or any single fixed color.

## Project Shape

The project is standalone under:

```text
/home/yuyuyu/桌面/goblet-liquid-ratio
```

The implementation will be a small Python package with:

- `src/goblet_liquid_ratio/` for library code.
- `tests/` for pytest tests.
- `examples/` for optional scripts and sample images.
- `pyproject.toml` for dependencies and test configuration.

Dependencies should stay small:

- `numpy`
- `opencv-python-headless`
- `Pillow` only if useful for image loading convenience
- `pytest` for tests

## Public API

The core function will accept a file path or an image array:

```python
estimate_liquid_height_ratio(image, *, debug=False)
```

Default return:

```python
float | None
```

- Returns `float` when the goblet and liquid surface are detected with enough confidence.
- Returns `None` when the goblet cannot be located or the liquid evidence is too weak.

When `debug=True`, return a structured result instead:

```python
LiquidHeightResult(
    ratio: float | None,
    confidence: float,
    goblet_bbox: tuple[int, int, int, int] | None,
    liquid_surface_y: int | None,
    cup_top_y: int | None,
    cup_bottom_y: int | None,
    debug_images: dict[str, np.ndarray],
)
```

This keeps the normal function simple while allowing inspection when tuning thresholds.

## Algorithm

> **Implementation note (as built).** The sections below describe the original
> edge-template plan. The shipped estimator instead localizes the goblet with a
> small YOLO detector (`yolo11n`, COCO `wine glass` / `cup`) and measures juice
> inside the chosen box. This change was driven by ground-truth evaluation: the
> pure colour-blob/template estimator scored MAE 0.125 with 3 hard failures on
> 15 real photos because red/orange neighbours (bottles) bled into the cup blob.
> YOLO localization isolates the cup and brought MAE to 0.068 with 0 hard
> failures. Key decisions that differ from the plan:
>
> - **Localization** is YOLO detection (not edge template matching, which never
>   matched the real cluttered edges). `method="opencv"` keeps a colour-blob
>   localizer as a torch-free fallback.
> - **Liquid band** is the widest near-peak-width run of the juice blob, which
>   excludes background bleed above and stem/base reflection below.
> - **Scale ruler** is the bowl inner width (a fixed-shape invariant), not the
>   detection box height, because YOLO boxes vary in how much stem they include.
> - **Colour scope** is orange/red juice specifically (hue-gated), matching the
>   actual drink, rather than a fully colour-agnostic signal.

### 1. Image Loading and Normalization

Accept either:

- path-like input
- RGB/BGR numpy array

Internally normalize to BGR or RGB consistently and resize only for detection speed if the input is very large. Preserve enough resolution for liquid surface detection.

### 2. Goblet Localization

Use template-based localization because the goblet shape is fixed and the camera viewpoint is similar.

The first implementation will support a stored goblet template or reference mask. It will use multi-scale OpenCV template matching over edge/gradient images instead of raw color. Edge-based matching is less sensitive to drink color and lighting.

Expected output:

```text
goblet_bbox = (x, y, w, h)
```

If the best match score is below the configured threshold, return `None`.

### 3. Cup Interior Region

Once the goblet bbox is found, resize the goblet crop to a canonical size. Apply a fixed cup-interior mask in canonical coordinates.

The mask excludes:

- stem
- base
- outside glass rim
- background outside the cup bowl

It keeps only the region where liquid can appear.

The canonical cup top and bottom y-coordinates define the height denominator.

### 4. Liquid Evidence

Because drink color varies but is opaque, do not use a fixed hue threshold. Combine color-agnostic signals inside the cup mask:

- saturation or chroma above local background
- brightness difference from nearby clear glass/background
- reduced background visibility or stronger filled-region texture
- horizontal edge response near the liquid surface

The implementation should compute a per-pixel liquid confidence mask, then aggregate confidence per horizontal row.

### 5. Liquid Surface Detection

Find the top boundary of the filled liquid region by scanning row confidence from bottom to top.

Use smoothing and minimum-width constraints so small highlights, reflections, and glass rim edges do not become the detected liquid line.

The ratio is:

```text
ratio = (cup_bottom_y - liquid_surface_y) / (cup_bottom_y - cup_top_y)
```

Clamp to `[0.0, 1.0]`.

### 6. Confidence and Failure Handling

Confidence combines:

- goblet template match score
- strength and continuity of liquid evidence
- plausibility of detected surface position

Return `None` when:

- goblet match is weak
- liquid mask is too sparse
- detected surface is outside the cup interior
- row evidence is ambiguous

## Testing Strategy

Follow test-first implementation.

Initial tests:

1. Synthetic image with known goblet-like mask and low liquid level returns a low ratio.
2. Synthetic image with known medium liquid level returns a medium ratio.
3. Synthetic image with known high liquid level returns a high ratio.
4. Same synthetic cup shifted in the image still returns the same ratio.
5. No goblet returns `None`.

Optional later tests:

- Use a real reference image once sample files are added.
- Save debug overlays for manual QA.

## CLI

Add a small command after the library works:

```bash
python -m goblet_liquid_ratio path/to/image.jpg --debug out/debug
```

It should print the ratio and optionally write debug masks/overlays.

## Risks

- Transparent glass edges can be hard to locate against bright backgrounds.
- Strong reflections on the glass may look like liquid boundaries.
- Very low-saturation opaque drinks, such as milk or black coffee in some lighting, may require threshold tuning.
- If scale or camera angle changes too much, template matching may need extra templates.

## Acceptance Criteria

- The project is standalone and does not modify the RL repository.
- The core function uses only lightweight CV dependencies.
- The public API returns a liquid height ratio or `None`.
- Unit tests cover synthetic heights and shifted cup position.
- Debug mode exposes enough intermediate data to tune real-image behavior.

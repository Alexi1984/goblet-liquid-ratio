# Goblet Liquid Ratio

Lightweight OpenCV utility for estimating the liquid surface height ratio in a fixed transparent goblet shape.

It is designed for images where:

- the goblet shape is the same as the reference goblet
- the camera viewpoint is similar
- the goblet may appear at different image positions
- the drink is opaque, but its color may vary

It reports height ratio, not physical volume:

```text
ratio = liquid_height / usable_inner_cup_height
```

## Install

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
```

Optional YOLO segmentation support:

```bash
.venv/bin/python -m pip install -e ".[test,yolo]"
```

## Python API

```python
from goblet_liquid_ratio import estimate_liquid_height_ratio

ratio = estimate_liquid_height_ratio("path/to/image.jpg")
print(ratio)
```

Use YOLO-seg as an optional first-stage goblet cutout when the dependency is installed:

```python
ratio = estimate_liquid_height_ratio("path/to/image.jpg", method="yolo")
```

The function returns:

- `float` in `[0.0, 1.0]` when the goblet and liquid surface are detected
- `None` when the goblet or liquid surface evidence is too weak

Debug mode returns a structured result:

```python
result = estimate_liquid_height_ratio("path/to/image.jpg", debug=True)
print(result.ratio, result.confidence, result.goblet_bbox)
print(result.debug_images.keys())
```

## CLI

```bash
.venv/bin/python -m goblet_liquid_ratio path/to/image.jpg
```

Process a directory of images:

```bash
.venv/bin/python -m goblet_liquid_ratio /home/yuyuyu/桌面 --debug /tmp/goblet-debug
```

Use YOLO mode:

```bash
.venv/bin/python -m goblet_liquid_ratio path/to/image.jpg --method yolo
```

With debug images:

```bash
.venv/bin/python -m goblet_liquid_ratio path/to/image.jpg --debug debug_outputs
```

If an image file name contains shell characters such as `&`, either quote the full path or pass the parent directory:

```bash
.venv/bin/python -m goblet_liquid_ratio 'path/to/image&from&wechat.jpeg'
```

## How It Works

The default estimator uses traditional CV only:

1. Detect opaque colored liquid evidence using saturation and chroma.
2. Split liquid candidates into connected components.
3. Ignore narrow bottles, huge background regions, and connected base reflections.
4. Estimate the liquid surface from the dominant filled row band and nearby horizontal edge evidence.
5. Fall back to fixed goblet edge template matching when color evidence is weak.

YOLO mode uses a small Ultralytics segmentation model first to cut out `wine glass` or `cup`, then applies the same liquid-surface measurement inside that mask.

## Limits

This is intentionally small and tunable, not a general object segmentation system.

Likely failure cases:

- strong camera angle or scale changes beyond the template search range
- very low-contrast opaque liquids against a similar background
- heavy reflections that form wide horizontal bands
- a different cup shape

Add real sample images and inspect `debug_images` or `--debug` outputs when tuning thresholds.

## Tests

```bash
.venv/bin/python -m pytest -q
```

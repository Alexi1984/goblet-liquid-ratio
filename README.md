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

## Python API

```python
from goblet_liquid_ratio import estimate_liquid_height_ratio

ratio = estimate_liquid_height_ratio("path/to/image.jpg")
print(ratio)
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

With debug images:

```bash
.venv/bin/python -m goblet_liquid_ratio path/to/image.jpg --debug debug_outputs
```

## How It Works

The estimator uses traditional CV only:

1. Generate a fixed goblet edge template.
2. Locate the goblet with multi-scale OpenCV edge template matching.
3. Resize the matched crop to canonical goblet coordinates.
4. Apply a fixed cup-interior mask.
5. Detect opaque liquid evidence using saturation, chroma, and brightness difference from local background.
6. Estimate the top of the bottom-connected filled row region as the liquid surface.

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

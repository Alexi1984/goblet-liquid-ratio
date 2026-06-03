# Goblet Liquid Ratio

Estimate the **liquid height ratio** of orange/red juice in a fixed transparent
goblet, from cluttered desktop photos.

```text
ratio = liquid_height / bowl_cavity_height        # in [0.0, 1.0]
```

It is designed for images where:

- the goblet shape is the same coupe/bowl shape as the reference photos
- the camera viewpoint is similar
- the goblet may appear anywhere in the frame, at different scales
- the drink is an opaque **orange/red** juice
- other objects (water bottles, red/blue bottles, cups) share the scene

## How it works

The hard part of these photos is that the goblet is transparent and other
red/orange objects sit right next to it, so a pure colour-blob segmenter bleeds
across them. The pipeline avoids that by **localizing the cup first**:

1. **Detect** `wine glass` / `cup` boxes with a small YOLO model (`yolo11n`,
   5 MB). This isolates the goblet from neighbouring bottles.
2. **Pick** the box that actually holds juice and is shape-consistent (an empty
   transparent rim band on top, a solid juice band below — not a full-frame
   fill like a bottle), preferring `wine glass` and down-weighting boxes that
   touch the image edge.
3. **Measure** inside the chosen box using orange/red colour evidence:
   - the **bottom** of the solid juice band is the bowl bottom (juice always
     pools at the bottom — a reliable anchor),
   - the **top** of the band is the liquid surface,
   - the band is taken as the widest near-peak-width run, so background bleed
     above and stem/base reflections below are excluded.
4. **Scale** to a ratio using a fixed-shape prior: the full bowl cavity height
   (rim → bowl bottom) is a calibrated multiple of the bowl inner width. Using
   the bowl width — not the YOLO box height — makes the ratio robust to how
   much stem/base the detector happened to include.

`method="opencv"` replaces step 1 with a lightweight colour-blob localizer (no
torch), used as a fallback and for synthetic unit tests.

## Accuracy

On 15 hand-annotated real photos (`eval/ground_truth.json`):

| Estimator | MAE | Hard failures (>0.20) | Within 0.10 |
|-----------|-----|------------------------|-------------|
| Old colour-blob | 0.125 | 3 / 15 | 6 / 15 |
| **YOLO-localized** | **0.068** | **0 / 15** | 11 / 15 |

```bash
python eval/run_eval.py --method yolo      # per-image error + MAE
python eval/render_overlays.py             # overlays + contact sheet
```

## Install

YOLO mode needs `torch` + `ultralytics`. The simplest path that does not touch
other environments is an overlay venv that reuses an existing torch install:

```bash
python -m venv --system-site-packages .venv   # reuse a torch-bearing base
.venv/bin/python -m pip install --no-deps ultralytics ultralytics-thop scipy
.venv/bin/python -m pip install --no-deps -e .
```

Lightweight (no torch), colour-blob only:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
```

## Python API

```python
from goblet_liquid_ratio import estimate_liquid_height_ratio

ratio = estimate_liquid_height_ratio("path/to/image.jpg")          # method="yolo" default
ratio = estimate_liquid_height_ratio("path/to/image.jpg", method="opencv")
```

Returns a `float` in `[0.0, 1.0]`, or `None` when no juice-bearing goblet is
found. Debug mode returns a structured result:

```python
r = estimate_liquid_height_ratio("img.jpg", debug=True)
print(r.ratio, r.confidence, r.goblet_bbox)
print(r.liquid_surface_y, r.cup_top_y, r.cup_bottom_y)
print(r.debug_images.keys())   # "juice_confidence", "overlay"
```

## CLI

```bash
.venv/bin/python -m goblet_liquid_ratio path/to/image.jpg
.venv/bin/python -m goblet_liquid_ratio /home/yuyuyu/桌面 --debug /tmp/goblet-debug
.venv/bin/python -m goblet_liquid_ratio path/to/image.jpg --method opencv
```

File names with shell characters such as `&` (e.g. WeChat downloads): quote the
full path, or pass the parent directory.

## Limits

- Tuned for **orange/red** juice; other hues are intentionally ignored.
- Deep-coloured juice with strong rim reflection reads slightly high; pale juice
  in shadow reads slightly low (remaining errors are in this 0.10–0.20 band).
- A very different cup shape would need re-calibrating `cavity_over_bowl_width`.
- If YOLO finds no cup, the function returns `None`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Pure-unit and colour-blob tests always run; the real-photo regression test is
skipped unless `ultralytics` and the sample photos are present.

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

### Use it in your own project (from GitHub)

The default `method="yolo"` needs the optional detector stack
(`torch` + `ultralytics`, a large download). Install **with** the `[yolo]`
extra to get it:

```bash
pip install "git+https://github.com/Alexi1984/goblet-liquid-ratio.git#egg=goblet-liquid-ratio[yolo]"
```

Torch-free install (colour-blob `method="opencv"` only — lighter, but less
robust to red/orange objects next to the cup):

```bash
pip install "git+https://github.com/Alexi1984/goblet-liquid-ratio.git"
```

On first use of `method="yolo"`, ultralytics downloads the `yolo11n` weights
(~5 MB) by name and caches them (needs network once). The weights are **not**
bundled in this repo on purpose — they are an Ultralytics AGPL-3.0 asset, and
shipping them would compromise this project's MIT license (see the license note
below). For offline use, drop a `yolo11n.pt` into
`src/goblet_liquid_ratio/models/` yourself.

> Heads-up: if you call the default `method="yolo"` without the `[yolo]` extra,
> the function raises an `ImportError` that tells you exactly how to fix it
> (install the extra, or pass `method="opencv"`).

### Local development

`torch` is heavy; the simplest path that does not touch other environments is an
overlay venv reusing an existing torch install:

```bash
python -m venv --system-site-packages .venv   # reuse a torch-bearing base
.venv/bin/python -m pip install --no-deps ultralytics ultralytics-thop scipy
.venv/bin/python -m pip install --no-deps -e ".[test]"
```

Torch-free dev install (colour-blob path + tests only):

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

Integrating into your own code, with a graceful fallback when the detector
stack is not installed:

```python
from goblet_liquid_ratio import estimate_liquid_height_ratio

def juice_level(image_path: str) -> float | None:
    try:
        return estimate_liquid_height_ratio(image_path)            # yolo (best)
    except ImportError:
        return estimate_liquid_height_ratio(image_path, method="opencv")  # torch-free fallback
```

A runnable version is in [`examples/quickstart.py`](examples/quickstart.py):

```bash
python examples/quickstart.py path/to/photo.jpg
python examples/quickstart.py path/to/photo.jpg --method opencv
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

## License

This project's source code is **MIT** (see [`LICENSE`](LICENSE)) — free for
commercial and closed-source use.

**Important caveat for the YOLO backend.** `method="yolo"` imports Ultralytics
at runtime and downloads YOLO11 weights, which are **AGPL-3.0**, not MIT. If you
use the YOLO backend, your usage falls under Ultralytics' AGPL-3.0 terms (or
their Enterprise License). To stay fully MIT / closed-source-friendly:

- use `method="opencv"` (NumPy BSD + OpenCV Apache-2.0, no Ultralytics), or
- swap in a non-AGPL detector behind the same interface.

The weights are intentionally **not** committed to this repo, so the repository
itself contains only MIT-licensed code.

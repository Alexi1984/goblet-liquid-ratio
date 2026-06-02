# Goblet Liquid Height Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone lightweight Python utility that estimates opaque liquid height ratio in a fixed-shape transparent goblet whose image position can vary.

**Architecture:** The package exposes one public API, `estimate_liquid_height_ratio`, backed by a generated goblet edge template and a generated canonical cup-interior mask. Detection is traditional OpenCV: multi-scale edge template matching locates the goblet, canonical crop analysis estimates liquid row evidence, and a bottom-connected filled run determines the liquid surface height.

**Tech Stack:** Python 3.12+, NumPy, OpenCV headless, pytest, optional Pillow-free path loading through OpenCV.

---

### File Structure

- Create `pyproject.toml`: package metadata, dependencies, pytest path, console entry point.
- Create `.gitignore`: ignore `.venv`, caches, build outputs, debug image outputs.
- Create `src/goblet_liquid_ratio/__init__.py`: public exports.
- Create `src/goblet_liquid_ratio/core.py`: dataclasses, image loading, template generation, goblet localization, liquid height estimation.
- Create `src/goblet_liquid_ratio/__main__.py`: CLI wrapper for image paths and debug output.
- Create `tests/test_goblet_liquid_ratio.py`: synthetic behavior tests.

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`

- [ ] **Step 1: Write packaging and dependency config**

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "goblet-liquid-ratio"
version = "0.1.0"
description = "Lightweight CV utility for estimating liquid height ratio in a fixed goblet."
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "opencv-python-headless>=4.9",
]

[project.optional-dependencies]
test = ["pytest>=8.0"]

[project.scripts]
goblet-liquid-ratio = "goblet_liquid_ratio.__main__:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
build/
dist/
*.egg-info/
debug/
debug_outputs/
```

- [ ] **Step 2: Create local environment and install package**

Run:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[test]"
```

Expected: dependencies install successfully; `python -c "import cv2"` succeeds inside `.venv`.

- [ ] **Step 3: Run baseline tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: pytest reports no tests collected, or exits successfully once tests exist.

- [ ] **Step 4: Commit scaffold**

```bash
git add pyproject.toml .gitignore
git commit -m "chore: scaffold goblet liquid package"
```

### Task 2: Public API and Synthetic TDD Tests

**Files:**
- Create: `tests/test_goblet_liquid_ratio.py`
- Create: `src/goblet_liquid_ratio/__init__.py`
- Create: `src/goblet_liquid_ratio/core.py`

- [ ] **Step 1: Write failing tests**

Create tests that draw a fixed-shape goblet in different positions and fill the cup interior to known height ratios. The tests call the public API only.

```python
from pathlib import Path

import cv2
import numpy as np

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
    stem = np.array([[ox + 73, oy + 164], [ox + 87, oy + 164], [ox + 87, oy + 226], [ox + 73, oy + 226]])
    base = np.array([[ox + 35, oy + 226], [ox + 125, oy + 226], [ox + 145, oy + 240], [ox + 15, oy + 240]])

    # Background clutter should not be mistaken for the goblet.
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


def test_estimates_low_liquid_height_ratio():
    image = synthetic_goblet_image(0.25, liquid_bgr=(30, 180, 40))
    ratio = estimate_liquid_height_ratio(image)
    assert ratio is not None
    assert ratio == pytest.approx(0.25, abs=0.10)
```

The actual test file will include `pytest` import and tests for low, medium, high, shifted, path input, debug result, and no-goblet behavior.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_goblet_liquid_ratio.py -q
```

Expected: FAIL because `goblet_liquid_ratio` cannot be imported or the public API is missing.

- [ ] **Step 3: Create minimal public module**

Create `__init__.py` exporting `LiquidHeightResult` and `estimate_liquid_height_ratio`. Create `core.py` with a dataclass and a temporary stub returning `None`.

```python
from .core import LiquidHeightResult, estimate_liquid_height_ratio

__all__ = ["LiquidHeightResult", "estimate_liquid_height_ratio"]
```

The stub exists only long enough to turn import errors into behavior failures.

- [ ] **Step 4: Run tests to verify behavioral RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_goblet_liquid_ratio.py -q
```

Expected: tests fail because known liquid ratios return `None`.

### Task 3: Lightweight CV Implementation

**Files:**
- Modify: `src/goblet_liquid_ratio/core.py`
- Modify: `tests/test_goblet_liquid_ratio.py`

- [ ] **Step 1: Implement generated goblet template and mask**

Add constants for canonical geometry and functions:

```python
CANONICAL_WIDTH = 160
CANONICAL_HEIGHT = 260
CUP_TOP_Y = 34
CUP_BOTTOM_Y = 154
```

Implement `_make_template_edges()` and `_make_cup_mask()` using OpenCV drawing calls matching the synthetic fixed goblet shape.

- [ ] **Step 2: Implement image loading and edge template matching**

Implement `_load_image_bgr`, `_edge_image`, and `_locate_goblet`. Use Canny edges and multi-scale `cv2.matchTemplate` with `cv2.TM_CCOEFF_NORMED`. Return `None` if edge count or match confidence is too low.

- [ ] **Step 3: Implement liquid evidence and ratio estimation**

In the canonical goblet crop:

1. Estimate background color from crop pixels outside the cup mask.
2. Compute per-pixel confidence from saturation/chroma and grayscale distance from background.
3. Aggregate row fill evidence within the cup mask.
4. Find the top of the bottom-connected filled region.
5. Compute ratio with `(CUP_BOTTOM_Y - surface_y) / (CUP_BOTTOM_Y - CUP_TOP_Y)`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_goblet_liquid_ratio.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Refactor while green**

Keep helper functions focused and make thresholds configurable through an internal `EstimatorConfig` dataclass. Do not add heavy model dependencies.

- [ ] **Step 6: Run tests again**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

### Task 4: CLI and Debug Output

**Files:**
- Create: `src/goblet_liquid_ratio/__main__.py`
- Modify: `tests/test_goblet_liquid_ratio.py`

- [ ] **Step 1: Write failing CLI test**

Add a test that writes a synthetic image, runs the module CLI, and checks that stdout contains `ratio=`.

```python
def test_cli_prints_ratio_for_image_path(tmp_path):
    image_path = tmp_path / "cup.png"
    cv2.imwrite(str(image_path), synthetic_goblet_image(0.5))
    completed = subprocess.run(
        [sys.executable, "-m", "goblet_liquid_ratio", str(image_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ratio=" in completed.stdout
```

- [ ] **Step 2: Run CLI test to verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_goblet_liquid_ratio.py::test_cli_prints_ratio_for_image_path -q
```

Expected: FAIL because `goblet_liquid_ratio.__main__` does not exist.

- [ ] **Step 3: Implement CLI**

Parse `image`, optional `--debug DIR`, call `estimate_liquid_height_ratio(image, debug=True)`, print ratio and confidence, and write debug images when requested.

- [ ] **Step 4: Run CLI test to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_goblet_liquid_ratio.py::test_cli_prints_ratio_for_image_path -q
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: PASS.

### Task 5: Verification, Docs, and Commit

**Files:**
- Create: `README.md`
- Modify: none else unless verification exposes a gap.

- [ ] **Step 1: Add README usage**

Document installation, API use, CLI use, and the algorithm limits.

- [ ] **Step 2: Run final verification**

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m goblet_liquid_ratio --help
git status --short
```

Expected: tests pass, CLI help prints, only intended project files are modified.

- [ ] **Step 3: Commit implementation**

```bash
git add .gitignore pyproject.toml README.md src tests docs/superpowers/plans/2026-06-02-goblet-liquid-height.md
git commit -m "feat: implement goblet liquid height estimator"
```

### Self-Review

- Spec coverage: standalone project, lightweight dependencies, public API, debug result, synthetic shifted tests, and CLI are all covered.
- Placeholder scan: no TBD or TODO placeholders are used.
- Type consistency: the public function name is consistently `estimate_liquid_height_ratio`; the debug dataclass is consistently `LiquidHeightResult`.

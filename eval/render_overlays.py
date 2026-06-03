"""Render a before/after contact sheet of liquid-line overlays on real photos."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from goblet_liquid_ratio import estimate_liquid_height_ratio  # noqa: E402

SAMPLES = Path("/home/yuyuyu/桌面")
GT = {s["id"]: s for s in json.loads((ROOT / "eval" / "ground_truth.json").read_text())["samples"]}
OUT = ROOT / "eval" / "results_new"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = str(ROOT / "models" / "yolo11n.pt")

paths = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
tiles = []
for i, p in enumerate(paths, 1):
    if i not in GT:
        continue
    res = estimate_liquid_height_ratio(p, debug=True, method="yolo", yolo_model=MODEL)
    img = res.debug_images.get("overlay", cv2.imread(str(p)))
    g = GT[i]["gt_ratio"]
    pred = res.ratio if res.ratio is not None else float("nan")
    err = abs(pred - g) if res.ratio is not None else float("nan")
    tile = cv2.resize(img, (320, 240))
    cv2.rectangle(tile, (0, 0), (320, 20), (0, 0, 0), -1)
    cv2.putText(tile, f"{i:02d} GT={g:.2f} pred={pred:.2f} e={err:.2f}", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"{i:02d}_overlay.jpg"), img)
    tiles.append(tile)

cols = 3
rows = (len(tiles) + cols - 1) // cols
sheet = np.full((rows * 240, cols * 320, 3), 255, dtype=np.uint8)
for idx, tile in enumerate(tiles):
    r, c = divmod(idx, cols)
    sheet[r * 240:(r + 1) * 240, c * 320:(c + 1) * 320] = tile
cv2.imwrite(str(OUT / "contact_sheet_new.jpg"), sheet)
print(f"wrote {len(tiles)} overlays + contact_sheet_new.jpg -> {OUT}")

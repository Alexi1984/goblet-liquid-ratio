"""Evaluate the estimator against hand-annotated ground truth on real photos.

Usage:
    python eval/run_eval.py [--method yolo|opencv|auto] [--samples DIR]

Reports per-image error vs eval/ground_truth.json and the MAE / failure count.
Sample image order matches the ids in ground_truth.json (sorted desktop jpegs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from goblet_liquid_ratio import estimate_liquid_height_ratio  # noqa: E402

DEFAULT_SAMPLES = Path("/home/yuyuyu/桌面")
GT_PATH = ROOT / "eval" / "ground_truth.json"
DEFAULT_MODEL = ROOT / "models" / "yolo11n.pt"


def sorted_samples(samples_dir: Path) -> list[Path]:
    return sorted(p for p in samples_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="yolo", choices=("yolo", "opencv", "auto"))
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--fail-threshold", type=float, default=0.20)
    args = ap.parse_args()

    gt = {s["id"]: s for s in json.loads(GT_PATH.read_text())["samples"]}
    paths = sorted_samples(args.samples)

    print(f"{'id':>2} {'GT':>5} {'pred':>6} {'err':>6} {'conf':>5}  status")
    errs = []
    n_none = 0
    for i, p in enumerate(paths, 1):
        if i not in gt:
            continue
        res = estimate_liquid_height_ratio(p, debug=True, method=args.method, yolo_model=str(args.model))
        pred = res.ratio
        g = gt[i]["gt_ratio"]
        if pred is None:
            n_none += 1
            print(f"{i:>2} {g:>5.2f} {'None':>6} {'--':>6} {res.confidence:>5.2f}  NONE")
            continue
        e = abs(pred - g)
        errs.append(e)
        status = "OK" if e <= 0.10 else ("BAD" if e <= args.fail_threshold else "FAIL")
        print(f"{i:>2} {g:>5.2f} {pred:>6.3f} {e:>6.3f} {res.confidence:>5.2f}  {status}")

    if errs:
        mae = sum(errs) / len(errs)
        n_fail = sum(e > args.fail_threshold for e in errs)
        n_ok = sum(e <= 0.10 for e in errs)
        print(f"\nMAE = {mae:.3f}   max = {max(errs):.3f}   "
              f"OK(<=0.10) = {n_ok}/{len(errs)}   FAIL(>{args.fail_threshold:.2f}) = {n_fail}   NONE = {n_none}")
    else:
        print(f"\nno scored samples (NONE={n_none})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

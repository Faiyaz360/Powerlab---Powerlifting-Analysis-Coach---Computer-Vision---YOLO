"""Evaluation harness — measure pipeline accuracy against ground-truth labels.

Why: "accurate" must be a number, not a vibe. Add one JSON per clip under eval/groundtruth/
(see eval/README.md), then run:

    .\.venv\Scripts\python.exe eval\run_eval.py

Reports rep-count accuracy and per-rep legal-pass accuracy (squat depth / deadlift lockout).
Runs the normal pipeline (uses the pose cache, so it's fast and inference-only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import analyze  # noqa: E402

GT_DIR = ROOT / "eval" / "groundtruth"
LEGAL_KEY = {"squat": "depth_pass", "deadlift": "lockout_pass"}  # the IPF red-light criterion


def _pred_legal(result, lift):
    return [bool(r[LEGAL_KEY[lift]]) for r in result["analysis"]["rep_metrics"]]


def main():
    ap = argparse.ArgumentParser(description="Score the pipeline against ground-truth labels")
    ap.add_argument("--pose", default="yolo", choices=["mediapipe", "yolo", "rtmpose"],
                    help="pose backend to evaluate")
    args = ap.parse_args()

    files = sorted(GT_DIR.glob("*.json"))
    if not files:
        print("No ground-truth files in eval/groundtruth/. Add some (see eval/README.md).")
        return

    print(f"Pose backend: {args.pose}\n")
    rep_ok = 0
    legal_total = legal_correct = 0
    for f in files:
        gt = json.loads(f.read_text(encoding="utf-8"))
        lift = gt["lift"]
        res = analyze(str(ROOT / gt["video"]), lift=lift, backend=args.pose)
        pred, true = res["rep_count"], gt["true_rep_count"]
        count_ok = pred == true
        rep_ok += int(count_ok)
        line = f"{f.stem}: reps pred={pred} true={true} {'OK' if count_ok else 'MISS'}"

        gt_pass = gt.get("per_rep_legal_pass")
        if count_ok and gt_pass is not None:
            pred_pass = _pred_legal(res, lift)
            if len(pred_pass) == len(gt_pass):
                correct = sum(int(a == b) for a, b in zip(pred_pass, gt_pass))
                legal_total += len(gt_pass)
                legal_correct += correct
                line += f" | depth/lockout {correct}/{len(gt_pass)}"
        print(line)

    n = len(files)
    print("\n=== Scorecard ===")
    print(f"Rep-count accuracy:      {rep_ok}/{n} ({100 * rep_ok / n:.0f}%)")
    if legal_total:
        pct = 100 * legal_correct / legal_total
        print(f"Depth/lockout accuracy:  {legal_correct}/{legal_total} ({pct:.0f}%)")
    else:
        print("Depth/lockout accuracy:  (no aligned reps — fill per_rep_legal_pass + match counts)")


if __name__ == "__main__":
    main()

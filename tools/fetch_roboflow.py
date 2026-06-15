"""Download a labelled barbell/plate dataset from Roboflow Universe (YOLO format).

Roboflow Universe hosts several CC-BY barbell/plate detection sets — already labelled images, so
`tools/train_plate.py` can train on them directly (no video, no hand-labeling). This is the fastest
legal way to bootstrap the plate detector.

Find a set: search "barbell" or "weight plate" on https://universe.roboflow.com , open a CC-BY one,
click **Download Dataset -> YOLOv11 -> show download code**. The snippet contains the three slugs
you pass below, e.g. `rf.workspace("ws").project("proj").version(3)`.

Needs a FREE Roboflow API key (roboflow.com -> sign in -> Settings -> API key):
    setx ROBOFLOW_API_KEY your_key      # then reopen the shell
Needs:  pip install roboflow

Usage:
    python tools/fetch_roboflow.py --workspace WS --project PROJ --version 3
    python tools/fetch_roboflow.py --workspace WS --project PROJ --version 3 --single-class
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def _collapse_to_single_class(root: Path) -> None:
    """Remap every label to class 0 'plate' so it matches tools/train_plate.py (nc:1)."""
    for txt in root.rglob("*.txt"):
        if "labels" not in txt.parts:  # skip READMEs etc., only touch label files
            continue
        out_lines = []
        for ln in txt.read_text().splitlines():
            parts = ln.split()
            if parts:
                parts[0] = "0"
                out_lines.append(" ".join(parts))
        txt.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

    yaml = root / "data.yaml"
    if yaml.exists():
        text = yaml.read_text()
        text = re.sub(r"nc:\s*\d+", "nc: 1", text)
        text = re.sub(r"names:.*", "names: ['plate']", text)
        yaml.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a Roboflow barbell/plate dataset (YOLO format)")
    ap.add_argument("--workspace", required=True, help="workspace slug (from the dataset's download snippet)")
    ap.add_argument("--project", required=True, help="project slug")
    ap.add_argument("--version", type=int, required=True, help="dataset version number")
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"),
                    help="Roboflow API key (or set ROBOFLOW_API_KEY)")
    ap.add_argument("--out", default="dataset_roboflow", help="download location")
    ap.add_argument("--single-class", action="store_true",
                    help="remap all labels to one class 'plate' (for tools/train_plate.py)")
    args = ap.parse_args()

    if not args.api_key:
        print("No API key. Get a free one: roboflow.com -> Settings -> API key.")
        print("Then:  setx ROBOFLOW_API_KEY your_key   (reopen shell), or pass --api-key.")
        return
    try:
        from roboflow import Roboflow
    except ImportError:
        print("roboflow not installed.  pip install roboflow")
        return

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    project.version(args.version).download("yolov11", location=args.out)
    print(f"downloaded to {args.out}/")

    if args.single_class:
        _collapse_to_single_class(Path(args.out))
        print("remapped to single class 'plate' -> train with data=" + args.out + "/data.yaml")
    print("Check the license on the dataset's Universe page and keep it noted (DATA.md).")


if __name__ == "__main__":
    main()

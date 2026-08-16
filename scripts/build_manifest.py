#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scan_geometry.data.manifest_builders import PRESETS, build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified dataset manifest from a known raw dataset layout.")
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS), help="Dataset preset to use.")
    parser.add_argument("--root", required=True, help="Root directory of the raw dataset.")
    parser.add_argument("--output", required=True, help="Output manifest CSV path.")
    parser.add_argument("--dataset-name", default=None, help="Override dataset name written to the manifest.")
    parser.add_argument("--modality", default=None, help="Override modality written to the manifest.")
    parser.add_argument("--task-type", default=None, help="Override task type written to the manifest.")
    parser.add_argument("--split", default="", help="Split label to write when the preset cannot infer one.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_manifest(
        preset=args.preset,
        root=args.root,
        output=args.output,
        dataset_name=args.dataset_name,
        modality=args.modality,
        task_type=args.task_type,
        split=args.split,
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

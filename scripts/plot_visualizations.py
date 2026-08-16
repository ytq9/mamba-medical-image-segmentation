#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scan_geometry.visualization import VisualizationConfig, generate_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase A/B/C visualizations when source CSVs are available.")
    parser.add_argument("--output-dir", default="results/visualizations", help="Directory for generated figures.")
    parser.add_argument("--dataset-geometry", default="results/dataset_audit/dataset_geometry.csv")
    parser.add_argument("--matching-contrast", default="results/dataset_audit/matching_contrast.csv")
    parser.add_argument("--metrics-summary", default="results/phase_b/metrics_summary.csv")
    parser.add_argument("--case-metrics", default=None, help="Case-level metrics CSV for the P5 association plot.")
    parser.add_argument("--class-id", default="macro", help="Class id to plot from metrics summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_visualizations(
        VisualizationConfig(
            output_dir=Path(args.output_dir),
            dataset_geometry=Path(args.dataset_geometry) if args.dataset_geometry else None,
            matching_contrast=Path(args.matching_contrast) if args.matching_contrast else None,
            metrics_summary=Path(args.metrics_summary) if args.metrics_summary else None,
            case_metrics=Path(args.case_metrics) if args.case_metrics else None,
            prefer_class_id=args.class_id,
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

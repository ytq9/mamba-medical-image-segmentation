#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scan_geometry.phase_b.metrics import aggregate_case_metrics, metric_files_from_runs, read_case_metrics, write_metrics_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase B case-level Dice/HD95 metric files.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Explicit case-level metric CSV files.")
    parser.add_argument("--runs-csv", default=None, help="Run matrix CSV used to discover <output_dir>/metrics.csv files.")
    parser.add_argument("--output", default="results/phase_b/metrics_summary.csv", help="Output summary CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_paths = [Path(path) for path in args.metrics or []]
    if args.runs_csv:
        metric_paths.extend(metric_files_from_runs(args.runs_csv))
    if not metric_paths:
        raise SystemExit("No metrics found. Provide --metrics files or --runs-csv with run output metrics.")
    metrics = read_case_metrics(metric_paths)
    summary = aggregate_case_metrics(metrics)
    output = write_metrics_summary(args.output, summary)
    print(f"Wrote Phase B metric summary to {output}")


if __name__ == "__main__":
    main()

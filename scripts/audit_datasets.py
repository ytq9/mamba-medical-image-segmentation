#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scan_geometry.audit import AuditConfig, run_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase A dataset geometry and suitability audit.")
    parser.add_argument("--manifest", nargs="+", default=None, help="One or more unified manifest CSV files.")
    parser.add_argument("--config", default=None, help="Optional YAML config containing manifests and audit settings.")
    parser.add_argument("--output-dir", default="results/dataset_audit", help="Directory for all audit outputs.")
    parser.add_argument("--target-size", type=int, default=256, help="Primary in-memory analysis size.")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap samples for dataset descriptor CIs.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional cap per dataset for fast dry runs.")
    parser.add_argument("--min-patients", type=int, default=10, help="Minimum unique patients/samples for primary use.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for bootstrap sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AuditConfig.from_args(args)
    outputs = run_audit(config)
    print(f"Audit complete: {outputs.output_dir}")
    print(f"Suitability summary: {outputs.suitability_summary}")


if __name__ == "__main__":
    main()

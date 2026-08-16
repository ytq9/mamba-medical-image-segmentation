#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scan_geometry.phase_b import PhaseBConfig, make_phase_b_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create locked Phase B 70/10/20 splits and 10% labeled subsets.")
    parser.add_argument("--config", default="configs/experiments/phase_b_10pct.yaml", help="Phase B YAML config.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Limit split generation to one dataset. Can be repeated. Accepts names like CAMUS or sanitized ids like m_ms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PhaseBConfig.from_yaml(args.config).only_datasets(args.dataset)
    summaries = make_phase_b_splits(config)
    print(f"Wrote {len(summaries)} dataset split files under {config.split_dir()}")
    print(f"Split summary: {config.output_dir / 'split_summary.csv'}")


if __name__ == "__main__":
    main()

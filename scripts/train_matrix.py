#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scan_geometry.phase_b import PhaseBConfig, generate_run_matrix, make_phase_b_splits, write_run_matrix
from scan_geometry.phase_b.matrix import write_command_file
from scan_geometry.phase_b.utils import sanitize_id


DEFAULT_TEMPLATE = (
    "python scripts/train_adapter.py --config configs/experiments/phase_b_10pct.yaml "
    '--dataset {dataset} --condition {condition} --scan "{scan}" --seed {seed} '
    "--split-file {split_file} --output-dir {output_dir}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Phase B run matrix.")
    parser.add_argument("--config", default="configs/experiments/phase_b_10pct.yaml", help="Phase B YAML config.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Limit run matrix generation to one dataset. Can be repeated. Accepts names like CAMUS or sanitized ids like m_ms.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=None,
        help="Limit run matrix generation to one seed. Can be repeated.",
    )
    parser.add_argument("--make-splits", action="store_true", help="Create split files before writing the run matrix.")
    parser.add_argument("--output", default=None, help="Output run matrix CSV. Defaults to <output_dir>/runs.csv.")
    parser.add_argument("--commands-output", default=None, help="Optional shell script containing one train command per row.")
    parser.add_argument("--command-template", default=DEFAULT_TEMPLATE, help="Python format string used for --commands-output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PhaseBConfig.from_yaml(args.config).only_datasets(args.dataset).only_seeds(args.seed)
    split_summaries = make_phase_b_splits(config) if args.make_splits else _read_split_summary(config.output_dir / "split_summary.csv")
    rows = generate_run_matrix(config, split_summaries=split_summaries)
    output = Path(args.output) if args.output else _default_run_matrix_path(config, dataset_filter=bool(args.dataset))
    write_run_matrix(output, config, rows)
    print(f"Wrote {len(rows)} planned runs to {output}")
    if args.commands_output:
        command_path = write_command_file(args.commands_output, rows, args.command_template)
        print(f"Wrote command template script to {command_path}")


def _default_run_matrix_path(config: PhaseBConfig, *, dataset_filter: bool) -> Path:
    if not dataset_filter:
        return config.output_dir / "runs.csv"
    suffix = "_".join(sanitize_id(dataset.name) for dataset in config.datasets)
    return config.output_dir / f"runs_{suffix}.csv"


def _read_split_summary(path: Path):
    if not path.exists():
        return None
    from scan_geometry.phase_b.splits import SplitSummary

    summaries = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            summaries.append(
                SplitSummary(
                    dataset=row["dataset"],
                    split_file=Path(row["split_file"]),
                    split_unit=row["split_unit"],
                    n_cases=int(row["n_cases"]),
                    n_groups=int(row["n_groups"]),
                    train_groups=int(row["train_groups"]),
                    val_groups=int(row["val_groups"]),
                    test_groups=int(row["test_groups"]),
                    labeled_train_groups=int(row["labeled_train_groups"]),
                )
            )
    return summaries


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scan_geometry.geometry.scan_profiles import matching_contrast, matching_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute geometry-matching scores from an existing dataset_geometry.csv. "
            "This avoids repeating mask-level dataset auditing when only the scan-profile "
            "matching rule changes."
        )
    )
    parser.add_argument("--dataset-geometry", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_geometry_path = Path(args.dataset_geometry)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = read_csv(dataset_geometry_path)
    matching_rows: list[dict[str, Any]] = []
    for row in dataset_rows:
        dataset = str(row["dataset"])
        table = matching_table(dataset, row)
        contrast = matching_contrast(table)
        for item in table:
            item.update(contrast)
            matching_rows.append(item)

    write_csv(output_dir / "dataset_geometry.csv", dataset_rows)
    write_csv(output_dir / "matching_contrast.csv", matching_rows)
    print(f"wrote {output_dir / 'matching_contrast.csv'}")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

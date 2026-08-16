#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scan_geometry.data.loaders import load_image_and_label
from scan_geometry.data.manifest import read_manifest
from scan_geometry.geometry.descriptors import aggregate_descriptors, axial_angle_mean, descriptors_for_mask_stack
from scan_geometry.geometry.scan_profiles import matching_contrast, matching_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast Phase A subset for H1 analysis. Computes 256px dataset geometry and "
            "matching_contrast.csv without bootstrap or resize-stability checks."
        )
    )
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/dataset_audit_fast")
    parser.add_argument("--analysis-size", type=int, default=256)
    parser.add_argument("--max-cases-per-dataset", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for manifest in args.manifests:
        records.extend(read_manifest(manifest))
    if args.max_cases_per_dataset is not None:
        records = cap_records(records, args.max_cases_per_dataset)

    case_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        if index == 1 or index % 25 == 0 or index == total:
            print(f"[fast-audit] {index}/{total} {record.dataset} {record.case_id}", flush=True)
        row: dict[str, Any] = {
            "dataset": record.dataset,
            "case_id": record.case_id,
            "patient_id": record.patient_id,
            "valid": False,
            "reason": "",
        }
        try:
            label = load_image_and_label(record)[1].array
            descriptors = descriptors_for_mask_stack(label, slice_axis=record.slice_axis, analysis_size=args.analysis_size)
            aggregated = aggregate_descriptors(descriptors)
            empty_count = sum(1 for item in descriptors if item.empty)
            row.update(
                {
                    "valid": bool(descriptors) and empty_count < len(descriptors),
                    "n_slices": len(descriptors),
                    "empty_slices": empty_count,
                    "all_empty": len(descriptors) == 0 or empty_count == len(descriptors),
                }
            )
            row.update(aggregated)
        except Exception as exc:  # noqa: BLE001
            row["reason"] = f"{type(exc).__name__}: {exc}"
        case_rows.append(row)
        valid_rows.append(
            {
                "dataset": record.dataset,
                "case_id": record.case_id,
                "patient_id": record.patient_id,
                "valid": row["valid"],
                "reason": row["reason"],
            }
        )

    dataset_rows = dataset_geometry(case_rows, valid_rows)
    matching_rows: list[dict[str, Any]] = []
    for row in dataset_rows:
        table = matching_table(str(row["dataset"]), row)
        contrast = matching_contrast(table)
        for item in table:
            item.update(contrast)
            matching_rows.append(item)

    write_csv(output_dir / "case_geometry.csv", case_rows)
    write_csv(output_dir / "dataset_geometry.csv", dataset_rows)
    write_csv(output_dir / "matching_contrast.csv", matching_rows)
    write_csv(output_dir / "manifest_validated.csv", valid_rows)
    print(f"[fast-audit] wrote {output_dir / 'matching_contrast.csv'}", flush=True)


def cap_records(records: list[Any], max_cases: int) -> list[Any]:
    by_dataset: dict[str, list[Any]] = {}
    for record in records:
        by_dataset.setdefault(record.dataset, []).append(record)
    output: list[Any] = []
    for dataset in sorted(by_dataset):
        output.extend(by_dataset[dataset][:max_cases])
    return output


def dataset_geometry(case_rows: list[dict[str, Any]], valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets = sorted({str(row["dataset"]) for row in valid_rows})
    output: list[dict[str, Any]] = []
    for dataset in datasets:
        rows = [row for row in case_rows if row.get("dataset") == dataset and bool(row.get("valid", False))]
        manifest_dataset = [row for row in valid_rows if row.get("dataset") == dataset]
        if not rows:
            output.append({"dataset": dataset, "n_cases": len(manifest_dataset), "n_valid_cases": 0})
            continue
        payload: dict[str, Any] = {
            "dataset": dataset,
            "n_cases": len(manifest_dataset),
            "n_valid_cases": len(rows),
            "n_patients": len({row.get("patient_id", "") for row in manifest_dataset if row.get("valid")}),
        }
        for column in ["D", "A_data", "L", "S", "B", "C", "foreground_ratio", "empty_slice_rate"]:
            payload[column] = float(np.nanmean([float(row.get(column, np.nan)) for row in rows]))
        payload["phi_data"] = axial_angle_mean([float(row.get("phi_data", np.nan)) for row in rows])
        output.append(payload)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import PhaseBConfig
from .model_protocol import condition_model_config_hash, condition_scan_order_hash
from .splits import SplitSummary, expected_split_path
from .utils import sanitize_id


RUN_COLUMNS = [
    "run_id",
    "dataset",
    "condition",
    "condition_family",
    "scan",
    "seed",
    "manifest",
    "split_file",
    "label_ratio",
    "labeled_train_units",
    "model_config_hash",
    "scan_order_hash",
    "output_dir",
    "status",
]


def generate_run_matrix(
    config: PhaseBConfig,
    split_summaries: list[SplitSummary] | None = None,
) -> list[dict[str, Any]]:
    split_by_dataset = {summary.dataset: summary for summary in split_summaries or []}
    rows: list[dict[str, Any]] = []
    for dataset in config.datasets:
        split_file = split_by_dataset.get(dataset.name).split_file if dataset.name in split_by_dataset else expected_split_path(config, dataset)
        labeled_train_units = (
            split_by_dataset[dataset.name].labeled_train_groups
            if dataset.name in split_by_dataset
            else dataset.labeled_train_units or ""
        )
        for condition in config.conditions:
            for seed in config.seeds:
                run_id = f"{sanitize_id(dataset.name)}__{sanitize_id(condition.name)}__seed{seed}"
                rows.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset.name,
                        "condition": condition.name,
                        "condition_family": condition.family,
                        "scan": condition.scan,
                        "seed": seed,
                        "manifest": str(dataset.manifest),
                        "split_file": str(split_file),
                        "label_ratio": config.low_label_ratio,
                        "labeled_train_units": labeled_train_units,
                        "model_config_hash": condition_model_config_hash(config, condition, dataset),
                        "scan_order_hash": condition_scan_order_hash(config, condition),
                        "output_dir": str(config.run_dir() / run_id),
                        "status": "planned",
                    }
                )
    return rows


def write_run_matrix(
    path: str | Path,
    config: PhaseBConfig,
    rows: list[dict[str, Any]],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RUN_COLUMNS})
    protocol_path = output.parent / "phase_b_protocol.json"
    protocol_payload = config.protocol_payload()
    protocol_payload["n_runs"] = len(rows)
    protocol_payload["run_matrix"] = str(output)
    with protocol_path.open("w", encoding="utf-8") as handle:
        json.dump(protocol_payload, handle, indent=2)
    return output


def write_command_file(
    path: str | Path,
    rows: list[dict[str, Any]],
    template: str,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for row in rows:
        lines.append(template.format(**{key: "" if value is None else value for key, value in row.items()}))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output.chmod(0o755)
    return output

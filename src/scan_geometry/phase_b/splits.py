from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scan_geometry.data.manifest import MANIFEST_COLUMNS, ManifestRecord, read_manifest

from .config import DatasetSpec, PhaseBConfig
from .utils import sanitize_id


SPLIT_COLUMNS = [
    "phase_b_group_id",
    "phase_b_split",
    "phase_b_labeled",
    "phase_b_label_subset",
    "phase_b_split_unit",
]


@dataclass(frozen=True, slots=True)
class SplitSummary:
    dataset: str
    split_file: Path
    split_unit: str
    n_cases: int
    n_groups: int
    train_groups: int
    val_groups: int
    test_groups: int
    labeled_train_groups: int

    def to_row(self) -> dict[str, str | int]:
        return {
            "dataset": self.dataset,
            "split_file": str(self.split_file),
            "split_unit": self.split_unit,
            "n_cases": self.n_cases,
            "n_groups": self.n_groups,
            "train_groups": self.train_groups,
            "val_groups": self.val_groups,
            "test_groups": self.test_groups,
            "labeled_train_groups": self.labeled_train_groups,
        }


def make_phase_b_splits(config: PhaseBConfig) -> list[SplitSummary]:
    config.validate()
    config.split_dir().mkdir(parents=True, exist_ok=True)
    summaries = [_write_dataset_split(config, dataset) for dataset in config.datasets]
    _write_split_summary(config.output_dir / "split_summary.csv", summaries)
    return summaries


def expected_split_path(config: PhaseBConfig, dataset: DatasetSpec) -> Path:
    return config.split_dir() / f"{sanitize_id(dataset.name)}_split.csv"


def _write_dataset_split(config: PhaseBConfig, dataset: DatasetSpec) -> SplitSummary:
    records = read_manifest(dataset.manifest)
    groups = _group_records(records, dataset.split_unit)
    assignments = _assign_splits(
        sorted(groups),
        seed=_dataset_seed(config.split_seed, dataset.name),
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
    )
    train_group_ids = [group_id for group_id, split in assignments.items() if split == "train"]
    labeled_group_ids = set(
        _choose_labeled_train_groups(
            train_group_ids,
            dataset=dataset,
            config=config,
            seed=_dataset_seed(config.split_seed + 7919, dataset.name),
        )
    )
    output_path = expected_split_path(config, dataset)
    fieldnames = [*MANIFEST_COLUMNS, *SPLIT_COLUMNS]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            group_id = _record_group_id(record, dataset.split_unit)
            split = assignments[group_id]
            row = record.to_row()
            row.update(
                {
                    "phase_b_group_id": group_id,
                    "phase_b_split": split,
                    "phase_b_labeled": "true" if group_id in labeled_group_ids and split == "train" else "false",
                    "phase_b_label_subset": "10pct" if group_id in labeled_group_ids and split == "train" else "",
                    "phase_b_split_unit": dataset.split_unit,
                }
            )
            writer.writerow(row)

    counts = _count_splits(assignments)
    return SplitSummary(
        dataset=dataset.name,
        split_file=output_path,
        split_unit=dataset.split_unit,
        n_cases=len(records),
        n_groups=len(groups),
        train_groups=counts["train"],
        val_groups=counts["val"],
        test_groups=counts["test"],
        labeled_train_groups=len(labeled_group_ids),
    )


def _group_records(records: list[ManifestRecord], split_unit: str) -> dict[str, list[ManifestRecord]]:
    groups: dict[str, list[ManifestRecord]] = {}
    for record in records:
        groups.setdefault(_record_group_id(record, split_unit), []).append(record)
    return groups


def _record_group_id(record: ManifestRecord, split_unit: str) -> str:
    if split_unit == "patient":
        return record.patient_id or record.case_id
    if split_unit == "case":
        return record.case_id
    raise ValueError(f"Unsupported split_unit={split_unit!r}.")


def _assign_splits(
    group_ids: list[str],
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, str]:
    n_groups = len(group_ids)
    if n_groups == 0:
        raise ValueError("Cannot split an empty manifest.")
    rng = np.random.default_rng(seed)
    shuffled = list(group_ids)
    rng.shuffle(shuffled)
    n_train = int(math.floor(n_groups * train_fraction))
    n_val = int(math.floor(n_groups * val_fraction))
    if n_groups >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
    if n_train + n_val > n_groups:
        n_val = max(0, n_groups - n_train)
    assignments: dict[str, str] = {}
    for index, group_id in enumerate(shuffled):
        if index < n_train:
            split = "train"
        elif index < n_train + n_val:
            split = "val"
        else:
            split = "test"
        assignments[group_id] = split
    return assignments


def _choose_labeled_train_groups(
    train_group_ids: list[str],
    dataset: DatasetSpec,
    config: PhaseBConfig,
    seed: int,
) -> list[str]:
    if not train_group_ids:
        return []
    requested = dataset.labeled_train_units
    if requested is None:
        requested = max(
            int(math.ceil(len(train_group_ids) * config.low_label_ratio)),
            int(config.min_labeled_train_units),
        )
    if requested > len(train_group_ids):
        raise ValueError(
            f"Dataset {dataset.name} has only {len(train_group_ids)} training {dataset.split_unit} groups, "
            f"but {requested} labeled groups were requested."
        )
    rng = np.random.default_rng(seed)
    shuffled = list(train_group_ids)
    rng.shuffle(shuffled)
    return sorted(shuffled[:requested])


def _count_splits(assignments: dict[str, str]) -> dict[str, int]:
    return {
        "train": sum(1 for value in assignments.values() if value == "train"),
        "val": sum(1 for value in assignments.values() if value == "val"),
        "test": sum(1 for value in assignments.values() if value == "test"),
    }


def _write_split_summary(path: Path, summaries: list[SplitSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].to_row()) if summaries else [])
        if summaries:
            writer.writeheader()
            for summary in summaries:
                writer.writerow(summary.to_row())


def _dataset_seed(seed: int, dataset_name: str) -> int:
    digest = hashlib.sha1(dataset_name.encode("utf-8")).hexdigest()
    return int(seed) + int(digest[:8], 16)

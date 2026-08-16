from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from scan_geometry.data.manifest import ManifestRecord, write_manifest
from scan_geometry.phase_b import PhaseBConfig, aggregate_case_metrics, generate_run_matrix, make_phase_b_splits


def test_phase_b_splits_lock_train_val_test_and_labeled_subset(tmp_path: Path) -> None:
    manifest = tmp_path / "camus.csv"
    _write_manifest(manifest, dataset="CAMUS", n_cases=20)
    config = PhaseBConfig.from_payload(
        {
            "output_dir": str(tmp_path / "phase_b"),
            "split": {"seed": 2026, "train": 0.70, "val": 0.10, "test": 0.20},
            "low_label": {"ratio": 0.10, "min_labeled_train_units": 10},
            "seeds": [2026, 2027, 2028],
            "conditions": ["CNN", "Raster-H", "Raster-V", "Hilbert", "LocalWindow", "RandomPermute"],
            "datasets": [{"name": "CAMUS", "manifest": str(manifest), "split_unit": "patient"}],
        }
    )

    summaries = make_phase_b_splits(config)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.train_groups == 14
    assert summary.val_groups == 2
    assert summary.test_groups == 4
    assert summary.labeled_train_groups == 10

    rows = list(csv.DictReader(summary.split_file.open("r", encoding="utf-8", newline="")))
    labeled_rows = [row for row in rows if row["phase_b_labeled"] == "true"]
    assert len({row["phase_b_group_id"] for row in labeled_rows}) == 10
    assert {row["phase_b_split"] for row in labeled_rows} == {"train"}

    matrix = generate_run_matrix(config, split_summaries=summaries)
    assert len(matrix) == 18
    assert {row["split_file"] for row in matrix} == {str(summary.split_file)}
    assert {row["labeled_train_units"] for row in matrix} == {10}
    mamba_rows = [row for row in matrix if row["condition_family"] == "mamba"]
    assert len({row["model_config_hash"] for row in mamba_rows}) == 1
    assert len({row["scan_order_hash"] for row in mamba_rows}) == 5


def test_phase_b_absolute_labeled_count_for_drive(tmp_path: Path) -> None:
    manifest = tmp_path / "drive.csv"
    _write_manifest(manifest, dataset="DRIVE", n_cases=40, patient_prefix="case")
    config = PhaseBConfig.from_payload(
        {
            "output_dir": str(tmp_path / "phase_b"),
            "split": {"seed": 2026, "train": 0.70, "val": 0.10, "test": 0.20},
            "low_label": {"ratio": 0.10, "min_labeled_train_units": 10},
            "seeds": [2026],
            "conditions": ["CNN"],
            "datasets": [
                {
                    "name": "DRIVE",
                    "manifest": str(manifest),
                    "split_unit": "case",
                    "labeled_train_units": 10,
                }
            ],
        }
    )

    summary = make_phase_b_splits(config)[0]
    assert summary.train_groups == 28
    assert summary.val_groups == 4
    assert summary.test_groups == 8
    assert summary.labeled_train_groups == 10


def test_phase_b_config_can_filter_datasets_by_name_or_sanitized_id(tmp_path: Path) -> None:
    camus_manifest = tmp_path / "camus.csv"
    mnms_manifest = tmp_path / "mnms.csv"
    _write_manifest(camus_manifest, dataset="CAMUS", n_cases=3)
    _write_manifest(mnms_manifest, dataset="M&Ms", n_cases=3)
    config = PhaseBConfig.from_payload(
        {
            "output_dir": str(tmp_path / "phase_b"),
            "seeds": [2026],
            "conditions": ["CNN"],
            "datasets": [
                {"name": "CAMUS", "manifest": str(camus_manifest), "split_unit": "patient"},
                {"name": "M&Ms", "manifest": str(mnms_manifest), "split_unit": "patient"},
            ],
        }
    )

    assert [dataset.name for dataset in config.only_datasets(["CAMUS"]).datasets] == ["CAMUS"]
    assert [dataset.name for dataset in config.only_datasets(["m_ms"]).datasets] == ["M&Ms"]


def test_phase_b_config_can_filter_seeds(tmp_path: Path) -> None:
    manifest = tmp_path / "isic2018.csv"
    _write_manifest(manifest, dataset="ISIC2018", n_cases=3)
    config = PhaseBConfig.from_payload(
        {
            "output_dir": str(tmp_path / "phase_b"),
            "seeds": [2026, 2027, 2028],
            "conditions": ["CNN", "Raster-H"],
            "datasets": [{"name": "ISIC2018", "manifest": str(manifest), "split_unit": "case"}],
        }
    )

    filtered = config.only_seeds([2026])
    assert filtered.seeds == [2026]
    assert len(generate_run_matrix(filtered)) == 2


def test_phase_b_config_preserves_training_options(tmp_path: Path) -> None:
    manifest = tmp_path / "isic2018.csv"
    _write_manifest(manifest, dataset="ISIC2018", n_cases=3)
    config = PhaseBConfig.from_payload(
        {
            "output_dir": str(tmp_path / "phase_b"),
            "seeds": [2026],
            "conditions": ["CNN"],
            "training": {"batch_size": 32, "num_workers": 8, "cache_dir": "results/cache"},
            "datasets": [{"name": "ISIC2018", "manifest": str(manifest), "split_unit": "case"}],
        }
    )

    assert config.training["batch_size"] == 32
    assert config.protocol_payload()["training"]["cache_dir"] == "results/cache"


def test_phase_b_metrics_aggregate_seed_level_variation() -> None:
    metrics = pd.DataFrame(
        [
            {"dataset": "CAMUS", "condition": "CNN", "seed": 1, "case_id": "a", "class_id": "macro", "dice": 0.8, "hd95": 5.0},
            {"dataset": "CAMUS", "condition": "CNN", "seed": 1, "case_id": "b", "class_id": "macro", "dice": 0.4, "hd95": 9.0},
            {"dataset": "CAMUS", "condition": "CNN", "seed": 2, "case_id": "a", "class_id": "macro", "dice": 0.9, "hd95": 4.0},
            {"dataset": "CAMUS", "condition": "CNN", "seed": 2, "case_id": "b", "class_id": "macro", "dice": 0.7, "hd95": 6.0},
        ]
    )

    summary = aggregate_case_metrics(metrics)
    row = summary.iloc[0]
    assert row["dice_mean"] == pytest.approx(0.7)
    assert row["n_seeds"] == 2
    assert row["n_cases"] == 2
    assert row["failure_rate"] == 0.25


def _write_manifest(path: Path, dataset: str, n_cases: int, patient_prefix: str = "patient") -> None:
    records = [
        ManifestRecord(
            dataset=dataset,
            case_id=f"case_{index:03d}",
            patient_id=f"{patient_prefix}_{index:03d}",
            image_path=f"/tmp/{dataset}/image_{index:03d}.png",
            label_path=f"/tmp/{dataset}/label_{index:03d}.png",
            modality="synthetic",
            task_type="unit_test",
            is_2d=True,
        )
        for index in range(n_cases)
    ]
    write_manifest(path, records)

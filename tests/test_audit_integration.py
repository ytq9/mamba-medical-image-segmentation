from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scan_geometry.audit import AuditConfig, run_audit


def test_audit_datasets_outputs_expected_files(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    label_path = tmp_path / "label.png"
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(image_path)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 20:44] = 1
    Image.fromarray(mask).save(label_path)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "case_id",
                "patient_id",
                "image_path",
                "label_path",
                "modality",
                "task_type",
                "split",
                "is_2d",
                "spacing_z",
                "spacing_y",
                "spacing_x",
                "slice_axis",
                "view",
                "phase",
                "label_values",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "SYNTH",
                "case_id": "case001",
                "patient_id": "patient001",
                "image_path": str(image_path),
                "label_path": str(label_path),
                "modality": "synthetic",
                "task_type": "unit_test",
                "split": "train",
                "is_2d": "true",
                "spacing_z": "1",
                "spacing_y": "1",
                "spacing_x": "1",
                "slice_axis": "-1",
                "view": "",
                "phase": "",
                "label_values": "1",
                "notes": "",
            }
        )

    config = AuditConfig.from_args(
        SimpleNamespace(
            manifest=[str(manifest)],
            config=None,
            output_dir=str(tmp_path / "audit"),
            target_size=64,
            bootstrap_samples=10,
            max_cases=None,
            min_patients=1,
            seed=1,
        )
    )
    outputs = run_audit(config)
    expected = [
        "manifest_validated.csv",
        "case_geometry.csv",
        "dataset_geometry.csv",
        "descriptor_stability.csv",
        "matching_contrast.csv",
        "suitability_summary.csv",
    ]
    for relative in expected:
        assert (outputs.output_dir / relative).exists()
    unexpected = [
        "audit_report.json",
        "audit_report.md",
        "figures/geometry_pca.png",
        "figures/descriptor_heatmap.png",
        "figures/matching_contrast.png",
        "figures/stability_intervals.png",
    ]
    for relative in unexpected:
        assert not (outputs.output_dir / relative).exists()

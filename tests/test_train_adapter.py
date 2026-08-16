from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


SPLIT_COLUMNS = [
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
    "phase_b_group_id",
    "phase_b_split",
    "phase_b_labeled",
    "phase_b_label_subset",
    "phase_b_split_unit",
]


def test_train_adapter_runs_tiny_cnn_end_to_end(tmp_path: Path) -> None:
    split_csv = tmp_path / "split.csv"
    rows = []
    for index, split in enumerate(["train", "train", "val", "test"]):
        image_path = tmp_path / f"image_{index}.png"
        label_path = tmp_path / f"label_{index}.png"
        image = np.zeros((16, 16), dtype=np.uint8)
        label = np.zeros((16, 16), dtype=np.uint8)
        image[4:12, 4:12] = 120 + index
        label[4:12, 4:12] = 1
        Image.fromarray(image).save(image_path)
        Image.fromarray(label).save(label_path)
        rows.append(_row(f"case_{index}", image_path, label_path, split=split, labeled=split == "train"))
    _write_split(split_csv, rows)

    config_path = tmp_path / "phase_b.yaml"
    output_dir = tmp_path / "run"
    config_path.write_text(
        f"""
output_dir: {tmp_path / "phase_b"}
split:
  seed: 2026
  train: 0.70
  val: 0.10
  test: 0.20
low_label:
  ratio: 0.10
  min_labeled_train_units: 1
model:
  architecture: umamba_scan_bot_2d
  input_size: [16, 16]
  bottleneck_scan_shape: [4, 4]
  features: [4, 8, 16]
  blocks_per_stage: 1
  num_classes: 2
  local_window_size: 4
  scan_seed: 2026
  cnn_bottleneck_blocks: 1
  mamba:
    d_state: 4
    d_conv: 2
    expand: 1
training:
  epochs: 1
  batch_size: 2
  learning_rate: 0.001
  weight_decay: 0.0
  num_workers: 0
  normalize: none
  device: cpu
seeds: [2026]
conditions:
  - name: CNN
    family: cnn_baseline
    scan: ""
datasets:
  - name: SYNTH
    manifest: unused.csv
    split_unit: case
    input_channels: 1
    num_classes: 2
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_adapter.py",
            "--config",
            str(config_path),
            "--dataset",
            "SYNTH",
            "--condition",
            "CNN",
            "--scan",
            "",
            "--seed",
            "2026",
            "--split-file",
            str(split_csv),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Training complete" in result.stdout
    assert (output_dir / "config.json").exists()
    assert (output_dir / "history.json").exists()
    assert (output_dir / "checkpoints" / "best.pt").exists()
    metrics_path = output_dir / "metrics.csv"
    assert metrics_path.exists()
    metrics = list(csv.DictReader(metrics_path.open("r", encoding="utf-8", newline="")))
    assert len(metrics) == 1
    assert metrics[0]["dataset"] == "SYNTH"
    assert metrics[0]["condition"] == "CNN"
    assert metrics[0]["class_id"] == "1"


def _write_split(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _row(case_id: str, image_path: Path, label_path: Path, *, split: str, labeled: bool) -> dict[str, str]:
    return {
        "dataset": "SYNTH",
        "case_id": case_id,
        "patient_id": case_id,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "modality": "synthetic",
        "task_type": "unit_test",
        "split": "",
        "is_2d": "true",
        "spacing_z": "1",
        "spacing_y": "1",
        "spacing_x": "1",
        "slice_axis": "-1",
        "view": "",
        "phase": "",
        "label_values": "1",
        "notes": "",
        "phase_b_group_id": case_id,
        "phase_b_split": split,
        "phase_b_labeled": "true" if labeled else "false",
        "phase_b_label_subset": "10pct" if labeled else "",
        "phase_b_split_unit": "case",
    }

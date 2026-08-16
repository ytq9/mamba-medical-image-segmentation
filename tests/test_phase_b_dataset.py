from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scan_geometry.phase_b import ManifestSegmentationDataset


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


def test_manifest_segmentation_dataset_loads_labeled_2d_rows(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    label_path = tmp_path / "label.png"
    Image.fromarray(np.full((20, 30), 7, dtype=np.uint8)).save(image_path)
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:12, 6:18] = 2
    Image.fromarray(mask).save(label_path)
    split_csv = tmp_path / "split.csv"
    _write_split(
        split_csv,
        [
            _row("case001", image_path, label_path, is_2d=True, split="train", labeled=True),
            _row("case002", image_path, label_path, is_2d=True, split="train", labeled=False),
            _row("case003", image_path, label_path, is_2d=True, split="val", labeled=False),
        ],
    )

    dataset = ManifestSegmentationDataset(split_csv, split="train", labeled_only=True, input_channels=1, target_size=32)

    assert len(dataset) == 1
    item = dataset[0]
    assert tuple(item["image"].shape) == (1, 32, 32)
    assert tuple(item["label"].shape) == (32, 32)
    assert item["label"].max().item() == 2
    assert item["case_id"] == "case001"
    assert item["labeled"] is True


def test_manifest_segmentation_dataset_expands_3d_foreground_slices(tmp_path: Path) -> None:
    image = np.zeros((4, 12, 16), dtype=np.float32)
    label = np.zeros((4, 12, 16), dtype=np.uint8)
    image[1] = 4
    image[3] = 8
    label[1, 3:8, 4:10] = 1
    label[3, 2:6, 1:5] = 2
    image_path = tmp_path / "volume.npy"
    label_path = tmp_path / "label.npy"
    np.save(image_path, image)
    np.save(label_path, label)
    split_csv = tmp_path / "split.csv"
    _write_split(
        split_csv,
        [_row("vol001", image_path, label_path, is_2d=False, split="train", labeled=True, slice_axis=0)],
    )

    dataset = ManifestSegmentationDataset(split_csv, split="train", labeled_only=True, input_channels=1, target_size=16)

    assert len(dataset) == 2
    assert [dataset[index]["slice_index"] for index in range(len(dataset))] == [1, 3]
    assert tuple(dataset[0]["image"].shape) == (1, 16, 16)
    assert tuple(dataset[0]["label"].shape) == (16, 16)


def test_manifest_segmentation_dataset_expands_4d_cmr_foreground_frames(tmp_path: Path) -> None:
    image = np.zeros((12, 16, 3, 2), dtype=np.float32)
    label = np.zeros((12, 16, 3, 2), dtype=np.uint8)
    image[:, :, 1, 0] = 4
    image[:, :, 2, 1] = 8
    label[3:8, 4:10, 1, 0] = 1
    label[2:6, 1:5, 2, 1] = 2
    image_path = tmp_path / "cmr.npy"
    label_path = tmp_path / "cmr_gt.npy"
    np.save(image_path, image)
    np.save(label_path, label)
    split_csv = tmp_path / "split.csv"
    _write_split(
        split_csv,
        [_row("cmr001", image_path, label_path, is_2d=False, split="train", labeled=True)],
    )

    dataset = ManifestSegmentationDataset(split_csv, split="train", labeled_only=True, input_channels=1, target_size=16)

    assert len(dataset) == 2
    assert [dataset[index]["slice_index"] for index in range(len(dataset))] == [10000, 20001]
    assert tuple(dataset[0]["image"].shape) == (1, 16, 16)
    assert tuple(dataset[0]["label"].shape) == (16, 16)


def test_manifest_segmentation_dataset_handles_rgb_input(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    label_path = tmp_path / "mask.png"
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    rgb[..., 0] = 20
    rgb[..., 1] = 40
    rgb[..., 2] = 60
    Image.fromarray(rgb).save(image_path)
    Image.fromarray((np.ones((10, 12), dtype=np.uint8) * 1)).save(label_path)
    split_csv = tmp_path / "split.csv"
    _write_split(split_csv, [_row("rgb001", image_path, label_path, is_2d=True, split="test", labeled=False)])

    dataset = ManifestSegmentationDataset(split_csv, split="test", input_channels=3, target_size=16, normalize="minmax")

    item = dataset[0]
    assert tuple(item["image"].shape) == (3, 16, 16)
    assert tuple(item["label"].shape) == (16, 16)


def test_binary_dataset_maps_255_mask_values_to_one(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    label_path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8)).save(image_path)
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:8, 3:9] = 255
    Image.fromarray(mask).save(label_path)
    split_csv = tmp_path / "split.csv"
    _write_split(split_csv, [_row("binary001", image_path, label_path, is_2d=True, split="train", labeled=True)])

    dataset = ManifestSegmentationDataset(
        split_csv,
        split="train",
        labeled_only=True,
        input_channels=3,
        num_classes=2,
        target_size=16,
    )

    label = dataset[0]["label"]
    assert set(label.unique().tolist()) <= {0, 1}
    assert label.max().item() == 1


def test_manifest_segmentation_dataset_writes_preprocessed_cache(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    label_path = tmp_path / "mask.png"
    Image.fromarray(np.full((10, 12), 9, dtype=np.uint8)).save(image_path)
    Image.fromarray((np.ones((10, 12), dtype=np.uint8) * 255)).save(label_path)
    split_csv = tmp_path / "split.csv"
    _write_split(split_csv, [_row("cache001", image_path, label_path, is_2d=True, split="train", labeled=True)])
    cache_dir = tmp_path / "cache"

    dataset = ManifestSegmentationDataset(
        split_csv,
        split="train",
        labeled_only=True,
        input_channels=1,
        num_classes=2,
        target_size=16,
        cache_dir=cache_dir,
    )

    item = dataset[0]
    cache_files = list(cache_dir.rglob("*.pt"))
    assert len(cache_files) == 1
    cached = torch.load(cache_files[0], map_location="cpu", weights_only=True)
    assert torch.equal(item["image"], cached["image"])
    assert torch.equal(item["label"], cached["label"])
    assert torch.equal(dataset[0]["label"], item["label"])


def test_multiclass_dataset_rejects_labels_outside_class_range(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    label_path = tmp_path / "label.png"
    Image.fromarray(np.zeros((10, 12), dtype=np.uint8)).save(image_path)
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:8, 3:9] = 4
    Image.fromarray(mask).save(label_path)
    split_csv = tmp_path / "split.csv"
    _write_split(split_csv, [_row("multi001", image_path, label_path, is_2d=True, split="train", labeled=True)])
    dataset = ManifestSegmentationDataset(
        split_csv,
        split="train",
        labeled_only=True,
        input_channels=1,
        num_classes=4,
        target_size=16,
    )

    with pytest.raises(ValueError, match=r"outside \[0, 3\]"):
        dataset[0]


def _write_split(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    case_id: str,
    image_path: Path,
    label_path: Path,
    *,
    is_2d: bool,
    split: str,
    labeled: bool,
    slice_axis: int = -1,
) -> dict[str, str]:
    return {
        "dataset": "SYNTH",
        "case_id": case_id,
        "patient_id": case_id,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "modality": "synthetic",
        "task_type": "unit_test",
        "split": "",
        "is_2d": "true" if is_2d else "false",
        "spacing_z": "1",
        "spacing_y": "1",
        "spacing_x": "1",
        "slice_axis": str(slice_axis),
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

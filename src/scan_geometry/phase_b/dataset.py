from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import Dataset

from scan_geometry.data.loaders import load_image_and_label
from scan_geometry.data.manifest import ManifestRecord


@dataclass(frozen=True, slots=True)
class SegmentationSample:
    record: ManifestRecord
    split: str
    labeled: bool
    slice_index: int | tuple[int, int] | None = None


class ManifestSegmentationDataset(Dataset):
    """Torch dataset backed by a Phase B split CSV."""

    def __init__(
        self,
        split_csv: str | Path,
        *,
        split: str,
        labeled_only: bool = False,
        input_channels: int = 1,
        num_classes: int | None = None,
        target_size: int = 256,
        include_empty_slices: bool = False,
        max_slices_per_volume: int | None = None,
        normalize: str = "zscore",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.split = split
        self.labeled_only = bool(labeled_only)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.target_size = int(target_size)
        self.include_empty_slices = bool(include_empty_slices)
        self.max_slices_per_volume = max_slices_per_volume
        self.normalize = normalize
        self.cache_dir = Path(cache_dir) if cache_dir not in {None, ""} else None
        self.samples = _build_samples(
            self.split_csv,
            split=split,
            labeled_only=labeled_only,
            include_empty_slices=include_empty_slices,
            max_slices_per_volume=max_slices_per_volume,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_tensor, label_tensor = self._load_or_prepare_tensors(sample)
        return {
            "image": image_tensor,
            "label": label_tensor,
            "dataset": sample.record.dataset,
            "case_id": sample.record.case_id,
            "patient_id": sample.record.patient_id,
            "slice_index": _encoded_slice_index(sample.slice_index),
            "split": sample.split,
            "labeled": sample.labeled,
        }

    def _load_or_prepare_tensors(self, sample: SegmentationSample) -> tuple[torch.Tensor, torch.Tensor]:
        cache_path = self._cache_path(sample)
        if cache_path is not None:
            cached = _read_cached_tensors(cache_path)
            if cached is not None:
                return cached

        image_tensor, label_tensor = self._prepare_tensors(sample)
        if cache_path is not None:
            _write_cached_tensors(cache_path, image_tensor, label_tensor)
        return image_tensor, label_tensor

    def _prepare_tensors(self, sample: SegmentationSample) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = load_image_and_label(sample.record)
        image_slice = _select_2d_slice(image.array, sample.slice_index, sample.record.slice_axis)
        label_slice = _select_2d_slice(label.array, sample.slice_index, sample.record.slice_axis)

        image_2d = _prepare_image(image_slice, channels=self.input_channels, target_size=self.target_size, normalize=self.normalize)
        label_2d = _prepare_label(label_slice, target_size=self.target_size, num_classes=self.num_classes)
        return (
            torch.from_numpy(image_2d.astype(np.float32, copy=False)).contiguous(),
            torch.from_numpy(label_2d.astype(np.int64, copy=False)).contiguous(),
        )

    def _cache_path(self, sample: SegmentationSample) -> Path | None:
        if self.cache_dir is None:
            return None
        key = _cache_key(
            sample,
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            target_size=self.target_size,
            normalize=self.normalize,
        )
        return self.cache_dir / _safe_cache_name(sample.record.dataset) / key[:2] / f"{key}.pt"


def _build_samples(
    split_csv: Path,
    *,
    split: str,
    labeled_only: bool,
    include_empty_slices: bool,
    max_slices_per_volume: int | None,
) -> list[SegmentationSample]:
    rows = _read_split_rows(split_csv)
    samples: list[SegmentationSample] = []
    for row in rows:
        if row.get("phase_b_split", row.get("split", "")) != split:
            continue
        labeled = _parse_bool(row.get("phase_b_labeled", "false"))
        if labeled_only and not labeled:
            continue
        record = ManifestRecord.from_row(row, split_csv.parent)
        if record.is_2d:
            samples.append(SegmentationSample(record=record, split=split, labeled=labeled))
            continue
        label = load_image_and_label(record)[1].array
        slice_indices = _volume_slice_indices(
            label,
            slice_axis=record.slice_axis,
            include_empty_slices=include_empty_slices,
            max_slices=max_slices_per_volume,
        )
        for slice_index in slice_indices:
            samples.append(SegmentationSample(record=record, split=split, labeled=labeled, slice_index=slice_index))
    return samples


def _read_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cache_key(
    sample: SegmentationSample,
    *,
    input_channels: int,
    num_classes: int | None,
    target_size: int,
    normalize: str,
) -> str:
    record = sample.record
    parts = [
        "phase_b_preprocessed_v1",
        record.dataset,
        record.case_id,
        record.patient_id,
        _path_signature(record.image_path),
        _path_signature(record.label_path),
        str(record.is_2d),
        str(record.slice_axis),
        str(_encoded_slice_index(sample.slice_index)),
        str(input_channels),
        "" if num_classes is None else str(num_classes),
        str(target_size),
        normalize,
    ]
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def _path_signature(path: str) -> str:
    source = Path(path)
    try:
        stat = source.stat()
    except OSError:
        return str(source)
    return f"{source}|{stat.st_size}|{stat.st_mtime_ns}"


def _safe_cache_name(value: str) -> str:
    text = "".join(char.lower() if char.isalnum() else "_" for char in str(value).strip())
    return "_".join(part for part in text.split("_") if part) or "dataset"


def _read_cached_tensors(path: Path) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not path.exists():
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch compatibility
            payload = torch.load(path, map_location="cpu")
        image = payload["image"]
        label = payload["label"]
    except Exception:
        return None
    if not torch.is_tensor(image) or not torch.is_tensor(label):
        return None
    return image.contiguous(), label.to(dtype=torch.int64).contiguous()


def _write_cached_tensors(path: Path, image: torch.Tensor, label: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        torch.save({"image": image.cpu(), "label": _compact_label_tensor(label)}, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _compact_label_tensor(label: torch.Tensor) -> torch.Tensor:
    label_cpu = label.detach().cpu()
    if label_cpu.numel() == 0:
        return label_cpu.to(dtype=torch.int64)
    min_value = int(label_cpu.min().item())
    max_value = int(label_cpu.max().item())
    if 0 <= min_value and max_value <= 255:
        return label_cpu.to(dtype=torch.uint8)
    if -32768 <= min_value and max_value <= 32767:
        return label_cpu.to(dtype=torch.int16)
    return label_cpu.to(dtype=torch.int64)


def _volume_slice_indices(
    label: np.ndarray,
    *,
    slice_axis: int,
    include_empty_slices: bool,
    max_slices: int | None,
) -> list[int | tuple[int, int]]:
    arr = np.asarray(label)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return [0]
    if arr.ndim == 4:
        return _volume_time_slice_indices(arr, include_empty_slices=include_empty_slices, max_slices=max_slices)
    if arr.ndim < 3:
        return []
    axis = slice_axis if slice_axis >= 0 else arr.ndim + slice_axis
    axis = min(max(axis, 0), arr.ndim - 1)
    moved = np.moveaxis(arr, axis, 0)
    indices = [index for index, item in enumerate(moved) if include_empty_slices or np.any(item > 0)]
    if max_slices is not None and len(indices) > max_slices:
        positions = np.linspace(0, len(indices) - 1, int(max_slices)).round().astype(int)
        indices = [indices[int(position)] for position in positions]
    return indices


def _volume_time_slice_indices(
    label: np.ndarray,
    *,
    include_empty_slices: bool,
    max_slices: int | None,
) -> list[tuple[int, int]]:
    arr = np.asarray(label)
    indices = [
        (z_index, time_index)
        for time_index in range(arr.shape[3])
        for z_index in range(arr.shape[2])
        if include_empty_slices or np.any(arr[:, :, z_index, time_index] > 0)
    ]
    if max_slices is not None and len(indices) > max_slices:
        positions = np.linspace(0, len(indices) - 1, int(max_slices)).round().astype(int)
        indices = [indices[int(position)] for position in positions]
    return indices


def _select_2d_slice(array: np.ndarray, slice_index: int | tuple[int, int] | None, slice_axis: int) -> np.ndarray:
    arr = np.asarray(array)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in {1, 3, 4} and arr.shape[0] > 8:
        return arr
    if arr.ndim == 4 and isinstance(slice_index, tuple):
        z_index, time_index = slice_index
        return np.asarray(arr[:, :, int(z_index), int(time_index)])
    if arr.ndim < 3:
        raise ValueError(f"Cannot select a 2D slice from shape {arr.shape}.")
    axis = slice_axis if slice_axis >= 0 else arr.ndim + slice_axis
    axis = min(max(axis, 0), arr.ndim - 1)
    moved = np.moveaxis(arr, axis, 0)
    index = 0 if slice_index is None else int(slice_index)
    return np.asarray(moved[index])


def _encoded_slice_index(slice_index: int | tuple[int, int] | None) -> int:
    if slice_index is None:
        return -1
    if isinstance(slice_index, tuple):
        z_index, time_index = slice_index
        return int(z_index) * 10000 + int(time_index)
    return int(slice_index)


def _prepare_image(image: np.ndarray, *, channels: int, target_size: int, normalize: str) -> np.ndarray:
    arr = np.asarray(image)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[-1] in {1, 3, 4}:
        arr = arr[..., :channels]
        arr = np.moveaxis(arr, -1, 0)
    elif arr.ndim == 3:
        arr = arr[:1, ...]
    else:
        raise ValueError(f"Unsupported image shape {arr.shape}.")

    if arr.shape[0] == 1 and channels == 3:
        arr = np.repeat(arr, 3, axis=0)
    if arr.shape[0] >= 3 and channels == 1:
        arr = arr[:1]
    if arr.shape[0] != channels:
        raise ValueError(f"Expected {channels} image channels, got {arr.shape[0]}.")

    resized = np.stack([_resize_2d(channel, target_size, order=1) for channel in arr], axis=0).astype(np.float32)
    if normalize == "zscore":
        mean = float(resized.mean())
        std = float(resized.std())
        if std > 1e-6:
            resized = (resized - mean) / std
        else:
            resized = resized - mean
    elif normalize == "minmax":
        low = float(resized.min())
        high = float(resized.max())
        resized = (resized - low) / max(high - low, 1e-6)
    elif normalize not in {"none", ""}:
        raise ValueError(f"Unsupported normalize mode {normalize!r}.")
    return resized


def _prepare_label(label: np.ndarray, *, target_size: int, num_classes: int | None = None) -> np.ndarray:
    arr = np.asarray(label)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[-1] in {1, 3, 4}:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported label shape {arr.shape}.")
    if arr.dtype.kind in {"f", "c"}:
        arr = np.rint(np.real(arr))
    resized = _resize_2d(arr.astype(np.int32, copy=False), target_size, order=0).astype(np.int64, copy=False)
    if num_classes is None:
        return resized
    if int(num_classes) <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")
    if int(num_classes) == 2:
        return (resized > 0).astype(np.int64, copy=False)

    invalid = resized[(resized < 0) | (resized >= int(num_classes))]
    if invalid.size:
        values = np.unique(invalid)[:10].astype(int).tolist()
        raise ValueError(f"Label values {values} are outside [0, {int(num_classes) - 1}].")
    return resized


def _resize_2d(array: np.ndarray, target_size: int, *, order: int) -> np.ndarray:
    arr = np.asarray(array)
    if arr.shape == (target_size, target_size):
        return arr
    zoom_y = target_size / float(arr.shape[0])
    zoom_x = target_size / float(arr.shape[1])
    return ndi.zoom(arr, zoom=(zoom_y, zoom_x), order=order)


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}

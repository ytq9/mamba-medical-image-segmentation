from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .manifest import ManifestRecord


@dataclass(slots=True)
class LoadedArray:
    array: np.ndarray
    spacing: tuple[float, float, float]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".bmp"}
NUMPY_EXTENSIONS = {".npy", ".npz"}


def load_array(path: str | Path, record: ManifestRecord | None = None) -> LoadedArray:
    source = Path(path)
    suffix = source.suffix.lower()
    suffixes = [item.lower() for item in source.suffixes]
    spacing = _spacing_from_record(record)
    if suffix in IMAGE_EXTENSIONS:
        with Image.open(source) as image:
            array = np.asarray(image)
        return LoadedArray(array=array, spacing=spacing)
    if suffix in NUMPY_EXTENSIONS:
        payload = np.load(source)
        if suffix == ".npz":
            key = "arr_0" if "arr_0" in payload else next(iter(payload.keys()))
            array = np.asarray(payload[key])
        else:
            array = np.asarray(payload)
        return LoadedArray(array=array, spacing=spacing)
    if suffixes[-2:] == [".nii", ".gz"] or suffix == ".nii":
        return _load_nifti(source)
    if suffix in {".mhd", ".mha"}:
        return _load_simpleitk(source, fallback_spacing=spacing)
    raise ValueError(f"Unsupported file format: {source}")


def load_image_and_label(record: ManifestRecord) -> tuple[LoadedArray, LoadedArray]:
    image = load_array(record.image_path, record)
    label = load_array(record.label_path, record)
    return image, label


def label_values_from_mask(mask: np.ndarray) -> list[int]:
    values = np.unique(np.asarray(mask))
    return [int(value) for value in values if int(value) != 0]


def spatial_shape(array: np.ndarray) -> tuple[int, int] | tuple[int, int, int]:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return (int(arr.shape[0]), int(arr.shape[1]))
    if arr.ndim == 3:
        if arr.shape[-1] in {1, 3, 4} and arr.shape[0] > 8:
            return (int(arr.shape[0]), int(arr.shape[1]))
        return tuple(int(x) for x in arr.shape[-3:])
    if arr.ndim >= 4:
        return tuple(int(x) for x in arr.shape[-3:])
    return tuple(int(x) for x in arr.shape)


def shapes_compatible(image: np.ndarray, label: np.ndarray) -> bool:
    image_shape = spatial_shape(image)
    label_shape = spatial_shape(label)
    if len(image_shape) == len(label_shape):
        return image_shape == label_shape
    if len(image_shape) == 3 and len(label_shape) == 2:
        return image_shape[-2:] == label_shape
    if len(image_shape) == 2 and len(label_shape) == 3:
        return image_shape == label_shape[-2:]
    return False


def _spacing_from_record(record: ManifestRecord | None) -> tuple[float, float, float]:
    if record is None:
        return (1.0, 1.0, 1.0)
    return (float(record.spacing_z), float(record.spacing_y), float(record.spacing_x))


def _load_nifti(path: Path) -> LoadedArray:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent
        raise ImportError("nibabel is required for .nii/.nii.gz files. Install project dependencies first.") from exc
    image = nib.load(str(path))
    array = np.asarray(image.get_fdata())
    zooms = image.header.get_zooms()
    if len(zooms) >= 3:
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    elif len(zooms) == 2:
        spacing = (1.0, float(zooms[1]), float(zooms[0]))
    else:
        spacing = (1.0, 1.0, 1.0)
    return LoadedArray(array=array, spacing=spacing)


def _load_simpleitk(path: Path, fallback_spacing: tuple[float, float, float]) -> LoadedArray:
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent
        raise ImportError("SimpleITK is required for .mhd/.mha files. Install project dependencies first.") from exc
    image = sitk.ReadImage(str(path))
    array = np.asarray(sitk.GetArrayFromImage(image))
    spacing_raw = image.GetSpacing()
    if len(spacing_raw) >= 3:
        spacing = (float(spacing_raw[2]), float(spacing_raw[1]), float(spacing_raw[0]))
    elif len(spacing_raw) == 2:
        spacing = (1.0, float(spacing_raw[1]), float(spacing_raw[0]))
    else:
        spacing = fallback_spacing
    return LoadedArray(array=array, spacing=spacing)

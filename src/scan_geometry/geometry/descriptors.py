from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


DEFAULT_LOCALITY_RADII = (0.005, 0.01, 0.02, 0.04)
MIN_COMPONENT_AREA = 20.0
DESCRIPTOR_COLUMNS = ["D", "A_data", "phi_data", "L", "S", "B", "C", "foreground_ratio"]


@dataclass(slots=True)
class DescriptorResult:
    D: float
    A_data: float
    phi_data: float
    L: float
    S: float
    B: float
    C: float
    foreground_ratio: float
    empty: bool = False

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "D": self.D,
            "A_data": self.A_data,
            "phi_data": self.phi_data,
            "L": self.L,
            "S": self.S,
            "B": self.B,
            "C": self.C,
            "foreground_ratio": self.foreground_ratio,
            "empty": self.empty,
        }


def compute_mask_descriptors(
    mask: np.ndarray,
    locality_radii: tuple[float, ...] = DEFAULT_LOCALITY_RADII,
) -> DescriptorResult:
    label = _as_label2d(mask)
    foreground = label > 0
    if not foreground.any():
        return DescriptorResult(
            D=1.0,
            A_data=0.0,
            phi_data=float("nan"),
            L=0.0,
            S=0.0,
            B=0.0,
            C=0.0,
            foreground_ratio=0.0,
            empty=True,
        )
    D, phi = directionality_entropy_and_phi(foreground)
    L = locality_index(label, locality_radii=locality_radii)
    S = scale_diversity(label)
    B = boundary_complexity(label)
    C = connectivity(label)
    return DescriptorResult(
        D=float(D),
        A_data=float(1.0 - D),
        phi_data=float(phi),
        L=float(L),
        S=float(S),
        B=float(B),
        C=float(C),
        foreground_ratio=float(foreground.mean()),
        empty=False,
    )


def directionality_entropy_and_phi(binary_mask: np.ndarray, bins: int = 18) -> tuple[float, float]:
    mask = np.asarray(binary_mask, dtype=bool)
    if not mask.any():
        return 1.0, float("nan")
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary = mask & ~eroded
    if int(boundary.sum()) < 4:
        return 1.0, float("nan")
    gy = ndi.sobel(mask.astype(np.float32), axis=0, mode="constant")
    gx = ndi.sobel(mask.astype(np.float32), axis=1, mode="constant")
    magnitude = np.hypot(gx, gy)
    active = boundary & (magnitude > 1e-6)
    if int(active.sum()) < 4:
        return 1.0, float("nan")
    angles = np.mod(np.arctan2(gy[active], gx[active]), np.pi)
    hist, edges = np.histogram(angles, bins=bins, range=(0.0, np.pi))
    probs = hist.astype(np.float64)
    if probs.sum() == 0:
        return 1.0, float("nan")
    probs /= probs.sum()
    nonzero = probs[probs > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero))) / float(np.log(bins))
    centers = 0.5 * (edges[:-1] + edges[1:])
    dominant = float(centers[int(np.argmax(hist))])
    return _clip01(entropy), dominant


def locality_index(label: np.ndarray, locality_radii: tuple[float, ...] = DEFAULT_LOCALITY_RADII) -> float:
    label2d = _as_label2d(label)
    height, width = label2d.shape
    diagonal = float(np.hypot(height, width))
    values: list[float] = []
    for class_value in _foreground_values(label2d):
        class_mask = label2d == class_value
        if not class_mask.any():
            continue
        for fraction in locality_radii:
            radius = max(1, int(round(diagonal * float(fraction))))
            kernel = _disk_kernel(radius)
            same_count = ndi.convolve(class_mask.astype(np.float32), kernel, mode="constant", cval=0.0)
            denom = ndi.convolve(np.ones_like(class_mask, dtype=np.float32), kernel, mode="constant", cval=0.0)
            local_fraction = same_count[class_mask] / np.maximum(denom[class_mask], 1.0)
            values.append(float(local_fraction.mean()))
    return _clip01(float(np.mean(values))) if values else 0.0


def scale_diversity(label: np.ndarray) -> float:
    areas = _component_areas(_as_label2d(label), min_area=MIN_COMPONENT_AREA)
    if len(areas) < 2:
        return 0.0
    return float(np.std(np.log(np.asarray(areas, dtype=np.float64))))


def boundary_complexity(label: np.ndarray) -> float:
    label2d = _as_label2d(label)
    complexities: list[float] = []
    for class_value in _foreground_values(label2d):
        components, count = ndi.label(label2d == class_value)
        for component_id in range(1, count + 1):
            component = components == component_id
            area = float(component.sum())
            if area < MIN_COMPONENT_AREA:
                continue
            eroded = ndi.binary_erosion(component, structure=np.ones((3, 3), dtype=bool), border_value=0)
            perimeter = float((component & ~eroded).sum())
            if perimeter < 1:
                continue
            compactness = 4.0 * np.pi * area / (perimeter * perimeter)
            complexities.append(_clip01(1.0 - float(compactness)))
    if not complexities:
        return 0.0
    return float(np.mean(complexities))


def connectivity(label: np.ndarray) -> float:
    label2d = _as_label2d(label)
    counts: list[int] = []
    for class_value in _foreground_values(label2d):
        _, count = ndi.label(label2d == class_value)
        counts.append(int(count))
    return float(np.mean(counts)) if counts else 0.0


def aggregate_descriptors(results: list[DescriptorResult]) -> dict[str, float]:
    valid = [result for result in results if not result.empty]
    if not valid:
        return {column: float("nan") for column in DESCRIPTOR_COLUMNS} | {"empty_slice_rate": 1.0, "n_units": 0}
    payload: dict[str, float] = {}
    for column in ["D", "A_data", "L", "S", "B", "C", "foreground_ratio"]:
        payload[column] = float(np.mean([getattr(result, column) for result in valid]))
    payload["phi_data"] = axial_angle_mean([result.phi_data for result in valid])
    payload["empty_slice_rate"] = float(1.0 - (len(valid) / max(len(results), 1)))
    payload["n_units"] = float(len(valid))
    return payload


def axial_angle_mean(angles: list[float]) -> float:
    clean = np.asarray([angle for angle in angles if np.isfinite(angle)], dtype=np.float64)
    if clean.size == 0:
        return float("nan")
    vector = np.mean(np.exp(2j * clean))
    return float(np.mod(np.angle(vector) / 2.0, np.pi))


def iter_informative_slices(mask: np.ndarray, slice_axis: int = -1) -> list[np.ndarray]:
    arr = np.asarray(mask)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return []
    if arr.ndim == 2:
        return [_as_label2d(arr)]
    if arr.ndim == 3:
        axis = slice_axis if slice_axis >= 0 else arr.ndim + slice_axis
        axis = min(max(axis, 0), arr.ndim - 1)
        moved = np.moveaxis(arr, axis, 0)
        return [_as_label2d(slice_) for slice_ in moved]
    if arr.ndim > 3:
        axis = slice_axis if slice_axis >= 0 else arr.ndim + slice_axis
        moved = np.moveaxis(arr, min(max(axis, 0), arr.ndim - 1), 0)
        output: list[np.ndarray] = []
        for block in moved:
            output.extend(iter_informative_slices(block, slice_axis=-1))
        return output
    return []


def resize_mask_nearest(mask: np.ndarray, target_size: int) -> np.ndarray:
    label = _as_label2d(mask)
    if label.shape == (target_size, target_size):
        return label
    zoom_y = target_size / float(label.shape[0])
    zoom_x = target_size / float(label.shape[1])
    return ndi.zoom(label, zoom=(zoom_y, zoom_x), order=0).astype(label.dtype, copy=False)


def descriptors_for_mask_stack(mask: np.ndarray, slice_axis: int, analysis_size: int | None = None) -> list[DescriptorResult]:
    slices = iter_informative_slices(mask, slice_axis=slice_axis)
    output: list[DescriptorResult] = []
    for slice_ in slices:
        label = resize_mask_nearest(slice_, analysis_size) if analysis_size else _as_label2d(slice_)
        output.append(compute_mask_descriptors(label))
    return output


def _component_areas(label: np.ndarray, min_area: float = 0.0) -> list[float]:
    areas: list[float] = []
    for class_value in _foreground_values(label):
        components, count = ndi.label(label == class_value)
        for component_id in range(1, count + 1):
            area = float((components == component_id).sum())
            if area >= min_area:
                areas.append(area)
    return areas


def _disk_kernel(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (xx * xx + yy * yy) <= radius * radius
    return kernel.astype(np.float32)


def _foreground_values(label: np.ndarray) -> list[int]:
    values = np.unique(label)
    return [int(value) for value in values if int(value) != 0]


def _as_label2d(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[-1] in {1, 3, 4}:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D label mask, got shape {arr.shape}.")
    if arr.dtype.kind in {"f", "c"}:
        arr = np.rint(np.real(arr))
    return arr.astype(np.int32, copy=False)


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))

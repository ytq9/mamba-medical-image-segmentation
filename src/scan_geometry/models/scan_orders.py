from __future__ import annotations

import hashlib
import math
from functools import lru_cache

import numpy as np


VALID_SCAN_ORDERS = {"Raster-H", "Raster-V", "Hilbert", "LocalWindow", "RandomPermute"}


def build_scan_order(
    name: str,
    height: int,
    width: int,
    *,
    local_window_size: int = 8,
    seed: int = 2026,
) -> np.ndarray:
    """Return row-major flat indices in the order consumed by the sequence model."""
    order_name = _normalize_scan_name(name)
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}.")
    return _cached_scan_order(order_name, int(height), int(width), int(local_window_size), int(seed)).copy()


def inverse_scan_order(order: np.ndarray) -> np.ndarray:
    order = np.asarray(order, dtype=np.int64)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size, dtype=np.int64)
    return inverse


def scan_order_hash(
    name: str,
    height: int,
    width: int,
    *,
    local_window_size: int = 8,
    seed: int = 2026,
) -> str:
    order = build_scan_order(name, height, width, local_window_size=local_window_size, seed=seed)
    payload = order.astype(np.int64, copy=False).tobytes()
    header = f"{_normalize_scan_name(name)}:{height}:{width}:{local_window_size}:{seed}:".encode("utf-8")
    return hashlib.sha256(header + payload).hexdigest()[:16]


def validate_scan_order(order: np.ndarray, length: int) -> None:
    order = np.asarray(order, dtype=np.int64)
    if order.shape != (length,):
        raise ValueError(f"Scan order must have shape ({length},), got {order.shape}.")
    if sorted(order.tolist()) != list(range(length)):
        raise ValueError("Scan order must be a bijection over flattened spatial positions.")


@lru_cache(maxsize=128)
def _cached_scan_order(name: str, height: int, width: int, local_window_size: int, seed: int) -> np.ndarray:
    if name == "Raster-H":
        return np.arange(height * width, dtype=np.int64)
    if name == "Raster-V":
        return np.asarray([y * width + x for x in range(width) for y in range(height)], dtype=np.int64)
    if name == "Hilbert":
        return _hilbert_order(height, width)
    if name == "LocalWindow":
        return _local_window_order(height, width, local_window_size)
    if name == "RandomPermute":
        rng = np.random.default_rng(seed)
        return rng.permutation(height * width).astype(np.int64, copy=False)
    raise ValueError(f"Unsupported scan order {name!r}. Expected one of {sorted(VALID_SCAN_ORDERS)}.")


def _local_window_order(height: int, width: int, window_size: int) -> np.ndarray:
    if window_size <= 0:
        raise ValueError("local_window_size must be positive.")
    if height % window_size != 0 or width % window_size != 0:
        raise ValueError(
            f"LocalWindow requires H and W divisible by window_size={window_size}; got {height}x{width}."
        )
    indices: list[int] = []
    for window_y in range(0, height, window_size):
        for window_x in range(0, width, window_size):
            for y in range(window_y, window_y + window_size):
                for x in range(window_x, window_x + window_size):
                    indices.append(y * width + x)
    return np.asarray(indices, dtype=np.int64)


def _hilbert_order(height: int, width: int) -> np.ndarray:
    if height != width:
        raise ValueError(f"Hilbert scan requires a square feature map, got {height}x{width}.")
    if not _is_power_of_two(height):
        raise ValueError(f"Hilbert scan requires a power-of-two size, got {height}.")
    side = int(height)
    indices = []
    for distance in range(side * side):
        x, y = _hilbert_d2xy(side, distance)
        indices.append(y * side + x)
    return np.asarray(indices, dtype=np.int64)


def _hilbert_d2xy(side: int, distance: int) -> tuple[int, int]:
    x = 0
    y = 0
    t = int(distance)
    scale = 1
    while scale < side:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = scale - 1 - x
                y = scale - 1 - y
            x, y = y, x
        x += scale * rx
        y += scale * ry
        t //= 4
        scale *= 2
    return int(x), int(y)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and math.log2(value).is_integer()


def _normalize_scan_name(name: str) -> str:
    clean = str(name).strip()
    aliases = {
        "raster_h": "Raster-H",
        "raster-h": "Raster-H",
        "rasterh": "Raster-H",
        "horizontal": "Raster-H",
        "raster_v": "Raster-V",
        "raster-v": "Raster-V",
        "rasterv": "Raster-V",
        "vertical": "Raster-V",
        "hilbert": "Hilbert",
        "local_window": "LocalWindow",
        "local-window": "LocalWindow",
        "localwindow": "LocalWindow",
        "random": "RandomPermute",
        "random_permute": "RandomPermute",
        "random-permute": "RandomPermute",
        "randompermute": "RandomPermute",
    }
    normalized = aliases.get(clean.lower(), clean)
    if normalized not in VALID_SCAN_ORDERS:
        raise ValueError(f"Unsupported scan order {name!r}. Expected one of {sorted(VALID_SCAN_ORDERS)}.")
    return normalized

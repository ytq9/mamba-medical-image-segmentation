from __future__ import annotations

import numpy as np

from .descriptors import DESCRIPTOR_COLUMNS


def bootstrap_dataset_ci(
    case_rows: list[dict[str, float | str | bool]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float]:
    numeric = [row for row in case_rows if not bool(row.get("empty", False))]
    if not numeric:
        return {f"{col}_ci_low": float("nan") for col in DESCRIPTOR_COLUMNS} | {
            f"{col}_ci_high": float("nan") for col in DESCRIPTOR_COLUMNS
        }
    rng = np.random.default_rng(seed)
    payload: dict[str, float] = {}
    for column in DESCRIPTOR_COLUMNS:
        values = np.asarray([float(row[column]) for row in numeric if _is_number(row.get(column))], dtype=np.float64)
        if values.size == 0:
            payload[f"{column}_ci_low"] = float("nan")
            payload[f"{column}_ci_high"] = float("nan")
            continue
        means = []
        for _ in range(max(1, int(bootstrap_samples))):
            sample = rng.choice(values, size=values.size, replace=True)
            means.append(float(np.mean(sample)))
        payload[f"{column}_ci_low"] = float(np.percentile(means, 2.5))
        payload[f"{column}_ci_high"] = float(np.percentile(means, 97.5))
    return payload


def resize_stability_delta(base: dict[str, float], alternate: dict[str, float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for column in ["D", "L", "S", "B", "C", "A_data"]:
        left = float(base.get(column, float("nan")))
        right = float(alternate.get(column, float("nan")))
        payload[f"{column}_resize_delta"] = float(abs(left - right)) if np.isfinite(left) and np.isfinite(right) else float("nan")
    finite = [value for value in payload.values() if np.isfinite(value)]
    payload["max_resize_delta"] = float(max(finite)) if finite else float("nan")
    return payload


def _is_number(value: object) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False

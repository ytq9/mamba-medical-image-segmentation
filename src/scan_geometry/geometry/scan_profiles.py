from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scan_geometry.models.scan_orders import VALID_SCAN_ORDERS, build_scan_order


DEFAULT_SCAN_SHAPE = (32, 32)
DEFAULT_LOCAL_WINDOW_SIZE = 8
DEFAULT_SCAN_SEED = 2026
CONTINUITY_SIGMA = 1.0
DIRECTIONALITY_MIN_CONTINUITY = 0.05


@dataclass(frozen=True, slots=True)
class ScanProfile:
    name: str
    A_scan: float
    phi_scan: float | None
    P_loc: float
    C_scan: float = 1.0
    mean_step: float = 1.0
    jump_rate: float = 0.0
    turn_rate: float = 0.0


def omega(phi_data: float, phi_scan: float | None) -> float:
    if phi_scan is None or not np.isfinite(phi_data):
        return 0.5
    return float((1.0 + np.cos(2.0 * (phi_data - phi_scan))) / 2.0)


def matching_score(dataset_geometry: dict[str, float], profile: ScanProfile) -> dict[str, float | str]:
    A_data = _clip01(float(dataset_geometry.get("A_data", 0.0)))
    L = _clip01(float(dataset_geometry.get("L", 0.0)))
    phi_data = float(dataset_geometry.get("phi_data", float("nan")))
    continuity_gate = _clip01(profile.C_scan)
    M_dir_raw = (1.0 - A_data) * (1.0 - profile.A_scan) + A_data * profile.A_scan * omega(phi_data, profile.phi_scan)
    M_loc_raw = 1.0 - abs(L - profile.P_loc)
    M_dir = continuity_gate * M_dir_raw
    M_loc = continuity_gate * M_loc_raw
    M_primary = 0.5 * M_dir + 0.5 * M_loc
    return {
        "scan": profile.name,
        "A_scan": profile.A_scan,
        "phi_scan": "" if profile.phi_scan is None else profile.phi_scan,
        "P_loc": profile.P_loc,
        "C_scan": profile.C_scan,
        "mean_step": profile.mean_step,
        "jump_rate": profile.jump_rate,
        "turn_rate": profile.turn_rate,
        "M_dir_raw": _clip01(M_dir_raw),
        "M_loc_raw": _clip01(M_loc_raw),
        "M_dir": _clip01(M_dir),
        "M_loc": _clip01(M_loc),
        "M_primary": _clip01(M_primary),
    }


def matching_table(dataset_name: str, dataset_geometry: dict[str, float]) -> list[dict[str, float | str]]:
    rows = []
    for profile in DEFAULT_SCAN_PROFILES:
        row = matching_score(dataset_geometry, profile)
        row["dataset"] = dataset_name
        rows.append(row)
    return rows


def matching_contrast(rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    if not rows:
        return {"matching_contrast": float("nan"), "best_scan": "", "worst_scan": ""}
    ordered = sorted(rows, key=lambda row: float(row["M_primary"]))
    scores = [float(row["M_primary"]) for row in ordered]
    return {
        "matching_contrast": float(max(scores) - min(scores)),
        "matching_std": float(np.std(scores)),
        "best_scan": str(ordered[-1]["scan"]),
        "worst_scan": str(ordered[0]["scan"]),
    }


def path_derived_scan_profiles(
    *,
    height: int = DEFAULT_SCAN_SHAPE[0],
    width: int = DEFAULT_SCAN_SHAPE[1],
    local_window_size: int = DEFAULT_LOCAL_WINDOW_SIZE,
    seed: int = DEFAULT_SCAN_SEED,
) -> list[ScanProfile]:
    return [
        scan_profile_from_path(name, height=height, width=width, local_window_size=local_window_size, seed=seed)
        for name in ["Raster-H", "Raster-V", "Hilbert", "LocalWindow", "RandomPermute"]
    ]


def scan_profile_from_path(
    name: str,
    *,
    height: int = DEFAULT_SCAN_SHAPE[0],
    width: int = DEFAULT_SCAN_SHAPE[1],
    local_window_size: int = DEFAULT_LOCAL_WINDOW_SIZE,
    seed: int = DEFAULT_SCAN_SEED,
) -> ScanProfile:
    if name not in VALID_SCAN_ORDERS:
        raise ValueError(f"Unsupported scan order {name!r}. Expected one of {sorted(VALID_SCAN_ORDERS)}.")
    order = build_scan_order(name, height, width, local_window_size=local_window_size, seed=seed)
    return scan_profile_from_order(name, order, height=height, width=width)


def scan_profile_from_order(name: str, order: np.ndarray, *, height: int, width: int) -> ScanProfile:
    order = np.asarray(order, dtype=np.int64)
    y = order // int(width)
    x = order % int(width)
    coords = np.stack([y, x], axis=1).astype(np.float64, copy=False)
    deltas = np.diff(coords, axis=0)
    if deltas.size == 0:
        return ScanProfile(name, A_scan=0.0, phi_scan=None, P_loc=0.0, C_scan=0.0)

    distances = np.linalg.norm(deltas, axis=1)
    local_weights = np.exp(-(np.maximum(distances - 1.0, 0.0) ** 2) / (2.0 * CONTINUITY_SIGMA**2))
    P_loc = _clip01(float(np.mean(local_weights)))
    C_scan = P_loc
    jump_rate = float(np.mean(distances > (np.sqrt(2.0) + 1e-9)))
    turn_rate = _turn_rate(deltas, local_weights)

    A_scan, phi_scan = _path_directionality(deltas, local_weights, C_scan)
    return ScanProfile(
        name=name,
        A_scan=A_scan,
        phi_scan=phi_scan,
        P_loc=P_loc,
        C_scan=C_scan,
        mean_step=float(np.mean(distances)),
        jump_rate=jump_rate,
        turn_rate=turn_rate,
    )


def _path_directionality(deltas: np.ndarray, weights: np.ndarray, continuity: float) -> tuple[float, float | None]:
    if continuity < DIRECTIONALITY_MIN_CONTINUITY or float(np.sum(weights)) < 1e-12:
        return 0.0, None
    angles = np.mod(np.arctan2(deltas[:, 0], deltas[:, 1]), np.pi)
    vector = np.sum(weights * np.exp(2j * angles)) / float(np.sum(weights))
    A_scan = _clip01(float(np.abs(vector)))
    if A_scan < 1e-12:
        return A_scan, None
    phi_scan = float(np.mod(np.angle(vector) / 2.0, np.pi))
    return A_scan, phi_scan


def _turn_rate(deltas: np.ndarray, weights: np.ndarray) -> float:
    if len(deltas) < 2:
        return 0.0
    local = weights > 0.5
    valid = local[:-1] & local[1:]
    if not bool(valid.any()):
        return 1.0
    previous = deltas[:-1][valid]
    current = deltas[1:][valid]
    previous_angles = np.mod(np.arctan2(previous[:, 0], previous[:, 1]), np.pi)
    current_angles = np.mod(np.arctan2(current[:, 0], current[:, 1]), np.pi)
    angle_delta = np.abs(np.angle(np.exp(2j * (current_angles - previous_angles))) / 2.0)
    return float(np.mean(angle_delta > (np.pi / 8.0)))


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


DEFAULT_SCAN_PROFILES = path_derived_scan_profiles()

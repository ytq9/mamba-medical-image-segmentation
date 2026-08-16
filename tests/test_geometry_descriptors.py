from __future__ import annotations

import numpy as np

from scan_geometry.geometry.descriptors import compute_mask_descriptors, directionality_entropy_and_phi
from scan_geometry.geometry.scan_profiles import matching_score, ScanProfile


def test_compact_circle_has_high_locality() -> None:
    yy, xx = np.ogrid[:128, :128]
    mask = ((yy - 64) ** 2 + (xx - 64) ** 2 <= 24**2).astype(np.uint8)
    result = compute_mask_descriptors(mask)
    assert result.empty is False
    assert result.L > 0.70
    assert 0.0 <= result.D <= 1.0
    assert result.C == 1.0


def test_fragmented_components_increase_connectivity_and_scale_diversity() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[40:70, 40:70] = 1
    mask[90:100, 90:100] = 1
    result = compute_mask_descriptors(mask)
    assert result.C == 3.0
    assert result.S > 0.0


def test_empty_mask_is_flagged_without_crashing() -> None:
    result = compute_mask_descriptors(np.zeros((32, 32), dtype=np.uint8))
    assert result.empty is True
    assert result.foreground_ratio == 0.0


def test_tiny_components_are_filtered_from_scale_and_boundary_complexity() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    with_tiny_noise = mask.copy()
    with_tiny_noise[40:42, 40:42] = 1

    base = compute_mask_descriptors(mask)
    noisy = compute_mask_descriptors(with_tiny_noise)

    assert noisy.C == 2.0
    assert noisy.S == base.S
    assert noisy.B == base.B


def test_directionality_entropy_uses_sobel_boundary_angles() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 20:44] = 1
    entropy, phi = directionality_entropy_and_phi(mask)
    assert 0.0 <= entropy <= 1.0
    assert 0.0 <= phi < np.pi


def test_matching_score_uses_only_geometry_and_scan_profile() -> None:
    dataset_geometry = {"A_data": 1.0, "phi_data": 0.0, "L": 0.8}
    horizontal = ScanProfile("Raster-H", A_scan=1.0, phi_scan=0.0, P_loc=0.5)
    vertical = ScanProfile("Raster-V", A_scan=1.0, phi_scan=np.pi / 2, P_loc=0.5)
    assert matching_score(dataset_geometry, horizontal)["M_primary"] > matching_score(dataset_geometry, vertical)["M_primary"]

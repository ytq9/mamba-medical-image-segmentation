from __future__ import annotations

import numpy as np
import pytest
import torch

from scan_geometry.models import ModelConfig, build_phase_b_model, build_scan_order, count_parameters
from scan_geometry.models.scan_orders import inverse_scan_order, scan_order_hash, validate_scan_order


@pytest.mark.parametrize("scan", ["Raster-H", "Raster-V", "Hilbert", "LocalWindow", "RandomPermute"])
def test_scan_orders_are_bijections_and_invertible(scan: str) -> None:
    order = build_scan_order(scan, 16, 16, local_window_size=4, seed=2026)
    validate_scan_order(order, 16 * 16)
    inverse = inverse_scan_order(order)
    values = np.arange(16 * 16)
    assert np.array_equal(values[order][inverse], values)


def test_random_scan_hash_is_seeded_and_deterministic() -> None:
    left = scan_order_hash("RandomPermute", 16, 16, local_window_size=4, seed=2026)
    right = scan_order_hash("RandomPermute", 16, 16, local_window_size=4, seed=2026)
    other = scan_order_hash("RandomPermute", 16, 16, local_window_size=4, seed=2027)
    assert left == right
    assert left != other


@pytest.mark.parametrize("scan", ["Raster-H", "Raster-V", "Hilbert", "LocalWindow", "RandomPermute"])
def test_umamba_scan_bot_forward_shape_with_fallback(scan: str) -> None:
    model = build_phase_b_model(
        condition_family="mamba",
        scan_order=scan,
        config=ModelConfig(
            input_channels=1,
            num_classes=3,
            features=(8, 16, 32),
            blocks_per_stage=1,
            local_window_size=4,
            allow_mamba_fallback=True,
        ),
    )
    x = torch.randn(2, 1, 32, 32)
    y = model(x)
    assert tuple(y.shape) == (2, 3, 32, 32)


def test_mamba_conditions_have_identical_parameter_counts_with_fallback() -> None:
    config = ModelConfig(
        input_channels=1,
        num_classes=2,
        features=(8, 16, 32),
        blocks_per_stage=1,
        local_window_size=4,
        allow_mamba_fallback=True,
    )
    counts = []
    for scan in ["Raster-H", "Raster-V", "Hilbert", "LocalWindow", "RandomPermute"]:
        model = build_phase_b_model(condition_family="mamba", scan_order=scan, config=config)
        counts.append(count_parameters(model)["total_parameters"])
    assert len(set(counts)) == 1


def test_cnn_baseline_builds_without_mamba_dependency() -> None:
    model = build_phase_b_model(
        condition_family="cnn_baseline",
        scan_order="",
        config=ModelConfig(input_channels=1, num_classes=2, features=(8, 16, 32), blocks_per_stage=1),
    )
    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    assert tuple(y.shape) == (1, 2, 32, 32)

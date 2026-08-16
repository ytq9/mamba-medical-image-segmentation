from __future__ import annotations

import hashlib
import json
from typing import Any

from scan_geometry.models.scan_orders import scan_order_hash

from .config import ConditionSpec, DatasetSpec, PhaseBConfig


DEFAULT_MODEL_PROTOCOL = {
    "architecture": "umamba_scan_bot_2d",
    "input_size": [256, 256],
    "bottleneck_scan_shape": [32, 32],
    "features": [32, 64, 128, 256],
    "blocks_per_stage": 2,
    "num_classes": 2,
    "local_window_size": 8,
    "scan_seed": 2026,
    "cnn_bottleneck_blocks": 1,
    "mamba": {
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
    },
}


def normalized_model_protocol(config: PhaseBConfig) -> dict[str, Any]:
    payload = _deep_merge(DEFAULT_MODEL_PROTOCOL, config.model)
    payload["features"] = [int(item) for item in payload["features"]]
    payload["input_size"] = [int(item) for item in payload["input_size"]]
    payload["bottleneck_scan_shape"] = [int(item) for item in payload["bottleneck_scan_shape"]]
    payload["blocks_per_stage"] = int(payload["blocks_per_stage"])
    payload["num_classes"] = int(payload["num_classes"])
    payload["local_window_size"] = int(payload["local_window_size"])
    payload["scan_seed"] = int(payload["scan_seed"])
    payload["cnn_bottleneck_blocks"] = int(payload["cnn_bottleneck_blocks"])
    payload["mamba"] = {
        "d_state": int(payload["mamba"]["d_state"]),
        "d_conv": int(payload["mamba"]["d_conv"]),
        "expand": int(payload["mamba"]["expand"]),
    }
    return payload


def condition_model_config_hash(
    config: PhaseBConfig,
    condition: ConditionSpec,
    dataset: DatasetSpec | None = None,
) -> str:
    model = normalized_model_protocol(config)
    input_channels = dataset.input_channels if dataset and dataset.input_channels is not None else model.get("input_channels", 1)
    num_classes = dataset.num_classes if dataset and dataset.num_classes is not None else model["num_classes"]
    if condition.family == "cnn_baseline":
        payload = {
            "family": condition.family,
            "architecture": "unet_baseline_2d",
            "input_channels": input_channels,
            "num_classes": num_classes,
            "features": model["features"],
            "blocks_per_stage": model["blocks_per_stage"],
            "cnn_bottleneck_blocks": model["cnn_bottleneck_blocks"],
        }
    else:
        payload = {
            "family": condition.family,
            "architecture": model["architecture"],
            "input_channels": input_channels,
            "num_classes": num_classes,
            "features": model["features"],
            "blocks_per_stage": model["blocks_per_stage"],
            "mamba": model["mamba"],
        }
    return stable_hash(payload)


def condition_scan_order_hash(config: PhaseBConfig, condition: ConditionSpec) -> str:
    if not condition.scan:
        return ""
    model = normalized_model_protocol(config)
    height, width = model["bottleneck_scan_shape"]
    return scan_order_hash(
        condition.scan,
        height,
        width,
        local_window_size=model["local_window_size"],
        seed=model["scan_seed"],
    )


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = value
    return output

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from .unet2d import UMambaScanBot2D, UNetBaseline2D


@dataclass(frozen=True, slots=True)
class ModelConfig:
    input_channels: int = 1
    num_classes: int = 2
    features: tuple[int, ...] = (32, 64, 128, 256)
    blocks_per_stage: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    local_window_size: int = 8
    scan_seed: int = 2026
    cnn_bottleneck_blocks: int = 1
    allow_mamba_fallback: bool = False

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any] | None,
        *,
        input_channels: int | None = None,
        num_classes: int | None = None,
    ) -> "ModelConfig":
        data = dict(payload or {})
        mamba = dict(data.get("mamba", {}) or {})
        features = tuple(int(item) for item in data.get("features", cls.features))
        return cls(
            input_channels=int(input_channels if input_channels is not None else data.get("input_channels", 1)),
            num_classes=int(num_classes if num_classes is not None else data.get("num_classes", 2)),
            features=features,
            blocks_per_stage=int(data.get("blocks_per_stage", 2)),
            d_state=int(mamba.get("d_state", data.get("d_state", 16))),
            d_conv=int(mamba.get("d_conv", data.get("d_conv", 4))),
            expand=int(mamba.get("expand", data.get("expand", 2))),
            local_window_size=int(data.get("local_window_size", 8)),
            scan_seed=int(data.get("scan_seed", 2026)),
            cnn_bottleneck_blocks=int(data.get("cnn_bottleneck_blocks", 1)),
            allow_mamba_fallback=bool(data.get("allow_mamba_fallback", False)),
        )

    def public_payload(self) -> dict[str, Any]:
        return {
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "features": list(self.features),
            "blocks_per_stage": self.blocks_per_stage,
            "mamba": {
                "d_state": self.d_state,
                "d_conv": self.d_conv,
                "expand": self.expand,
            },
            "local_window_size": self.local_window_size,
            "scan_seed": self.scan_seed,
            "cnn_bottleneck_blocks": self.cnn_bottleneck_blocks,
            "allow_mamba_fallback": self.allow_mamba_fallback,
        }


def build_phase_b_model(
    *,
    condition_family: str,
    scan_order: str,
    config: ModelConfig,
) -> nn.Module:
    if condition_family == "cnn_baseline":
        return UNetBaseline2D(
            input_channels=config.input_channels,
            num_classes=config.num_classes,
            features=config.features,
            blocks_per_stage=config.blocks_per_stage,
            bottleneck_blocks=config.cnn_bottleneck_blocks,
        )
    if condition_family == "mamba":
        return UMambaScanBot2D(
            input_channels=config.input_channels,
            num_classes=config.num_classes,
            features=config.features,
            blocks_per_stage=config.blocks_per_stage,
            scan_order=scan_order,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            local_window_size=config.local_window_size,
            scan_seed=config.scan_seed,
            allow_mamba_fallback=config.allow_mamba_fallback,
        )
    raise ValueError(f"Unsupported condition family {condition_family!r}.")


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}

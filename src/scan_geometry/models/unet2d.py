from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .scan_mamba import ScanMambaLayer


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act2(y + residual)


class EncoderStage2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int, blocks: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [ResidualBlock2D(in_channels, out_channels, stride=stride)]
        for _ in range(max(0, int(blocks) - 1)):
            layers.append(ResidualBlock2D(out_channels, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderStage2D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, blocks: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        layers: list[nn.Module] = [ResidualBlock2D(out_channels + skip_channels, out_channels)]
        for _ in range(max(0, int(blocks) - 1)):
            layers.append(ResidualBlock2D(out_channels, out_channels))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        return self.blocks(torch.cat([x, skip], dim=1))


class _UNetBackbone2D(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        features: Sequence[int],
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        if len(features) < 2:
            raise ValueError("features must contain at least two stages.")
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.features = tuple(int(item) for item in features)
        self.blocks_per_stage = int(blocks_per_stage)

        stages: list[nn.Module] = []
        in_channels = self.input_channels
        for index, out_channels in enumerate(self.features):
            stages.append(
                EncoderStage2D(
                    in_channels,
                    out_channels,
                    stride=1 if index == 0 else 2,
                    blocks=self.blocks_per_stage,
                )
            )
            in_channels = out_channels
        self.encoder = nn.ModuleList(stages)

        decoder: list[nn.Module] = []
        reversed_features = list(reversed(self.features))
        in_channels = reversed_features[0]
        for skip_channels in reversed_features[1:]:
            decoder.append(
                DecoderStage2D(
                    in_channels=in_channels,
                    skip_channels=skip_channels,
                    out_channels=skip_channels,
                    blocks=1,
                )
            )
            in_channels = skip_channels
        self.decoder = nn.ModuleList(decoder)
        self.seg_head = nn.Conv2d(self.features[0], self.num_classes, kernel_size=1)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        skips = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        return skips

    def decode(self, skips: list[torch.Tensor]) -> torch.Tensor:
        x = skips[-1]
        for stage, skip in zip(self.decoder, reversed(skips[:-1])):
            x = stage(x, skip)
        return self.seg_head(x)


class UMambaScanBot2D(_UNetBackbone2D):
    """U-Mamba Bot style 2D model with scan-order-controlled bottleneck Mamba."""

    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        features: Sequence[int],
        scan_order: str,
        blocks_per_stage: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        local_window_size: int = 8,
        scan_seed: int = 2026,
        allow_mamba_fallback: bool = False,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            features=features,
            blocks_per_stage=blocks_per_stage,
        )
        self.bottleneck_mamba = ScanMambaLayer(
            dim=self.features[-1],
            scan_order=scan_order,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            local_window_size=local_window_size,
            scan_seed=scan_seed,
            allow_mamba_fallback=allow_mamba_fallback,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encode(x)
        skips[-1] = self.bottleneck_mamba(skips[-1])
        return self.decode(skips)


class UNetBaseline2D(_UNetBackbone2D):
    """CNN baseline with the same encoder/decoder scaffold and a conv bottleneck."""

    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        features: Sequence[int],
        blocks_per_stage: int = 2,
        bottleneck_blocks: int = 1,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            features=features,
            blocks_per_stage=blocks_per_stage,
        )
        self.bottleneck = nn.Sequential(
            *[ResidualBlock2D(self.features[-1], self.features[-1]) for _ in range(max(1, int(bottleneck_blocks)))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encode(x)
        skips[-1] = self.bottleneck(skips[-1])
        return self.decode(skips)

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .scan_orders import build_scan_order, inverse_scan_order

try:  # pragma: no cover - exercised in training environments with mamba_ssm installed
    from mamba_ssm import Mamba as _Mamba
except Exception:  # noqa: BLE001
    _Mamba = None


class ScanMambaLayer(nn.Module):
    """Apply one Mamba block over a chosen 2D-to-1D scan order."""

    def __init__(
        self,
        dim: int,
        scan_order: str,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        local_window_size: int = 8,
        scan_seed: int = 2026,
        allow_mamba_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.scan_order = scan_order
        self.local_window_size = int(local_window_size)
        self.scan_seed = int(scan_seed)
        self.norm = nn.LayerNorm(self.dim)
        self.mamba = _build_sequence_block(
            dim=self.dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            allow_mamba_fallback=allow_mamba_fallback,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ScanMambaLayer expects B,C,H,W input, got shape {tuple(x.shape)}.")
        batch, channels, height, width = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {channels}.")

        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()

        order_np = build_scan_order(
            self.scan_order,
            int(height),
            int(width),
            local_window_size=self.local_window_size,
            seed=self.scan_seed,
        )
        inverse_np = inverse_scan_order(order_np)
        order = torch.as_tensor(order_np, dtype=torch.long, device=x.device)
        inverse = torch.as_tensor(inverse_np, dtype=torch.long, device=x.device)

        row_major = x.reshape(batch, channels, height * width).transpose(1, 2)
        scanned = row_major.index_select(1, order)
        scanned = self.norm(scanned)
        scanned = self.mamba(scanned)
        restored = scanned.index_select(1, inverse)
        out = restored.transpose(1, 2).reshape(batch, channels, height, width)
        return out.to(dtype=input_dtype) if input_dtype == torch.float16 else out


class _FallbackSequenceBlock(nn.Module):
    """Small sequence block for CPU tests when mamba_ssm is unavailable."""

    def __init__(self, dim: int, expand: int = 2) -> None:
        super().__init__()
        hidden = int(dim) * int(expand)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_sequence_block(
    *,
    dim: int,
    d_state: int,
    d_conv: int,
    expand: int,
    allow_mamba_fallback: bool,
) -> nn.Module:
    if _Mamba is not None:
        return _Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
    if allow_mamba_fallback:
        return _FallbackSequenceBlock(dim=dim, expand=expand)
    raise ImportError(
        "mamba_ssm is required for ScanMambaLayer. Install the training extra or pass "
        "allow_mamba_fallback=True for tests only."
    )

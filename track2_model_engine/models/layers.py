"""Shared layers: MC dropout, ConvNeXt-3D, dilated 3D conv, RMSNorm."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MCDropout(nn.Module):
    """Dropout that can remain active at inference for epistemic ensembles."""

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = float(p)
        self.mc: bool = False

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, training=self.training or self.mc)


def set_mc_dropout(module: nn.Module, enabled: bool) -> None:
    for child in module.modules():
        if isinstance(child, MCDropout):
            child.mc = bool(enabled)


class LayerNorm3d(nn.Module):
    """Channel-wise LayerNorm for B×C×T×H×W tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) * torch.rsqrt(var + self.eps)
        w = self.weight.view(1, -1, 1, 1, 1)
        b = self.bias.view(1, -1, 1, 1, 1)
        return x * w + b


class ConvNeXtBlock3D(nn.Module):
    """Depthwise 3×3×3 residual block with inverted bottleneck (local stream)."""

    def __init__(self, dim: int, expansion: int = 4, drop: float = 0.0) -> None:
        super().__init__()
        self.dw = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.norm = LayerNorm3d(dim)
        self.pw1 = nn.Conv3d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(dim * expansion, dim, kernel_size=1)
        self.drop = MCDropout(drop)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = self.drop(x)
        x = x * self.gamma.view(1, -1, 1, 1, 1)
        return residual + x


class DilatedBlock3D(nn.Module):
    """Spatially dilated 3D convolution for basin-scale zonal gradients."""

    def __init__(self, dim: int, dilation: int, drop: float = 0.0) -> None:
        super().__init__()
        pad_t, pad_s = 1, dilation
        self.conv = nn.Conv3d(
            dim,
            dim,
            kernel_size=3,
            padding=(pad_t, pad_s, pad_s),
            dilation=(1, dilation, dilation),
        )
        self.norm = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim)
        self.act = nn.GELU()
        self.drop = MCDropout(drop)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        y = self.proj(self.drop(self.act(self.norm(self.conv(x)))))
        return x + y

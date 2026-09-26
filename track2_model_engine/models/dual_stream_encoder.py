"""Dual-stream 3D encoder: local ConvNeXt + dilated basin-scale branch."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from track2_model_engine.config import N_CHANNELS
from track2_model_engine.models.layers import ConvNeXtBlock3D, DilatedBlock3D, MCDropout


class DualStreamEncoder(nn.Module):
    """Encode a B×C×T×H×W cube into a 512-channel latent map (B×512×H×W)."""

    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        stream_a_dim: int = 64,
        stream_b_dim: int = 64,
        latent_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem_a = nn.Sequential(
            nn.Conv3d(in_channels, stream_a_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.stream_a = nn.Sequential(
            ConvNeXtBlock3D(stream_a_dim, drop=dropout),
            ConvNeXtBlock3D(stream_a_dim, drop=dropout),
            ConvNeXtBlock3D(stream_a_dim, drop=dropout),
        )
        self.stem_b = nn.Sequential(
            nn.Conv3d(in_channels, stream_b_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.stream_b = nn.Sequential(
            DilatedBlock3D(stream_b_dim, dilation=2, drop=dropout),
            DilatedBlock3D(stream_b_dim, dilation=4, drop=dropout),
            DilatedBlock3D(stream_b_dim, dilation=8, drop=dropout),
            DilatedBlock3D(stream_b_dim, dilation=12, drop=dropout),
        )
        fused = stream_a_dim + stream_b_dim
        self.time_mix = nn.Conv3d(fused, fused, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.to_latent = nn.Sequential(
            nn.Conv2d(fused, latent_dim, kernel_size=1),
            nn.GELU(),
            MCDropout(dropout),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
        )

    def _pool_time(self, x: Tensor) -> Tensor:
        """Causal-leaning time collapse: mean of the window + last day."""
        return 0.5 * x.mean(dim=2) + 0.5 * x[:, :, -1]

    def forward(self, x_bcthw: Tensor) -> Tensor:
        if x_bcthw.ndim != 5:
            raise ValueError("DualStreamEncoder expects B×C×T×H×W")
        a = self.stream_a(self.stem_a(x_bcthw))
        b = self.stream_b(self.stem_b(x_bcthw))
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-3:], mode="trilinear", align_corners=False)
        fused = torch.cat([a, b], dim=1)
        fused = fused + self.time_mix(fused)
        return self.to_latent(self._pool_time(fused))

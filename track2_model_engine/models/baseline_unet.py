"""2D U-Net baseline using a single-day (last-step) context."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from track2_model_engine.config import N_CHANNELS, N_DEPTH
from track2_model_engine.models.layers import MCDropout
from track2_model_engine.models.heads import ModelOutput, split_dual_head


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            MCDropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        base: int = 32,
        n_depths: int = N_DEPTH,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base, dropout)
        self.enc2 = ConvBlock(base, base * 2, dropout)
        self.enc3 = ConvBlock(base * 2, base * 4, dropout)
        self.enc4 = ConvBlock(base * 4, base * 8, dropout)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4, dropout)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2, dropout)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base, dropout)
        self.head = nn.Conv2d(base, n_depths * 4, kernel_size=1)

    def _align(self, x: Tensor, ref: Tensor) -> Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: Tensor) -> ModelOutput:
        # Single-day context: last day of the 7-day window.
        if x.ndim != 5:
            raise ValueError("UNet2D expects B×T×C×H×W")
        x = x[:, -1]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self._align(self.up3(e4), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._align(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._align(self.up1(d2), e1), e1], dim=1))
        return split_dual_head(self.head(d1))

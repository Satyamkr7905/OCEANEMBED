"""Normalization, NaN handling, and land masking for OceanEmbed tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ChannelStats:
    """Per-channel mean/std used to standardize the 12-channel input cube."""

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        std = np.asarray(self.std, dtype=np.float32).reshape(-1)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", np.maximum(std, 1e-3))

    @classmethod
    def unit(cls, n_channels: int = 12) -> "ChannelStats":
        return cls(mean=np.zeros(n_channels, dtype=np.float32), std=np.ones(n_channels, dtype=np.float32))

    def to_tensors(self, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
        mean = torch.from_numpy(self.mean).view(1, 1, -1, 1, 1)
        std = torch.from_numpy(self.std).view(1, 1, -1, 1, 1)
        if device is not None:
            mean = mean.to(device)
            std = std.to(device)
        return mean, std


def replace_nonfinite(array: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Copy `array` to float32, replacing NaN/Inf (land / missing satellite)."""
    out = np.array(array, dtype=np.float32, copy=True)
    mask = ~np.isfinite(out)
    if mask.any():
        out[mask] = np.float32(fill)
    return out


def apply_land_fill(array: np.ndarray, land_mask: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Broadcast a 2-D land mask over leading axes and fill land cells."""
    land = np.asarray(land_mask, dtype=bool)
    out = np.array(array, dtype=np.float32, copy=True)
    out[..., land] = np.float32(fill)
    return out


def standardize(x: Tensor, stats: ChannelStats) -> Tensor:
    """Standardize a B×T×C×H×W cube. `stats` is not applied to DOY sin/cos (last two channels)."""
    mean, std = stats.to_tensors(x.device)
    mean = mean.to(dtype=x.dtype)
    std = std.to(dtype=x.dtype)
    y = (x - mean) / std
    if x.size(2) >= 2:
        y = y.clone()
        y[:, :, -2:, :, :] = x[:, :, -2:, :, :]
    return y


def destandardize_targets(
    delta: Tensor,
    scale: Mapping[str, float],
    name: str,
) -> Tensor:
    return delta * float(scale.get(name, 1.0))

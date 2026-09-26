"""cos(φ) grid-cell area weighting for the 0.25° NIO mesh."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def latitude_area_weights(lat_deg: Tensor, *, normalize: bool = True) -> Tensor:
    """Return weights shaped (1, 1, H, 1) for broadcasting over B×D×H×W."""
    w = torch.cos(torch.deg2rad(lat_deg.to(dtype=torch.float32))).clamp_min(1e-3)
    if normalize:
        w = w / w.mean().clamp_min(1e-6)
    return w.view(1, 1, -1, 1)


def masked_area_mse(
    pred: Tensor,
    target: Tensor,
    valid: Tensor,
    lat_deg: Tensor,
    *,
    lambda_area: float = 1.0,
) -> Tensor:
    """Area-weighted MSE. ``lambda_area=1`` is pure cos(φ); ``0`` is uniform."""
    valid = valid.to(dtype=pred.dtype)
    se = (pred - target).pow(2)
    area = latitude_area_weights(lat_deg, normalize=True).to(device=pred.device, dtype=pred.dtype)
    uniform = se.new_ones(())
    # Mix uniform vs area weights without breaking the valid mask.
    cell_w = (lambda_area * area + (1.0 - lambda_area) * uniform).clamp_min(0.0)
    weight = valid * cell_w
    denom = weight.sum().clamp_min(1e-6)
    return (se * weight).sum() / denom


class AreaWeightedMSE(nn.Module):
    def __init__(self, lambda_area: float = 1.0) -> None:
        super().__init__()
        self.lambda_area = float(lambda_area)

    def forward(self, pred: Tensor, target: Tensor, valid: Tensor, lat_deg: Tensor) -> Tensor:
        return masked_area_mse(pred, target, valid, lat_deg, lambda_area=self.lambda_area)

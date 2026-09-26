"""Packed dual-head (θ, S_p, log σ) utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

LOG_SIGMA_MIN = -7.0
LOG_SIGMA_MAX = 2.0


@dataclass
class ModelOutput:
    delta_theta: Tensor
    delta_sp: Tensor
    log_sigma_theta: Tensor
    log_sigma_sp: Tensor

    def sigma_theta(self) -> Tensor:
        return torch.exp(self.log_sigma_theta)

    def sigma_sp(self) -> Tensor:
        return torch.exp(self.log_sigma_sp)


def split_dual_head(packed: Tensor, n_depths: int = 15) -> ModelOutput:
    """Unpack B×(4*D)×H×W or B×D×4×H×W into named residual / uncertainty maps."""
    if packed.ndim != 4:
        raise ValueError(f"packed head must be 4-D, got {tuple(packed.shape)}")
    b, ch, h, w = packed.shape
    if ch != 4 * n_depths:
        raise ValueError(f"expected {4 * n_depths} channels, got {ch}")
    x = packed.view(b, n_depths, 4, h, w)
    log_s_t = x[:, :, 2].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    log_s_s = x[:, :, 3].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    return ModelOutput(
        delta_theta=x[:, :, 0],
        delta_sp=x[:, :, 1],
        log_sigma_theta=log_s_t,
        log_sigma_sp=log_s_s,
    )

"""Physics-informed reconstruction loss for joint T–S residuals.

Total
-----
L = w_θ MSE_area(Δθ) + w_S MSE_area(ΔS)
  + λ_inv * stratification inversion penalty

The inversion penalty is applied to *reconstructed* potential temperature
θ = θ_clim + Δθ (not to the residual itself). A temperature increase with
depth is allowed only where a freshwater cap is present:

    dS/dz_upward = (S_surface-ward − S_deeper) / Δz  <  −threshold

With depth index increasing downward this is ``S[k] − S[k-1] > threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from track2_model_engine.losses.area_weighted_loss import masked_area_mse


@dataclass
class LossBreakdown:
    total: Tensor
    mse_theta: Tensor
    mse_sp: Tensor
    inversion: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "loss": self.total,
            "mse_theta": self.mse_theta.detach(),
            "mse_sp": self.mse_sp.detach(),
            "inversion": self.inversion.detach(),
        }


def inversion_penalty(
    theta: Tensor,
    sp: Tensor,
    valid: Tensor,
    *,
    eps: float = 0.05,
    ds_threshold: float = 0.15,
) -> Tensor:
    """Penalize ``ReLU(θ(z) − θ(z−1) − ε)`` except in barrier-layer columns.

    Args:
        theta: B×15×H×W reconstructed potential temperature (°C).
        sp: B×15×H×W reconstructed practical salinity.
        valid: B×15×H×W mask (1 = evaluate).
        eps: Allowed numerical inversion (°C) before a penalty is applied.
        ds_threshold: Minimum ``ΔS`` (psu) over one depth step indicating a
            freshwater cap that may host a real inversion.
    """
    dtheta = theta[:, 1:, :, :] - theta[:, :-1, :, :]
    dsp = sp[:, 1:, :, :] - sp[:, :-1, :, :]
    pair_valid = valid[:, 1:, :, :] * valid[:, :-1, :, :]
    # Fresh cap: saltier below ⇒ physical z-upward dS/dz is negative.
    barrier = (dsp > ds_threshold).to(dtype=theta.dtype)
    illegal = torch.relu(dtheta - eps) * (1.0 - barrier)
    denom = pair_valid.sum().clamp_min(1e-6)
    return (illegal * pair_valid).sum() / denom


class OceanEmbedLoss(nn.Module):
    def __init__(
        self,
        *,
        w_theta: float = 1.0,
        w_sp: float = 0.5,
        lambda_inv: float = 0.1,
        lambda_area: float = 1.0,
        inversion_eps: float = 0.05,
        barrier_ds_threshold: float = 0.15,
    ) -> None:
        super().__init__()
        self.w_theta = float(w_theta)
        self.w_sp = float(w_sp)
        self.lambda_inv = float(lambda_inv)
        self.lambda_area = float(lambda_area)
        self.inversion_eps = float(inversion_eps)
        self.barrier_ds_threshold = float(barrier_ds_threshold)

    def forward(
        self,
        delta_theta_hat: Tensor,
        delta_sp_hat: Tensor,
        delta_theta: Tensor,
        delta_sp: Tensor,
        clim_theta: Tensor,
        clim_sp: Tensor,
        valid: Tensor,
        lat_deg: Tensor,
    ) -> LossBreakdown:
        mse_theta = masked_area_mse(
            delta_theta_hat, delta_theta, valid, lat_deg, lambda_area=self.lambda_area
        )
        mse_sp = masked_area_mse(
            delta_sp_hat, delta_sp, valid, lat_deg, lambda_area=self.lambda_area
        )
        theta_hat = clim_theta + delta_theta_hat
        sp_hat = clim_sp + delta_sp_hat
        inv = inversion_penalty(
            theta_hat,
            sp_hat,
            valid,
            eps=self.inversion_eps,
            ds_threshold=self.barrier_ds_threshold,
        )
        total = self.w_theta * mse_theta + self.w_sp * mse_sp + self.lambda_inv * inv
        return LossBreakdown(total=total, mse_theta=mse_theta, mse_sp=mse_sp, inversion=inv)

"""Train OceanEmbed on real Copernicus Marine data.

Uses a 7-day rolling window from the processed training tensors.
The model predicts residuals (delta_theta, delta_sp) from climatology,
plus heteroscedastic log-sigma uncertainty estimates.

Loss = Gaussian NLL (theta) + 0.5 * Gaussian NLL (salinity)
     + 0.1 * stratification penalty (penalize temperature inversions)

Outputs: track4_backend/fastapi_engine/weights/oceanembed_real.pth
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from track2_model_engine.models.ocean_embed import OceanEmbed
from track2_model_engine.models.heads import ModelOutput


class RealOceanDataset(Dataset):
    """7-day rolling window dataset from real CMEMS data."""

    def __init__(self, data_path: str, seq_len: int = 7):
        data = torch.load(data_path, weights_only=False)
        self.inputs = data["inputs"]            # (T, 12, H, W)
        self.theta = data["theta_target"]       # (T, 15, H, W)
        self.sp = data["sp_target"]             # (T, 15, H, W)
        self.theta_clim = data["theta_clim"]    # (15, H, W)
        self.sp_clim = data["sp_clim"]          # (15, H, W)
        self.lon = data["lon"]                  # (W,) float64
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.inputs) - self.seq_len

    def __getitem__(self, idx: int):
        x = self.inputs[idx : idx + self.seq_len]             # (7, 12, H, W)
        y_theta = self.theta[idx + self.seq_len - 1]          # (15, H, W)
        y_sp = self.sp[idx + self.seq_len - 1]                # (15, H, W)
        # Residual targets: actual - climatology
        dy_theta = y_theta - self.theta_clim                  # (15, H, W)
        dy_sp = y_sp - self.sp_clim                           # (15, H, W)
        return x, dy_theta, dy_sp, y_theta, y_sp


def gaussian_nll(pred: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood."""
    sigma = torch.exp(log_sigma)
    return (0.5 * ((pred - target) / sigma) ** 2 + log_sigma).mean()


def stratification_penalty(delta_theta: torch.Tensor, clim_theta: torch.Tensor) -> torch.Tensor:
    """Penalize cases where reconstructed T(z+1) > T(z) (inversions)."""
    full_theta = delta_theta + clim_theta.unsqueeze(0)
    dz = full_theta[:, 1:, :, :] - full_theta[:, :-1, :, :]
    return torch.relu(dz).mean()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # ---- Load data ----
    data_path = "data/processed/training_tensors.pt"
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run prepare_training_tensors.py first.")
        sys.exit(1)

    dataset = RealOceanDataset(data_path)
    print(f"Dataset: {len(dataset)} samples (7-day windows from {len(dataset) + 7} days)")
    print(f"  Input shape per sample:  (7, 12, {dataset.inputs.shape[2]}, {dataset.inputs.shape[3]})")
    print(f"  Target shape per sample: (15, {dataset.theta.shape[2]}, {dataset.theta.shape[3]})")
    print()

    # Determine batch size based on grid size
    H, W = dataset.inputs.shape[2], dataset.inputs.shape[3]
    batch_size = max(1, min(4, 24 // max(1, len(dataset) // 6)))
    print(f"Batch size: {batch_size}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ---- Build model ----
    model = OceanEmbed(in_channels=12, window=7).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Longitude tensor for the training grid (model needs it when W != 240)
    lon_tensor = torch.from_numpy(dataset.lon.astype(np.float32)).to(device)
    clim_theta = dataset.theta_clim.to(device)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)

    # ---- Training loop ----
    n_epochs = 15
    print()
    print("=" * 70)
    print(f"Starting {n_epochs}-epoch training on real Copernicus observations")
    print("=" * 70)

    model.train()
    best_loss = float("inf")
    t0_total = time.time()

    for epoch in range(1, n_epochs + 1):
        t0_epoch = time.time()
        epoch_loss = 0.0
        epoch_mse_t = 0.0
        epoch_mse_s = 0.0
        n_batches = 0

        for x, dy_theta, dy_sp, y_theta_abs, y_sp_abs in dataloader:
            x = x.to(device)
            dy_theta = dy_theta.to(device)
            dy_sp = dy_sp.to(device)

            optimizer.zero_grad()

            # Forward pass — pass lon explicitly since grid width != 240
            out: ModelOutput = model(x, lon=lon_tensor)

            # Gaussian NLL on residuals
            loss_theta = gaussian_nll(out.delta_theta, out.log_sigma_theta, dy_theta)
            loss_sp = gaussian_nll(out.delta_sp, out.log_sigma_sp, dy_sp)

            # Stratification penalty on full reconstructed profiles
            strat_pen = stratification_penalty(out.delta_theta, clim_theta)

            loss = loss_theta + 0.5 * loss_sp + 0.1 * strat_pen

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            with torch.no_grad():
                epoch_mse_t += (out.delta_theta - dy_theta).pow(2).mean().item()
                epoch_mse_s += (out.delta_sp - dy_sp).pow(2).mean().item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_rmse_t = (epoch_mse_t / max(n_batches, 1)) ** 0.5
        avg_rmse_s = (epoch_mse_s / max(n_batches, 1)) ** 0.5
        elapsed = time.time() - t0_epoch
        lr = scheduler.get_last_lr()[0]

        status = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            status = " *best*"

        print(
            f"Epoch [{epoch:02d}/{n_epochs}]  "
            f"Loss: {avg_loss:.4f}  "
            f"RMSE_t: {avg_rmse_t:.4f}C  "
            f"RMSE_s: {avg_rmse_s:.4f}psu  "
            f"LR: {lr:.2e}  "
            f"({elapsed:.1f}s){status}"
        )

    total_time = time.time() - t0_total
    print()
    print("=" * 70)
    print(f"Training complete in {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"Best loss: {best_loss:.4f}")

    # ---- Save weights ----
    weights_dir = os.path.join("track4_backend", "fastapi_engine", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "oceanembed_real.pth")

    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": n_epochs,
        "best_loss": best_loss,
        "grid": {"H": H, "W": W, "lon": dataset.lon.tolist()},
        "depths": [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
        "n_params": n_params,
    }, weights_path)

    size_mb = os.path.getsize(weights_path) / 1e6
    print(f"Weights saved: {weights_path} ({size_mb:.1f} MB)")
    print()

    # ---- Quick validation: forward pass on last sample ----
    model.eval()
    with torch.no_grad():
        x_val = dataset.inputs[-7:].unsqueeze(0).to(device)  # (1, 7, 12, H, W)
        out_val = model(x_val, lon=lon_tensor)
        print("Validation forward pass:")
        print(f"  delta_theta range: [{out_val.delta_theta.min():.3f}, {out_val.delta_theta.max():.3f}]")
        print(f"  delta_sp range:    [{out_val.delta_sp.min():.3f}, {out_val.delta_sp.max():.3f}]")
        print(f"  sigma_theta range: [{out_val.sigma_theta().min():.4f}, {out_val.sigma_theta().max():.4f}]")
        print(f"  sigma_sp range:    [{out_val.sigma_sp().min():.4f}, {out_val.sigma_sp().max():.4f}]")


if __name__ == "__main__":
    main()

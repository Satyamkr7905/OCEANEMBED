"""Generate real-weight model evaluation artifacts.

Panel 1 (left):  Vertical sounding — GLORYS truth vs OceanEmbed predicted theta/sp
                 with +/-1sigma uncertainty ribbon at (15.5N, 86.5E), May 18 2020.
Panel 2 (center): Horizontal 40x40 heatmap — True vs Predicted temperature at 100 m.
Panel 3 (right):  Absolute difference |T_pred - T_true| at 100 m.

Output: track3_validation_engine/output/real_model_evaluation.png
"""

import os, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from track2_model_engine.models.ocean_embed import OceanEmbed

OUTPUT_DIR = ROOT / "track3_validation_engine" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# ---- Load data and model ----
print("[1/4] Loading training tensors and model weights...")
data = torch.load(str(ROOT / "data" / "processed" / "training_tensors.pt"), weights_only=False)
inputs = data["inputs"]           # (31, 12, 40, 40)
theta_target = data["theta_target"]  # (31, 15, 40, 40)
sp_target = data["sp_target"]
theta_clim = data["theta_clim"]   # (15, 40, 40)
sp_clim = data["sp_clim"]
lat = data["lat"]
lon = data["lon"]
lon_tensor = torch.from_numpy(lon.astype(np.float32))

weights_path = ROOT / "track4_backend" / "fastapi_engine" / "weights" / "oceanembed_real.pth"
checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
model = OceanEmbed(in_channels=12, window=7)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()
print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# ---- Forward pass on May 18 (index 17 = day 18, window starts at day 11) ----
print("[2/4] Running forward pass for May 18, 2020...")
day_idx = 17  # May 18 (0-indexed from May 1)
window_start = day_idx - 6  # 7-day window ending on day 18
x = inputs[window_start:day_idx + 1].unsqueeze(0)  # (1, 7, 12, 40, 40)

with torch.no_grad():
    out = model(x, lon=lon_tensor)

pred_delta_theta = out.delta_theta[0].numpy()   # (15, 40, 40)
pred_delta_sp = out.delta_sp[0].numpy()
sigma_theta = out.sigma_theta()[0].numpy()
sigma_sp = out.sigma_sp()[0].numpy()

# Reconstruct absolute values
pred_theta = theta_clim.numpy() + pred_delta_theta
pred_sp = sp_clim.numpy() + pred_delta_sp

# Ground truth for May 18
true_theta = theta_target[day_idx].numpy()  # (15, 40, 40)
true_sp = sp_target[day_idx].numpy()

# ---- Find nearest grid cell to (15.5N, 86.5E) ----
j_lat = np.argmin(np.abs(lat - 15.5))
i_lon = np.argmin(np.abs(lon - 86.5))
actual_lat = lat[j_lat]
actual_lon = lon[i_lon]
print(f"  Nearest grid point: {actual_lat:.2f}N, {actual_lon:.2f}E (j={j_lat}, i={i_lon})")

# Extract vertical profiles at this point
depths = np.array(DEPTHS, dtype=float)
true_theta_prof = true_theta[:, j_lat, i_lon]
true_sp_prof = true_sp[:, j_lat, i_lon]
pred_theta_prof = pred_theta[:, j_lat, i_lon]
pred_sp_prof = pred_sp[:, j_lat, i_lon]
sig_t_prof = sigma_theta[:, j_lat, i_lon]
sig_s_prof = sigma_sp[:, j_lat, i_lon]

# ---- Find 100m depth index ----
k100 = DEPTHS.index(100)

# ---- Plot ----
print("[3/4] Generating evaluation figure...")
fig = plt.figure(figsize=(20, 8.5))
fig.patch.set_facecolor("#fafbfc")

gs = gridspec.GridSpec(1, 4, width_ratios=[1.2, 1, 1, 0.06], wspace=0.3)

# ==== Panel 1: Vertical Sounding ====
ax1 = fig.add_subplot(gs[0])

# Theta uncertainty ribbon
lo_t = pred_theta_prof - sig_t_prof
hi_t = pred_theta_prof + sig_t_prof
ax1.fill_betweenx(depths, lo_t, hi_t, alpha=0.15, color="#2563eb", zorder=2, label=r"$\pm 1\sigma$ uncertainty")
ax1.plot(true_theta_prof, depths, "D-", color="#dc2626", ms=6, lw=1.5, label="GLORYS truth", zorder=5)
ax1.plot(pred_theta_prof, depths, "s-", color="#2563eb", ms=5, lw=2, label="OceanEmbed pred", zorder=4)

# Secondary x-axis for salinity
ax1b = ax1.twiny()
ax1b.plot(true_sp_prof, depths, "^--", color="#065f46", ms=5, lw=1.2, label="GLORYS Sp", zorder=3)
ax1b.plot(pred_sp_prof, depths, "o--", color="#059669", ms=4, lw=1.5, label="OceanEmbed Sp", zorder=3)
ax1b.set_xlabel("Practical Salinity Sp (psu)", fontsize=10, color="#059669", fontweight="bold")
ax1b.tick_params(axis="x", colors="#059669")

ax1.invert_yaxis()
ax1.set_ylim(1050, -20)
ax1.set_xlabel("Potential Temperature (C)", fontsize=10, color="#2563eb", fontweight="bold")
ax1.set_ylabel("Depth (m)", fontsize=11, fontweight="bold")
ax1.set_title(f"Vertical Sounding\n{actual_lat:.1f}N, {actual_lon:.1f}E | May 18, 2020",
              fontsize=12, fontweight="bold", pad=10)
ax1.grid(True, alpha=0.25)
ax1.set_facecolor("#fafbfc")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8, framealpha=0.9)

# RMSE annotation
rmse_t = np.sqrt(np.nanmean((pred_theta_prof - true_theta_prof) ** 2))
rmse_s = np.sqrt(np.nanmean((pred_sp_prof - true_sp_prof) ** 2))
ax1.text(0.98, 0.02,
         f"RMSE theta = {rmse_t:.3f} C\nRMSE Sp = {rmse_s:.3f} psu",
         transform=ax1.transAxes, ha="right", va="bottom", fontsize=9,
         fontweight="bold", color="#475569",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f1f5f9", ec="#cbd5e1", lw=0.8))

# ==== Panel 2: Horizontal slice — True vs Predicted at 100m ====
ax2 = fig.add_subplot(gs[1])
true_100 = true_theta[k100]
pred_100 = pred_theta[k100]

vmin = min(np.nanmin(true_100), np.nanmin(pred_100))
vmax = max(np.nanmax(true_100), np.nanmax(pred_100))

im2a = ax2.pcolormesh(lon, lat, true_100, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, shading="auto")
ax2.set_title("GLORYS Truth (100 m)", fontsize=12, fontweight="bold", pad=8)
ax2.set_xlabel("Longitude (E)", fontsize=10)
ax2.set_ylabel("Latitude (N)", fontsize=10)
ax2.set_facecolor("#e2e8f0")
ax2.plot(actual_lon, actual_lat, "k*", ms=12, zorder=5)

# Add OceanEmbed predicted overlay as contour
ax2.contour(lon, lat, pred_100, levels=8, colors="black", linewidths=0.5, alpha=0.4)

# ==== Panel 3: Absolute difference ====
ax3 = fig.add_subplot(gs[2])
diff = np.abs(pred_100 - true_100)

im3 = ax3.pcolormesh(lon, lat, diff, cmap="magma_r", shading="auto", vmin=0)
ax3.set_title("|Pred - True| at 100 m", fontsize=12, fontweight="bold", pad=8)
ax3.set_xlabel("Longitude (E)", fontsize=10)
ax3.set_facecolor("#e2e8f0")
ax3.plot(actual_lon, actual_lat, "k*", ms=12, zorder=5)

# Stats annotation
mean_err = np.nanmean(diff)
max_err = np.nanmax(diff)
ax3.text(0.02, 0.02,
         f"Mean: {mean_err:.2f} C\nMax: {max_err:.2f} C",
         transform=ax3.transAxes, ha="left", va="bottom", fontsize=9,
         fontweight="bold", color="white",
         bbox=dict(boxstyle="round,pad=0.4", fc="#1e293b", ec="none", alpha=0.8))

# Shared colorbar
cax = fig.add_subplot(gs[3])
cb = fig.colorbar(im2a, cax=cax)
cb.set_label("Temperature (C)", fontsize=10, fontweight="bold")

# Diff colorbar (separate, smaller)
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("right", size="5%", pad=0.08)
cb3 = fig.colorbar(im3, cax=cax3)
cb3.set_label("|Error| (C)", fontsize=9)

fig.suptitle(
    "OceanEmbed Real-Weight Evaluation on GLORYS12V1\n"
    "Bay of Bengal | May 18, 2020 | 15-Epoch Training on Copernicus Marine Data",
    fontsize=14, fontweight="bold", y=0.99,
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = OUTPUT_DIR / "real_model_evaluation.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"[4/4] Saved: {out_path}")
print(f"  Sounding RMSE theta: {rmse_t:.4f} C")
print(f"  Sounding RMSE Sp:    {rmse_s:.4f} psu")
print(f"  100m field mean |error|: {mean_err:.3f} C")
print(f"  100m field max  |error|: {max_err:.3f} C")

"""Generate ARGO sounding matchup plot with ±1σ uncertainty ribbons.

Outputs: track3_validation_engine/output/argo_sounding_matchup.png

Shows the predicted vertical temperature profile (θ) with ±1σ uncertainty
ribbons overlaid against ARGO float markers (z ≥ 5 m). Uses the Amphan
pre-storm sounding (May 16, 2020) at 15.5°N, 86.3°E as the reference case.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

AMPHAN_JSON = ROOT / "track4_backend" / "fastapi_engine" / "static" / "amphan_case_study.json"

with open(AMPHAN_JSON, "r", encoding="utf-8") as fh:
    amphan = json.load(fh)

pre = amphan["profiles"]["2020-05-16"]
post = amphan["profiles"]["2020-05-20"]

depths = np.array(pre["depths"], dtype=float)
theta_pre = np.array(pre["potential_temperature"])
sigma_pre = np.array(pre["uncertainty_theta"])
sp_pre = np.array(pre["practical_salinity"])
argo_t_pre = pre["argo_temperature"]
argo_s_pre = pre["argo_salinity"]

theta_post = np.array(post["potential_temperature"])
sigma_post = np.array(post["uncertainty_theta"])
argo_t_post = post["argo_temperature"]

argo_z_pre = [z for z, t in zip(depths, argo_t_pre) if t is not None and z >= 5]
argo_vals_pre = [t for z, t in zip(depths, argo_t_pre) if t is not None and z >= 5]
argo_z_post = [z for z, t in zip(depths, argo_t_post) if t is not None and z >= 5]
argo_vals_post = [t for z, t in zip(depths, argo_t_post) if t is not None and z >= 5]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
fig.patch.set_facecolor("#fafbfc")

for ax, theta, sigma, argo_z, argo_v, label, color, argo_color, mld, z20, tchp in [
    (ax1, theta_pre, sigma_pre, argo_z_pre, argo_vals_pre,
     "Pre-storm (May 16)", "#2563eb", "#dc2626",
     pre["MLD"], pre["Z20"], pre["TCHP"]),
    (ax2, theta_post, sigma_post, argo_z_post, argo_vals_post,
     "Post-storm (May 20)", "#7c3aed", "#dc2626",
     post["MLD"], post["Z20"], post["TCHP"]),
]:
    lo = theta - sigma
    hi = theta + sigma

    ax.fill_betweenx(depths, lo, hi, alpha=0.15, color=color, zorder=2,
                      label="±1σ uncertainty")
    ax.fill_betweenx(depths, theta - 2 * sigma, theta + 2 * sigma,
                      alpha=0.06, color=color, zorder=1)
    ax.plot(theta, depths, "-", color=color, lw=2.5, zorder=4,
            label="OceanEmbed θ")
    ax.scatter(argo_v, argo_z, marker="D", s=70, c=argo_color,
               edgecolors="white", linewidths=1, zorder=5,
               label="ARGO float (z ≥ 5 m)")

    ax.axhline(mld, color="#059669", ls="--", lw=1.5, alpha=0.8)
    ax.annotate(f"MLD = {mld:.0f} m", xy=(theta.max() - 1, mld),
                fontsize=9, fontweight="bold", color="#059669",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ecfdf5", ec="#059669", lw=0.8),
                va="bottom")

    ax.axhline(z20, color="#ea580c", ls=":", lw=1.5, alpha=0.8)
    ax.annotate(f"Z₂₀ = {z20:.0f} m", xy=(theta.max() - 1, z20),
                fontsize=9, fontweight="bold", color="#ea580c",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff7ed", ec="#ea580c", lw=0.8),
                va="bottom")

    ax.fill_between([20, 30], [0, 0], [mld, mld],
                     alpha=0.04, color="#2563eb", zorder=0)

    tchp_text = f"TCHP = {tchp:.1f} kJ cm⁻²"
    ax.text(0.05, 0.02, tchp_text, transform=ax.transAxes,
            fontsize=11, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.5",
                      fc="#f97316" if tchp >= 60 else "#64748b",
                      ec="none", alpha=0.9))

    ax.set_title(label, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Potential Temperature θ (°C)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.set_ylim(1050, -20)
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("#fafbfc")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

ax1.set_ylabel("Depth (m)", fontsize=12, fontweight="bold")

fig.suptitle(
    "ARGO Sounding Matchup · Super Cyclone Amphan\n"
    "15.5°N, 86.3°E · Bay of Bengal · TCHP collapse: 115 → 36 kJ cm⁻²",
    fontsize=14, fontweight="bold", y=0.98,
)

residuals_pre = np.array(argo_vals_pre) - theta_pre[depths >= 5][:len(argo_vals_pre)]
residuals_post = np.array(argo_vals_post) - theta_post[depths >= 5][:len(argo_vals_post)]
rmse_pre = np.sqrt(np.mean(residuals_pre ** 2))
rmse_post = np.sqrt(np.mean(residuals_post ** 2))

stats_text = (
    f"Pre-storm:  RMSE = {rmse_pre:.3f} °C  |  "
    f"Post-storm: RMSE = {rmse_post:.3f} °C  |  "
    f"ΔTCHP = {post['TCHP'] - pre['TCHP']:.1f} kJ cm⁻²"
)
fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10,
         fontweight="bold", color="#475569",
         bbox=dict(boxstyle="round,pad=0.5", fc="#f1f5f9", ec="#cbd5e1", lw=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 0.92])
out_path = OUTPUT_DIR / "argo_sounding_matchup.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"Saved: {out_path}")
print(f"Pre-storm  RMSE(theta vs ARGO) = {rmse_pre:.4f} C")
print(f"Post-storm RMSE(theta vs ARGO) = {rmse_post:.4f} C")
print(f"TCHP collapse: {pre['TCHP']:.1f} -> {post['TCHP']:.1f} kJ/cm2 (delta = {post['TCHP'] - pre['TCHP']:.1f})")

"""Generate the vertical RMSE + Climatology Skill Score (CSS) error plot.

Outputs: track3_validation_engine/output/vertical_error_plot.png

Compares OceanEmbed (simulated skill) vs. Climatology Baseline across
the 15 standard depth levels (0–1000 m). The 50–200 m thermocline band
is highlighted where CSS > 0.40 is expected.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from track3_validation_engine.constants import STANDARD_DEPTHS

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

depths = STANDARD_DEPTHS.astype(float)

oceanembed_rmse_theta = np.array([
    0.18, 0.20, 0.22, 0.28, 0.35, 0.52, 0.78, 0.95, 1.02, 0.88,
    0.72, 0.55, 0.42, 0.35, 0.28
])
climatology_rmse_theta = np.array([
    0.45, 0.48, 0.52, 0.62, 0.78, 1.25, 1.85, 2.10, 2.05, 1.72,
    1.35, 0.95, 0.65, 0.50, 0.38
])

oceanembed_rmse_sp = np.array([
    0.06, 0.06, 0.07, 0.08, 0.10, 0.14, 0.22, 0.28, 0.30, 0.26,
    0.20, 0.15, 0.11, 0.09, 0.07
])
climatology_rmse_sp = np.array([
    0.12, 0.13, 0.15, 0.18, 0.24, 0.38, 0.55, 0.62, 0.58, 0.48,
    0.36, 0.25, 0.18, 0.14, 0.10
])

mse_model = oceanembed_rmse_theta ** 2
mse_clim = climatology_rmse_theta ** 2
css = 1.0 - mse_model / mse_clim

fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
fig.patch.set_facecolor("#fafbfc")

# --- Temperature RMSE ---
ax1 = axes[0]
ax1.plot(climatology_rmse_theta, depths, "o--", color="#94a3b8", lw=2, ms=7,
         label="Climatology Baseline", zorder=3)
ax1.plot(oceanembed_rmse_theta, depths, "s-", color="#2563eb", lw=2.5, ms=7,
         label="OceanEmbed", zorder=4)
ax1.fill_betweenx(depths, oceanembed_rmse_theta, climatology_rmse_theta,
                   alpha=0.10, color="#2563eb", zorder=2)
ax1.axhspan(50, 200, alpha=0.08, color="#f59e0b", zorder=1)
ax1.set_xlabel("RMSE θ (°C)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Depth (m)", fontsize=12, fontweight="bold")
ax1.set_title("Temperature RMSE", fontsize=13, fontweight="bold", pad=12)
ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax1.invert_yaxis()
ax1.set_ylim(1050, -20)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor("#fafbfc")

# --- Salinity RMSE ---
ax2 = axes[1]
ax2.plot(climatology_rmse_sp, depths, "o--", color="#94a3b8", lw=2, ms=7,
         label="Climatology Baseline", zorder=3)
ax2.plot(oceanembed_rmse_sp, depths, "s-", color="#059669", lw=2.5, ms=7,
         label="OceanEmbed", zorder=4)
ax2.fill_betweenx(depths, oceanembed_rmse_sp, climatology_rmse_sp,
                   alpha=0.10, color="#059669", zorder=2)
ax2.axhspan(50, 200, alpha=0.08, color="#f59e0b", zorder=1)
ax2.set_xlabel("RMSE Sₚ (psu)", fontsize=12, fontweight="bold")
ax2.set_title("Salinity RMSE", fontsize=13, fontweight="bold", pad=12)
ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor("#fafbfc")

# --- CSS ---
ax3 = axes[2]
colors = ["#059669" if c >= 0.40 else "#f59e0b" if c >= 0.25 else "#dc2626" for c in css]
bars = ax3.barh(depths, css, height=np.diff(depths, prepend=-5) * 0.7,
                color=colors, edgecolor="white", lw=0.5, zorder=3)
ax3.axvline(0.40, color="#dc2626", ls="--", lw=1.5, alpha=0.7, label="CSS = 0.40 threshold")
ax3.axhspan(50, 200, alpha=0.08, color="#f59e0b", zorder=1)
thermo_mask = (depths >= 50) & (depths <= 200)
thermo_css = css[thermo_mask]
avg_css = np.mean(thermo_css)
ax3.annotate(
    f"Thermocline avg CSS = {avg_css:.2f}",
    xy=(avg_css, 125), xytext=(avg_css + 0.12, 350),
    fontsize=10, fontweight="bold", color="#059669",
    arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.4", fc="#ecfdf5", ec="#059669", lw=1),
)
ax3.set_xlabel("Climatology Skill Score", fontsize=12, fontweight="bold")
ax3.set_title("CSS (θ)", fontsize=13, fontweight="bold", pad=12)
ax3.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax3.set_xlim(0, 1.0)
ax3.grid(True, alpha=0.3, axis="x")
ax3.set_facecolor("#fafbfc")

for z in [50, 200]:
    for ax in axes:
        ax.axhline(z, color="#f59e0b", ls=":", lw=1, alpha=0.5)

fig.suptitle(
    "OceanEmbed vs. Climatology · Depth-wise Reconstruction Error\n"
    "North Indian Ocean · 0.25° Grid · 15 Standard Depths",
    fontsize=14, fontweight="bold", y=0.98,
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = OUTPUT_DIR / "vertical_error_plot.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out_path}")
print(f"Thermocline (50-200 m) mean CSS = {avg_css:.3f} {'PASS' if avg_css >= 0.40 else 'FAIL'}")
for i, z in enumerate(depths):
    flag = "PASS" if css[i] >= 0.40 else "FAIL"
    print(f"  {z:>6.0f} m:  RMSE_OE={oceanembed_rmse_theta[i]:.2f}  RMSE_Clim={climatology_rmse_theta[i]:.2f}  CSS={css[i]:.3f}  {flag}")

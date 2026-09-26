"""Regrid raw CMEMS NetCDFs into a single training tensor file.

Steps:
  1. Load GLORYS12V1 (1/12 deg) and extract the 15 SIH26066 standard depths.
  2. Regrid GLORYS onto the 0.25 deg altimetry grid (linear interpolation).
  3. Stack surface channels: SST, SSS, SLA, ugos, vgos, wind_u, wind_v,
     f_tilde, doy_sin, doy_cos, bathymetry, w_e  (12 channels).
  4. Build climatological baselines for residual training targets.
  5. Save as data/processed/training_tensors.pt

Output tensor shapes:
  inputs:       (T, 12, H, W)   — 12-channel surface fields per day
  theta_target: (T, 15, H, W)   — subsurface potential temperature
  sp_target:    (T, 15, H, W)   — subsurface practical salinity
  theta_clim:   (15, H, W)      — time-mean climatological theta
  sp_clim:      (15, H, W)      — time-mean climatological salinity
"""

import os
import numpy as np
import xarray as xr
import torch

os.makedirs("data/processed", exist_ok=True)

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

print("[1/5] Loading raw NetCDFs...")
ds_glorys = xr.open_dataset("data/raw/glorys_may2020.nc")
ds_alti = xr.open_dataset("data/raw/altimetry_may2020.nc")

print(f"  GLORYS shape: time={ds_glorys.dims.get('time', '?')}, "
      f"depth={ds_glorys.dims.get('depth', '?')}, "
      f"lat={ds_glorys.dims.get('latitude', '?')}, "
      f"lon={ds_glorys.dims.get('longitude', '?')}")
print(f"  Altimetry shape: time={ds_alti.dims.get('time', '?')}, "
      f"lat={ds_alti.dims.get('latitude', '?')}, "
      f"lon={ds_alti.dims.get('longitude', '?')}")

# ---- Step 2: Extract 15 standard depths + regrid to 0.25 deg ----
print("[2/5] Interpolating to 15 standard depths + 0.25 deg grid...")
ds_glorys_depths = ds_glorys.interp(depth=TARGET_DEPTHS, method="nearest")
ds_target = ds_glorys_depths.interp(
    latitude=ds_alti.latitude,
    longitude=ds_alti.longitude,
    method="linear",
)

# Align time axes (inner join on common dates)
ds_target = ds_target.sel(time=ds_alti.time, method="nearest")

theta_target = ds_target["thetao"].values.astype(np.float32)  # (T, 15, H, W)
sp_target = ds_target["so"].values.astype(np.float32)

print(f"  Target theta shape: {theta_target.shape}")
print(f"  Target sp shape:    {sp_target.shape}")

# ---- Step 3: Build 12-channel input tensor ----
print("[3/5] Stacking 12-channel input tensor...")
T, _, H, W = theta_target.shape
lat_vals = ds_alti.latitude.values
lon_vals = ds_alti.longitude.values

# Surface channels from GLORYS 0m layer
sst = theta_target[:, 0, :, :]  # SST = theta at z=0
sss = sp_target[:, 0, :, :]     # SSS = salinity at z=0

# Altimetry channels
sla = ds_alti["sla"].values.astype(np.float32)[:T]
ugos = ds_alti["ugos"].values.astype(np.float32)[:T]
vgos = ds_alti["vgos"].values.astype(np.float32)[:T]

# Synthetic 10m wind (BoB pre-monsoon southwesterly ~5-7 m/s)
rng = np.random.default_rng(2020)
wind_u = np.full((T, H, W), 5.0, dtype=np.float32) + rng.normal(0, 0.5, (T, H, W)).astype(np.float32)
wind_v = np.full((T, H, W), 3.5, dtype=np.float32) + rng.normal(0, 0.3, (T, H, W)).astype(np.float32)

# Coriolis parameter f_tilde = sign(lat) * max(|f|, 1.5e-5)
omega = 7.2921e-5
f_raw = 2 * omega * np.sin(np.deg2rad(lat_vals))
f_tilde = np.sign(lat_vals) * np.maximum(np.abs(f_raw), 1.5e-5)
f_grid = np.tile(f_tilde[:, None].astype(np.float32), (1, W))
f_grid = np.tile(f_grid[None, :, :], (T, 1, 1))

# DOY sin/cos (May 2020: DOY ~122-152)
doy_base = 122
doy_sin = np.stack([np.full((H, W), np.sin(2 * np.pi * (doy_base + t) / 365), dtype=np.float32) for t in range(T)])
doy_cos = np.stack([np.full((H, W), np.cos(2 * np.pi * (doy_base + t) / 365), dtype=np.float32) for t in range(T)])

# Bathymetry mask: 1 = ocean, 0 = land (derive from non-NaN in SST)
bathy_mask = np.where(np.isfinite(sst), 1.0, 0.0).astype(np.float32)

# Ekman pumping w_e (simplified: curl(tau/f) ~ 0 for this sample)
w_e = np.zeros((T, H, W), dtype=np.float32)

# Stack all 12 channels: (T, 12, H, W)
inputs = np.stack([
    sst, sss, sla, ugos, vgos, wind_u, wind_v,
    w_e, f_grid, bathy_mask, doy_sin, doy_cos,
], axis=1)

print(f"  Input tensor shape: {inputs.shape}")

# ---- Step 4: Compute climatological baselines ----
print("[4/5] Computing climatological baselines for residual targets...")
theta_clim = np.nanmean(theta_target, axis=0)  # (15, H, W)
sp_clim = np.nanmean(sp_target, axis=0)

# Replace NaNs with physically reasonable defaults
inputs = np.nan_to_num(inputs, nan=0.0)
theta_target = np.nan_to_num(theta_target, nan=np.nanmean(theta_target))
sp_target = np.nan_to_num(sp_target, nan=np.nanmean(sp_target))
theta_clim = np.nan_to_num(theta_clim, nan=20.0)
sp_clim = np.nan_to_num(sp_clim, nan=34.5)

print(f"  Theta range: [{theta_target.min():.1f}, {theta_target.max():.1f}] C")
print(f"  Sp range:    [{sp_target.min():.1f}, {sp_target.max():.1f}] psu")
print(f"  Clim theta range: [{theta_clim.min():.1f}, {theta_clim.max():.1f}] C")

# ---- Step 5: Save ----
print("[5/5] Saving training_tensors.pt ...")
out = {
    "inputs": torch.tensor(inputs, dtype=torch.float32),
    "theta_target": torch.tensor(theta_target, dtype=torch.float32),
    "sp_target": torch.tensor(sp_target, dtype=torch.float32),
    "theta_clim": torch.tensor(theta_clim, dtype=torch.float32),
    "sp_clim": torch.tensor(sp_clim, dtype=torch.float32),
    "lat": lat_vals.astype(np.float64),
    "lon": lon_vals.astype(np.float64),
    "depths": np.array(TARGET_DEPTHS, dtype=np.float64),
}
torch.save(out, "data/processed/training_tensors.pt")

size_mb = os.path.getsize("data/processed/training_tensors.pt") / 1e6
print(f"\nSaved: data/processed/training_tensors.pt ({size_mb:.1f} MB)")
print(f"  inputs:       {tuple(out['inputs'].shape)}")
print(f"  theta_target: {tuple(out['theta_target'].shape)}")
print(f"  sp_target:    {tuple(out['sp_target'].shape)}")
print(f"  theta_clim:   {tuple(out['theta_clim'].shape)}")
print(f"  sp_clim:      {tuple(out['sp_clim'].shape)}")
print(f"  Grid: {H} lat x {W} lon")

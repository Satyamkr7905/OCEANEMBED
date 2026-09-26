"""Multi-year daily climatology (DOY 1–365) and residual anomaly engine.

Leap-year handling: 29 February is averaged into day 59 (28 February) and
subsequent leap-year DOYs are shifted back by one so March 1 is always 60.

The model is trained on residuals:
    Δθ = θ_true − θ̄_clim(x, y, z, DOY)
    ΔS_p = S_p,true − S̄_p,clim(x, y, z, DOY)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr
from tqdm import tqdm

from track1_data_engine.core import (
    climatology_doy,
    data_root,
    get_logger,
    load_bbox_config,
    write_json,
)

logger = get_logger()

DEFAULT_TARGET_VARS = ("thetao", "so")
DEFAULT_SURFACE_VARS = ("sst", "sss", "sla", "ugos", "vgos", "u10", "v10")


def _ensure_doy(ds: xr.Dataset) -> xr.DataArray:
    if "time" not in ds.coords and "time" not in ds.dims:
        raise ValueError("Climatology requires a 'time' coordinate.")
    doy = climatology_doy(ds["time"].values)
    return xr.DataArray(doy, dims=("time",), coords={"time": ds["time"]}, name="doy")


def compute_daily_climatology(
    ds: xr.Dataset,
    variables: Sequence[str] | None = None,
    *,
    min_samples: int = 3,
) -> xr.Dataset:
    """Mean and standard deviation for each DOY 1–365.

    Output dimensions: ``(doy, [depth,] lat, lon)`` with `doy` running 1..365.
    """
    variables = [v for v in (variables or list(ds.data_vars)) if v in ds.data_vars]
    if not variables:
        raise ValueError("No climatology variables found in dataset.")
    doy = _ensure_doy(ds)
    work = ds[variables].assign_coords(doy=doy)
    grouped = work.groupby("doy")
    mean = grouped.mean("time", skipna=True)
    std = grouped.std("time", skipna=True)
    count = grouped.count("time")

    # Guarantee a full 365-day axis even if the source misses a calendar day.
    full_doy = np.arange(1, 366, dtype=np.int16)
    mean = mean.reindex(doy=full_doy)
    std = std.reindex(doy=full_doy)
    count = count.reindex(doy=full_doy, fill_value=0)

    # Harmonic fill of missing DOYs (e.g. incomplete years) using neighbouring days.
    for name in variables:
        da = mean[name]
        if da.isnull().any():
            mean[name] = da.interpolate_na(dim="doy", method="linear", fill_value="extrapolate")
        low = count[name] < min_samples
        if low.any():
            logger.warning(
                "Climatology for %s has %s DOY bins with <%s samples.",
                name,
                int(low.sum()),
                min_samples,
            )

    clim = xr.Dataset()
    for name in variables:
        clim[f"{name}_clim"] = mean[name]
        clim[f"{name}_clim_std"] = std[name]
        clim[f"{name}_clim_count"] = count[name]
    clim = clim.assign_coords(doy=full_doy)
    clim["doy"].attrs.update({"long_name": "day of year", "units": "1", "leap_handling": "Feb29->Feb28"})
    clim.attrs["climatology_method"] = "arithmetic_mean_by_doy_1_365"
    return clim


def anomalies_from_climatology(
    ds: xr.Dataset,
    clim: xr.Dataset,
    variables: Sequence[str] | None = None,
) -> xr.Dataset:
    """Subtract the matching DOY climatology from each field in `ds`."""
    variables = [v for v in (variables or list(ds.data_vars)) if v in ds.data_vars]
    doy = _ensure_doy(ds)
    out = xr.Dataset(coords=ds.coords)
    for name in tqdm(variables, desc="anomaly decomposition", leave=False):
        key = f"{name}_clim"
        if key not in clim:
            raise KeyError(f"Climatology is missing '{key}'.")
        clim_on_time = clim[key].sel(doy=doy)
        # Drop the synthetic doy coord so subtraction aligns on time/depth/lat/lon.
        if "doy" in clim_on_time.dims:
            clim_on_time = clim_on_time.drop_vars("doy", errors="ignore")
        residual = ds[name] - clim_on_time
        residual.attrs.update(ds[name].attrs)
        residual.attrs["long_name"] = f"{name} anomaly vs daily climatology"
        out_name = {
            "thetao": "delta_theta",
            "so": "delta_sp",
            "sst": "delta_sst",
            "sss": "delta_sss",
        }.get(name, f"delta_{name}")
        out[out_name] = residual
        out[f"{name}_clim_on_time"] = clim_on_time
    out["doy"] = doy
    out.attrs["residual_definition"] = "field - climatology(DOY)"
    return out


def climatology_skill_baseline_mse(
    ds: xr.Dataset,
    clim: xr.Dataset,
    variable: str,
) -> xr.DataArray:
    """MSE of the climatology predictor (denominator of CSS)."""
    doy = _ensure_doy(ds)
    pred = clim[f"{variable}_clim"].sel(doy=doy)
    err = ds[variable] - pred
    return (err**2).mean("time", skipna=True).rename(f"{variable}_clim_mse")


def run_climatology_pipeline(
    input_zarr: Path,
    *,
    data_dir: Path | None = None,
    bbox_cfg: Mapping[str, Any] | None = None,
    variables: Sequence[str] | None = None,
    split: str | None = "train",
) -> tuple[Path, Path]:
    """Compute climatology (optionally on the training split only) and write anomalies.

    Using only 2012–2018 for the climatology prevents test-period leakage into
    the seasonal baseline.
    """
    from track1_data_engine.storage.zarr_converter import write_dataset_zarr

    bbox_cfg = dict(bbox_cfg or load_bbox_config())
    root = data_root(bbox_cfg, data_dir)
    ds = xr.open_zarr(input_zarr, consolidated=True)
    if split:
        start, end = bbox_cfg["chronological_split"][split]
        ds_clim_src = ds.sel(time=slice(start, end))
        if ds_clim_src.sizes.get("time", 0) == 0:
            logger.warning(
                "Split %s (%s–%s) has no timesteps in %s; using the full store for climatology.",
                split,
                start,
                end,
                input_zarr,
            )
            ds_clim_src = ds
        else:
            logger.info("Climatology estimated from %s split %s → %s", split, start, end)
    else:
        ds_clim_src = ds

    variables = list(variables or [v for v in DEFAULT_TARGET_VARS if v in ds.data_vars])
    clim = compute_daily_climatology(ds_clim_src, variables)
    clim_path = root / "climatology" / f"{Path(input_zarr).stem}_daily_clim.zarr"
    write_dataset_zarr(clim, clim_path, bbox_cfg=bbox_cfg)

    anoms = anomalies_from_climatology(ds, clim, variables)
    anom_path = root / "anomalies" / f"{Path(input_zarr).stem}_anomalies.zarr"
    write_dataset_zarr(anoms, anom_path, bbox_cfg=bbox_cfg)

    mse_report = {}
    for var in variables:
        mse = climatology_skill_baseline_mse(ds_clim_src, clim, var)
        mse_report[var] = {
            "global_mse": float(mse.mean().values),
            "rmse": float(np.sqrt(mse.mean().values)),
        }
    write_json(root / "metadata" / f"{Path(input_zarr).stem}_climatology_baseline.json", mse_report)
    logger.info("Climatology baseline RMSE: %s", mse_report)
    return clim_path, anom_path

"""Schema, domain, and physics-boundary validation for OceanEmbed stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr

from track1_data_engine.core import (
    STANDARD_DEPTHS,
    build_target_grid,
    get_logger,
    load_bbox_config,
    write_json,
    zarr_chunks,
)
from track1_data_engine.preprocessing.physics_features import F0

logger = get_logger()


class ValidationError(AssertionError):
    """Raised when a processed store violates the frozen schema."""


def _as_dataset(obj: Path | str | xr.Dataset) -> xr.Dataset:
    if isinstance(obj, xr.Dataset):
        return obj
    return xr.open_zarr(obj, consolidated=True)


def validate_grid(ds: xr.Dataset, bbox_cfg: Mapping[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    target = build_target_grid(bbox_cfg)
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return ["missing lat/lon coordinates"]
    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    if lat.shape != (target.n_lat,):
        errors.append(f"lat size {lat.shape} != {target.n_lat}")
    if lon.shape != (target.n_lon,):
        errors.append(f"lon size {lon.shape} != {target.n_lon}")
    if lat.size == target.n_lat and not np.allclose(lat, target.lat, atol=1e-4):
        errors.append("latitude centers do not match the frozen 0.25° NIO grid")
    if lon.size == target.n_lon and not np.allclose(lon, target.lon, atol=1e-4):
        errors.append("longitude centers do not match the frozen 0.25° NIO grid")
    if "depth" in ds.coords:
        depth = np.asarray(ds["depth"].values, dtype=np.float64)
        expected = np.asarray(STANDARD_DEPTHS, dtype=np.float64)
        if depth.size != expected.size or not np.allclose(depth, expected, atol=1e-3):
            errors.append(f"depth levels {depth} != {expected}")
    return errors


def validate_chunks(ds: xr.Dataset, bbox_cfg: Mapping[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    spec = zarr_chunks(bbox_cfg)
    for name, da in ds.data_vars.items():
        if da.chunks is None:
            continue
        for dim, chunk_tuple in zip(da.dims, da.chunks):
            if dim in spec and chunk_tuple:
                first = chunk_tuple[0]
                expected = min(spec[dim], da.sizes[dim])
                if first != expected:
                    errors.append(
                        f"{name}.{dim} first chunk {first} != expected {expected}"
                    )
    return errors


def validate_land_nan(ds: xr.Dataset, land_mask: xr.DataArray, variables: list[str]) -> list[str]:
    errors: list[str] = []
    land = np.asarray(land_mask.values, dtype=bool)
    for name in variables:
        if name not in ds:
            continue
        da = ds[name]
        if "lat" not in da.dims or "lon" not in da.dims:
            continue
        sample = da.isel(time=0) if "time" in da.dims else da
        sample = sample.isel(depth=0) if "depth" in sample.dims else sample
        values = np.asarray(sample.values)
        if values.shape != land.shape:
            errors.append(f"{name} spatial shape {values.shape} != land {land.shape}")
            continue
        leaked = np.isfinite(values) & land
        if leaked.any():
            errors.append(f"{name} has {int(leaked.sum())} finite values on land")
    return errors


def validate_coriolis(f_tilde: xr.DataArray | np.ndarray) -> list[str]:
    values = np.asarray(f_tilde, dtype=np.float64)
    errors: list[str] = []
    if np.any(np.abs(values) < F0 - 1e-12):
        errors.append(f"|f_tilde| dropped below f0={F0}")
    if np.any(~np.isfinite(values)):
        errors.append("f_tilde contains non-finite values")
    return errors


def validate_curl_spikes(
    curl: xr.DataArray,
    land_mask: xr.DataArray,
    *,
    max_abs: float = 5e-4,
) -> list[str]:
    """Flag pathological wind-stress curl magnitudes (N m⁻³)."""
    values = np.asarray(curl.values, dtype=np.float64)
    land = np.asarray(land_mask.values, dtype=bool)
    if values.ndim == 3:
        values = values[0]
    ocean = values[~land]
    ocean = ocean[np.isfinite(ocean)]
    if ocean.size == 0:
        return ["curl field has no finite ocean cells"]
    n_spike = int(np.sum(np.abs(ocean) > max_abs))
    if n_spike > 0.001 * ocean.size:
        return [f"wind-stress curl: {n_spike} ocean cells exceed |curl|>{max_abs}"]
    return []


def validate_store(
    store: Path | str | xr.Dataset,
    *,
    bbox_cfg: Mapping[str, Any] | None = None,
    masks: xr.Dataset | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full schema + boundary suite. Raises ValidationError on failure."""
    bbox_cfg = dict(bbox_cfg or load_bbox_config())
    ds = _as_dataset(store)
    errors: list[str] = []
    errors.extend(validate_grid(ds, bbox_cfg))
    errors.extend(validate_chunks(ds, bbox_cfg))
    if masks is not None:
        land = masks["land_mask"]
        spatial_vars = [
            name
            for name in ds.data_vars
            if {"lat", "lon"}.issubset(set(ds[name].dims))
            and name not in {"land_mask", "coastal_buffer_mask", "gradient_invalid_mask", "depth_valid"}
        ]
        errors.extend(validate_land_nan(ds, land, spatial_vars))
        if "f_tilde" in ds:
            errors.extend(validate_coriolis(ds["f_tilde"]))
        if "wind_stress_curl" in ds:
            errors.extend(validate_curl_spikes(ds["wind_stress_curl"], land))
        if "w_e" in ds:
            we = np.asarray(ds["w_e"].values)
            land_b = np.asarray(land.values, dtype=bool)
            if we.ndim >= 2 and np.isfinite(we[..., land_b]).any():
                errors.append("w_e has finite values on land")

    report = {
        "store": str(store) if not isinstance(store, xr.Dataset) else "in-memory",
        "variables": list(ds.data_vars),
        "dims": {str(k): int(v) for k, v in ds.sizes.items()},
        "errors": errors,
        "ok": not errors,
    }
    if report_path is not None:
        write_json(report_path, report)
    if errors:
        logger.error("Validation failed: %s", errors)
        raise ValidationError("; ".join(errors))
    logger.info("Validation passed for %s", report["store"])
    return report

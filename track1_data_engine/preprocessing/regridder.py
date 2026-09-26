"""Conservative regridding onto the OceanEmbed 0.25° NIO grid.

Prefers xESMF (`method='conservative'`) when ESMF is available. On platforms
where esmpy cannot be installed (typical Windows), a first-order conservative
remap for regular lat–lon grids is used. Both paths weight by spherical cell
overlap area and therefore avoid bilinear land–sea bleeding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np
import xarray as xr
from scipy.interpolate import pchip_interpolate
from tqdm import tqdm

from track1_data_engine.core import (
    TargetGrid,
    build_target_grid,
    cell_bounds_1d,
    get_logger,
    load_bbox_config,
)

logger = get_logger()

try:
    import xesmf as xe

    _HAS_XESMF = True
except Exception:  # pragma: no cover - optional on Windows
    xe = None  # type: ignore[assignment]
    _HAS_XESMF = False


LAT_NAMES = ("lat", "latitude", "nav_lat", "y")
LON_NAMES = ("lon", "longitude", "nav_lon", "x")
DEPTH_NAMES = ("depth", "deptht", "lev", "z", "elevation")
TIME_NAMES = ("time", "t", "valid_time")


class RegridError(RuntimeError):
    """Raised when a source field cannot be aligned to the target grid."""


def _first_present(names: Iterable[Hashable], candidates: Sequence[str]) -> Hashable:
    lower = {str(n).lower(): n for n in names}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise KeyError(f"None of {candidates} found in {list(names)}")


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename latitude/longitude/depth/time to `lat`, `lon`, `depth`, `time`."""
    rename: dict[Hashable, str] = {}
    dims_and_coords = list(ds.dims) + [c for c in ds.coords if c not in ds.dims]
    try:
        lat_name = _first_present(ds.coords, LAT_NAMES)
        if lat_name != "lat":
            rename[lat_name] = "lat"
    except KeyError:
        lat_name = _first_present(ds.dims, LAT_NAMES)
        if lat_name != "lat":
            rename[lat_name] = "lat"
    try:
        lon_name = _first_present(ds.coords, LON_NAMES)
        if lon_name != "lon":
            rename[lon_name] = "lon"
    except KeyError:
        lon_name = _first_present(ds.dims, LON_NAMES)
        if lon_name != "lon":
            rename[lon_name] = "lon"
    for dim in ds.dims:
        if str(dim).lower() in DEPTH_NAMES and dim != "depth":
            rename[dim] = "depth"
        if str(dim).lower() in TIME_NAMES and dim != "time":
            rename[dim] = "time"
    if rename:
        ds = ds.rename(rename)
    if "lon" in ds.coords:
        lon = ds["lon"].values
        if np.nanmax(lon) > 180.0 and np.nanmin(lon) > 0:
            ds = ds.assign_coords(lon=(((ds["lon"] + 180) % 360) - 180)).sortby("lon")
    if "lat" in ds.coords and ds["lat"].ndim == 1 and ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")
    return ds


def _lat_lon_1d(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    lon = np.asarray(ds["lon"].values, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1:
        raise RegridError("Conservative fallback requires 1-D lat/lon coordinates.")
    return lat, lon


def _overlap_matrix(src_bounds: np.ndarray, dst_bounds: np.ndarray, spherical_lat: bool) -> np.ndarray:
    """Return (n_dst, n_src) overlap weights.

    For latitude (`spherical_lat=True`) the weight is Δsin(φ).
    For longitude it is Δλ in radians.
    """
    src_a = src_bounds[:-1]
    src_b = src_bounds[1:]
    dst_a = dst_bounds[:-1]
    dst_b = dst_bounds[1:]
    lo = np.maximum(dst_a[:, None], src_a[None, :])
    hi = np.minimum(dst_b[:, None], src_b[None, :])
    if spherical_lat:
        overlap = np.clip(np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo)), 0.0, None)
    else:
        overlap = np.clip(np.deg2rad(hi - lo), 0.0, None)
    return overlap.astype(np.float64)


def conservative_remap_regular(
    field: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
) -> np.ndarray:
    """First-order conservative remap of a regular lat–lon array onto `dst`.

    `field` may be 2-D (lat, lon) or have leading extra axes (e.g. time, depth).
    """
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    dst_lat = np.asarray(dst_lat, dtype=np.float64)
    dst_lon = np.asarray(dst_lon, dtype=np.float64)
    w_lat = _overlap_matrix(cell_bounds_1d(src_lat), cell_bounds_1d(dst_lat), spherical_lat=True)
    w_lon = _overlap_matrix(cell_bounds_1d(src_lon), cell_bounds_1d(dst_lon), spherical_lat=False)

    extra = field.ndim - 2
    if extra < 0:
        raise ValueError("field must be at least 2-D (lat, lon).")
    leading = field.shape[:extra]
    src = np.reshape(field, (-1, field.shape[-2], field.shape[-1]))
    out = np.empty((src.shape[0], dst_lat.size, dst_lon.size), dtype=np.float64)
    for i in range(src.shape[0]):
        slab = src[i]
        valid = np.isfinite(slab)
        src0 = np.where(valid, slab, 0.0)
        weight = valid.astype(np.float64)
        numerator = w_lat @ src0 @ w_lon.T
        denominator = w_lat @ weight @ w_lon.T
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(denominator > 0.0, numerator / denominator, np.nan)
    return out.reshape((*leading, dst_lat.size, dst_lon.size))


def _xesmf_regrid(
    da: xr.DataArray,
    target: TargetGrid,
    weight_path: Path | None,
) -> xr.DataArray:
    if not _HAS_XESMF:
        raise RegridError("xesmf is not available.")
    ds_in = da.to_dataset(name=str(da.name or "var"))
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], target.lat),
            "lon": (["lon"], target.lon),
            "lat_b": (["lat_b"], target.lat_bounds),
            "lon_b": (["lon_b"], target.lon_bounds),
        }
    )
    reuse = bool(weight_path and weight_path.exists())
    regridder = xe.Regridder(
        ds_in,
        ds_out,
        method="conservative",
        periodic=False,
        reuse_weights=reuse,
        filename=str(weight_path) if weight_path is not None else None,
    )
    result = regridder(ds_in)
    return result[str(da.name or "var")]


def regrid_dataarray(
    da: xr.DataArray,
    target: TargetGrid | None = None,
    *,
    weight_path: Path | None = None,
    prefer_xesmf: bool = True,
) -> xr.DataArray:
    """Conservatively regrid a DataArray onto the OceanEmbed target grid."""
    target = target or build_target_grid()
    da = da.load() if da.chunks else da
    ds = standardize_coords(da.to_dataset(name=str(da.name or "var")))
    da = ds[str(da.name or "var")]
    if "lat" not in da.dims or "lon" not in da.dims:
        # 2-D geographic coordinates stored as aux coords
        raise RegridError(f"Expected lat/lon dimensions, got {da.dims}")

    if prefer_xesmf and _HAS_XESMF:
        try:
            out = _xesmf_regrid(da, target, weight_path)
            return _attach_target_coords(out, da, target)
        except Exception as exc:
            logger.warning("xESMF conservative regrid failed (%s); using spherical overlap remap.", exc)

    src_lat = np.asarray(da["lat"].values, dtype=np.float64)
    src_lon = np.asarray(da["lon"].values, dtype=np.float64)
    # Move lat, lon to trailing axes
    da_t = da.transpose(..., "lat", "lon")
    remapped = conservative_remap_regular(np.asarray(da_t.values), src_lat, src_lon, target.lat, target.lon)
    coords = {k: v for k, v in da_t.coords.items() if k not in {"lat", "lon"}}
    coords["lat"] = ("lat", target.lat)
    coords["lon"] = ("lon", target.lon)
    out = xr.DataArray(remapped, dims=da_t.dims, coords=coords, attrs=dict(da.attrs))
    out.attrs["regrid_method"] = "first_order_conservative_spherical"
    return out


def _attach_target_coords(out: xr.DataArray, original: xr.DataArray, target: TargetGrid) -> xr.DataArray:
    out = out.assign_coords(lat=target.lat, lon=target.lon)
    out.attrs.update(original.attrs)
    out.attrs["regrid_method"] = "xesmf_conservative"
    return out


def pchip_to_standard_depths(
    da: xr.DataArray,
    target_depths: np.ndarray | None = None,
) -> xr.DataArray:
    """Interpolate a depth-dependent field onto the 15 standard levels with PCHIP."""
    if "depth" not in da.dims:
        raise RegridError("PCHIP vertical interpolation requires a 'depth' dimension.")
    target = np.asarray(
        target_depths if target_depths is not None else build_target_grid().depths,
        dtype=np.float64,
    )
    src_z = np.asarray(da["depth"].values, dtype=np.float64)
    order = np.argsort(src_z)
    src_z = src_z[order]
    da_t = da.transpose("depth", ...)
    values = np.take(np.asarray(da_t.values, dtype=np.float64), order, axis=0)
    flat = values.reshape(src_z.size, -1)
    out_flat = np.full((target.size, flat.shape[1]), np.nan, dtype=np.float64)
    z_min, z_max = float(src_z[0]), float(src_z[-1])
    query = np.clip(target, z_min, z_max)
    complete = np.all(np.isfinite(flat), axis=0)
    if np.any(complete):
        out_flat[:, complete] = pchip_interpolate(src_z, flat[:, complete], query, axis=0)
        out_flat[target > z_max, :][:, complete] = np.nan
    partial_idx = np.flatnonzero((~complete) & (np.isfinite(flat).sum(axis=0) >= 3))
    for col in partial_idx:
        good = np.isfinite(flat[:, col])
        z_g, y_g = src_z[good], flat[good, col]
        z0, z1 = float(z_g[0]), float(z_g[-1])
        out_flat[:, col] = pchip_interpolate(z_g, y_g, np.clip(target, z0, z1))
        out_flat[target > z1, col] = np.nan
    interpolated = out_flat.reshape((target.size, *values.shape[1:]))
    coords = {k: v for k, v in da_t.coords.items() if k != "depth"}
    coords["depth"] = ("depth", target, {"units": "m", "positive": "down"})
    return xr.DataArray(
        interpolated,
        dims=("depth", *da_t.dims[1:]),
        coords=coords,
        attrs=dict(da.attrs),
    )


def apply_coastal_buffer(
    land_mask: xr.DataArray | np.ndarray,
    buffer_cells: int = 2,
) -> np.ndarray:
    """Dilate a land mask by `buffer_cells` (True = land or coastal buffer)."""
    from scipy.ndimage import binary_dilation

    mask = np.asarray(land_mask, dtype=bool)
    if buffer_cells <= 0:
        return mask
    structure = np.ones((2 * buffer_cells + 1, 2 * buffer_cells + 1), dtype=bool)
    return binary_dilation(mask, structure=structure)


def regrid_dataset(
    ds: xr.Dataset,
    target: TargetGrid | None = None,
    *,
    variables: Sequence[str] | None = None,
    interpolate_depth: bool = False,
) -> xr.Dataset:
    """Regrid every (or selected) variable in a dataset onto the target grid."""
    target = target or build_target_grid()
    ds = standardize_coords(ds)
    variables = list(variables or [name for name in ds.data_vars])
    out_vars: dict[str, xr.DataArray] = {}
    for name in tqdm(variables, desc="conservative regrid", leave=False):
        da = ds[name]
        if interpolate_depth and "depth" in da.dims:
            da = pchip_to_standard_depths(da, target.depths)
        if "lat" in da.dims and "lon" in da.dims:
            out_vars[name] = regrid_dataarray(da, target)
        else:
            out_vars[name] = da
    out = xr.Dataset(out_vars)
    out.attrs.update(ds.attrs)
    out.attrs["target_grid"] = "OceanEmbed 0.25deg NIO 100x240"
    return out


def build_land_mask_from_sst(sst: xr.DataArray, target: TargetGrid | None = None) -> xr.DataArray:
    """Land where SST is NaN on a climatological or snapshot field."""
    target = target or build_target_grid()
    if "time" in sst.dims:
        frac_finite = sst.notnull().mean("time")
        land = frac_finite < 0.5
    else:
        land = ~np.isfinite(sst)
    land = land.astype(bool)
    coastal = apply_coastal_buffer(land.values, int(load_bbox_config()["masks"]["coastal_buffer_cells"]))
    return xr.Dataset(
        {
            "land_mask": (("lat", "lon"), np.asarray(land, dtype=bool)),
            "coastal_buffer_mask": (("lat", "lon"), coastal),
        },
        coords={"lat": target.lat, "lon": target.lon},
    )


def process_raw_directory(
    raw_glob: str,
    output_zarr: Path,
    *,
    variables: Sequence[str] | None = None,
    interpolate_depth: bool = False,
    bbox_cfg: Mapping[str, Any] | None = None,
) -> Path:
    """Open a collection of raw NetCDF files, regrid, and stream to Zarr via the converter."""
    from track1_data_engine.storage.zarr_converter import write_dataset_zarr

    target = build_target_grid(bbox_cfg)
    ds = xr.open_mfdataset(raw_glob, combine="by_coords", parallel=False, chunks={"time": 30})
    ds = standardize_coords(ds)
    regridded = regrid_dataset(ds, target, variables=variables, interpolate_depth=interpolate_depth)
    return write_dataset_zarr(regridded, output_zarr, bbox_cfg=bbox_cfg)

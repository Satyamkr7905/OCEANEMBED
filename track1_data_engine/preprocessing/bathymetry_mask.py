"""ETOPO1 bathymetry ingestion and seabed truncation masks.

Grid cells where the seabed is shallower than a target depth are set to NaN
in loss/evaluation tensors. Elevation convention: ETOPO1 bedrock `z` is
positive above sea level and negative in the ocean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import numpy as np
import xarray as xr

from track1_data_engine.core import (
    TargetGrid,
    build_target_grid,
    data_root,
    ensure_dir,
    get_logger,
    load_bbox_config,
    load_sources_config,
    write_json,
)
from track1_data_engine.ingestion.http_util import DownloadError, http_get_to_file
from track1_data_engine.preprocessing.regridder import (
    apply_coastal_buffer,
    regrid_dataarray,
    standardize_coords,
)

logger = get_logger()


def download_etopo1(
    *,
    data_dir: Path | None = None,
    bbox_cfg: Mapping[str, Any] | None = None,
    sources_cfg: Mapping[str, Any] | None = None,
    skip_existing: bool = True,
) -> Path:
    """Download a regional ETOPO1 bedrock subset via ERDDAP."""
    bbox_cfg = dict(bbox_cfg or load_bbox_config())
    sources = dict(sources_cfg or load_sources_config())
    spec = sources["etopo1"]
    box = bbox_cfg["bbox"]
    root = data_root(bbox_cfg, data_dir)
    dest = ensure_dir(root / "raw" / "bathymetry") / "etopo1_bedrock_nio.nc"

    pad = 0.5
    lat_min = float(box["lat_min"]) - pad
    lat_max = float(box["lat_max"]) + pad
    lon_min = float(box["lon_min"]) - pad
    lon_max = float(box["lon_max"]) + pad
    # PACIOOS etopo1_bedrock uses latitude descending in some deployments;
    # ERDDAP accepts (start):(stop) in either order.
    query = (
        f"{spec['variable']}"
        f"[({lat_max}):1:({lat_min})]"
        f"[({lon_min}):1:({lon_max})]"
    )
    url = f"{spec['erddap_url']}?{quote(query, safe='[]():,')}"
    try:
        return http_get_to_file(url, dest, skip_existing=skip_existing, timeout=300, min_bytes=2048)
    except DownloadError as exc:
        logger.warning("ETOPO1 ERDDAP failed (%s); trying NGDC gzip fallback.", exc)
        gz_path = dest.with_suffix(".grd.gz")
        http_get_to_file(spec["fallback_url"], gz_path, skip_existing=skip_existing, timeout=600, min_bytes=1024)
        return gz_path


def _open_etopo(path: Path) -> xr.DataArray:
    if str(path).endswith(".gz"):
        import gzip
        import tempfile

        with gzip.open(path, "rb") as src, tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(src.read())
            tmp_path = Path(tmp.name)
        ds = xr.open_dataset(tmp_path)
    else:
        ds = xr.open_dataset(path)
    ds = standardize_coords(ds)
    for name in ("z", "altitude", "Band1", "elevation", "topo"):
        if name in ds.data_vars:
            da = ds[name]
            break
    else:
        data_vars = [v for v in ds.data_vars if set(ds[v].dims) >= {"lat", "lon"}]
        if not data_vars:
            raise KeyError(f"No bathymetry variable in {path}: {list(ds.data_vars)}")
        da = ds[data_vars[0]]
    return da


def build_bathymetry_masks(
    *,
    data_dir: Path | None = None,
    bbox_cfg: Mapping[str, Any] | None = None,
    etopo_path: Path | None = None,
    coastal_buffer_cells: int | None = None,
) -> xr.Dataset:
    """Regrid ETOPO1 to the target grid and build land / depth-truncation masks.

    Returns a dataset with:
      * `elevation_m` — ETOPO1 elevation (m), ocean negative
      * `water_column_m` — positive ocean depth
      * `land_mask` — True on land (elevation >= 0)
      * `coastal_buffer_mask` — land dilated by 2 cells
      * `gradient_invalid_mask` — land dilated by 1 cell (curl safeguard)
      * `depth_valid` — True where water_column > target depth (time-independent)
    """
    bbox_cfg = dict(bbox_cfg or load_bbox_config())
    target = build_target_grid(bbox_cfg)
    root = data_root(bbox_cfg, data_dir)
    if etopo_path is None:
        default = root / "raw" / "bathymetry" / "etopo1_bedrock_nio.nc"
        etopo_path = default if default.exists() else download_etopo1(data_dir=root, bbox_cfg=bbox_cfg)

    elevation = regrid_dataarray(_open_etopo(etopo_path), target)
    elevation = elevation.squeeze(drop=True)
    if "lat" not in elevation.dims:
        elevation = elevation.rename({elevation.dims[-2]: "lat", elevation.dims[-1]: "lon"})

    land_threshold = float(bbox_cfg["masks"]["land_elevation_threshold_m"])
    land = np.asarray(elevation.values >= land_threshold)
    water_column = np.where(land, 0.0, np.abs(np.minimum(elevation.values, 0.0)))

    coastal_n = int(
        coastal_buffer_cells
        if coastal_buffer_cells is not None
        else bbox_cfg["masks"]["coastal_buffer_cells"]
    )
    dilate_n = int(bbox_cfg["masks"]["land_dilation_cells_for_gradients"])
    coastal = apply_coastal_buffer(land, coastal_n)
    gradient_invalid = apply_coastal_buffer(land, dilate_n)

    depth_valid = water_column[None, :, :] > target.depths[:, None, None]
    depth_valid[:, land] = False

    ds = xr.Dataset(
        {
            "elevation_m": (("lat", "lon"), np.asarray(elevation.values, dtype=np.float32)),
            "water_column_m": (("lat", "lon"), water_column.astype(np.float32)),
            "land_mask": (("lat", "lon"), land),
            "coastal_buffer_mask": (("lat", "lon"), coastal),
            "gradient_invalid_mask": (("lat", "lon"), gradient_invalid),
            "depth_valid": (("depth", "lat", "lon"), depth_valid),
        },
        coords={
            "lat": target.lat,
            "lon": target.lon,
            "depth": target.depths,
        },
        attrs={
            "source": "ETOPO1 bedrock",
            "convention": "elevation positive upward; ocean negative",
            "coastal_buffer_cells": coastal_n,
            "gradient_dilation_cells": dilate_n,
            "n_land_cells": int(land.sum()),
            "n_coastal_buffer_cells": int(coastal.sum() - land.sum()),
            "n_gradient_buffer_cells": int(gradient_invalid.sum() - land.sum()),
        },
    )
    out_path = root / "masks" / "bathymetry_masks.zarr"
    from track1_data_engine.storage.zarr_converter import write_dataset_zarr

    write_dataset_zarr(ds, out_path, bbox_cfg=bbox_cfg)
    write_json(
        root / "metadata" / "bathymetry_mask_stats.json",
        {
            "n_land_cells": int(land.sum()),
            "n_ocean_cells": int((~land).sum()),
            "n_coastal_buffer_cells": int((coastal & ~land).sum()),
            "n_gradient_buffer_cells": int((gradient_invalid & ~land).sum()),
            "valid_cells_per_depth": {
                str(int(z)): int(depth_valid[i].sum()) for i, z in enumerate(target.depths)
            },
        },
    )
    logger.info(
        "Bathymetry masks written to %s (land=%s, coastal_buffer=%s, gradient_buffer=%s)",
        out_path,
        int(land.sum()),
        int((coastal & ~land).sum()),
        int((gradient_invalid & ~land).sum()),
    )
    return ds


def apply_depth_truncation(field: xr.DataArray, masks: xr.Dataset) -> xr.DataArray:
    """Set NaN where the seabed is shallower than the evaluated depth."""
    if "depth" not in field.dims:
        return field.where(~masks["land_mask"])
    return field.where(masks["depth_valid"])

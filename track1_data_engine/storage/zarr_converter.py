"""Chunked Zarr exporter for OceanEmbed tensors.

Default chunks: time=30, depth=15, lat=50, lon=50, matching the frozen blueprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr
from numcodecs import Blosc

from track1_data_engine.core import get_logger, load_bbox_config, zarr_chunks

logger = get_logger()

COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def _chunks_for(da: xr.DataArray, chunk_spec: Mapping[str, int]) -> tuple[int, ...]:
    chunks: list[int] = []
    for dim, size in zip(da.dims, da.shape):
        if dim in chunk_spec:
            chunks.append(int(min(chunk_spec[dim], size)))
        elif str(dim) in {"time", "t"}:
            chunks.append(int(min(chunk_spec.get("time", 30), size)))
        elif str(dim) in {"depth", "z"}:
            chunks.append(int(min(chunk_spec.get("depth", 15), size)))
        elif str(dim) in {"lat", "latitude"}:
            chunks.append(int(min(chunk_spec.get("lat", 50), size)))
        elif str(dim) in {"lon", "longitude"}:
            chunks.append(int(min(chunk_spec.get("lon", 50), size)))
        else:
            chunks.append(int(size))
    return tuple(chunks)


def _encoding(ds: xr.Dataset, chunk_spec: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    encoding: dict[str, dict[str, Any]] = {}
    for name, da in ds.data_vars.items():
        dtype = da.dtype
        if np.issubdtype(dtype, np.floating):
            dtype = np.float32
        elif np.issubdtype(dtype, np.bool_):
            dtype = np.uint8
        encoding[name] = {
            "chunks": _chunks_for(da, chunk_spec),
            "compressor": COMPRESSOR,
            "dtype": dtype,
        }
    return encoding


def write_dataset_zarr(
    ds: xr.Dataset,
    path: Path | str,
    *,
    bbox_cfg: Mapping[str, Any] | None = None,
    mode: str = "w",
    consolidated: bool = True,
) -> Path:
    """Write `ds` to a chunked, compressed Zarr store."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk_spec = zarr_chunks(bbox_cfg)
    encoding = _encoding(ds, chunk_spec)
    try:
        import dask  # noqa: F401

        ds = ds.chunk({dim: chunk_spec[dim] for dim in ds.dims if dim in chunk_spec})
    except ImportError:
        logger.warning("dask is not installed; writing Zarr from in-memory arrays with encoded chunks.")
    if path.exists() and mode == "w":
        import shutil

        shutil.rmtree(path)
    logger.info("Writing Zarr %s vars=%s chunks=%s", path, list(ds.data_vars), chunk_spec)
    ds.to_zarr(path, mode=mode, encoding=encoding, consolidated=consolidated)
    return path


def append_time_zarr(
    ds: xr.Dataset,
    path: Path | str,
    *,
    bbox_cfg: Mapping[str, Any] | None = None,
) -> Path:
    """Append along `time` to an existing store (same variables/grid required)."""
    path = Path(path)
    if not path.exists():
        return write_dataset_zarr(ds, path, bbox_cfg=bbox_cfg, mode="w")
    ds.to_zarr(path, mode="a", append_dim="time", consolidated=True)
    return path


def open_store(path: Path | str) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=True)

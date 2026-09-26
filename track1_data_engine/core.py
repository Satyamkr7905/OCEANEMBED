"""Shared configuration, target grid, logging, and I/O helpers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import yaml

STANDARD_DEPTHS: tuple[int, ...] = (
    0,
    5,
    10,
    20,
    30,
    50,
    75,
    100,
    125,
    150,
    200,
    300,
    500,
    700,
    1000,
)

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_BBOX_YAML = PACKAGE_DIR / "configs" / "bounding_box.yaml"
DEFAULT_SOURCES_YAML = PACKAGE_DIR / "configs" / "data_sources.yaml"

LOGGER_NAME = "oceanembed.track1"


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> logging.Logger:
    """Configure a module logger with console (and optional file) handlers."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must contain a mapping.")
    return data


def load_bbox_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or DEFAULT_BBOX_YAML)


def load_sources_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or DEFAULT_SOURCES_YAML)


def parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def daterange(start: date, end: date) -> Iterator[date]:
    """Inclusive calendar-day iterator."""
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def monthly_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield inclusive [window_start, window_end] month-aligned chunks."""
    cursor = start
    while cursor <= end:
        if cursor.month == 12:
            month_end = date(cursor.year, 12, 31)
        else:
            month_end = date(cursor.year, cursor.month + 1, 1) - timedelta(days=1)
        yield cursor, min(month_end, end)
        cursor = month_end + timedelta(days=1)


@dataclass(frozen=True)
class TargetGrid:
    """NIO 0.25° cell-center grid (100 × 240)."""

    lat: np.ndarray
    lon: np.ndarray
    depths: np.ndarray
    resolution_deg: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @property
    def n_lat(self) -> int:
        return int(self.lat.size)

    @property
    def n_lon(self) -> int:
        return int(self.lon.size)

    @property
    def lat_bounds(self) -> np.ndarray:
        return cell_bounds_1d(self.lat)

    @property
    def lon_bounds(self) -> np.ndarray:
        return cell_bounds_1d(self.lon)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.lon, self.lat, indexing="xy")

    def cell_area_m2(self, earth_radius_m: float = 6_371_000.0) -> np.ndarray:
        """Spherical cell area on the target grid, shape (n_lat, n_lon)."""
        lat_b = np.deg2rad(self.lat_bounds)
        lon_b = np.deg2rad(self.lon_bounds)
        dsin_lat = np.diff(np.sin(lat_b))  # (n_lat,)
        dlon = np.diff(lon_b)  # (n_lon,)
        return (earth_radius_m**2) * dsin_lat[:, None] * dlon[None, :]

    def to_dataset(self):
        import xarray as xr

        lon2d, lat2d = self.mesh()
        return xr.Dataset(
            coords={
                "lat": ("lat", self.lat, {"units": "degrees_north", "standard_name": "latitude"}),
                "lon": ("lon", self.lon, {"units": "degrees_east", "standard_name": "longitude"}),
                "depth": ("depth", self.depths, {"units": "m", "positive": "down"}),
            },
            data_vars={
                "lat_b": ("lat_b", self.lat_bounds, {"long_name": "latitude cell bounds"}),
                "lon_b": ("lon_b", self.lon_bounds, {"long_name": "longitude cell bounds"}),
                "area": (("lat", "lon"), self.cell_area_m2(), {"units": "m2"}),
            },
            attrs={
                "grid_registration": "cell_centers",
                "resolution_deg": self.resolution_deg,
                "domain": "North Indian Ocean",
            },
        )


def cell_bounds_1d(centers: np.ndarray) -> np.ndarray:
    """Infer n+1 cell edges from n monotonically increasing cell centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("centers must be a 1-D array with at least 2 values.")
    if np.any(np.diff(centers) <= 0):
        raise ValueError("centers must be strictly increasing.")
    interior = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (interior[0] - centers[0])
    last = centers[-1] + (centers[-1] - interior[-1])
    return np.concatenate([[first], interior, [last]])


def build_target_grid(bbox_cfg: Mapping[str, Any] | None = None) -> TargetGrid:
    cfg = dict(bbox_cfg or load_bbox_config())
    grid = cfg["grid"]
    box = cfg["bbox"]
    res = float(grid["resolution_deg"])
    n_lat = int(grid["n_lat"])
    n_lon = int(grid["n_lon"])
    lat = float(grid["lat_start"]) + res * np.arange(n_lat, dtype=np.float64)
    lon = float(grid["lon_start"]) + res * np.arange(n_lon, dtype=np.float64)
    depths = np.asarray(cfg.get("depths_m", STANDARD_DEPTHS), dtype=np.float64)
    if lat.size != n_lat or lon.size != n_lon:
        raise RuntimeError("Target grid construction produced unexpected sizes.")
    if not np.isclose(lat[-1] + res / 2.0, float(box["lat_max"]), atol=1e-6):
        raise RuntimeError(
            f"Latitude grid does not tile bbox: last edge {lat[-1] + res / 2:.4f} "
            f"vs lat_max {box['lat_max']}."
        )
    if not np.isclose(lon[-1] + res / 2.0, float(box["lon_max"]), atol=1e-6):
        raise RuntimeError(
            f"Longitude grid does not tile bbox: last edge {lon[-1] + res / 2:.4f} "
            f"vs lon_max {box['lon_max']}."
        )
    return TargetGrid(
        lat=lat,
        lon=lon,
        depths=depths,
        resolution_deg=res,
        lat_min=float(box["lat_min"]),
        lat_max=float(box["lat_max"]),
        lon_min=float(box["lon_min"]),
        lon_max=float(box["lon_max"]),
    )


def data_root(bbox_cfg: Mapping[str, Any] | None = None, override: Path | None = None) -> Path:
    if override is not None:
        root = Path(override)
    else:
        cfg = dict(bbox_cfg or load_bbox_config())
        root = Path(cfg["storage"]["data_root"])
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("raw", "regridded", "climatology", "anomalies", "masks", "metadata"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root.resolve()


def zarr_chunks(bbox_cfg: Mapping[str, Any] | None = None) -> dict[str, int]:
    cfg = dict(bbox_cfg or load_bbox_config())
    return {str(k): int(v) for k, v in cfg["storage"]["zarr_chunks"].items()}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def env_first(names: Sequence[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def climatology_doy(times: np.ndarray | Sequence[np.datetime64]) -> np.ndarray:
    """Map timestamps to DOY 1–365 with 29 Feb averaged into 28 Feb.

    Leap-year days after 29 Feb are shifted back by one so March 1 is always DOY 60.
    """
    import pandas as pd

    index = pd.DatetimeIndex(times)
    doy = index.dayofyear.to_numpy().astype(np.int16)
    leap = index.is_leap_year.to_numpy()
    doy = np.where(leap & (doy == 60), 59, doy)
    doy = np.where(leap & (doy > 60), doy - 1, doy)
    if np.any((doy < 1) | (doy > 365)):
        raise ValueError("Climatology DOY mapping produced values outside 1–365.")
    return doy

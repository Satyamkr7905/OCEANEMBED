"""Shared constants and configuration for the OceanEmbed model engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_YAML = PACKAGE_DIR / "configs" / "model.yaml"

N_LAT = 100
N_LON = 240
N_DEPTH = 15
WINDOW_DAYS = 7
N_SURFACE = 7
N_CHANNELS = 12
LAT_START = 5.125
LON_START = 45.125
RESOLUTION = 0.25
BOB_LON = 80.0
STANDARD_DEPTHS = (
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
SURFACE_NAMES = ("sst", "sss", "sla", "ugos", "vgos", "u10", "v10")
CHANNEL_NAMES = SURFACE_NAMES + ("w_e", "f_tilde", "bathymetry", "doy_sin", "doy_cos")


def load_model_config(path: Path | None = None) -> dict[str, Any]:
    with (path or DEFAULT_MODEL_YAML).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("model.yaml must contain a mapping")
    return data


def target_latitudes() -> np.ndarray:
    return LAT_START + RESOLUTION * np.arange(N_LAT, dtype=np.float64)


def target_longitudes() -> np.ndarray:
    return LON_START + RESOLUTION * np.arange(N_LON, dtype=np.float64)


def arabian_sea_mask(lon: np.ndarray | None = None) -> np.ndarray:
    """True on the Arabian Sea side of 80°E (broadcastable to H×W)."""
    if lon is None:
        lon = target_longitudes()
    return (lon < BOB_LON)[None, :]


def climatology_doy(times: np.ndarray) -> np.ndarray:
    import pandas as pd

    index = pd.DatetimeIndex(times)
    doy = index.dayofyear.to_numpy().astype(np.int16)
    leap = index.is_leap_year.to_numpy()
    doy = np.where(leap & (doy == 60), 59, doy)
    doy = np.where(leap & (doy > 60), doy - 1, doy)
    return doy

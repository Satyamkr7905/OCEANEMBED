"""Shared constants for OceanEmbed validation and TEOS-10 diagnostics."""

from __future__ import annotations

import numpy as np

N_LAT = 100
N_LON = 240
N_DEPTH = 15
LAT_START = 5.125
LON_START = 45.125
RESOLUTION = 0.25
STANDARD_DEPTHS = np.asarray(
    [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
    dtype=np.float64,
)
Z_1M_MAX = 1000
CP_TCHP = 4178.0  # J kg-1 K-1 (NOAA AOML / Leipper–Volgenau operational TCHP)
RHO0 = 1025.0
MLD_DRHO = 0.03  # kg m-3, TEOS-10 density criterion
ILD_DT = 0.2  # °C
TCHP_ISOTHERM = 26.0
Z20_ISOTHERM = 20.0
REF_DEPTH_MLD = 10.0
ARGO_MIN_DEPTH_M = 5.0
ARGO_QC_GOOD = frozenset({"1", "2", 1, 2, b"1", b"2"})
RAMA_SITES = {
    "rama_15n90e": (15.0, 90.0),
    "rama_12n90e": (12.0, 90.0),
}
RAMA_SENSOR_DEPTHS = np.asarray(
    [1.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 140.0, 200.0, 300.0, 500.0],
    dtype=np.float64,
)
DT_BINS_HOURS = {
    "dt_0d": (0.0, 12.0),
    "dt_1d": (12.0, 36.0),
    "dt_2d": (36.0, 60.0),
}
KJ_CM2_PER_J_M2 = 1.0e-7  # 1 kJ cm-2 = 1e7 J m-2


def target_latitudes() -> np.ndarray:
    return LAT_START + RESOLUTION * np.arange(N_LAT, dtype=np.float64)


def target_longitudes() -> np.ndarray:
    return LON_START + RESOLUTION * np.arange(N_LON, dtype=np.float64)


def z_1m(z_max: int = Z_1M_MAX) -> np.ndarray:
    return np.arange(0, int(z_max) + 1, dtype=np.float64)

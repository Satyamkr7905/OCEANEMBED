"""Fast 7-day antecedent cube assembly from Track-1 Zarr stores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import (
    LAT_START,
    LON_START,
    N_CHANNELS,
    N_LAT,
    N_LON,
    RESOLUTION,
    STANDARD_DEPTHS,
    WINDOW_DAYS,
    get_settings,
)

SURFACE_ALIASES: dict[str, tuple[str, ...]] = {
    "sst": ("delta_sst", "sst"),
    "sss": ("delta_sss", "sss", "sos"),
    "sla": ("delta_sla", "sla"),
    "ugos": ("delta_ugos", "ugos"),
    "vgos": ("delta_vgos", "vgos"),
    "u10": ("delta_u10", "u10"),
    "v10": ("delta_v10", "v10"),
}


def latitudes() -> np.ndarray:
    return LAT_START + RESOLUTION * np.arange(N_LAT, dtype=np.float64)


def longitudes() -> np.ndarray:
    return LON_START + RESOLUTION * np.arange(N_LON, dtype=np.float64)


def nearest_index(coord: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(coord - value)))


def climatology_doy(day: date) -> int:
    doy = int(day.strftime("%j"))
    leap = day.year % 4 == 0 and (day.year % 100 != 0 or day.year % 400 == 0)
    if leap and doy == 60:
        return 59
    if leap and doy > 60:
        return doy - 1
    return doy


@dataclass
class AntecedentCube:
    inputs: np.ndarray  # (1, 7, 12, 100, 240) float32
    clim_theta: np.ndarray  # (15, 100, 240)
    clim_sp: np.ndarray
    land_mask: np.ndarray  # (100, 240) bool
    sst: np.ndarray  # (100, 240) last-day SST
    sla: np.ndarray
    date: date
    synthetic: bool


def _open(path: Path):
    import zarr

    return zarr.open_group(str(path), mode="r")


def _first(group, names: tuple[str, ...]):
    keys = set(group.array_keys()) if hasattr(group, "array_keys") else set(group.keys())
    for name in names:
        if name in keys:
            return group[name]
    raise KeyError(f"None of {list(names)} in {sorted(keys)}")


def _decode_times(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    if np.issubdtype(raw.dtype, np.datetime64):
        return raw.astype("datetime64[D]")
    return raw.astype("datetime64[ns]").astype("datetime64[D]")


@lru_cache(maxsize=8)
def _handles(root: str) -> dict[str, object]:
    root_p = Path(root)
    mapping = {
        "sst": root_p / "regridded" / "sst.zarr",
        "sss": root_p / "regridded" / "sss.zarr",
        "sealevel": root_p / "regridded" / "sealevel.zarr",
        "winds": root_p / "regridded" / "winds.zarr",
        "physics": root_p / "regridded" / "physics_features.zarr",
        "clim": root_p / "climatology" / "glorys12v1_daily_clim.zarr",
        "masks": root_p / "masks" / "bathymetry_masks.zarr",
    }
    opened: dict[str, object] = {}
    for key, path in mapping.items():
        if path.exists():
            opened[key] = _open(path)
    return opened


def stores_available() -> bool:
    handles = _handles(str(get_settings().data_root.resolve()))
    return {"sst", "clim", "masks"}.issubset(handles)


def _synthetic_cube(target: date) -> AntecedentCube:
    lat = latitudes()
    rng = np.random.default_rng(int(target.strftime("%Y%m%d")))
    land = np.zeros((N_LAT, N_LON), dtype=bool)
    land[:2] = True
    land[:, :2] = True
    cube = rng.normal(0.0, 0.15, size=(WINDOW_DAYS, N_CHANNELS, N_LAT, N_LON)).astype(np.float32)
    cube[..., land] = 0.0
    doy = climatology_doy(target)
    angle = 2.0 * np.pi * (doy - 1) / 365.0
    cube[:, -2, :, :] = np.float32(np.sin(angle))
    cube[:, -1, :, :] = np.float32(np.cos(angle))
    z = np.asarray(STANDARD_DEPTHS, dtype=np.float32)
    clim_t = (28.5 - 0.016 * z)[:, None, None] * np.ones((1, N_LAT, N_LON), dtype=np.float32)
    clim_s = (34.6 + 0.0008 * z)[:, None, None] * np.ones((1, N_LAT, N_LON), dtype=np.float32)
    sst = np.full((N_LAT, N_LON), 29.2, dtype=np.float32)
    sla = rng.normal(0.04, 0.03, size=(N_LAT, N_LON)).astype(np.float32)
    sst[land] = np.nan
    return AntecedentCube(
        inputs=cube[None, ...],
        clim_theta=clim_t,
        clim_sp=clim_s,
        land_mask=land,
        sst=sst,
        sla=sla,
        date=target,
        synthetic=True,
    )


def load_antecedent(target: date) -> AntecedentCube:
    """Load the 7-day window ending on ``target`` (inclusive)."""
    settings = get_settings()
    root = str(settings.data_root.resolve())
    handles = _handles(root)
    if not stores_available():
        if settings.allow_synthetic:
            return _synthetic_cube(target)
        raise FileNotFoundError(
            f"Zarr stores missing under {root}. Run track1 or set ALLOW_SYNTHETIC=true."
        )

    sst_g = handles["sst"]
    times = _decode_times(np.asarray(sst_g["time"][:]))
    target_d = np.datetime64(target.isoformat(), "D")
    matches = np.flatnonzero(times == target_d)
    if matches.size == 0:
        raise KeyError(f"No satellite fields for {target.isoformat()}")
    t_end = int(matches[0])
    if t_end < WINDOW_DAYS - 1:
        raise KeyError(f"Need {WINDOW_DAYS} antecedent days ending {target.isoformat()}")
    t0, t1 = t_end - WINDOW_DAYS + 1, t_end + 1

    def window_2d(group, aliases: tuple[str, ...]) -> np.ndarray:
        return np.array(_first(group, aliases)[t0:t1], dtype=np.float32, copy=True)

    sst = window_2d(handles["sst"], SURFACE_ALIASES["sst"])
    sss = window_2d(handles["sss"], SURFACE_ALIASES["sss"]) if "sss" in handles else np.zeros_like(sst)
    sla = window_2d(handles["sealevel"], SURFACE_ALIASES["sla"]) if "sealevel" in handles else np.zeros_like(sst)
    ugos = window_2d(handles["sealevel"], SURFACE_ALIASES["ugos"]) if "sealevel" in handles else np.zeros_like(sst)
    vgos = window_2d(handles["sealevel"], SURFACE_ALIASES["vgos"]) if "sealevel" in handles else np.zeros_like(sst)
    u10 = window_2d(handles["winds"], SURFACE_ALIASES["u10"]) if "winds" in handles else np.zeros_like(sst)
    v10 = window_2d(handles["winds"], SURFACE_ALIASES["v10"]) if "winds" in handles else np.zeros_like(sst)
    we = window_2d(handles["physics"], ("w_e",)) if "physics" in handles else np.zeros_like(sst)

    masks = handles["masks"]
    land = np.array(masks["land_mask"][:], dtype=bool, copy=True)
    bathy = np.array(masks["water_column_m"][:], dtype=np.float32, copy=True)
    phy = handles["physics"]
    f_tilde = np.array(_first(phy, ("f_tilde", "f"))[:], dtype=np.float32, copy=True)
    f_map = np.broadcast_to(f_tilde[:, None] if f_tilde.ndim == 1 else f_tilde, (N_LAT, N_LON)).astype(np.float32)

    days = [target - timedelta(days=WINDOW_DAYS - 1 - i) for i in range(WINDOW_DAYS)]
    doy = np.asarray([climatology_doy(d) for d in days], dtype=np.float32)
    angle = 2.0 * np.pi * (doy - 1.0) / 365.0
    doy_sin = np.broadcast_to(np.sin(angle)[:, None, None], sst.shape).astype(np.float32)
    doy_cos = np.broadcast_to(np.cos(angle)[:, None, None], sst.shape).astype(np.float32)
    bathy_t = np.broadcast_to(bathy[None], sst.shape).astype(np.float32)
    f_t = np.broadcast_to(f_map[None], sst.shape).astype(np.float32)

    cube = np.stack([sst, sss, sla, ugos, vgos, u10, v10, we, f_t, bathy_t, doy_sin, doy_cos], axis=1)
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    cube[..., land] = 0.0

    clim = handles["clim"]
    doy_end = climatology_doy(target)
    clim_t = np.array(_first(clim, ("thetao_clim", "theta_clim"))[doy_end - 1], dtype=np.float32, copy=True)
    clim_s = np.array(_first(clim, ("so_clim", "sp_clim"))[doy_end - 1], dtype=np.float32, copy=True)
    if clim_t.shape[0] != len(STANDARD_DEPTHS):
        raise ValueError(f"climatology depth {clim_t.shape} != 15")

    return AntecedentCube(
        inputs=cube[None, ...],
        clim_theta=np.nan_to_num(clim_t).astype(np.float32),
        clim_sp=np.nan_to_num(clim_s).astype(np.float32),
        land_mask=land,
        sst=np.array(sst[-1], dtype=np.float32, copy=True),
        sla=np.array(sla[-1], dtype=np.float32, copy=True),
        date=target,
        synthetic=False,
    )

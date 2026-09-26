"""RAMA mooring validation at 15°N, 90°E and 12°N, 90°E.

RAMA sensors sit at 1, 10, 20, 40, 60, 80, 100, 140, 200, 300, 500 m — not
on the 15-level OceanEmbed grid. Model profiles are PCHIP-interpolated to
those exact sensor depths. A reverse mapping (RAMA → 15 standard depths)
is also reported so interpolation direction is not a hidden degree of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from track3_validation_engine.constants import (
    RAMA_SENSOR_DEPTHS,
    RAMA_SITES,
    STANDARD_DEPTHS,
    target_latitudes,
    target_longitudes,
)
from track3_validation_engine.diagnostics.pchip_profiler import pchip_sample
from track3_validation_engine.benchmarks.skill_scores import SkillReport, depthwise_skill

Array = NDArray[np.floating]


@dataclass
class RamaRecord:
    site: str
    lat: float
    lon: float
    time: np.ndarray  # datetime64[ns], (T,)
    depth: np.ndarray  # (Z,)
    temp: np.ndarray  # (T, Z)
    psal: np.ndarray  # (T, Z)


def _pick(ds, names: Sequence[str]):
    for name in names:
        if name in ds:
            return ds[name]
        if name.upper() in ds:
            return ds[name.upper()]
        if name.lower() in ds:
            return ds[name.lower()]
    raise KeyError(f"None of {list(names)} in {list(ds.data_vars) + list(ds.coords)}")


def read_rama_netcdf(path: Path | str, site: str | None = None) -> RamaRecord:
    """Read a PMEL/INCOIS RAMA mooring NetCDF (flexible variable names)."""
    import xarray as xr

    path = Path(path)
    ds = xr.open_dataset(path)
    try:
        time = _pick(ds, ("TIME", "time", "JULD")).values
        time = np.asarray(time).astype("datetime64[ns]")
        depth = np.asarray(_pick(ds, ("DEPTH", "depth", "PRES", "plev")).values, dtype=np.float64).reshape(-1)
        temp_da = _pick(ds, ("TEMP", "T_20", "temperature", "T_11", "theta"))
        try:
            salt_da = _pick(ds, ("PSAL", "S_41", "salinity", "S_12", "practical_salinity"))
        except KeyError:
            salt_da = None
        temp = np.asarray(temp_da.values, dtype=np.float64)
        psal = np.full_like(temp, np.nan) if salt_da is None else np.asarray(salt_da.values, dtype=np.float64)
        if temp.ndim == 1:
            temp = temp[None, :] if time.size == 1 else temp.reshape(time.size, -1)
            psal = psal.reshape(temp.shape)
        if temp.shape[0] != time.size and temp.shape[-1] == time.size:
            temp = np.moveaxis(temp, -1, 0)
            psal = np.moveaxis(psal, -1, 0)
        lat = float(np.nanmean(_pick(ds, ("LATITUDE", "lat", "latitude")).values))
        lon = float(np.nanmean(_pick(ds, ("LONGITUDE", "lon", "longitude")).values))
    finally:
        ds.close()

    if site is None:
        site = _nearest_site(lat, lon)
    return RamaRecord(site=site, lat=lat, lon=lon, time=time, depth=depth, temp=temp, psal=psal)


def _nearest_site(lat: float, lon: float) -> str:
    best = min(RAMA_SITES.items(), key=lambda kv: (kv[1][0] - lat) ** 2 + (kv[1][1] - lon) ** 2)
    return best[0]


def ingest_rama(paths: Iterable[Path | str]) -> list[RamaRecord]:
    return [read_rama_netcdf(p) for p in paths]


def _nearest_index(coord: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(coord - value)))


@dataclass
class RamaMatch:
    site: str
    model_on_rama: pd.DataFrame
    rama_on_standard: pd.DataFrame
    notes: dict[str, Any] = field(default_factory=dict)


def match_rama_to_model(
    record: RamaRecord,
    model_theta: Array,
    model_sp: Array,
    model_time: Array,
    *,
    lat: Array | None = None,
    lon: Array | None = None,
    model_depth: Array | None = None,
    sensor_depths: Array | None = None,
) -> RamaMatch:
    """Interpolate the model to RAMA sensors and RAMA to the 15 standard depths."""
    lat = target_latitudes() if lat is None else np.asarray(lat, dtype=np.float64)
    lon = target_longitudes() if lon is None else np.asarray(lon, dtype=np.float64)
    z_model = STANDARD_DEPTHS if model_depth is None else np.asarray(model_depth, dtype=np.float64)
    z_rama = RAMA_SENSOR_DEPTHS if sensor_depths is None else np.asarray(sensor_depths, dtype=np.float64)
    times = np.asarray(model_time).astype("datetime64[ns]")
    theta = np.asarray(model_theta, dtype=np.float64)
    sp = np.asarray(model_sp, dtype=np.float64)

    iy = _nearest_index(lat, record.lat)
    ix = _nearest_index(lon, record.lon)
    rama_times = np.asarray(record.time).astype("datetime64[ns]")

    # Map RAMA depths onto the record's native sensor axis via nearest, then PCHIP in z.
    rows_a: list[dict[str, Any]] = []
    rows_b: list[dict[str, Any]] = []

    for i, t_r in enumerate(rama_times):
        it = int(np.argmin(np.abs(times.astype("int64") - np.datetime64(t_r, "ns").astype("int64"))))
        abs_h = abs(int(times[it].astype("int64") - np.datetime64(t_r, "ns").astype("int64"))) / 3.6e12
        if abs_h > 12.0:
            continue
        col_t = theta[it, :, iy, ix]
        col_s = sp[it, :, iy, ix]
        t_on_rama = np.asarray(pchip_sample(col_t[:, None], z_model, z_rama, depth_axis=0)).reshape(-1)
        s_on_rama = np.asarray(pchip_sample(col_s[:, None], z_model, z_rama, depth_axis=0)).reshape(-1)

        obs_t = _sample_record_at(record.temp[i], record.depth, z_rama)
        obs_s = _sample_record_at(record.psal[i], record.depth, z_rama)
        for k, z in enumerate(z_rama):
            rows_a.append(
                {
                    "site": record.site,
                    "time": t_r,
                    "depth": float(z),
                    "temp_model": float(t_on_rama[k]),
                    "temp_rama": float(obs_t[k]),
                    "sp_model": float(s_on_rama[k]),
                    "sp_rama": float(obs_s[k]),
                    "direction": "model_to_rama",
                }
            )

        t_on_std = _sample_record_at(record.temp[i], record.depth, z_model)
        s_on_std = _sample_record_at(record.psal[i], record.depth, z_model)
        for k, z in enumerate(z_model):
            rows_b.append(
                {
                    "site": record.site,
                    "time": t_r,
                    "depth": float(z),
                    "temp_model": float(col_t[k]),
                    "temp_rama": float(t_on_std[k]),
                    "sp_model": float(col_s[k]),
                    "sp_rama": float(s_on_std[k]),
                    "direction": "rama_to_standard",
                }
            )

    notes = {
        "site": record.site,
        "grid_lat": float(lat[iy]),
        "grid_lon": float(lon[ix]),
        "sensor_depths_m": z_rama.tolist(),
        "primary_window_h": 12,
        "in_situ_includes_1m": True,
        "argo_contrast": "RAMA 1 m is valid in-situ; ARGO validation still starts at 5 m.",
    }
    return RamaMatch(
        site=record.site,
        model_on_rama=pd.DataFrame.from_records(rows_a),
        rama_on_standard=pd.DataFrame.from_records(rows_b),
        notes=notes,
    )


def _sample_record_at(values: np.ndarray, z_src: np.ndarray, z_out: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    zs = np.asarray(z_src, dtype=np.float64).reshape(-1)
    if v.size != zs.size:
        n = min(v.size, zs.size)
        v, zs = v[:n], zs[:n]
    finite = np.isfinite(v) & np.isfinite(zs)
    if finite.sum() < 3:
        return np.full(z_out.shape, np.nan)
    return np.asarray(pchip_sample(v[finite][:, None], zs[finite], z_out, depth_axis=0)).reshape(-1)


def rama_skill(match: RamaMatch) -> dict[str, SkillReport]:
    reports: dict[str, SkillReport] = {}
    for name, df in ("model_to_rama", match.model_on_rama), ("rama_to_standard", match.rama_on_standard):
        if df.empty:
            continue
        reports[f"{name}_theta"] = depthwise_skill(
            df["temp_model"].to_numpy(), df["temp_rama"].to_numpy(), df["depth"].to_numpy()
        )
        reports[f"{name}_sp"] = depthwise_skill(
            df["sp_model"].to_numpy(), df["sp_rama"].to_numpy(), df["depth"].to_numpy()
        )
    return reports

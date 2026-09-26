"""ARGO co-location with Δt-stratified metrics and the z ≥ 5 m in-situ guard.

Rules (frozen SIH26066 blueprint)
---------------------------------
* Only QC flags 1 and 2.
* Primary match: same calendar day or |Δt| ≤ 12 h.
* Metrics reported separately for Δt = 0 d, ±1 d, ±2 d.
* In-situ evaluation begins at z ≥ 5 m (ARGO pump cutoff).
* 0 m reconstruction is scored against OISST, never against ARGO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from track3_validation_engine.constants import (
    ARGO_MIN_DEPTH_M,
    ARGO_QC_GOOD,
    DT_BINS_HOURS,
    STANDARD_DEPTHS,
    target_latitudes,
    target_longitudes,
)
from track3_validation_engine.diagnostics.pchip_profiler import pchip_sample
from track3_validation_engine.benchmarks.skill_scores import SkillReport, depthwise_skill

Array = NDArray[np.floating]


def _as_datetime64(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]")
    # Argo JULD: days since 1950-01-01
    epoch = np.datetime64("1950-01-01", "ns")
    days = values.astype(np.float64)
    return epoch + (days * 86_400_000_000_000).astype("timedelta64[ns]")


def _qc_is_good(flags: np.ndarray) -> np.ndarray:
    flags = np.asarray(flags)
    if flags.dtype.kind in {"U", "S", "O"}:
        decoded = np.array(
            [f.decode() if isinstance(f, (bytes, bytearray)) else str(f).strip() for f in flags.ravel()],
            dtype=object,
        ).reshape(flags.shape)
        return np.isin(decoded, ["1", "2"])
    return np.isin(flags, [1, 2])


def _dt_bin(abs_hours: float) -> str | None:
    for name, (lo, hi) in DT_BINS_HOURS.items():
        if lo <= abs_hours <= hi:
            return name
    return None


@dataclass
class ArgoProfile:
    time: np.datetime64
    lat: float
    lon: float
    depth: np.ndarray
    temp: np.ndarray
    psal: np.ndarray
    source: str


def read_argo_netcdf(path: Path | str) -> list[ArgoProfile]:
    """Read one Coriolis/INCOIS Argo profile file (multi-profile NetCDF supported)."""
    import xarray as xr

    path = Path(path)
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        lat = np.atleast_1d(ds["LATITUDE"].values.astype(np.float64))
        lon = np.atleast_1d(ds["LONGITUDE"].values.astype(np.float64))
        juld = np.atleast_1d(ds["JULD"].values)
        times = _as_datetime64(juld)
        pres = np.atleast_2d(ds["PRES"].values.astype(np.float64))
        temp = np.atleast_2d(ds["TEMP"].values.astype(np.float64))
        psal = np.atleast_2d(ds["PSAL"].values.astype(np.float64)) if "PSAL" in ds else np.full_like(temp, np.nan)
        t_qc = ds["TEMP_QC"].values if "TEMP_QC" in ds else np.full(temp.shape, "1")
        s_qc = ds["PSAL_QC"].values if "PSAL_QC" in ds else np.full(psal.shape, "1")
        p_qc = ds["PRES_QC"].values if "PRES_QC" in ds else np.full(pres.shape, "1")
        pos_qc = ds["POSITION_QC"].values if "POSITION_QC" in ds else np.full(lat.shape, "1")
        juld_qc = ds["JULD_QC"].values if "JULD_QC" in ds else np.full(lat.shape, "1")
    finally:
        ds.close()

    if pres.ndim == 1:
        pres = pres[None, :]
        temp = temp[None, :]
        psal = psal[None, :]
        t_qc = np.atleast_2d(t_qc)
        s_qc = np.atleast_2d(s_qc)
        p_qc = np.atleast_2d(p_qc)

    profiles: list[ArgoProfile] = []
    n_prof = lat.size
    for i in range(n_prof):
        if not bool(_qc_is_good(np.atleast_1d(pos_qc)[i])):
            continue
        if not bool(_qc_is_good(np.atleast_1d(juld_qc)[i])):
            continue
        good = _qc_is_good(np.atleast_1d(t_qc[i])) & _qc_is_good(np.atleast_1d(p_qc[i]))
        s_good = _qc_is_good(np.atleast_1d(s_qc[i]))
        depth = pres[i].copy()
        # PRES is dbar; near-surface 1 dbar ≈ 1 m. Keep hydrostatic metres.
        t = temp[i].copy()
        s = psal[i].copy()
        t[~good] = np.nan
        s[~s_good] = np.nan
        depth[~good] = np.nan
        if np.isfinite(t).sum() < 3:
            continue
        profiles.append(
            ArgoProfile(
                time=times[i],
                lat=float(lat[i]),
                lon=float(lon[i]),
                depth=depth,
                temp=t,
                psal=s,
                source=str(path),
            )
        )
    return profiles


def ingest_argo(paths: Iterable[Path | str]) -> list[ArgoProfile]:
    profiles: list[ArgoProfile] = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            files = sorted(p.glob("*.nc"))
        else:
            files = [p]
        for f in files:
            profiles.extend(read_argo_netcdf(f))
    return profiles


def _nearest_index(coord: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(coord - value)))


@dataclass
class ArgoMatchTable:
    rows: pd.DataFrame
    notes: dict[str, Any] = field(default_factory=dict)

    def by_dt(self, bin_name: str) -> pd.DataFrame:
        return self.rows[self.rows["dt_bin"] == bin_name]


def match_argo_to_model(
    profiles: Sequence[ArgoProfile],
    model_theta: Array,
    model_sp: Array,
    model_time: Array,
    *,
    lat: Array | None = None,
    lon: Array | None = None,
    model_depth: Array | None = None,
    max_radius_deg: float = 0.35,
) -> ArgoMatchTable:
    """Co-locate ARGO profiles with a model cube (time, z, y, x).

    Model depths are interpolated onto each float's observed levels with PCHIP.
    Samples shallower than 5 m are dropped (pump-cutoff guard).
    """
    lat = target_latitudes() if lat is None else np.asarray(lat, dtype=np.float64)
    lon = target_longitudes() if lon is None else np.asarray(lon, dtype=np.float64)
    z_model = STANDARD_DEPTHS if model_depth is None else np.asarray(model_depth, dtype=np.float64)
    times = np.asarray(model_time).astype("datetime64[ns]")
    theta = np.asarray(model_theta, dtype=np.float64)
    sp = np.asarray(model_sp, dtype=np.float64)
    if theta.shape[0] != times.size:
        raise ValueError("model_theta time axis must match model_time")

    records: list[dict[str, Any]] = []
    skipped_shallow = 0
    skipped_far = 0
    skipped_time = 0

    for prof in profiles:
        if not (lat.min() <= prof.lat <= lat.max() and lon.min() <= prof.lon <= lon.max()):
            skipped_far += 1
            continue
        iy = _nearest_index(lat, prof.lat)
        ix = _nearest_index(lon, prof.lon)
        if abs(lat[iy] - prof.lat) > max_radius_deg or abs(lon[ix] - prof.lon) > max_radius_deg:
            skipped_far += 1
            continue
        dt_ns = times.astype("datetime64[ns]").astype("int64") - np.datetime64(prof.time, "ns").astype("int64")
        it = int(np.argmin(np.abs(dt_ns)))
        abs_h = abs(dt_ns[it]) / 3.6e12
        bin_name = _dt_bin(abs_h)
        if bin_name is None:
            skipped_time += 1
            continue
        col_t = theta[it, :, iy, ix]
        col_s = sp[it, :, iy, ix]
        if np.isfinite(col_t).sum() < 3:
            continue
        good = np.isfinite(prof.depth) & np.isfinite(prof.temp) & (prof.depth >= ARGO_MIN_DEPTH_M)
        skipped_shallow += int(np.sum(np.isfinite(prof.temp) & (np.asarray(prof.depth) < ARGO_MIN_DEPTH_M)))
        if good.sum() == 0:
            continue
        z_obs = prof.depth[good]
        t_hat = np.asarray(pchip_sample(col_t[:, None], z_model, z_obs, depth_axis=0)).reshape(-1)
        s_hat = np.asarray(pchip_sample(col_s[:, None], z_model, z_obs, depth_axis=0)).reshape(-1)
        t_obs = prof.temp[good]
        s_obs = prof.psal[good]
        for k in range(z_obs.size):
            records.append(
                {
                    "time_argo": prof.time,
                    "time_model": times[it],
                    "lat": prof.lat,
                    "lon": prof.lon,
                    "depth": float(z_obs[k]),
                    "temp_argo": float(t_obs[k]),
                    "temp_model": float(t_hat[k]),
                    "sp_argo": float(s_obs[k]) if np.isfinite(s_obs[k]) else np.nan,
                    "sp_model": float(s_hat[k]),
                    "dt_hours": float(abs_h),
                    "dt_bin": bin_name,
                    "source": prof.source,
                }
            )

    table = pd.DataFrame.from_records(records)
    notes = {
        "in_situ_min_depth_m": ARGO_MIN_DEPTH_M,
        "zero_metre_policy": "ARGO never validates 0 m; use match_surface_to_oisst.",
        "skipped_far": skipped_far,
        "skipped_time": skipped_time,
        "skipped_shallow_samples": skipped_shallow,
        "n_pairs": int(len(table)),
    }
    return ArgoMatchTable(rows=table, notes=notes)


def argo_skill_by_dt(table: ArgoMatchTable) -> dict[str, dict[str, SkillReport]]:
    """Depth-wise RMSE / bias / r stratified by Δt bin, for temperature and salinity."""
    out: dict[str, dict[str, SkillReport]] = {}
    if table.rows.empty:
        return out
    for bin_name in DT_BINS_HOURS:
        sub = table.by_dt(bin_name)
        if sub.empty:
            continue
        out[bin_name] = {
            "theta": depthwise_skill(sub["temp_model"].to_numpy(), sub["temp_argo"].to_numpy(), sub["depth"].to_numpy()),
            "sp": depthwise_skill(
                sub["sp_model"].to_numpy(),
                sub["sp_argo"].to_numpy(),
                sub["depth"].to_numpy(),
            ),
        }
    return out


def match_surface_to_oisst(
    model_theta0: Array,
    oisst: Array,
    valid: Array | None = None,
) -> SkillReport:
    """Score the 0 m bulk reconstruction against OISST (skin/sub-skin).

    Documented limitation: skin–bulk difference is typically 0.1–0.3 °C.
    """
    pred = np.asarray(model_theta0, dtype=np.float64).ravel()
    obs = np.asarray(oisst, dtype=np.float64).ravel()
    if valid is not None:
        m = np.asarray(valid, dtype=bool).ravel()
        pred, obs = pred[m], obs[m]
    depth = np.zeros_like(pred)
    return depthwise_skill(pred, obs, depth)

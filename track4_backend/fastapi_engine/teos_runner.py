"""Vectorized TEOS-10 + TCHP/MLD/Z20/BLT/CIP for inference payloads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import STANDARD_DEPTHS
from .zarr_reader import latitudes, longitudes


@dataclass
class DerivedMaps:
    sa: np.ndarray
    ct: np.ndarray
    rho: np.ndarray
    tchp: np.ndarray
    mld: np.ndarray
    z20: np.ndarray
    blt: np.ndarray
    cip: np.ndarray
    teos_ms: float


def _cip(sst: np.ndarray, sla: np.ndarray, tchp: np.ndarray) -> np.ndarray:
    """Cyclone Intensification Potential in [0, 1] from SST, SLA and TCHP thresholds."""
    score = np.zeros(sst.shape, dtype=np.float32)
    score += (sst >= 28.0).astype(np.float32)
    score += (tchp >= 60.0).astype(np.float32)
    score += (sla >= 0.05).astype(np.float32)
    return score / 3.0


def _vectorized_z20(ct: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Find depth (in meters) where temperature drops to 20°C via linear interpolation."""
    n_depths, n_lat, n_lon = ct.shape
    z20 = np.full((n_lat, n_lon), 120.0, dtype=np.float32)
    for k in range(n_depths - 1):
        t1, t2 = ct[k], ct[k + 1]
        z1, z2 = depths[k], depths[k + 1]
        mask = (t1 >= 20.0) & (t2 < 20.0)
        frac = np.where(mask, (20.0 - t1) / (t2 - t1 + 1e-6), 0.0)
        interp_z = z1 + frac * (z2 - z1)
        z20[mask] = interp_z[mask]
    return z20


def _vectorized_mld(ct: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Find Mixed Layer Depth (ΔT = 0.2°C from surface)."""
    n_depths, n_lat, n_lon = ct.shape
    t_surf = ct[0]
    t_target = t_surf - 0.2
    mld = np.full((n_lat, n_lon), 40.0, dtype=np.float32)
    for k in range(n_depths - 1):
        t1, t2 = ct[k], ct[k + 1]
        z1, z2 = depths[k], depths[k + 1]
        mask = (t1 >= t_target) & (t2 < t_target)
        frac = np.where(mask, (t_target - t1) / (t2 - t1 + 1e-6), 0.0)
        interp_z = z1 + frac * (z2 - z1)
        mld[mask] = interp_z[mask]
    return mld


def _vectorized_tchp(ct: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Tropical Cyclone Heat Potential (kJ/cm²) using trapezoidal integration for T >= 26°C."""
    n_depths, n_lat, n_lon = ct.shape
    tchp = np.zeros((n_lat, n_lon), dtype=np.float32)
    cp_rho = 0.428245  # kJ / (cm² * m * °C)
    for k in range(n_depths - 1):
        t1, t2 = ct[k], ct[k + 1]
        z1, z2 = depths[k], depths[k + 1]
        dz = z2 - z1
        ex1 = np.clip(t1 - 26.0, 0, None)
        ex2 = np.clip(t2 - 26.0, 0, None)
        avg_ex = 0.5 * (ex1 + ex2)
        tchp += (avg_ex * dz * cp_rho).astype(np.float32)
    return tchp


def derive(
    theta: np.ndarray,
    sp: np.ndarray,
    sst: np.ndarray,
    sla: np.ndarray,
    land: np.ndarray,
    sigma_sp: np.ndarray | None = None,
) -> DerivedMaps:
    from time import perf_counter

    t0 = perf_counter()
    z = np.asarray(STANDARD_DEPTHS, dtype=np.float32)
    sa = (np.asarray(sp, dtype=np.float32) * (35.16504 / 35.0)).astype(np.float32)
    ct = np.asarray(theta, dtype=np.float32)
    rho = (1025.0 - 0.15 * (ct - 10.0) + 0.76 * (sa - 35.0)).astype(np.float32)

    tchp = _vectorized_tchp(ct, z)
    mld = _vectorized_mld(ct, z)
    z20 = _vectorized_z20(ct, z)
    blt = np.clip(mld - 10.0, 0, None).astype(np.float32)

    land_b = np.asarray(land, dtype=bool)
    tchp[land_b] = np.nan
    mld[land_b] = np.nan
    z20[land_b] = np.nan
    blt[land_b] = np.nan
    cip = _cip(np.asarray(sst, dtype=np.float32), np.asarray(sla, dtype=np.float32), tchp)
    cip[land_b] = np.nan

    return DerivedMaps(
        sa=sa,
        ct=ct,
        rho=rho,
        tchp=tchp,
        mld=mld,
        z20=z20,
        blt=blt,
        cip=cip,
        teos_ms=(perf_counter() - t0) * 1000.0,
    )


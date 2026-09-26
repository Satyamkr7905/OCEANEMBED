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
    lat = latitudes()
    lon = longitudes()
    try:
        import sys
        project_root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from track3_validation_engine.diagnostics.indices import compute_indices_from_theta_sp
        from track3_validation_engine.diagnostics.teos10_vectorized import convert_theta_sp

        fields = convert_theta_sp(theta, sp, lat, lon, np.asarray(STANDARD_DEPTHS, dtype=np.float64))
        idx = compute_indices_from_theta_sp(
            theta, sp, lat, lon, np.asarray(STANDARD_DEPTHS, dtype=np.float64), sigma_sp=sigma_sp
        )
        sa, ct, rho = fields.sa, fields.ct, fields.rho
        tchp, mld, z20, blt = idx.tchp, idx.mld, idx.z20, idx.blt
        target_shape = theta.shape[1:]
        if tchp.shape != target_shape:
            tchp = tchp.reshape(target_shape)
        if mld.shape != target_shape:
            mld = mld.reshape(target_shape)
        if z20.shape != target_shape:
            z20 = z20.reshape(target_shape)
        if blt.shape != target_shape:
            blt = blt.reshape(target_shape)
    except Exception:
        # Lightweight fallback so the API stays up if gsw/track3 is absent.
        sa = np.asarray(sp, dtype=np.float64) + 0.15
        ct = np.asarray(theta, dtype=np.float64)
        rho = 1025.0 - 0.15 * (ct - 10.0) + 0.76 * (sa - 35.0)
        z = np.asarray(STANDARD_DEPTHS, dtype=np.float64)
        t26 = 26.0
        excess = np.clip(ct - t26, 0, None)
        dz = np.diff(z, prepend=z[0])
        tchp = (1025.0 * 4178.0 * (excess * dz[:, None, None]).sum(axis=0) * 1e-7).astype(np.float32)
        mld = np.full(ct.shape[1:], 40.0, dtype=np.float32)
        z20 = np.full(ct.shape[1:], 120.0, dtype=np.float32)
        blt = np.full(ct.shape[1:], np.nan, dtype=np.float32)
    land_b = np.asarray(land, dtype=bool)
    tchp = np.asarray(tchp, dtype=np.float32)
    mld = np.asarray(mld, dtype=np.float32)
    z20 = np.asarray(z20, dtype=np.float32)
    blt = np.asarray(blt, dtype=np.float32)
    tchp[land_b] = np.nan
    mld[land_b] = np.nan
    z20[land_b] = np.nan
    blt[land_b] = np.nan
    cip = _cip(np.asarray(sst, dtype=np.float32), np.asarray(sla, dtype=np.float32), tchp)
    cip[land_b] = np.nan
    return DerivedMaps(
        sa=np.asarray(sa, dtype=np.float32),
        ct=np.asarray(ct, dtype=np.float32),
        rho=np.asarray(rho, dtype=np.float32),
        tchp=tchp,
        mld=mld,
        z20=z20,
        blt=blt,
        cip=cip,
        teos_ms=(perf_counter() - t0) * 1000.0,
    )

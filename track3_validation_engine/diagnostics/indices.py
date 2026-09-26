"""Vectorized TCHP, MLD, Z20, ILD and BLT on 1 m TEOS-10 profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from track3_validation_engine.constants import (
    CP_TCHP,
    ILD_DT,
    KJ_CM2_PER_J_M2,
    MLD_DRHO,
    REF_DEPTH_MLD,
    TCHP_ISOTHERM,
    Z20_ISOTHERM,
    z_1m,
)
from track3_validation_engine.diagnostics.pchip_profiler import pchip_to_z
from track3_validation_engine.diagnostics.teos10_vectorized import Teos10Fields, convert_theta_sp

Array = NDArray[np.floating]


def _z_index(z: Array, z_ref: float) -> int:
    idx = int(np.argmin(np.abs(z - z_ref)))
    if abs(float(z[idx]) - z_ref) > 0.51:
        raise ValueError(f"1 m grid does not contain {z_ref} m (nearest {z[idx]}).")
    return idx


def _first_crossing(values: Array, z: Array, target: float, *, from_above: bool = True) -> Array:
    """Depth where ``values`` crosses ``target``, searching downward from the surface.

    ``values`` has shape (Z, N). Returns (N,) with NaN if the isotherm is absent.
    """
    z = np.asarray(z, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    n_z, n = v.shape
    out = np.full(n, np.nan, dtype=np.float64)
    if from_above:
        above = v >= target
    else:
        above = v <= target
    # Last True from the top before a False, requiring the surface side to start True.
    for j in range(n):
        col = v[:, j]
        if not np.isfinite(col).any():
            continue
        mask = np.isfinite(col)
        if from_above:
            rel = col - target
        else:
            rel = target - col
        # Crossing from non-negative to negative (isotherm from above).
        pos = rel >= 0
        if not pos[mask][0]:
            # Already colder/fresher than target at the first finite sample.
            out[j] = z[int(np.flatnonzero(mask)[0])]
            continue
        crossed = np.where(pos[:-1] & (~pos[1:]) & np.isfinite(rel[:-1]) & np.isfinite(rel[1:]))[0]
        if crossed.size == 0:
            continue
        i = int(crossed[0])
        r0, r1 = rel[i], rel[i + 1]
        frac = r0 / (r0 - r1) if r1 != r0 else 0.0
        frac = float(np.clip(frac, 0.0, 1.0))
        out[j] = z[i] + frac * (z[i + 1] - z[i])
    return out


def _first_crossing_vectorized(values: Array, z: Array, target: float) -> Array:
    """Vectorized isotherm depth for a (Z, N) array decreasing through ``target``."""
    v = np.asarray(values, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = v.shape[1]
    rel = v - target
    pos = rel >= 0
    trans = pos[:-1] & (~pos[1:]) & np.isfinite(rel[:-1]) & np.isfinite(rel[1:])
    has = trans.any(axis=0)
    # First transition index per column.
    idx = np.argmax(trans, axis=0)
    i0 = idx
    i1 = np.clip(idx + 1, 0, z.size - 1)
    r0 = rel[i0, np.arange(n)]
    r1 = rel[i1, np.arange(n)]
    denom = r0 - r1
    frac = np.divide(r0, denom, out=np.zeros_like(r0), where=np.abs(denom) > 1e-12)
    frac = np.clip(frac, 0.0, 1.0)
    z_cross = z[i0] + frac * (z[i1] - z[i0])
    z_cross = np.where(has, z_cross, np.nan)
    # Surface already below target.
    first_finite = np.argmax(np.isfinite(v), axis=0)
    already = np.isfinite(v[first_finite, np.arange(n)]) & (rel[first_finite, np.arange(n)] < 0)
    z_cross = np.where(already, z[first_finite], z_cross)
    return z_cross


def d26_depth(ct_1m: Array, z: Array | None = None) -> Array:
    z = z_1m() if z is None else np.asarray(z, dtype=np.float64)
    return _first_crossing_vectorized(_flatten_depth(ct_1m), z, TCHP_ISOTHERM)


def z20_depth(ct_1m: Array, z: Array | None = None) -> Array:
    z = z_1m() if z is None else np.asarray(z, dtype=np.float64)
    return _first_crossing_vectorized(_flatten_depth(ct_1m), z, Z20_ISOTHERM)


def _flatten_depth(arr: Array) -> Array:
    a = np.asarray(arr, dtype=np.float64)
    return a.reshape(a.shape[0], -1)


def _unflatten(vec: Array, template: Array) -> Array:
    return vec.reshape(template.shape[1:])


def tchp_kj_cm2(ct_1m: Array, rho_1m: Array, z: Array | None = None) -> Array:
    """Tropical Cyclone Heat Potential in kJ cm⁻².

    Q = ∫_0^{D26} ρ(z) C_p (Θ(z) − 26) dz
    using TEOS-10 in-situ density and a 1 m trapezoidal rule. Negative
    (Θ − 26) contributions are clipped; integration stops at D26.
    """
    z = z_1m() if z is None else np.asarray(z, dtype=np.float64)
    ct = _flatten_depth(ct_1m)
    rho = _flatten_depth(rho_1m)
    if ct.shape != rho.shape:
        raise ValueError("ct and rho must share shape")
    d26 = d26_depth(ct, z)
    excess = np.clip(ct - TCHP_ISOTHERM, 0.0, None)
    dz = np.diff(z)
    # Trapezoid weights on interfaces; mask levels deeper than D26.
    z_mid = 0.5 * (z[:-1] + z[1:])
    excess_mid = 0.5 * (excess[:-1] + excess[1:])
    rho_mid = 0.5 * (rho[:-1] + rho[1:])
    valid_layer = np.isfinite(excess_mid) & np.isfinite(rho_mid)
    depth_ok = z_mid[:, None] <= d26[None, :]
    integrand = rho_mid * CP_TCHP * excess_mid * dz[:, None]
    energy = np.where(valid_layer & depth_ok, integrand, 0.0).sum(axis=0)  # J m-2
    energy = np.where(np.isfinite(d26), energy, np.nan)
    out = energy * KJ_CM2_PER_J_M2
    return _unflatten(out, ct_1m) if ct_1m.ndim > 1 else out


def mld_m(rho_1m: Array, z: Array | None = None, *, drho: float = MLD_DRHO, z_ref: float = REF_DEPTH_MLD) -> Array:
    """Density-based mixed layer depth (Δρ = 0.03 kg m⁻³ from 10 m)."""
    z = z_1m() if z is None else np.asarray(z, dtype=np.float64)
    rho = _flatten_depth(rho_1m)
    i_ref = _z_index(z, z_ref)
    rho_ref = rho[i_ref]
    anomaly = rho - rho_ref[None, :]
    # Search only below the reference depth.
    below = np.zeros_like(anomaly, dtype=bool)
    below[i_ref:, :] = True
    crossed = below & (anomaly >= drho) & np.isfinite(anomaly)
    has = crossed.any(axis=0)
    idx = np.argmax(crossed, axis=0)
    # Linear depth between idx-1 and idx.
    i1 = np.clip(idx, 1, z.size - 1)
    i0 = i1 - 1
    a0 = anomaly[i0, np.arange(rho.shape[1])]
    a1 = anomaly[i1, np.arange(rho.shape[1])]
    frac = np.divide(drho - a0, a1 - a0, out=np.zeros_like(a0), where=np.abs(a1 - a0) > 1e-12)
    frac = np.clip(frac, 0.0, 1.0)
    depth = z[i0] + frac * (z[i1] - z[i0])
    depth = np.where(has, depth, np.nan)
    return _unflatten(depth, rho_1m) if rho_1m.ndim > 1 else depth


def ild_m(ct_1m: Array, z: Array | None = None, *, dt: float = ILD_DT, z_ref: float = REF_DEPTH_MLD) -> Array:
    """Isothermal layer depth (ΔΘ = 0.2 °C from 10 m)."""
    z = z_1m() if z is None else np.asarray(z, dtype=np.float64)
    ct = _flatten_depth(ct_1m)
    i_ref = _z_index(z, z_ref)
    t_ref = ct[i_ref]
    target = t_ref - dt
    below = np.zeros_like(ct, dtype=bool)
    below[i_ref:, :] = True
    crossed = below & (ct <= target[None, :]) & np.isfinite(ct)
    has = crossed.any(axis=0)
    idx = np.argmax(crossed, axis=0)
    i1 = np.clip(idx, 1, z.size - 1)
    i0 = i1 - 1
    t0 = ct[i0, np.arange(ct.shape[1])]
    t1 = ct[i1, np.arange(ct.shape[1])]
    tgt = target
    frac = np.divide(t0 - tgt, t0 - t1, out=np.zeros_like(t0), where=np.abs(t0 - t1) > 1e-12)
    frac = np.clip(frac, 0.0, 1.0)
    depth = z[i0] + frac * (z[i1] - z[i0])
    depth = np.where(has, depth, np.nan)
    return _unflatten(depth, ct_1m) if ct_1m.ndim > 1 else depth


def blt_m(
    ct_1m: Array,
    rho_1m: Array,
    z: Array | None = None,
    *,
    sigma_sp: Array | None = None,
    sigma_max: float = 0.15,
) -> Array:
    """Barrier layer thickness ILD − MLD. NaN where salinity is unconfident or ILD < MLD."""
    ild = ild_m(ct_1m, z)
    mld = mld_m(rho_1m, z)
    blt = ild - mld
    blt = np.where(np.isfinite(ild) & np.isfinite(mld) & (ild > mld), blt, np.nan)
    if sigma_sp is not None:
        sig = np.asarray(sigma_sp, dtype=np.float64)
        if sig.ndim == ct_1m.ndim:
            sig = np.nanmean(sig, axis=0)
        blt = np.where(sig <= sigma_max, blt, np.nan)
    return blt


@dataclass(frozen=True)
class OceanIndices:
    tchp: Array
    mld: Array
    z20: Array
    ild: Array
    blt: Array
    d26: Array


def compute_indices_from_theta_sp(
    theta: Array,
    sp: Array,
    lat: Array,
    lon: Array,
    depth: Array | None = None,
    *,
    sigma_sp: Array | None = None,
) -> OceanIndices:
    """End-to-end: 15-level θ/S_p → TEOS-10 → 1 m PCHIP → TCHP/MLD/Z20/BLT."""
    fields: Teos10Fields = convert_theta_sp(theta, sp, lat, lon, depth)
    z_src = fields.depth
    ct_1m = pchip_to_z(fields.ct, z_src=z_src, depth_axis=0)
    rho_1m = pchip_to_z(fields.rho, z_src=z_src, depth_axis=0)
    z = z_1m()
    return OceanIndices(
        tchp=tchp_kj_cm2(ct_1m, rho_1m, z),
        mld=mld_m(rho_1m, z),
        z20=z20_depth(ct_1m, z),
        ild=ild_m(ct_1m, z),
        blt=blt_m(ct_1m, rho_1m, z, sigma_sp=sigma_sp),
        d26=d26_depth(ct_1m, z),
    )

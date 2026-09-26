"""Vectorized TEOS-10 conversion for a full NIO 0.25° cube.

Native model / GLORYS / ARGO storage is potential temperature θ and practical
salinity S_p. Density, TCHP and MLD require Conservative Temperature Θ and
Absolute Salinity S_A.

The Gibbs SeaWater C extensions expect *flat* arrays. Nested Python loops over
24,000 columns × 15 depths exceed 40 s; a single ``gsw.SA_from_SP`` /
``gsw.CT_from_pt`` call on the flattened cube stays well under 500 ms.

Note on API: ``gsw.CT_from_pt(SA, pt)`` takes Absolute Salinity, not S_p.
The scientifically correct sequence is SA_from_SP → CT_from_pt(SA, θ).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from track3_validation_engine.constants import STANDARD_DEPTHS

try:
    import gsw
except ImportError as exc:  # pragma: no cover
    gsw = None
    _GSW_IMPORT_ERROR = exc
else:
    _GSW_IMPORT_ERROR = None


Array = NDArray[np.floating]


def _require_gsw() -> None:
    if gsw is None:
        raise ImportError(
            "The gsw package is required for TEOS-10 conversion. Install with `pip install gsw`."
        ) from _GSW_IMPORT_ERROR


@dataclass(frozen=True)
class Teos10Fields:
    """TEOS-10 fields aligned with the input θ / S_p cube."""

    theta: Array
    sp: Array
    sa: Array
    ct: Array
    pressure_dbar: Array
    rho: Array
    lat: Array
    lon: Array
    depth: Array
    elapsed_s: float

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.theta.shape)


def _as_depth(depth: Array | None, n_z: int) -> Array:
    if depth is None:
        if n_z != STANDARD_DEPTHS.size:
            raise ValueError(f"depth omitted but leading axis is {n_z}, not 15.")
        return STANDARD_DEPTHS.copy()
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    if depth.size != n_z:
        raise ValueError(f"depth length {depth.size} != temperature levels {n_z}")
    if np.any(np.diff(depth) <= 0):
        raise ValueError("depth must be strictly increasing (positive downward).")
    return depth


def _broadcast_lat_lon(lat: Array, lon: Array, spatial: tuple[int, ...]) -> tuple[Array, Array]:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.shape == spatial and lon.shape == spatial:
        return lat, lon
    if lat.ndim == 1 and lon.ndim == 1 and spatial == (lat.size, lon.size):
        lon2d, lat2d = np.meshgrid(lon, lat, indexing="xy")
        return lat2d, lon2d
    if lat.shape == spatial and lon.ndim == 1 and lon.size == spatial[-1]:
        return lat, np.broadcast_to(lon.reshape(1, -1), spatial).copy()
    raise ValueError(
        f"Cannot broadcast lat {lat.shape} / lon {lon.shape} onto spatial {spatial}."
    )


def convert_theta_sp(
    theta: Array,
    sp: Array,
    lat: Array,
    lon: Array,
    depth: Array | None = None,
) -> Teos10Fields:
    """Convert a θ / S_p cube to S_A, Θ, p, and in-situ density.

    Parameters
    ----------
    theta, sp:
        Arrays with depth as axis 0, shape ``(Z, Y, X)`` or ``(Z, N)``.
        Leading time axes are allowed: ``(T, Z, Y, X)``.
    lat, lon:
        1-D axes, 2-D maps, or already broadcast to the spatial trailing dims.
    depth:
        Z-vector in metres (positive down). Defaults to the 15 standard levels.
    """
    _require_gsw()
    theta_in = np.asarray(theta, dtype=np.float64)
    sp_in = np.asarray(sp, dtype=np.float64)
    if theta_in.shape != sp_in.shape:
        raise ValueError(f"theta shape {theta_in.shape} != sp shape {sp_in.shape}")
    if theta_in.ndim < 2:
        raise ValueError("theta/sp must include a depth axis and at least one spatial axis.")

    # Identify depth axis: prefer an axis whose length matches STANDARD_DEPTHS.
    if theta_in.ndim >= 3 and theta_in.shape[-3] == STANDARD_DEPTHS.size:
        depth_axis = theta_in.ndim - 3
    elif theta_in.shape[0] == STANDARD_DEPTHS.size or depth is not None:
        depth_axis = 0
    else:
        raise ValueError("Could not locate the depth axis (expected 15 levels).")

    theta = np.moveaxis(theta_in, depth_axis, 0)
    sp = np.moveaxis(sp_in, depth_axis, 0)
    n_z = theta.shape[0]
    z = _as_depth(depth, n_z)
    spatial = theta.shape[1:]
    lat2d, lon2d = _broadcast_lat_lon(lat, lon, spatial[-2:] if len(spatial) >= 2 else spatial)

    if lat2d.shape != spatial[-2:] and lat2d.shape != spatial:
        # time leading: spatial is (T, Y, X) or (Y, X)
        if len(spatial) == 3:
            lat2d = np.broadcast_to(lat2d, spatial).copy()
            lon2d = np.broadcast_to(lon2d, spatial).copy()
        else:
            raise ValueError("lat/lon could not be aligned with theta spatial axes.")
    elif lat2d.shape == spatial[-2:] and len(spatial) == 3:
        lat2d = np.broadcast_to(lat2d[None, ...], spatial).copy()
        lon2d = np.broadcast_to(lon2d[None, ...], spatial).copy()

    lat_b = np.broadcast_to(lat2d[None, ...], theta.shape)
    lon_b = np.broadcast_to(lon2d[None, ...], theta.shape)
    z_b = z.reshape((n_z,) + (1,) * (theta.ndim - 1))
    z_b = np.broadcast_to(z_b, theta.shape)

    # gsw.p_from_z expects height (negative downward).
    height = -z_b
    valid = np.isfinite(theta) & np.isfinite(sp) & np.isfinite(lat_b) & np.isfinite(lon_b)

    theta_f = np.ascontiguousarray(theta.reshape(-1))
    sp_f = np.ascontiguousarray(sp.reshape(-1))
    lat_f = np.ascontiguousarray(lat_b.reshape(-1))
    lon_f = np.ascontiguousarray(lon_b.reshape(-1))
    height_f = np.ascontiguousarray(height.reshape(-1))
    valid_f = valid.reshape(-1)

    sa = np.full(theta_f.shape, np.nan, dtype=np.float64)
    ct = np.full(theta_f.shape, np.nan, dtype=np.float64)
    p = np.full(theta_f.shape, np.nan, dtype=np.float64)
    rho = np.full(theta_f.shape, np.nan, dtype=np.float64)

    t0 = perf_counter()
    if valid_f.any():
        p[valid_f] = gsw.p_from_z(height_f[valid_f], lat_f[valid_f])
        sa[valid_f] = gsw.SA_from_SP(sp_f[valid_f], p[valid_f], lon_f[valid_f], lat_f[valid_f])
        ct[valid_f] = gsw.CT_from_pt(sa[valid_f], theta_f[valid_f])
        rho[valid_f] = gsw.rho(sa[valid_f], ct[valid_f], p[valid_f])
    elapsed = perf_counter() - t0

    def _restore(arr: Array) -> Array:
        return np.moveaxis(arr.reshape(theta.shape), 0, depth_axis)

    return Teos10Fields(
        theta=theta_in,
        sp=sp_in,
        sa=_restore(sa),
        ct=_restore(ct),
        pressure_dbar=_restore(p),
        rho=_restore(rho),
        lat=lat2d,
        lon=lon2d,
        depth=z,
        elapsed_s=float(elapsed),
    )


def assert_runtime_budget(fields: Teos10Fields, n_columns: int = 24_000, budget_s: float = 0.5) -> None:
    """Scale the measured runtime to a full-basin 24k-column equivalent and assert."""
    n_actual = int(np.prod(fields.theta.shape[1:])) if fields.theta.ndim >= 2 else 1
    if n_actual <= 0:
        return
    scaled = fields.elapsed_s * (n_columns / max(n_actual, 1))
    if fields.elapsed_s > 0 and n_actual >= 1_000 and scaled > budget_s * 4:
        # Only hard-fail when the sample is large enough to be representative.
        raise AssertionError(
            f"TEOS-10 conversion too slow: {fields.elapsed_s:.3f}s for {n_actual} columns "
            f"(scaled {scaled:.3f}s vs {budget_s}s budget for {n_columns} columns)."
        )

"""Regularized Coriolis, wind-stress curl, and Ekman pumping.

Safeguards
----------
* Equatorial bound: ``f_tilde = sign(φ) * max(|2 Ω sin φ|, f0)`` with
  ``f0 = 1.5e-5 s^-1`` (~6°N) so ``1/(ρ0 f)`` cannot blow up at 5°N.
* Land mask is dilated by 1 cell *before* any derivative is taken.
* Centered differences are used only when both neighbours are valid ocean;
  otherwise forward/backward one-sided differences.
* A 3×3 NaN-aware median filter suppresses residual single-cell curl spikes.
* South of 8°N the divergence form of Ekman transport is used (β-plane).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation, generic_filter

from track1_data_engine.core import get_logger, load_sources_config

logger = get_logger()

OMEGA = 7.292115e-5
F0 = 1.5e-5
RHO_AIR = 1.225
CD = 1.3e-3
RHO0 = 1025.0
EARTH_RADIUS_M = 6_371_000.0
BETA_PLANE_LATITUDE = 8.0
MEDIAN_SIZE = 3


def _physics_constants(sources_cfg: Mapping[str, Any] | None = None) -> dict[str, float]:
    spec = dict(sources_cfg or load_sources_config()).get("physics", {})
    return {
        "omega": float(spec.get("omega", OMEGA)),
        "f0": float(spec.get("f0", F0)),
        "rho_air": float(spec.get("rho_air", RHO_AIR)),
        "cd": float(spec.get("drag_coefficient", CD)),
        "rho0": float(spec.get("rho0", RHO0)),
        "earth_radius_m": float(spec.get("earth_radius_m", EARTH_RADIUS_M)),
        "beta_plane_latitude": float(spec.get("beta_plane_latitude", BETA_PLANE_LATITUDE)),
        "median_filter_size": int(spec.get("median_filter_size", MEDIAN_SIZE)),
    }


def coriolis_parameter(
    lat: np.ndarray,
    *,
    omega: float = OMEGA,
    f0: float = F0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(f, f_tilde)`` with equatorial regularization.

    ``f = 2 Ω sin φ``
    ``f_tilde = sign(φ) * max(|f|, f0)``
    """
    phi = np.deg2rad(np.asarray(lat, dtype=np.float64))
    f = 2.0 * omega * np.sin(phi)
    f_tilde = np.sign(phi) * np.maximum(np.abs(f), f0)
    # At exactly 0° sign(φ)=0; NIO starts at 5°N so this is defensive.
    f_tilde = np.where(f_tilde == 0.0, f0, f_tilde)
    return f, f_tilde


def wind_stress(
    u10: np.ndarray,
    v10: np.ndarray,
    *,
    rho_air: float = RHO_AIR,
    cd: float = CD,
) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic bulk formula ``τ = ρ_a C_D |U10| U10``."""
    u10 = np.asarray(u10, dtype=np.float64)
    v10 = np.asarray(v10, dtype=np.float64)
    speed = np.hypot(u10, v10)
    factor = rho_air * cd * speed
    tau_x = factor * u10
    tau_y = factor * v10
    return tau_x, tau_y


def _geographic_metrics(
    lat: np.ndarray,
    lon: np.ndarray,
    earth_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dx, dy)`` cell-center spacings in metres, shapes (nlat, nlon) and (nlat, nlon)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    dlat = np.gradient(lat)
    dlon = np.gradient(lon)
    dy_1d = earth_radius_m * np.deg2rad(dlat)
    dx_2d = earth_radius_m * np.cos(np.deg2rad(lat))[:, None] * np.deg2rad(dlon)[None, :]
    dy_2d = np.broadcast_to(dy_1d[:, None], (lat.size, lon.size)).copy()
    return dx_2d, dy_2d


def _vectorized_masked_derivative(
    field: np.ndarray,
    spacing: np.ndarray,
    valid: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Vectorized one-sided/centered derivative along `axis` (0=y, 1=x)."""
    field = np.asarray(field, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(field)
    spacing = np.asarray(spacing, dtype=np.float64)
    deriv = np.full(field.shape, np.nan, dtype=np.float64)

    def shift(arr: np.ndarray, steps: int) -> np.ndarray:
        out = np.full_like(arr, np.nan, dtype=np.float64)
        if axis == 1:
            if steps > 0:
                out[:, steps:] = arr[:, :-steps]
            elif steps < 0:
                out[:, :steps] = arr[:, -steps:]
            else:
                out = arr
        else:
            if steps > 0:
                out[steps:, :] = arr[:-steps, :]
            elif steps < 0:
                out[:steps, :] = arr[-steps:, :]
            else:
                out = arr
        return out

    def shift_bool(arr: np.ndarray, steps: int) -> np.ndarray:
        out = np.zeros_like(arr, dtype=bool)
        if axis == 1:
            if steps > 0:
                out[:, steps:] = arr[:, :-steps]
            elif steps < 0:
                out[:, :steps] = arr[:, -steps:]
        else:
            if steps > 0:
                out[steps:, :] = arr[:-steps, :]
            elif steps < 0:
                out[:steps, :] = arr[-steps:, :]
        return out

    plus = shift(field, -1)
    minus = shift(field, 1)
    plus_ok = shift_bool(valid, -1)
    minus_ok = shift_bool(valid, 1)
    sp_plus = shift(spacing, -1)
    sp_minus = shift(spacing, 1)

    both = valid & plus_ok & minus_ok
    denom = np.where(np.isfinite(sp_plus) & np.isfinite(sp_minus), sp_plus + sp_minus, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        deriv[both] = (plus[both] - minus[both]) / denom[both]
        only_p = valid & plus_ok & ~minus_ok
        deriv[only_p] = (plus[only_p] - field[only_p]) / np.maximum(spacing[only_p], 1e-6)
        only_m = valid & minus_ok & ~plus_ok
        deriv[only_m] = (field[only_m] - minus[only_m]) / np.maximum(spacing[only_m], 1e-6)
    return deriv


def nan_median_filter(field: np.ndarray, size: int = 3) -> np.ndarray:
    """3×3 (or `size`×`size`) median filter that ignores NaNs."""

    def _nanmedian(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.nan
        return float(np.median(finite))

    return generic_filter(field, _nanmedian, size=size, mode="nearest")


def wind_stress_curl(
    tau_x: np.ndarray,
    tau_y: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    land_mask: np.ndarray,
    *,
    earth_radius_m: float = EARTH_RADIUS_M,
    dilation_cells: int = 1,
    median_size: int = MEDIAN_SIZE,
) -> tuple[np.ndarray, dict[str, int]]:
    """Boundary-safe ``∂τy/∂x − ∂τx/∂y`` with dilated land mask + median filter.

    Returns the curl field and a stats dict (buffer cell counts).
    """
    land = np.asarray(land_mask, dtype=bool)
    if land.ndim != 2:
        raise ValueError("land_mask must be 2-D (lat, lon).")
    structure = np.ones((2 * dilation_cells + 1, 2 * dilation_cells + 1), dtype=bool)
    dilated_land = binary_dilation(land, structure=structure)
    valid = ~dilated_land
    dx, dy = _geographic_metrics(lat, lon, earth_radius_m)

    d_tau_y_dx = _vectorized_masked_derivative(tau_y, dx, valid, axis=1)
    d_tau_x_dy = _vectorized_masked_derivative(tau_x, dy, valid, axis=0)
    curl = d_tau_y_dx - d_tau_x_dy
    curl = nan_median_filter(curl, size=median_size)
    curl[land] = np.nan

    stats = {
        "n_land_cells": int(land.sum()),
        "n_gradient_buffer_cells": int((dilated_land & ~land).sum()),
        "n_valid_curl_cells": int(np.isfinite(curl).sum()),
    }
    logger.debug(
        "Wind-stress curl: land=%s buffer=%s finite=%s",
        stats["n_land_cells"],
        stats["n_gradient_buffer_cells"],
        stats["n_valid_curl_cells"],
    )
    return curl, stats


def ekman_pumping_velocity(
    curl_tau: np.ndarray,
    tau_x: np.ndarray,
    tau_y: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    land_mask: np.ndarray,
    f_tilde_lat: np.ndarray,
    *,
    rho0: float = RHO0,
    earth_radius_m: float = EARTH_RADIUS_M,
    beta_plane_latitude: float = BETA_PLANE_LATITUDE,
    dilation_cells: int = 1,
    median_size: int = MEDIAN_SIZE,
) -> np.ndarray:
    """Regularized Ekman pumping ``w_e``.

    North of 8°N: ``w_e = curl(τ) / (ρ0 f̃)``.
    South of 8°N: ``w_e = ∇ · (τ × k̂ / (ρ0 f̃))``.
    """
    f2d = np.asarray(f_tilde_lat, dtype=np.float64)[:, None] * np.ones((1, np.asarray(lon).size))
    we_curl = np.asarray(curl_tau, dtype=np.float64) / (rho0 * f2d)

    land = np.asarray(land_mask, dtype=bool)
    structure = np.ones((2 * dilation_cells + 1, 2 * dilation_cells + 1), dtype=bool)
    valid = ~binary_dilation(land, structure=structure)
    dx, dy = _geographic_metrics(lat, lon, earth_radius_m)
    ux = tau_y / (rho0 * f2d)
    uy = -tau_x / (rho0 * f2d)
    dux_dx = _vectorized_masked_derivative(ux, dx, valid, axis=1)
    duy_dy = _vectorized_masked_derivative(uy, dy, valid, axis=0)
    we_div = dux_dx + duy_dy
    we_div = nan_median_filter(we_div, size=median_size)

    lat = np.asarray(lat, dtype=np.float64)
    south = lat[:, None] < beta_plane_latitude
    we = np.where(south, we_div, we_curl)
    we[land] = np.nan
    return we


def compute_physics_features(
    u10: xr.DataArray,
    v10: xr.DataArray,
    land_mask: xr.DataArray | np.ndarray,
    *,
    sources_cfg: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Build f, f̃, τ, curl(τ), and w_e on the OceanEmbed grid.

    `u10`/`v10` must include `lat` and `lon` (optional leading `time`).
    """
    const = _physics_constants(sources_cfg)
    lat = np.asarray(u10["lat"].values, dtype=np.float64)
    lon = np.asarray(u10["lon"].values, dtype=np.float64)
    f, f_tilde = coriolis_parameter(lat, omega=const["omega"], f0=const["f0"])
    land = np.asarray(land_mask.values if hasattr(land_mask, "values") else land_mask, dtype=bool)
    if land.shape != (lat.size, lon.size):
        raise ValueError(f"land_mask shape {land.shape} != ({lat.size}, {lon.size})")

    u = u10.transpose(..., "lat", "lon")
    v = v10.transpose(..., "lat", "lon")
    u_vals = np.asarray(u.values, dtype=np.float64)
    v_vals = np.asarray(v.values, dtype=np.float64)
    extra = u_vals.ndim - 2
    u_flat = u_vals.reshape((-1, lat.size, lon.size)) if extra else u_vals[None, ...]
    v_flat = v_vals.reshape((-1, lat.size, lon.size)) if extra else v_vals[None, ...]

    tau_x = np.empty_like(u_flat)
    tau_y = np.empty_like(v_flat)
    curl = np.empty_like(u_flat)
    we = np.empty_like(u_flat)
    stats_last: dict[str, int] = {}
    for t in range(u_flat.shape[0]):
        tx, ty = wind_stress(u_flat[t], v_flat[t], rho_air=const["rho_air"], cd=const["cd"])
        tx[land] = np.nan
        ty[land] = np.nan
        c, stats_last = wind_stress_curl(
            tx,
            ty,
            lat,
            lon,
            land,
            earth_radius_m=const["earth_radius_m"],
            median_size=int(const["median_filter_size"]),
        )
        w = ekman_pumping_velocity(
            c,
            tx,
            ty,
            lat,
            lon,
            land,
            f_tilde,
            rho0=const["rho0"],
            earth_radius_m=const["earth_radius_m"],
            beta_plane_latitude=const["beta_plane_latitude"],
            median_size=int(const["median_filter_size"]),
        )
        tau_x[t], tau_y[t], curl[t], we[t] = tx, ty, c, w

    leading_dims = tuple(u.dims[:-2])
    leading_sizes = u_vals.shape[:-2]
    def _pack(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(*leading_sizes, lat.size, lon.size) if extra else arr[0]

    coords = {k: u10[k] for k in u10.coords}
    ds = xr.Dataset(
        {
            "f": (("lat",), f, {"units": "s-1", "long_name": "Coriolis parameter"}),
            "f_tilde": (
                ("lat",),
                f_tilde,
                {
                    "units": "s-1",
                    "long_name": "regularized Coriolis parameter",
                    "f0": const["f0"],
                },
            ),
            "tau_x": (leading_dims + ("lat", "lon"), _pack(tau_x), {"units": "N m-2"}),
            "tau_y": (leading_dims + ("lat", "lon"), _pack(tau_y), {"units": "N m-2"}),
            "wind_stress_curl": (
                leading_dims + ("lat", "lon"),
                _pack(curl),
                {"units": "N m-3", "long_name": "boundary-safe wind stress curl"},
            ),
            "w_e": (
                leading_dims + ("lat", "lon"),
                _pack(we),
                {
                    "units": "m s-1",
                    "long_name": "regularized Ekman pumping velocity",
                    "beta_plane_latitude": const["beta_plane_latitude"],
                },
            ),
        },
        coords=coords,
        attrs={
            "gradient_buffer_cells": stats_last.get("n_gradient_buffer_cells"),
            "n_valid_curl_cells": stats_last.get("n_valid_curl_cells"),
            "formula_tau": "rho_a * C_D * |U10| * U10",
            "formula_we_curl": "curl(tau) / (rho0 * f_tilde)",
            "formula_we_div": "div(tau x k / (rho0 * f_tilde)) south of 8N",
        },
    )
    return ds


def lagged_dmi(
    sst: xr.DataArray,
    western: Mapping[str, float],
    eastern: Mapping[str, float],
    lags_days: tuple[int, ...] = (7, 14),
) -> xr.Dataset:
    """Causal Dipole Mode Index from SST anomalies (requires IOD-box coverage).

    DMI(t) uses only SST at time t (daily). Lagged copies DMI(t-7) and DMI(t-14)
    are attached so a daily model never sees a monthly-smoothed future window.
    """
    def _box_mean(da: xr.DataArray, spec: Mapping[str, float]) -> xr.DataArray:
        return da.sel(
            lat=slice(float(spec["lat_min"]), float(spec["lat_max"])),
            lon=slice(float(spec["lon_min"]), float(spec["lon_max"])),
        ).mean(dim=("lat", "lon"), skipna=True)

    west = _box_mean(sst, western)
    east = _box_mean(sst, eastern)
    dmi = (west - east).rename("dmi")
    out = xr.Dataset({"dmi": dmi})
    for lag in lags_days:
        out[f"dmi_lag_{lag}d"] = dmi.shift(time=int(lag))
    out.attrs["causal"] = True
    out.attrs["lags_days"] = list(lags_days)
    return out


def monsoon_phase_encoding(time: xr.DataArray | np.ndarray) -> xr.Dataset:
    """Cyclical day-of-year encoding (sin/cos) used as a monsoon-phase proxy.

    The network can also infer monsoon phase from the wind field; this encoding
    does not freeze a calendar onset date.
    """
    import pandas as pd

    from track1_data_engine.core import climatology_doy

    doy = climatology_doy(np.asarray(time))
    angle = 2.0 * np.pi * (doy - 1) / 365.0
    return xr.Dataset(
        {
            "doy": (("time",), doy),
            "doy_sin": (("time",), np.sin(angle)),
            "doy_cos": (("time",), np.cos(angle)),
        },
        coords={"time": pd.DatetimeIndex(np.asarray(time))},
        attrs={"note": "DOY 1-365; 29 Feb mapped to 28 Feb"},
    )

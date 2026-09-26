"""Depth-wise RMSE, bias, Pearson r, climatology skill score, and inversion Brier."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from track3_validation_engine.constants import STANDARD_DEPTHS

Array = NDArray[np.floating]


@dataclass
class SkillReport:
    name: str
    depths: list[float]
    n: list[int]
    rmse: list[float]
    bias: list[float]
    pearson_r: list[float]
    rmse_all: float
    bias_all: float
    pearson_all: float
    css: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def _finite_pair(pred: Array, obs: Array) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    obs = np.asarray(obs, dtype=np.float64).ravel()
    if pred.shape != obs.shape:
        raise ValueError(f"pred {pred.shape} != obs {obs.shape}")
    mask = np.isfinite(pred) & np.isfinite(obs)
    return pred[mask], obs[mask]


def rmse(pred: Array, obs: Array) -> float:
    p, o = _finite_pair(pred, obs)
    if p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - o) ** 2)))


def mean_bias(pred: Array, obs: Array) -> float:
    p, o = _finite_pair(pred, obs)
    if p.size == 0:
        return float("nan")
    return float(np.mean(p - o))


def pearson_r(pred: Array, obs: Array) -> float:
    p, o = _finite_pair(pred, obs)
    if p.size < 3:
        return float("nan")
    p = p - p.mean()
    o = o - o.mean()
    denom = float(np.sqrt(np.sum(p**2) * np.sum(o**2)))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(p * o) / denom)


def climatology_skill_score(mse_model: float, mse_clim: float) -> float:
    """CSS = 1 − MSE_model / MSE_clim. Predicting climatology scores exactly 0."""
    if not np.isfinite(mse_model) or not np.isfinite(mse_clim) or mse_clim <= 0:
        return float("nan")
    return float(1.0 - mse_model / mse_clim)


def mse(pred: Array, obs: Array) -> float:
    p, o = _finite_pair(pred, obs)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - o) ** 2))


def brier_score(probability: Array, binary_truth: Array) -> float:
    """Brier score for inversion (or any binary) events. Perfect = 0."""
    p, y = _finite_pair(probability, binary_truth)
    if p.size == 0:
        return float("nan")
    y = np.clip(y, 0.0, 1.0)
    p = np.clip(p, 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def inversion_flag(theta: Array, depth_axis: int = 0, *, eps: float = 0.05) -> Array:
    """True where temperature increases with depth by more than ``eps`` °C."""
    t = np.asarray(theta, dtype=np.float64)
    dt = np.diff(t, axis=depth_axis)
    return dt > eps


def inversion_brier(model_theta: Array, obs_theta: Array, *, eps: float = 0.05) -> float:
    """Brier score of false/missed inversions, column-wise (any illegal dT/dz)."""
    m = inversion_flag(model_theta, eps=eps)
    o = inversion_flag(obs_theta, eps=eps)
    # Reduce over depth: event if any inversion in the column.
    if m.ndim == 1:
        p = np.array([float(m.any())])
        y = np.array([float(o.any())])
    else:
        p = m.any(axis=0).astype(np.float64).ravel()
        y = o.any(axis=0).astype(np.float64).ravel()
    return brier_score(p, y)


def _bin_to_standard(depth: Array) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    idx = np.abs(depth[:, None] - STANDARD_DEPTHS[None, :]).argmin(axis=1)
    return STANDARD_DEPTHS[idx]


def depthwise_skill(
    pred: Array,
    obs: Array,
    depth: Array,
    *,
    clim: Array | None = None,
    name: str = "skill",
) -> SkillReport:
    """RMSE, bias and r grouped by nearest standard depth, plus all-level totals."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    obs = np.asarray(obs, dtype=np.float64).ravel()
    depth = np.asarray(depth, dtype=np.float64).ravel()
    if not (pred.size == obs.size == depth.size):
        raise ValueError("pred, obs and depth must have the same length")
    bins = _bin_to_standard(depth)
    depths: list[float] = []
    ns: list[int] = []
    rmses: list[float] = []
    biases: list[float] = []
    rs: list[float] = []
    for z in STANDARD_DEPTHS:
        sel = bins == z
        depths.append(float(z))
        p, o = pred[sel], obs[sel]
        ns.append(int(np.isfinite(p).sum() if p.size else 0))
        rmses.append(rmse(p, o))
        biases.append(mean_bias(p, o))
        rs.append(pearson_r(p, o))
    css_val = None
    if clim is not None:
        css_val = climatology_skill_score(mse(pred, obs), mse(np.asarray(clim).ravel(), obs))
    return SkillReport(
        name=name,
        depths=depths,
        n=ns,
        rmse=rmses,
        bias=biases,
        pearson_r=rs,
        rmse_all=rmse(pred, obs),
        bias_all=mean_bias(pred, obs),
        pearson_all=pearson_r(pred, obs),
        css=css_val,
    )


def cube_skill(
    pred: Array,
    obs: Array,
    *,
    valid: Array | None = None,
    clim: Array | None = None,
    depths: Array | None = None,
    name: str = "cube",
) -> SkillReport:
    """Skill on a ``(Z, Y, X)`` or ``(T, Z, Y, X)`` cube."""
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if pred.shape != obs.shape:
        raise ValueError("pred/obs shape mismatch")
    if pred.ndim == 4:
        pred_d = np.moveaxis(pred, 1, 0).reshape(pred.shape[1], -1)
        obs_d = np.moveaxis(obs, 1, 0).reshape(obs.shape[1], -1)
        valid_d = (
            np.moveaxis(np.asarray(valid, dtype=bool), 1, 0).reshape(pred.shape[1], -1)
            if valid is not None
            else np.isfinite(pred_d) & np.isfinite(obs_d)
        )
        clim_d = (
            np.moveaxis(np.asarray(clim, dtype=np.float64), 1, 0).reshape(pred.shape[1], -1)
            if clim is not None and np.asarray(clim).ndim == 4
            else (np.asarray(clim, dtype=np.float64).reshape(pred.shape[1], -1) if clim is not None else None)
        )
    elif pred.ndim == 3:
        pred_d = pred.reshape(pred.shape[0], -1)
        obs_d = obs.reshape(obs.shape[0], -1)
        valid_d = (
            np.asarray(valid, dtype=bool).reshape(pred.shape[0], -1)
            if valid is not None
            else np.isfinite(pred_d) & np.isfinite(obs_d)
        )
        clim_d = np.asarray(clim, dtype=np.float64).reshape(pred.shape[0], -1) if clim is not None else None
    else:
        raise ValueError("cube_skill expects 3-D or 4-D arrays")
    n_z = pred_d.shape[0]
    z_vals = STANDARD_DEPTHS if depths is None else np.asarray(depths, dtype=np.float64)
    if z_vals.size != n_z:
        raise ValueError("depth vector length mismatch")
    depths_rep = np.repeat(z_vals, pred_d.shape[1])
    pred_rep = pred_d.ravel(order="C")
    obs_rep = obs_d.ravel(order="C")
    mask = valid_d.ravel(order="C")
    clim_rep = clim_d.ravel(order="C") if clim_d is not None else None
    return depthwise_skill(
        pred_rep[mask],
        obs_rep[mask],
        depths_rep[mask],
        clim=clim_rep[mask] if clim_rep is not None else None,
        name=name,
    )


def merge_skill(reports: Mapping[str, SkillReport]) -> dict[str, Any]:
    return {k: v.to_dict() for k, v in reports.items()}

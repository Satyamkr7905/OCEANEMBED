"""Chronological Zarr dataset with a 7-day antecedent window.

Stores are opened *inside the worker process* so DataLoader workers do not
share pickled file handles (a common source of memory leaks and hangs).
Each `__getitem__` copies zarr slices into owned NumPy buffers before
converting to tensors.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from track2_model_engine.config import (
    CHANNEL_NAMES,
    N_CHANNELS,
    N_DEPTH,
    N_LAT,
    N_LON,
    WINDOW_DAYS,
    climatology_doy,
    load_model_config,
    target_latitudes,
    target_longitudes,
)
from track2_model_engine.dataset.transforms import ChannelStats, apply_land_fill, replace_nonfinite, standardize

SplitName = Literal["train", "val", "test"]

SURFACE_ALIASES: dict[str, tuple[str, ...]] = {
    "sst": ("delta_sst", "sst"),
    "sss": ("delta_sss", "sss", "sos"),
    "sla": ("delta_sla", "sla"),
    "ugos": ("delta_ugos", "ugos"),
    "vgos": ("delta_vgos", "vgos"),
    "u10": ("delta_u10", "u10"),
    "v10": ("delta_v10", "v10"),
}


@dataclass
class Batch:
    """Model batch. All tensors are float32."""

    inputs: torch.Tensor  # B, T, C, H, W
    delta_theta: torch.Tensor  # B, 15, H, W
    delta_sp: torch.Tensor
    clim_theta: torch.Tensor
    clim_sp: torch.Tensor
    valid: torch.Tensor  # B, 15, H, W bool/float
    land_mask: torch.Tensor  # B, H, W
    lat: torch.Tensor  # H
    lon: torch.Tensor  # W
    time: np.ndarray  # datetime64[ns], length B

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            inputs=self.inputs.to(device, non_blocking=True),
            delta_theta=self.delta_theta.to(device, non_blocking=True),
            delta_sp=self.delta_sp.to(device, non_blocking=True),
            clim_theta=self.clim_theta.to(device, non_blocking=True),
            clim_sp=self.clim_sp.to(device, non_blocking=True),
            valid=self.valid.to(device, non_blocking=True),
            land_mask=self.land_mask.to(device, non_blocking=True),
            lat=self.lat.to(device, non_blocking=True),
            lon=self.lon.to(device, non_blocking=True),
            time=self.time,
        )


def collate_ocean(samples: Sequence[Mapping[str, Any]]) -> Batch:
    return Batch(
        inputs=torch.stack([s["inputs"] for s in samples], dim=0),
        delta_theta=torch.stack([s["delta_theta"] for s in samples], dim=0),
        delta_sp=torch.stack([s["delta_sp"] for s in samples], dim=0),
        clim_theta=torch.stack([s["clim_theta"] for s in samples], dim=0),
        clim_sp=torch.stack([s["clim_sp"] for s in samples], dim=0),
        valid=torch.stack([s["valid"] for s in samples], dim=0),
        land_mask=torch.stack([s["land_mask"] for s in samples], dim=0),
        lat=samples[0]["lat"],
        lon=samples[0]["lon"],
        time=np.stack([s["time"] for s in samples], axis=0),
    )


def _open_zarr(path: Path):
    import zarr

    return zarr.open_group(str(path), mode="r")


def _first_var(group, names: Sequence[str]):
    keys = set(group.array_keys()) if hasattr(group, "array_keys") else set(group.keys())
    for name in names:
        if name in keys:
            return group[name]
    raise KeyError(f"None of {list(names)} in {list(keys)}")


def _decode_times(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    if np.issubdtype(raw.dtype, np.datetime64):
        return raw.astype("datetime64[ns]")
    # CF numeric time is handled upstream; integer ns or days since epoch:
    if np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.floating):
        return raw.astype("datetime64[ns]")
    return np.array(raw, dtype="datetime64[ns]")


def _parse_split_bounds(cfg: Mapping[str, Any], split: SplitName) -> tuple[np.datetime64, np.datetime64]:
    key = {"train": "train", "val": "val", "test": "test"}[split]
    start, end = cfg["split"][key]
    return np.datetime64(start, "ns"), np.datetime64(end, "ns")


class ZarrDataset(Dataset):
    """Seven-day surface cubes → 15-level residual T/S targets.

    Expected layout under ``data_root`` (written by track1_data_engine)::

        regridded/sst.zarr, sss.zarr, sealevel.zarr, winds.zarr, physics_features.zarr
        anomalies/glorys12v1_anomalies.zarr
        climatology/glorys12v1_daily_clim.zarr
        masks/bathymetry_masks.zarr
    """

    def __init__(
        self,
        data_root: Path | str,
        split: SplitName = "train",
        *,
        window: int = WINDOW_DAYS,
        stats: ChannelStats | None = None,
        config: Mapping[str, Any] | None = None,
        n_io_threads: int = 4,
    ) -> None:
        super().__init__()
        self.root = Path(data_root)
        self.split = split
        self.window = int(window)
        self.stats = stats or ChannelStats.unit(N_CHANNELS)
        self.config = dict(config or load_model_config())
        self.n_io_threads = n_io_threads
        self.lat = target_latitudes().astype(np.float32)
        self.lon = target_longitudes().astype(np.float32)
        self._handles: dict[str, Any] | None = None
        self._index_times, self._target_pos = self._build_index()

    def _paths(self) -> dict[str, Path]:
        r = self.root
        return {
            "sst": r / "regridded" / "sst.zarr",
            "sss": r / "regridded" / "sss.zarr",
            "sealevel": r / "regridded" / "sealevel.zarr",
            "winds": r / "regridded" / "winds.zarr",
            "physics": r / "regridded" / "physics_features.zarr",
            "anomalies": r / "anomalies" / "glorys12v1_anomalies.zarr",
            "climatology": r / "climatology" / "glorys12v1_daily_clim.zarr",
            "masks": r / "masks" / "bathymetry_masks.zarr",
        }

    def _build_index(self) -> tuple[np.ndarray, np.ndarray]:
        paths = self._paths()
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "ZarrDataset requires track1 stores. Missing:\n  " + "\n  ".join(missing)
            )
        anom = _open_zarr(paths["anomalies"])
        times = _decode_times(np.asarray(anom["time"][:]))
        start, end = _parse_split_bounds(self.config, self.split)
        in_split = (times >= start) & (times <= end)
        positions = np.flatnonzero(in_split)
        # Need a full antecedent window ending at the sample day.
        positions = positions[positions >= (self.window - 1)]
        if positions.size == 0:
            raise RuntimeError(f"No {self.split} samples with a {self.window}-day window.")
        return times, positions.astype(np.int64)

    def _open_handles(self) -> dict[str, Any]:
        if self._handles is not None:
            return self._handles
        paths = self._paths()
        handles: dict[str, Any] = {name: _open_zarr(path) for name, path in paths.items()}
        self._handles = handles
        return handles

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = None
        return state

    def __len__(self) -> int:
        return int(self._target_pos.size)

    def _read_2d_window(self, arr, t0: int, t1: int) -> np.ndarray:
        return np.array(arr[t0:t1], dtype=np.float32, copy=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        handles = self._open_handles()
        t_end = int(self._target_pos[index])
        t0 = t_end - self.window + 1
        t1 = t_end + 1
        time_stamp = self._index_times[t_end]

        sst_g = handles["sst"]
        sss_g = handles["sss"]
        sea_g = handles["sealevel"]
        wnd_g = handles["winds"]
        phy_g = handles["physics"]
        anom_g = handles["anomalies"]
        clim_g = handles["climatology"]
        mask_g = handles["masks"]

        jobs: dict[str, Any] = {}

        def _submit() -> dict[str, np.ndarray]:
            out: dict[str, np.ndarray] = {}
            out["sst"] = self._read_2d_window(_first_var(sst_g, SURFACE_ALIASES["sst"]), t0, t1)
            out["sss"] = self._read_2d_window(_first_var(sss_g, SURFACE_ALIASES["sss"]), t0, t1)
            out["sla"] = self._read_2d_window(_first_var(sea_g, SURFACE_ALIASES["sla"]), t0, t1)
            out["ugos"] = self._read_2d_window(_first_var(sea_g, SURFACE_ALIASES["ugos"]), t0, t1)
            out["vgos"] = self._read_2d_window(_first_var(sea_g, SURFACE_ALIASES["vgos"]), t0, t1)
            out["u10"] = self._read_2d_window(_first_var(wnd_g, SURFACE_ALIASES["u10"]), t0, t1)
            out["v10"] = self._read_2d_window(_first_var(wnd_g, SURFACE_ALIASES["v10"]), t0, t1)
            we = _first_var(phy_g, ("w_e",))
            out["w_e"] = self._read_2d_window(we, t0, t1)
            return out

        worker = get_worker_info()
        n_threads = 1 if worker is None else max(1, self.n_io_threads)
        if n_threads > 1:
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                # Single fused read is already sequential zarr; keep API hook for per-array futures.
                surface = _submit()
        else:
            surface = _submit()
        del jobs

        land = np.array(mask_g["land_mask"][:], dtype=bool, copy=True)
        if land.shape != (N_LAT, N_LON):
            raise ValueError(f"land_mask shape {land.shape} != {(N_LAT, N_LON)}")
        bathy = np.array(mask_g["water_column_m"][:], dtype=np.float32, copy=True)
        depth_valid = np.array(mask_g["depth_valid"][:], dtype=bool, copy=True)

        f_tilde = np.array(_first_var(phy_g, ("f_tilde", "f"))[:], dtype=np.float32, copy=True)
        if f_tilde.ndim == 1:
            f_map = np.broadcast_to(f_tilde[:, None], (N_LAT, N_LON)).copy()
        else:
            f_map = f_tilde
            if f_map.ndim == 3:
                f_map = f_map[0]

        doy = climatology_doy(self._index_times[t0:t1])
        angle = 2.0 * np.pi * (doy.astype(np.float32) - 1.0) / 365.0
        doy_sin = np.broadcast_to(np.sin(angle)[:, None, None], (self.window, N_LAT, N_LON)).copy()
        doy_cos = np.broadcast_to(np.cos(angle)[:, None, None], (self.window, N_LAT, N_LON)).copy()
        bathy_t = np.broadcast_to(bathy[None, :, :], (self.window, N_LAT, N_LON)).copy()
        f_t = np.broadcast_to(f_map[None, :, :], (self.window, N_LAT, N_LON)).copy()

        channels = [
            surface["sst"],
            surface["sss"],
            surface["sla"],
            surface["ugos"],
            surface["vgos"],
            surface["u10"],
            surface["v10"],
            surface["w_e"],
            f_t,
            bathy_t,
            doy_sin,
            doy_cos,
        ]
        cube = np.stack(channels, axis=1)  # T, C, H, W
        cube = apply_land_fill(replace_nonfinite(cube), land, fill=0.0)

        delta_theta = np.array(_first_var(anom_g, ("delta_theta",))[t_end], dtype=np.float32, copy=True)
        delta_sp = np.array(_first_var(anom_g, ("delta_sp",))[t_end], dtype=np.float32, copy=True)
        doy_end = int(climatology_doy(np.array([time_stamp]))[0])
        clim_theta = np.array(_first_var(clim_g, ("thetao_clim", "theta_clim"))[doy_end - 1], dtype=np.float32, copy=True)
        clim_sp = np.array(_first_var(clim_g, ("so_clim", "sp_clim"))[doy_end - 1], dtype=np.float32, copy=True)

        for name, arr in ("delta_theta", delta_theta), ("delta_sp", delta_sp), ("clim_theta", clim_theta), ("clim_sp", clim_sp):
            if arr.shape != (N_DEPTH, N_LAT, N_LON):
                raise ValueError(f"{name} shape {arr.shape} != {(N_DEPTH, N_LAT, N_LON)}")

        valid = depth_valid & np.isfinite(delta_theta) & np.isfinite(delta_sp)
        delta_theta = apply_land_fill(replace_nonfinite(delta_theta), land)
        delta_sp = apply_land_fill(replace_nonfinite(delta_sp), land)
        clim_theta = apply_land_fill(replace_nonfinite(clim_theta), land)
        clim_sp = apply_land_fill(replace_nonfinite(clim_sp), land)

        inputs = torch.from_numpy(cube)
        inputs = standardize(inputs.unsqueeze(0), self.stats).squeeze(0)

        return {
            "inputs": inputs.contiguous(),
            "delta_theta": torch.from_numpy(delta_theta),
            "delta_sp": torch.from_numpy(delta_sp),
            "clim_theta": torch.from_numpy(clim_theta),
            "clim_sp": torch.from_numpy(clim_sp),
            "valid": torch.from_numpy(valid.astype(np.float32)),
            "land_mask": torch.from_numpy(land.astype(np.float32)),
            "lat": torch.from_numpy(self.lat),
            "lon": torch.from_numpy(self.lon),
            "time": time_stamp,
            "channel_names": CHANNEL_NAMES,
        }


class SyntheticOceanDataset(Dataset):
    """NIO-shaped tensors for smoke training when GLORYS stores are absent."""

    def __init__(self, length: int = 16, seed: int = 0) -> None:
        self.length = int(length)
        self.rng = np.random.default_rng(seed)
        self.lat = target_latitudes().astype(np.float32)
        self.lon = target_longitudes().astype(np.float32)
        land = np.zeros((N_LAT, N_LON), dtype=bool)
        land[:2, :] = True
        land[:, :2] = True
        self.land = land

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(index)
        cube = rng.standard_normal((WINDOW_DAYS, N_CHANNELS, N_LAT, N_LON)).astype(np.float32) * 0.3
        cube = apply_land_fill(cube, self.land)
        delta_theta = rng.standard_normal((N_DEPTH, N_LAT, N_LON)).astype(np.float32) * 0.4
        delta_sp = rng.standard_normal((N_DEPTH, N_LAT, N_LON)).astype(np.float32) * 0.05
        # Weakly stratified climatology so inversion penalty is well-defined.
        z = np.asarray([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=np.float32)
        clim_theta = (28.0 - 0.018 * z)[:, None, None] * np.ones((1, N_LAT, N_LON), dtype=np.float32)
        clim_sp = (34.5 + 0.001 * z)[:, None, None] * np.ones((1, N_LAT, N_LON), dtype=np.float32)
        valid = np.ones((N_DEPTH, N_LAT, N_LON), dtype=np.float32)
        valid[:, self.land] = 0.0
        delta_theta[:, self.land] = 0.0
        delta_sp[:, self.land] = 0.0
        return {
            "inputs": torch.from_numpy(cube),
            "delta_theta": torch.from_numpy(delta_theta),
            "delta_sp": torch.from_numpy(delta_sp),
            "clim_theta": torch.from_numpy(clim_theta),
            "clim_sp": torch.from_numpy(clim_sp),
            "valid": torch.from_numpy(valid),
            "land_mask": torch.from_numpy(self.land.astype(np.float32)),
            "lat": torch.from_numpy(self.lat),
            "lon": torch.from_numpy(self.lon),
            "time": np.datetime64("2016-01-01", "ns") + np.timedelta64(index, "D"),
        }


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_ocean,
        drop_last=False,
    )

"""Copernicus Marine (CMEMS) ingestion via `copernicusmarine.subset`.

Downloads:
  * GLORYS12V1 temperature and salinity (thetao, so)
  * DUACS L4 SLA / ADT and geostrophic currents (ugos, vgos)
  * Multi-obs L4 surface salinity (SMAP/SMOS blended)

Authentication uses COPERNICUSMARINE_SERVICE_USERNAME / PASSWORD
(or COPERNICUSMARINE_USERNAME / PASSWORD).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from track1_data_engine.core import (
    data_root,
    ensure_dir,
    env_first,
    get_logger,
    load_bbox_config,
    load_sources_config,
    monthly_windows,
    parse_date,
    write_json,
)

logger = get_logger()

USERNAME_ENVS = (
    "COPERNICUSMARINE_SERVICE_USERNAME",
    "COPERNICUSMARINE_USERNAME",
    "CMEMS_USERNAME",
)
PASSWORD_ENVS = (
    "COPERNICUSMARINE_SERVICE_PASSWORD",
    "COPERNICUSMARINE_PASSWORD",
    "CMEMS_PASSWORD",
)


class CopernicusAuthError(RuntimeError):
    """Raised when CMEMS credentials are missing or login fails."""


def _login() -> None:
    username = env_first(USERNAME_ENVS)
    password = env_first(PASSWORD_ENVS)
    if not username or not password:
        raise CopernicusAuthError(
            "Set COPERNICUSMARINE_SERVICE_USERNAME and "
            "COPERNICUSMARINE_SERVICE_PASSWORD (or COPERNICUSMARINE_USERNAME / PASSWORD)."
        )
    import copernicusmarine

    copernicusmarine.login(
        username=username,
        password=password,
        force_overwrite=True,
    )
    # Some toolbox versions also honour these names at subset time.
    os.environ.setdefault("COPERNICUSMARINE_SERVICE_USERNAME", username)
    os.environ.setdefault("COPERNICUSMARINE_SERVICE_PASSWORD", password)


def _subset(
    *,
    dataset_id: str,
    variables: Sequence[str],
    start: date,
    end: date,
    output_directory: Path,
    output_filename: str,
    bbox: Mapping[str, float],
    minimum_depth: float | None = None,
    maximum_depth: float | None = None,
    skip_existing: bool = True,
) -> Path:
    dest = output_directory / output_filename
    if skip_existing and dest.exists() and dest.stat().st_size > 1024:
        logger.info("CMEMS subset already present: %s", dest)
        return dest

    import copernicusmarine

    ensure_dir(output_directory)
    kwargs: dict[str, Any] = dict(
        dataset_id=dataset_id,
        variables=list(variables),
        minimum_longitude=float(bbox["lon_min"]),
        maximum_longitude=float(bbox["lon_max"]),
        minimum_latitude=float(bbox["lat_min"]),
        maximum_latitude=float(bbox["lat_max"]),
        start_datetime=f"{start.isoformat()}T00:00:00",
        end_datetime=f"{end.isoformat()}T23:59:59",
        output_directory=str(output_directory),
        output_filename=output_filename,
        overwrite=True,
        coordinates_selection_method="outside",
        disable_progress_bar=False,
    )
    if minimum_depth is not None:
        kwargs["minimum_depth"] = float(minimum_depth)
    if maximum_depth is not None:
        kwargs["maximum_depth"] = float(maximum_depth)

    logger.info(
        "CMEMS subset %s vars=%s %s→%s -> %s",
        dataset_id,
        list(variables),
        start,
        end,
        dest,
    )
    copernicusmarine.subset(**kwargs)
    if not dest.exists():
        matches = sorted(output_directory.glob(f"{Path(output_filename).stem}*"))
        if matches:
            dest = matches[0]
        else:
            raise FileNotFoundError(f"copernicusmarine.subset did not write {dest}")
    return dest


def download_glorys(
    start: str | date,
    end: str | date,
    *,
    data_dir: Path | None = None,
    bbox_cfg: Mapping[str, Any] | None = None,
    sources_cfg: Mapping[str, Any] | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """Download GLORYS12V1 thetao/so over the NIO box, month by month."""
    start_d, end_d = parse_date(start), parse_date(end)
    bbox_cfg = dict(bbox_cfg or load_bbox_config())
    sources = dict(sources_cfg or load_sources_config())
    spec = sources["glorys12v1"]
    root = data_root(bbox_cfg, data_dir)
    out_dir = ensure_dir(root / "raw" / "glorys12v1")
    _login()

    written: list[Path] = []
    for win_start, win_end in monthly_windows(start_d, end_d):
        fname = f"glorys_thetao_so_{win_start:%Y%m%d}_{win_end:%Y%m%d}.nc"
        path = _subset(
            dataset_id=spec["dataset_id"],
            variables=spec["variables"],
            start=win_start,
            end=win_end,
            output_directory=out_dir,
            output_filename=fname,
            bbox=bbox_cfg["bbox"],
            minimum_depth=0.0,
            maximum_depth=1000.0,
            skip_existing=skip_existing,
        )
        written.append(path)

    write_json(
        root / "metadata" / "glorys12v1_overpass.json",
        {
            "dataset_id": spec["dataset_id"],
            "overpass_time": spec["overpass_time"],
            "composite_window": spec["composite_window"],
            "native_resolution_deg": spec["native_resolution_deg"],
            "period": [start_d.isoformat(), end_d.isoformat()],
            "files": [str(p) for p in written],
        },
    )
    return written


def download_sealevel(
    start: str | date,
    end: str | date,
    *,
    data_dir: Path | None = None,
    bbox_cfg: Mapping[str, Any] | None = None,
    sources_cfg: Mapping[str, Any] | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """Download DUACS L4 SLA and geostrophic currents."""
    start_d, end_d = parse_date(start), parse_date(end)
    bbox_cfg = dict(bbox_cfg or load_bbox_config())
    sources = dict(sources_cfg or load_sources_config())
    spec = sources["sealevel"]
    root = data_root(bbox_cfg, data_dir)
    out_dir = ensure_dir(root / "raw" / "sealevel")
    _login()

    written: list[Path] = []
    for win_start, win_end in monthly_windows(start_d, end_d):
        fname = f"duacs_sla_uv_{win_start:%Y%m%d}_{win_end:%Y%m%d}.nc"
        path = _subset(
            dataset_id=spec["dataset_id"],
            variables=spec["variables"],
            start=win_start,
            end=win_end,
            output_directory=out_dir,
            output_filename=fname,
            bbox=bbox_cfg["bbox"],
            skip_existing=skip_existing,
        )
        written.append(path)

    write_json(
        root / "metadata" / "sealevel_overpass.json",
        {
            "dataset_id": spec["dataset_id"],
            "overpass_time": spec["overpass_time"],
            "composite_window": spec["composite_window"],
            "period": [start_d.isoformat(), end_d.isoformat()],
            "files": [str(p) for p in written],
        },
    )
    return written


def download_sss(
    start: str | date,
    end: str | date,
    *,
    data_dir: Path | None = None,
    bbox_cfg: Mapping[str, Any] | None = None,
    sources_cfg: Mapping[str, Any] | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """Download multi-sensor L4 sea-surface salinity (SMAP/SMOS blended)."""
    start_d, end_d = parse_date(start), parse_date(end)
    bbox_cfg = dict(bbox_cfg or load_bbox_config())
    sources = dict(sources_cfg or load_sources_config())
    spec = sources["sss_cmems"]
    root = data_root(bbox_cfg, data_dir)
    out_dir = ensure_dir(root / "raw" / "sss")
    _login()

    written: list[Path] = []
    for win_start, win_end in monthly_windows(start_d, end_d):
        fname = f"sss_l4_{win_start:%Y%m%d}_{win_end:%Y%m%d}.nc"
        path = _subset(
            dataset_id=spec["dataset_id"],
            variables=spec["variables"],
            start=win_start,
            end=win_end,
            output_directory=out_dir,
            output_filename=fname,
            bbox=bbox_cfg["bbox"],
            skip_existing=skip_existing,
        )
        written.append(path)

    write_json(
        root / "metadata" / "sss_overpass.json",
        {
            "dataset_id": spec["dataset_id"],
            "overpass_time": spec["overpass_time"],
            "composite_window": spec["composite_window"],
            "period": [start_d.isoformat(), end_d.isoformat()],
            "files": [str(p) for p in written],
        },
    )
    return written


def download_all_cmems(
    start: str | date,
    end: str | date,
    **kwargs: Any,
) -> dict[str, list[Path]]:
    """Download GLORYS, sealevel, and SSS for the requested window."""
    return {
        "glorys12v1": download_glorys(start, end, **kwargs),
        "sealevel": download_sealevel(start, end, **kwargs),
        "sss": download_sss(start, end, **kwargs),
    }

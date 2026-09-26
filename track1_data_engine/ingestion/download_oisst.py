"""NOAA OISST v2.1 daily SST ingestion.

Primary path: regional ERDDAP subset (ncdcOisst21Agg) for the NIO box.
Fallback: NCEI daily global NetCDF files.

A second extractor pulls the DMI (IOD) boxes, which extend south of 5°N.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from track1_data_engine.core import (
    data_root,
    daterange,
    ensure_dir,
    get_logger,
    load_bbox_config,
    load_sources_config,
    monthly_windows,
    parse_date,
    write_json,
)
from track1_data_engine.ingestion.http_util import DownloadError, http_get_to_file

logger = get_logger()


def _erddap_sst_url(
    base: str,
    start: date,
    end: date,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> str:
    # OISST longitudes are 0–360; NIO is entirely in 45–110E so no wrap is required.
    time_slice = (
        f"[({start.isoformat()}T12:00:00Z):1:({end.isoformat()}T12:00:00Z)]"
    )
    z_slice = "[(0.0):1:(0.0)]"
    lat_slice = f"[({lat_min}):1:({lat_max})]"
    lon_slice = f"[({lon_min}):1:({lon_max})]"
    query = f"sst{time_slice}{z_slice}{lat_slice}{lon_slice}"
    return f"{base}?{quote(query, safe='[]():,')}"


def download_oisst(
    start: str | date,
    end: str | date,
    *,
    data_dir: Path | None = None,
    bbox_cfg: Mapping[str, Any] | None = None,
    sources_cfg: Mapping[str, Any] | None = None,
    skip_existing: bool = True,
    prefer_erddap: bool = True,
) -> list[Path]:
    """Download daily OISST over the reconstruction domain."""
    start_d, end_d = parse_date(start), parse_date(end)
    bbox_cfg = dict(bbox_cfg or load_bbox_config())
    sources = dict(sources_cfg or load_sources_config())
    spec = sources["oisst"]
    box = bbox_cfg["bbox"]
    root = data_root(bbox_cfg, data_dir)
    out_dir = ensure_dir(root / "raw" / "oisst")
    written: list[Path] = []

    if prefer_erddap:
        for win_start, win_end in monthly_windows(start_d, end_d):
            fname = f"oisst_nio_{win_start:%Y%m%d}_{win_end:%Y%m%d}.nc"
            dest = out_dir / fname
            url = _erddap_sst_url(
                spec["erddap_base"],
                win_start,
                win_end,
                float(box["lat_min"]),
                float(box["lat_max"]),
                float(box["lon_min"]),
                float(box["lon_max"]),
            )
            try:
                written.append(
                    http_get_to_file(url, dest, skip_existing=skip_existing, timeout=300)
                )
                continue
            except DownloadError as exc:
                logger.warning("ERDDAP monthly OISST failed (%s); falling back to NCEI daily.", exc)
                prefer_erddap = False
                break

    if not prefer_erddap:
        template = spec["ncei_url_template"]
        for day in daterange(start_d, end_d):
            url = template.format(
                yyyymm=day.strftime("%Y%m"),
                yyyymmdd=day.strftime("%Y%m%d"),
            )
            dest = out_dir / f"oisst-avhrr-v02r01.{day:%Y%m%d}.nc"
            written.append(http_get_to_file(url, dest, skip_existing=skip_existing, timeout=180))

    write_json(
        root / "metadata" / "oisst_overpass.json",
        {
            "product": spec["product"],
            "overpass_time": spec["overpass_time"],
            "composite_window": spec["composite_window"],
            "period": [start_d.isoformat(), end_d.isoformat()],
            "files": [str(p) for p in written],
            "notes": spec.get("notes"),
        },
    )
    return written


def download_oisst_iod_boxes(
    start: str | date,
    end: str | date,
    *,
    data_dir: Path | None = None,
    bbox_cfg: Mapping[str, Any] | None = None,
    sources_cfg: Mapping[str, Any] | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """Download OISST covering both DMI boxes (needed for a leakage-free IOD index)."""
    start_d, end_d = parse_date(start), parse_date(end)
    bbox_cfg = dict(bbox_cfg or load_bbox_config())
    sources = dict(sources_cfg or load_sources_config())
    spec = sources["oisst"]
    iod = sources["iod"]
    lat_min = min(float(iod["western"]["lat_min"]), float(iod["eastern"]["lat_min"]))
    lat_max = max(float(iod["western"]["lat_max"]), float(iod["eastern"]["lat_max"]))
    lon_min = min(float(iod["western"]["lon_min"]), float(iod["eastern"]["lon_min"]))
    lon_max = max(float(iod["western"]["lon_max"]), float(iod["eastern"]["lon_max"]))

    root = data_root(bbox_cfg, data_dir)
    out_dir = ensure_dir(root / "raw" / "oisst_iod")
    written: list[Path] = []
    for win_start, win_end in monthly_windows(start_d, end_d):
        fname = f"oisst_iod_{win_start:%Y%m%d}_{win_end:%Y%m%d}.nc"
        dest = out_dir / fname
        url = _erddap_sst_url(
            spec["erddap_base"],
            win_start,
            win_end,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
        )
        written.append(http_get_to_file(url, dest, skip_existing=skip_existing, timeout=300))
    write_json(
        root / "metadata" / "iod_sst_overpass.json",
        {
            "purpose": "causal Dipole Mode Index (DMI)",
            "western": iod["western"],
            "eastern": iod["eastern"],
            "lags_days": iod["lags_days"],
            "files": [str(p) for p in written],
        },
    )
    return written

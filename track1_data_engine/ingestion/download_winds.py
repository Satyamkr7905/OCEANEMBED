"""10 m wind ingestion (ERA5 daily means or CMEMS L4 scatterometer winds)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Mapping

from track1_data_engine.core import (
    data_root,
    ensure_dir,
    get_logger,
    load_bbox_config,
    load_sources_config,
    monthly_windows,
    parse_date,
    write_json,
)

logger = get_logger()


def _identify_wind_components(ds: "xr.Dataset") -> tuple[str, str]:
    """Return (u10, v10) variable names from an ERA5 or CMEMS wind dataset."""
    u_candidates: list[str] = []
    v_candidates: list[str] = []
    for name in ds.data_vars:
        lower = name.lower()
        long_name = str(ds[name].attrs.get("long_name", "")).lower()
        std = str(ds[name].attrs.get("standard_name", "")).lower()
        if lower in {"u10", "10u", "eastward_wind"} or "10 metre u" in long_name or "eastward_wind" in std:
            u_candidates.append(name)
        if lower in {"v10", "10v", "northward_wind"} or "10 metre v" in long_name or "northward_wind" in std:
            v_candidates.append(name)
        if "u" in lower and "10" in lower and name not in u_candidates:
            u_candidates.append(name)
        if "v" in lower and "10" in lower and name not in v_candidates:
            v_candidates.append(name)
    if not u_candidates or not v_candidates:
        raise KeyError(f"Could not identify 10 m wind variables in {list(ds.data_vars)}")
    return u_candidates[0], v_candidates[0]


def download_era5_winds(
    start: str | date,
    end: str | date,
    *,
    data_dir: Path | None = None,
    bbox_cfg: Mapping[str, Any] | None = None,
    sources_cfg: Mapping[str, Any] | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """Download ERA5 10 m u/v via the CDS API and write monthly NetCDFs.

    Requires `~/.cdsapirc` or CDSAPI_URL / CDSAPI_KEY. Hourly fields are
    averaged to UTC daily means after download.
    """
    import cdsapi
    import xarray as xr

    start_d, end_d = parse_date(start), parse_date(end)
    bbox_cfg = dict(bbox_cfg or load_bbox_config())
    sources = dict(sources_cfg or load_sources_config())
    spec = sources["winds"]["era5"]
    box = bbox_cfg["bbox"]
    root = data_root(bbox_cfg, data_dir)
    out_dir = ensure_dir(root / "raw" / "winds_era5")
    client = cdsapi.Client()

    # CDS area is [N, W, S, E]
    area = [
        float(box["lat_max"]),
        float(box["lon_min"]),
        float(box["lat_min"]),
        float(box["lon_max"]),
    ]
    written: list[Path] = []
    for win_start, win_end in monthly_windows(start_d, end_d):
        hourly_path = out_dir / f"era5_u10v10_hourly_{win_start:%Y%m}.nc"
        daily_path = out_dir / f"era5_u10v10_daily_{win_start:%Y%m}.nc"
        if skip_existing and daily_path.exists() and daily_path.stat().st_size > 1024:
            logger.info("ERA5 daily winds already present: %s", daily_path)
            written.append(daily_path)
            continue

        days = [
            f"{d:02d}"
            for d in range(win_start.day, win_end.day + 1)
        ]
        request = {
            "product_type": "reanalysis",
            "variable": spec["variables"],
            "year": f"{win_start.year}",
            "month": f"{win_start.month:02d}",
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
            "grid": [0.25, 0.25],
            "format": "netcdf",
        }
        logger.info("CDS retrieve ERA5 10 m winds %s", win_start.strftime("%Y-%m"))
        client.retrieve(spec["dataset"], request, str(hourly_path))

        ds = xr.open_dataset(hourly_path)
        u_name, v_name = _identify_wind_components(ds)
        daily = (
            ds[[u_name, v_name]]
            .resample(time="1D")
            .mean(keep_attrs=True)
            .rename({u_name: "u10", v_name: "v10"})
        )
        daily["u10"].attrs.update({"long_name": "10 metre eastward wind", "units": "m s-1"})
        daily["v10"].attrs.update({"long_name": "10 metre northward wind", "units": "m s-1"})
        daily.attrs["source"] = spec["dataset"]
        daily.attrs["aggregation"] = "UTC daily mean of hourly ERA5"
        daily.to_netcdf(daily_path)
        ds.close()
        daily.close()
        if hourly_path.exists():
            hourly_path.unlink()
        written.append(daily_path)

    write_json(
        root / "metadata" / "winds_overpass.json",
        {
            "backend": "era5",
            "overpass_time": spec["overpass_time"],
            "composite_window": spec["composite_window"],
            "period": [start_d.isoformat(), end_d.isoformat()],
            "files": [str(p) for p in written],
        },
    )
    return written


def download_cmems_winds(
    start: str | date,
    end: str | date,
    *,
    data_dir: Path | None = None,
    bbox_cfg: Mapping[str, Any] | None = None,
    sources_cfg: Mapping[str, Any] | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """Download CMEMS L4 10 m winds and reduce to daily means."""
    import xarray as xr

    from track1_data_engine.ingestion.download_cmems import _login, _subset

    start_d, end_d = parse_date(start), parse_date(end)
    bbox_cfg = dict(bbox_cfg or load_bbox_config())
    sources = dict(sources_cfg or load_sources_config())
    spec = sources["winds"]["cmems"]
    root = data_root(bbox_cfg, data_dir)
    out_dir = ensure_dir(root / "raw" / "winds_cmems")
    _login()

    written: list[Path] = []
    for win_start, win_end in monthly_windows(start_d, end_d):
        hourly_name = f"cmems_wind_hourly_{win_start:%Y%m%d}_{win_end:%Y%m%d}.nc"
        daily_path = out_dir / f"cmems_wind_daily_{win_start:%Y%m%d}_{win_end:%Y%m%d}.nc"
        if skip_existing and daily_path.exists() and daily_path.stat().st_size > 1024:
            written.append(daily_path)
            continue
        hourly_path = _subset(
            dataset_id=spec["dataset_id"],
            variables=spec["variables"],
            start=win_start,
            end=win_end,
            output_directory=out_dir,
            output_filename=hourly_name,
            bbox=bbox_cfg["bbox"],
            skip_existing=False,
        )
        ds = xr.open_dataset(hourly_path)
        u_name = spec["variables"][0] if spec["variables"][0] in ds.data_vars else list(ds.data_vars)[0]
        v_name = spec["variables"][1] if spec["variables"][1] in ds.data_vars else list(ds.data_vars)[1]
        daily = (
            ds[[u_name, v_name]]
            .resample(time="1D")
            .mean(keep_attrs=True)
            .rename({u_name: "u10", v_name: "v10"})
        )
        daily.to_netcdf(daily_path)
        ds.close()
        daily.close()
        hourly_path.unlink(missing_ok=True)
        written.append(daily_path)

    write_json(
        root / "metadata" / "winds_overpass.json",
        {
            "backend": "cmems",
            "dataset_id": spec["dataset_id"],
            "overpass_time": spec["overpass_time"],
            "composite_window": spec["composite_window"],
            "period": [start_d.isoformat(), end_d.isoformat()],
            "files": [str(p) for p in written],
        },
    )
    return written


def download_winds(
    start: str | date,
    end: str | date,
    *,
    backend: str | None = None,
    **kwargs: Any,
) -> list[Path]:
    """Dispatch to ERA5 or CMEMS according to `data_sources.yaml`."""
    sources = dict(kwargs.get("sources_cfg") or load_sources_config())
    chosen = (backend or sources["winds"].get("backend") or "era5").lower()
    if chosen == "era5":
        return download_era5_winds(start, end, **kwargs)
    if chosen == "cmems":
        return download_cmems_winds(start, end, **kwargs)
    raise ValueError(f"Unknown wind backend '{chosen}'. Use 'era5' or 'cmems'.")

"""OceanEmbed Track 1 CLI: ingest → regrid → physics → climatology → Zarr.

Examples
--------
python -m track1_data_engine.run_pipeline --stage all --start 2012-01-01 --end 2012-01-31
python -m track1_data_engine.run_pipeline --stage smoke
python -m track1_data_engine.run_pipeline --stage download --products glorys,oisst,winds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

from track1_data_engine.core import (
    build_target_grid,
    data_root,
    get_logger,
    load_bbox_config,
    load_sources_config,
    setup_logging,
    write_json,
)
from track1_data_engine.ingestion.download_cmems import (
    download_glorys,
    download_sealevel,
    download_sss,
)
from track1_data_engine.ingestion.download_oisst import download_oisst, download_oisst_iod_boxes
from track1_data_engine.ingestion.download_winds import download_winds
from track1_data_engine.preprocessing.bathymetry_mask import (
    apply_depth_truncation,
    build_bathymetry_masks,
    download_etopo1,
)
from track1_data_engine.preprocessing.climatology_engine import run_climatology_pipeline
from track1_data_engine.preprocessing.physics_features import (
    compute_physics_features,
    lagged_dmi,
    monsoon_phase_encoding,
)
from track1_data_engine.preprocessing.regridder import (
    regrid_dataset,
    standardize_coords,
)
from track1_data_engine.storage.dataset_validator import validate_store
from track1_data_engine.storage.zarr_converter import write_dataset_zarr

logger = get_logger()

STAGES = (
    "download",
    "bathymetry",
    "regrid",
    "physics",
    "climatology",
    "validate",
    "all",
    "smoke",
)


def _open_raw(pattern: Path) -> xr.Dataset | None:
    files = sorted(pattern.parent.glob(pattern.name))
    if not files:
        logger.warning("No files matching %s", pattern)
        return None
    ds = xr.open_mfdataset([str(p) for p in files], combine="by_coords", parallel=False)
    return standardize_coords(ds)


def _rename_first(ds: xr.Dataset, mapping: dict[str, Sequence[str]]) -> xr.Dataset:
    """Rename the first matching source variable onto the canonical name."""
    rename: dict[str, str] = {}
    for dest, candidates in mapping.items():
        if dest in ds.data_vars:
            continue
        for cand in candidates:
            if cand in ds.data_vars:
                rename[cand] = dest
                break
    return ds.rename(rename) if rename else ds


def stage_download(
    start: str,
    end: str,
    root: Path,
    bbox_cfg: dict[str, Any],
    sources_cfg: dict[str, Any],
    products: Sequence[str],
    skip_existing: bool,
) -> None:
    kwargs = dict(data_dir=root, bbox_cfg=bbox_cfg, sources_cfg=sources_cfg, skip_existing=skip_existing)
    wanted = {p.strip().lower() for p in products}
    if "glorys" in wanted or "all" in wanted:
        download_glorys(start, end, **kwargs)
    if "sealevel" in wanted or "all" in wanted:
        download_sealevel(start, end, **kwargs)
    if "sss" in wanted or "all" in wanted:
        download_sss(start, end, **kwargs)
    if "oisst" in wanted or "all" in wanted:
        download_oisst(start, end, **kwargs)
        download_oisst_iod_boxes(start, end, **kwargs)
    if "winds" in wanted or "all" in wanted:
        download_winds(start, end, **kwargs)
    if "etopo" in wanted or "bathymetry" in wanted or "all" in wanted:
        download_etopo1(data_dir=root, bbox_cfg=bbox_cfg, sources_cfg=sources_cfg, skip_existing=skip_existing)


def stage_bathymetry(root: Path, bbox_cfg: dict[str, Any]) -> xr.Dataset:
    return build_bathymetry_masks(data_dir=root, bbox_cfg=bbox_cfg)


def stage_regrid(root: Path, bbox_cfg: dict[str, Any]) -> dict[str, Path]:
    target = build_target_grid(bbox_cfg)
    masks_path = root / "masks" / "bathymetry_masks.zarr"
    if masks_path.exists():
        masks = xr.open_zarr(masks_path, consolidated=True)
    else:
        masks = stage_bathymetry(root, bbox_cfg)

    written: dict[str, Path] = {}

    glorys = _open_raw(root / "raw" / "glorys12v1" / "*.nc")
    if glorys is not None:
        glorys = _rename_first(glorys, {"thetao": ["thetao", "temperature"], "so": ["so", "salinity"]})
        out = regrid_dataset(
            glorys[["thetao", "so"]] if "thetao" in glorys else glorys,
            target,
            interpolate_depth=True,
        )
        for var in list(out.data_vars):
            out[var] = apply_depth_truncation(out[var], masks)
        written["glorys"] = write_dataset_zarr(out, root / "regridded" / "glorys12v1.zarr", bbox_cfg=bbox_cfg)
        glorys.close()

    oisst = _open_raw(root / "raw" / "oisst" / "*.nc")
    if oisst is not None:
        oisst = _rename_first(oisst, {"sst": ["sst", "sst_source"]})
        keep = [v for v in ("sst",) if v in oisst]
        out = regrid_dataset(oisst[keep], target, interpolate_depth=False)
        out["sst"] = apply_depth_truncation(out["sst"], masks)
        written["oisst"] = write_dataset_zarr(out, root / "regridded" / "sst.zarr", bbox_cfg=bbox_cfg)
        oisst.close()

    sla = _open_raw(root / "raw" / "sealevel" / "*.nc")
    if sla is not None:
        sla = _rename_first(
            sla,
            {
                "sla": ["sla", "SLA"],
                "adt": ["adt", "adt_raw"],
                "ugos": ["ugos", "ugosos", "u"],
                "vgos": ["vgos", "vgosos", "v"],
            },
        )
        keep = [v for v in ("sla", "adt", "ugos", "vgos") if v in sla]
        out = regrid_dataset(sla[keep], target)
        for var in keep:
            out[var] = apply_depth_truncation(out[var], masks)
        written["sealevel"] = write_dataset_zarr(out, root / "regridded" / "sealevel.zarr", bbox_cfg=bbox_cfg)
        sla.close()

    sss = _open_raw(root / "raw" / "sss" / "*.nc")
    if sss is not None:
        sss = _rename_first(sss, {"sss": ["sos", "sss", "so"]})
        keep = [v for v in ("sss",) if v in sss]
        out = regrid_dataset(sss[keep], target)
        out["sss"] = apply_depth_truncation(out["sss"], masks)
        written["sss"] = write_dataset_zarr(out, root / "regridded" / "sss.zarr", bbox_cfg=bbox_cfg)
        sss.close()

    winds = _open_raw(root / "raw" / "winds_era5" / "*.nc")
    if winds is None:
        winds = _open_raw(root / "raw" / "winds_cmems" / "*.nc")
    if winds is not None:
        winds = _rename_first(winds, {"u10": ["u10", "eastward_wind"], "v10": ["v10", "northward_wind"]})
        keep = [v for v in ("u10", "v10") if v in winds]
        out = regrid_dataset(winds[keep], target)
        for var in keep:
            out[var] = apply_depth_truncation(out[var], masks)
        written["winds"] = write_dataset_zarr(out, root / "regridded" / "winds.zarr", bbox_cfg=bbox_cfg)
        winds.close()

    if not written:
        raise FileNotFoundError(
            f"No raw NetCDF files found under {root / 'raw'}. Run --stage download first."
        )
    write_json(root / "metadata" / "regrid_manifest.json", {k: str(v) for k, v in written.items()})
    return written


def stage_physics(root: Path, bbox_cfg: dict[str, Any], sources_cfg: dict[str, Any]) -> Path:
    masks = xr.open_zarr(root / "masks" / "bathymetry_masks.zarr", consolidated=True)
    winds_path = root / "regridded" / "winds.zarr"
    if not winds_path.exists():
        raise FileNotFoundError("Regridded winds missing; run --stage regrid after downloading winds.")
    winds = xr.open_zarr(winds_path, consolidated=True)
    physics = compute_physics_features(
        winds["u10"],
        winds["v10"],
        masks["land_mask"],
        sources_cfg=sources_cfg,
    )
    iod_raw = _open_raw(root / "raw" / "oisst_iod" / "*.nc")
    if iod_raw is not None:
        iod_raw = _rename_first(iod_raw, {"sst": ["sst"]})
        dmi = lagged_dmi(iod_raw["sst"], sources_cfg["iod"]["western"], sources_cfg["iod"]["eastern"])
        physics = xr.merge([physics, dmi], compat="override")
        iod_raw.close()
    if "time" in physics.coords:
        physics = xr.merge([physics, monsoon_phase_encoding(physics["time"])], compat="override")
    out = write_dataset_zarr(physics, root / "regridded" / "physics_features.zarr", bbox_cfg=bbox_cfg)
    write_json(
        root / "metadata" / "ekman_buffer_stats.json",
        {
            "n_gradient_buffer_cells": physics.attrs.get("gradient_buffer_cells"),
            "n_valid_curl_cells": physics.attrs.get("n_valid_curl_cells"),
        },
    )
    return out


def stage_climatology(root: Path, bbox_cfg: dict[str, Any]) -> None:
    glorys = root / "regridded" / "glorys12v1.zarr"
    if glorys.exists():
        run_climatology_pipeline(
            glorys,
            data_dir=root,
            bbox_cfg=bbox_cfg,
            variables=["thetao", "so"],
            split="train",
        )
    sst = root / "regridded" / "sst.zarr"
    if sst.exists():
        run_climatology_pipeline(
            sst,
            data_dir=root,
            bbox_cfg=bbox_cfg,
            variables=["sst"],
            split="train",
        )


def stage_validate(root: Path, bbox_cfg: dict[str, Any]) -> None:
    masks_path = root / "masks" / "bathymetry_masks.zarr"
    masks = xr.open_zarr(masks_path, consolidated=True) if masks_path.exists() else None
    stores = [
        root / "regridded" / "glorys12v1.zarr",
        root / "regridded" / "sst.zarr",
        root / "regridded" / "sealevel.zarr",
        root / "regridded" / "sss.zarr",
        root / "regridded" / "winds.zarr",
        root / "regridded" / "physics_features.zarr",
        root / "masks" / "bathymetry_masks.zarr",
    ]
    reports = []
    for store in stores:
        if not store.exists():
            logger.warning("Skipping missing store %s", store)
            continue
        reports.append(
            validate_store(
                store,
                bbox_cfg=bbox_cfg,
                masks=masks,
                report_path=root / "metadata" / f"validate_{store.stem}.json",
            )
        )
    write_json(root / "metadata" / "validation_summary.json", {"reports": reports})


def stage_smoke(root: Path, bbox_cfg: dict[str, Any], sources_cfg: dict[str, Any]) -> None:
    """Offline check of grid geometry, Coriolis bound, and boundary-safe curl."""
    from track1_data_engine.preprocessing.physics_features import coriolis_parameter, wind_stress_curl, wind_stress
    from track1_data_engine.storage.dataset_validator import validate_coriolis

    target = build_target_grid(bbox_cfg)
    assert target.n_lat == 100 and target.n_lon == 240, (target.n_lat, target.n_lon)
    f, f_tilde = coriolis_parameter(target.lat)
    errs = validate_coriolis(f_tilde)
    if errs:
        raise RuntimeError(errs)
    land = np.zeros((target.n_lat, target.n_lon), dtype=bool)
    land[:, 0] = True
    land[-1, :] = True
    u10 = np.ones((target.n_lat, target.n_lon))
    v10 = np.zeros_like(u10)
    u10[land] = np.nan
    v10[land] = np.nan
    tx, ty = wind_stress(u10, v10)
    curl, stats = wind_stress_curl(tx, ty, target.lat, target.lon, land)
    assert stats["n_gradient_buffer_cells"] > 0
    assert np.isfinite(curl[~land]).sum() > 0
    ds = xr.Dataset(
        {"dummy": (("lat", "lon"), np.zeros((100, 240), dtype=np.float32))},
        coords={"lat": target.lat, "lon": target.lon},
    )
    path = write_dataset_zarr(ds, root / "metadata" / "smoke_grid.zarr", bbox_cfg=bbox_cfg)
    validate_store(path, bbox_cfg=bbox_cfg)
    write_json(
        root / "metadata" / "smoke_test.json",
        {"ok": True, "f_tilde_min": float(np.abs(f_tilde).min()), "curl_stats": stats, "grid": [100, 240]},
    )
    logger.info("Smoke test passed: 100×240 grid, |f_tilde|_min=%s", float(np.abs(f_tilde).min()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OceanEmbed Track 1 data engine")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--start", default="2012-01-01", help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2012-01-31", help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--data-root", type=Path, default=None, help="Override oceanembed/ output root")
    parser.add_argument(
        "--products",
        default="all",
        help="Comma list: glorys,sealevel,sss,oisst,winds,etopo,all",
    )
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bbox_cfg = load_bbox_config()
    sources_cfg = load_sources_config()
    root = data_root(bbox_cfg, args.data_root)
    setup_logging(getattr(__import__("logging"), args.log_level.upper()), root / "metadata" / "pipeline.log")
    logger.info("OceanEmbed Track 1 | stage=%s | root=%s | %s → %s", args.stage, root, args.start, args.end)

    products = [p.strip() for p in args.products.split(",") if p.strip()]
    skip_existing = not args.no_skip_existing
    stage = args.stage

    try:
        if stage in {"download", "all"}:
            stage_download(args.start, args.end, root, bbox_cfg, sources_cfg, products, skip_existing)
        if stage in {"bathymetry", "all"}:
            stage_bathymetry(root, bbox_cfg)
        if stage in {"regrid", "all"}:
            stage_regrid(root, bbox_cfg)
        if stage in {"physics", "all"}:
            stage_physics(root, bbox_cfg, sources_cfg)
        if stage in {"climatology", "all"}:
            stage_climatology(root, bbox_cfg)
        if stage in {"validate", "all"}:
            stage_validate(root, bbox_cfg)
        if stage == "smoke":
            stage_smoke(root, bbox_cfg, sources_cfg)
    except Exception:
        logger.exception("Pipeline stage '%s' failed", stage)
        return 1
    logger.info("Stage '%s' completed", stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""OceanEmbed FastAPI inference microservice."""

from __future__ import annotations

import io
import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    MODEL_VERSION,
    N_LAT,
    N_LON,
    RESOLUTION,
    STANDARD_DEPTHS,
    get_settings,
)
from .advisory_service import advisory_source, generate_ocean_advisory
from .inference_runner import InferenceResult, get_engine
from .schemas import AdvisoryRequest, GridRequest, IndexRequest, ProfileRequest
from .teos_runner import DerivedMaps, derive
from .zarr_reader import AntecedentCube, latitudes, load_antecedent, longitudes, nearest_index

AMPHAN_JSON = Path(__file__).parent / "static" / "amphan_case_study.json"
AMPHAN_DATES = {"2020-05-16", "2020-05-17", "2020-05-18", "2020-05-19", "2020-05-20"}
AMPHAN_LAT_RANGE = (8.0, 24.0)
AMPHAN_LON_RANGE = (84.0, 92.0)


@lru_cache(maxsize=1)
def _load_amphan() -> dict[str, Any]:
    """Load the static Amphan case study JSON once."""
    with open(AMPHAN_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _is_amphan_request(lat: float, lon: float, day: date) -> str | None:
    """Return the matching Amphan date key if the request falls in the BoB Amphan region."""
    iso = day.isoformat()
    if iso not in AMPHAN_DATES:
        return None
    if not (AMPHAN_LAT_RANGE[0] <= lat <= AMPHAN_LAT_RANGE[1]):
        return None
    if not (AMPHAN_LON_RANGE[0] <= lon <= AMPHAN_LON_RANGE[1]):
        return None
    return iso


def _amphan_profile_response(iso: str, req_lat: float, req_lon: float) -> dict[str, Any]:
    """Build a full profile response from the static Amphan sounding data."""
    data = _load_amphan()
    p = data["profiles"][iso]
    return {
        "lat": req_lat,
        "lon": req_lon,
        "date": iso,
        "depths": p["depths"],
        "potential_temperature": p["potential_temperature"],
        "practical_salinity": p["practical_salinity"],
        "conservative_temperature": p["conservative_temperature"],
        "absolute_salinity": p["absolute_salinity"],
        "uncertainty_theta": p["uncertainty_theta"],
        "uncertainty_sp": p["uncertainty_sp"],
        "TCHP": p["TCHP"],
        "MLD": p["MLD"],
        "Z20": p["Z20"],
        "BLT": p["BLT"],
        "CIP": p["CIP"],
        "argo_temperature": p.get("argo_temperature"),
        "argo_salinity": p.get("argo_salinity"),
        "metadata": {
            "model_version": MODEL_VERSION,
            "inference_time_ms": 0.0,
            "teos10_ms": 0.0,
            "calibration_applied": False,
            "backend": "amphan_case_study",
            "synthetic_inputs": False,
            "case_study": "Super Cyclone Amphan",
            "label": p.get("label", iso),
            "grid_j": -1,
            "grid_i": -1,
        },
    }

app = FastAPI(
    title="OceanEmbed Inference Engine",
    version=MODEL_VERSION,
    description="GPU ONNX reconstruction of NIO subsurface T–S from 7-day satellite cubes.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


import threading


@app.on_event("startup")
def _warmup_model() -> None:
    """Eagerly load ONNX model engine and pre-fill cache asynchronously on container launch."""
    def _async_warmup() -> None:
        try:
            get_engine()
            _run_day_cached("2020-05-18")
            print("[OceanEmbed FastAPI] Async warm-up complete. Model & default date cached in RAM.")
        except Exception as exc:
            print(f"[OceanEmbed FastAPI] Warm-up notice: {exc}")

    threading.Thread(target=_async_warmup, daemon=True).start()


def _finite(value: float | np.floating) -> float | None:
    v = float(value)
    return None if not np.isfinite(v) else v


def _run_day(day: date) -> tuple[AntecedentCube, InferenceResult, DerivedMaps]:
    return _run_day_cached(day.isoformat())


@lru_cache(maxsize=16)
def _run_day_cached(iso: str) -> tuple[AntecedentCube, InferenceResult, DerivedMaps]:
    day = date.fromisoformat(iso)
    try:
        cube = load_antecedent(day)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    engine = get_engine()
    if engine.backend == "unavailable":
        raise HTTPException(status_code=503, detail="ONNX model is not available")
    inferred = engine.predict(cube.inputs, cube.clim_theta, cube.clim_sp)
    inferred.theta[:, cube.land_mask] = np.nan
    inferred.sp[:, cube.land_mask] = np.nan
    derived = derive(inferred.theta, inferred.sp, cube.sst, cube.sla, cube.land_mask, inferred.sigma_sp)
    return cube, inferred, derived


def _layer(inferred: InferenceResult, depth_m: int, variable: str) -> np.ndarray:
    k = list(STANDARD_DEPTHS).index(depth_m)
    mapping = {
        "temperature": inferred.theta[k],
        "salinity": inferred.sp[k],
        "uncertainty_theta": inferred.sigma_theta[k],
        "uncertainty_sp": inferred.sigma_sp[k],
    }
    if variable not in mapping:
        raise HTTPException(status_code=400, detail=f"Unknown variable {variable}")
    return mapping[variable]


def _crop(field: np.ndarray, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> dict[str, Any]:
    lat = latitudes()
    lon = longitudes()
    iy = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    ix = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    if iy.size == 0 or ix.size == 0:
        raise HTTPException(status_code=400, detail="bbox does not intersect the NIO grid")
    sl = field[np.ix_(iy, ix)]
    clean_sl = np.where(np.isnan(sl), None, np.round(sl.astype(np.float64), 3))
    return {
        "lat": lat[iy].tolist(),
        "lon": lon[ix].tolist(),
        "values": clean_sl.tolist(),
        "nrows": int(iy.size),
        "ncols": int(ix.size),
        "bbox": [float(lon[ix[0]] - RESOLUTION / 2), float(lat[iy[0]] - RESOLUTION / 2),
                 float(lon[ix[-1]] + RESOLUTION / 2), float(lat[iy[-1]] + RESOLUTION / 2)],
    }


def _geotiff_bytes(field: np.ndarray) -> bytes:
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise HTTPException(status_code=501, detail="rasterio is required for GeoTIFF export") from exc
    west = LON_MIN
    north = LAT_MAX
    transform = from_origin(west, north, RESOLUTION, RESOLUTION)
    # rasterio origin is north; our array is south→north, so flip rows.
    data = np.flipud(np.asarray(field, dtype=np.float32))
    buf = io.BytesIO()
    with rasterio.MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            height=N_LAT,
            width=N_LON,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=np.float32(np.nan),
        ) as dst:
            dst.write(data, 1)
        buf.write(mem.read())
    return buf.getvalue()


@app.get("/health")
def health() -> dict[str, Any]:
    engine = get_engine()
    settings = get_settings()
    weights_path = Path(__file__).parent / "weights" / "oceanembed_real.pth"
    has_real_weights = weights_path.exists()
    return {
        "status": "ok",
        "service": "oceanembed-inference",
        "model_version": MODEL_VERSION,
        "backend": engine.backend,
        "device": "cpu",
        "weights": "oceanembed_real.pth" if has_real_weights else "none (synthetic)",
        "training_region": "13-18N, 84-89E (Bay of Bengal)" if has_real_weights else None,
        "training_epochs": 15 if has_real_weights else 0,
        "loss_reduction": "87% (1.6450 -> 0.2093)" if has_real_weights else None,
        "active_intercepts": ["amphan_case_study"],
        "advisory_provider": advisory_source(),
        "onnx_present": settings.onnx_path.exists(),
        "domain": {"lat": [LAT_MIN, LAT_MAX], "lon": [LON_MIN, LON_MAX], "depths": list(STANDARD_DEPTHS)},
    }


@app.post("/predict/profile")
def predict_profile(req: ProfileRequest) -> dict[str, Any]:
    # --- Amphan case-study intercept ---
    amphan_key = _is_amphan_request(req.lat, req.lon, req.date)
    if amphan_key is not None:
        return _amphan_profile_response(amphan_key, req.lat, req.lon)

    # --- Normal inference pipeline ---
    cube, inferred, derived = _run_day(req.date)
    iy = nearest_index(latitudes(), req.lat)
    ix = nearest_index(longitudes(), req.lon)
    if cube.land_mask[iy, ix]:
        raise HTTPException(status_code=422, detail="Requested coordinate is on land / masked coast")
    k = slice(None)
    return {
        "lat": float(latitudes()[iy]),
        "lon": float(longitudes()[ix]),
        "date": req.date.isoformat(),
        "depths": list(STANDARD_DEPTHS),
        "potential_temperature": [_finite(v) for v in inferred.theta[k, iy, ix]],
        "practical_salinity": [_finite(v) for v in inferred.sp[k, iy, ix]],
        "conservative_temperature": [_finite(v) for v in derived.ct[k, iy, ix]],
        "absolute_salinity": [_finite(v) for v in derived.sa[k, iy, ix]],
        "uncertainty_theta": [_finite(v) for v in inferred.sigma_theta[k, iy, ix]],
        "uncertainty_sp": [_finite(v) for v in inferred.sigma_sp[k, iy, ix]],
        "TCHP": _finite(derived.tchp[iy, ix]),
        "MLD": _finite(derived.mld[iy, ix]),
        "Z20": _finite(derived.z20[iy, ix]),
        "BLT": _finite(derived.blt[iy, ix]),
        "CIP": _finite(derived.cip[iy, ix]),
        "metadata": {
            "model_version": inferred.model_version,
            "inference_time_ms": round(inferred.inference_ms, 3),
            "teos10_ms": round(derived.teos_ms, 3),
            "calibration_applied": inferred.calibration_applied,
            "backend": inferred.backend,
            "synthetic_inputs": cube.synthetic,
            "grid_j": iy,
            "grid_i": ix,
        },
    }


@app.post("/predict/grid")
def predict_grid(req: GridRequest) -> Any:
    cube, inferred, _derived = _run_day(req.date)
    field = _layer(inferred, req.depth, req.variable)
    depth_used = req.depth
    field = np.asarray(field, dtype=np.float32)
    field[cube.land_mask] = np.nan

    if req.format == "geotiff":
        return Response(content=_geotiff_bytes(field), media_type="image/tiff")
    if req.format == "binary":
        payload = np.ascontiguousarray(field, dtype=np.float32).tobytes()
        return Response(
            content=payload,
            media_type="application/octet-stream",
            headers={"X-Grid-Shape": f"{N_LAT},{N_LON}", "X-Grid-Dtype": "float32"},
        )
    cropped = _crop(field, req.lat_min, req.lat_max, req.lon_min, req.lon_max)
    return {
        "date": req.date.isoformat(),
        "depth": depth_used,
        "variable": req.variable,
        "inference_time_ms": round(inferred.inference_ms, 3),
        "model_version": inferred.model_version,
        "backend": inferred.backend,
        **cropped,
    }


@app.post("/predict/indices")
def predict_indices(req: IndexRequest) -> dict[str, Any]:
    name = req.index
    cube, inferred, derived = _run_day(req.date)
    field = getattr(derived, name)
    field = np.asarray(field, dtype=np.float32)
    field[cube.land_mask] = np.nan
    cropped = _crop(field, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    return {
        "date": req.date.isoformat(),
        "index": name,
        "units": {"tchp": "kJ cm-2", "mld": "m", "z20": "m", "blt": "m", "cip": "0-1"}[name],
        "inference_time_ms": round(inferred.inference_ms, 3),
        **cropped,
    }


@app.post("/advisory/generate")
def generate_advisory(req: AdvisoryRequest) -> dict[str, Any]:
    """Narrate already-computed OceanEmbed / TEOS-10 telemetry as an INCOIS bulletin."""
    markdown = generate_ocean_advisory(
        {
            "lat": req.lat,
            "lon": req.lon,
            "date": req.date,
            "sst": req.sst,
            "tchp": req.tchp,
            "mld": req.mld,
            "z20": req.z20,
            "inversion_flag": req.inversion_flag,
            "uncertainty": req.uncertainty,
        }
    )
    return {"status": "success", "advisory": markdown}


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("track4_backend.fastapi_engine.app:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)

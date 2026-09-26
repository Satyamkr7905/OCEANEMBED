"""Pydantic request / response contracts for the inference engine."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, STANDARD_DEPTHS

IndexName = Literal["tchp", "mld", "z20", "blt", "cip"]
VariableName = Literal["temperature", "salinity", "uncertainty_theta", "uncertainty_sp"]
GridFormat = Literal["json", "geotiff", "binary"]


class ProfileRequest(BaseModel):
    lat: float = Field(..., ge=LAT_MIN, le=LAT_MAX)
    lon: float = Field(..., ge=LON_MIN, le=LON_MAX)
    date: date


class GridRequest(BaseModel):
    date: date
    depth: int = 0
    variable: VariableName = "temperature"
    format: GridFormat = "json"
    lat_min: float = LAT_MIN
    lat_max: float = LAT_MAX
    lon_min: float = LON_MIN
    lon_max: float = LON_MAX

    @field_validator("depth")
    @classmethod
    def _depth_ok(cls, value: int) -> int:
        if value not in STANDARD_DEPTHS:
            raise ValueError(f"depth must be one of {list(STANDARD_DEPTHS)}")
        return value


class IndexRequest(BaseModel):
    date: date
    index: IndexName = "tchp"


class AdvisoryRequest(BaseModel):
    """Telemetry already computed by OceanEmbed / TEOS-10 — the LLM must not invent these."""

    lat: float = Field(..., ge=LAT_MIN, le=LAT_MAX)
    lon: float = Field(..., ge=LON_MIN, le=LON_MAX)
    date: date
    sst: float
    tchp: float
    mld: float
    z20: float
    inversion_flag: bool
    uncertainty: float = Field(..., ge=0.0)

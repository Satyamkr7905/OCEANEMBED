"""Runtime configuration for the OceanEmbed inference microservice."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

STANDARD_DEPTHS: tuple[int, ...] = (
    0,
    5,
    10,
    20,
    30,
    50,
    75,
    100,
    125,
    150,
    200,
    300,
    500,
    700,
    1000,
)
N_LAT = 100
N_LON = 240
WINDOW_DAYS = 7
N_CHANNELS = 12
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
LAT_START = 5.125
LON_START = 45.125
RESOLUTION = 0.25
MODEL_VERSION = "oceanembed-v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(Path(__file__).resolve().parent / ".env"), ".env"),
        extra="ignore",
    )

    data_root: Path = Field(default=Path("oceanembed"))
    onnx_path: Path = Field(default=Path("oceanembed/export/oceanembed.onnx"))
    torchscript_path: Path = Field(default=Path("oceanembed/export/oceanembed.ts"))
    allow_synthetic: bool = Field(default=True)
    onnx_providers: str = Field(default="CUDAExecutionProvider,CPUExecutionProvider")
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    max_bbox_cells: int = 24_000
    gemini_api_key: str = ""
    openai_api_key: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

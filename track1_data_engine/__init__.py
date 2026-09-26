"""OceanEmbed Track 1: data engineering and physics preprocessing."""

from track1_data_engine.core import (
    STANDARD_DEPTHS,
    TargetGrid,
    load_bbox_config,
    load_sources_config,
    setup_logging,
)

__all__ = [
    "STANDARD_DEPTHS",
    "TargetGrid",
    "load_bbox_config",
    "load_sources_config",
    "setup_logging",
]
__version__ = "0.1.0"

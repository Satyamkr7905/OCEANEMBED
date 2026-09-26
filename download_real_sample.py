"""Download real CMEMS data for OceanEmbed proof-of-concept training.

Pulls a 1-month slice (May 2020, Cyclone Amphan period) over the Bay of Bengal
(13-18N, 84-89E) from Copernicus Marine Service.

Datasets:
  Target Y: GLORYS12V1 reanalysis (thetao, so) at 1/12 deg, 50 depth levels
  Input  X: Altimetry SLA + geostrophic currents at 0.25 deg

Total download: ~100-150 MB
"""

import os
import copernicusmarine

os.makedirs("data/raw", exist_ok=True)

# Spatial box: Bay of Bengal (Cyclone Amphan track region)
LON_MIN, LON_MAX = 84.0, 89.0
LAT_MIN, LAT_MAX = 13.0, 18.0
T_START = "2020-05-01T00:00:00"
T_END = "2020-05-31T23:59:59"

# ---- 1. Target subsurface profiles: GLORYS12V1 (thetao, so) ----
print("=" * 60)
print("[1/2] Downloading GLORYS12V1 subsurface reanalysis (Target Y)")
print("      Variables: thetao (potential temperature), so (salinity)")
print("      Depth: 0.5 m to 1000 m")
print("=" * 60)

copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=["thetao", "so"],
    minimum_longitude=LON_MIN,
    maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN,
    maximum_latitude=LAT_MAX,
    start_datetime=T_START,
    end_datetime=T_END,
    minimum_depth=0.5,
    maximum_depth=1000.0,
    output_filename="glorys_may2020.nc",
    output_directory="data/raw",
    force_download=True,
)
print("  -> glorys_may2020.nc saved.")

# ---- 2. Surface altimetry: SLA + geostrophic currents ----
print()
print("=" * 60)
print("[2/2] Downloading Altimetry SLA + geostrophic currents (Input X)")
print("      Variables: sla, ugos, vgos")
print("=" * 60)

copernicusmarine.subset(
    dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
    variables=["sla", "ugos", "vgos"],
    minimum_longitude=LON_MIN,
    maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN,
    maximum_latitude=LAT_MAX,
    start_datetime=T_START,
    end_datetime=T_END,
    output_filename="altimetry_may2020.nc",
    output_directory="data/raw",
    force_download=True,
)
print("  -> altimetry_may2020.nc saved.")

print()
print("=" * 60)
print("CMEMS download complete! Files in data/raw/")
print("=" * 60)

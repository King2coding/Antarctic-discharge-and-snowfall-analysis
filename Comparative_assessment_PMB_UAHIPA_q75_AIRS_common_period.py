#%%
"""Focused PMB/UA-HIPA q75/AIRS all-product comparison.

Interactive use
---------------
1. Run the established GPM V7/V8 comparison workflow through preparation of
   ``pmb_mon_01``, ``era5_mnth_01``, ``gpcp_mon_01``,
   ``gpm_pmw_v08_mon_01``, ``target_template_01deg``,
   ``basin_mask_01deg``, and ``region_masks_01deg``.
2. Run this file one numbered cell at a time.

This file writes only to a new UA-HIPA/AIRS analysis directory. It does not
write into the existing JGR-Atmospheres manuscript figure directories.
"""

#%%
# =============================================================================
# SECTION 1. IMPORTS AND ISOLATED OUTPUT PATHS
# =============================================================================

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - activates the xarray ``rio`` accessor
import xarray as xr
from rasterio.enums import Resampling

from comparative_uahipa_airs_common_period_utils import (
    paired_product_metrics,
    plot_regional_mean_annual_comparison,
    plot_regional_monthly_comparison,
    prepare_all_product_common_period,
    cosine_weighted_regional_series,
)


RUN_DATE = date.today().strftime("%Y%m%d")
MONTHLY_DATA_DIR = Path(
    "/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/"
    "data/uahipa_airs_monthly"
)
ANALYSIS_ROOT = Path(
    "/home/kkumah/Projects/Antarctic_discharge_work/"
    "uahipa_airs_comparison"
)
PLOT_DIR = ANALYSIS_ROOT / "plots"
TABLE_DIR = ANALYSIS_ROOT / "tables"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


#%%
# =============================================================================
# SECTION 2. VERIFY PREREQUISITE WORKSPACE VARIABLES
# =============================================================================

REQUIRED_EXISTING_VARIABLES = [
    "pmb_mon_01",
    "era5_mnth_01",
    "gpcp_mon_01",
    "gpm_pmw_v08_mon_01",
    "target_template_01deg",
    "basin_mask_01deg",
    "region_masks_01deg",
]

missing_variables = [name for name in REQUIRED_EXISTING_VARIABLES if name not in globals()]
if missing_variables:
    raise RuntimeError(
        "Run the established comparison workflow through monthly common-grid "
        f"product preparation first. Missing variables: {missing_variables}"
    )


#%%
# =============================================================================
# SECTION 3. FIND THE LATEST MONTHLY UA-HIPA q75 AND AIRS PRODUCTS
# =============================================================================

def latest_matching_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")
    return matches[-1]


UAHIPA_MONTHLY_FILE = latest_matching_file(
    MONTHLY_DATA_DIR,
    "uahipa_q75_equal_platform_monthly_*.nc",
)
AIRS_MONTHLY_FILE = latest_matching_file(MONTHLY_DATA_DIR, "airs_monthly_*.nc")

print("UA-HIPA monthly input:", UAHIPA_MONTHLY_FILE)
print("AIRS monthly input:    ", AIRS_MONTHLY_FILE)


#%%
# =============================================================================
# SECTION 4. LOAD AND REMAP UA-HIPA q75 TO THE COMMON 0.1-DEGREE GRID
# =============================================================================

with xr.open_dataset(UAHIPA_MONTHLY_FILE) as dataset:
    uahipa_q75_monthly = dataset["uahipa_q75_mm_month"].load()

uahipa_q75_monthly = (
    uahipa_q75_monthly
    .rename({"y": "lat", "x": "lon"})
    .sortby("lon")
    .sortby("lat", ascending=False)
)
uahipa_q75_monthly = uahipa_q75_monthly.rio.set_spatial_dims(
    x_dim="lon", y_dim="lat", inplace=False
)
uahipa_q75_monthly = uahipa_q75_monthly.rio.write_crs("EPSG:4326", inplace=False)

uahipa_q75_mon_01 = uahipa_q75_monthly.rio.reproject_match(
    target_template_01deg,
    resampling=Resampling.nearest,
)
uahipa_q75_mon_01 = uahipa_q75_mon_01.rename(
    {name: replacement for name, replacement in {"x": "lon", "y": "lat"}.items()
     if name in uahipa_q75_mon_01.dims}
)
uahipa_q75_mon_01 = (
    uahipa_q75_mon_01
    .sortby("lon")
    .sortby("lat", ascending=False)
    .where(basin_mask_01deg.notnull())
    .where(lambda field: field.lat < -60)
)


#%%
# =============================================================================
# SECTION 5. LOAD AND REMAP AIRS TO THE COMMON 0.1-DEGREE GRID
# =============================================================================

with xr.open_dataset(AIRS_MONTHLY_FILE) as dataset:
    airs_monthly = dataset["airs_precip_mm_month"].load()

airs_monthly = airs_monthly.sortby("lon").sortby("lat", ascending=False)
airs_monthly = airs_monthly.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
airs_monthly = airs_monthly.rio.write_crs("EPSG:4326", inplace=False)

airs_mon_01 = airs_monthly.rio.reproject_match(
    target_template_01deg,
    resampling=Resampling.nearest,
)
airs_mon_01 = airs_mon_01.rename(
    {name: replacement for name, replacement in {"x": "lon", "y": "lat"}.items()
     if name in airs_mon_01.dims}
)
airs_mon_01 = (
    airs_mon_01
    .sortby("lon")
    .sortby("lat", ascending=False)
    .where(basin_mask_01deg.notnull())
    .where(lambda field: field.lat < -60)
)


#%%
# =============================================================================
# SECTION 6. SELECT THE STRICT ALL-PRODUCT COMMON PERIOD
# =============================================================================

common_products, common_months = prepare_all_product_common_period(
    pmb=pmb_mon_01,
    era5=era5_mnth_01,
    gpcp_v33=gpcp_mon_01,
    gpm_pmw_v08=gpm_pmw_v08_mon_01,
    uahipa_q75=uahipa_q75_mon_01,
    airs=airs_mon_01,
)

COMMON_PERIOD_TAG = (
    f"{common_months.min():%Y%m}_{common_months.max():%Y%m}"
)

print("\nMonths used in every product:")
print([timestamp.strftime("%Y-%m") for timestamp in common_months])


#%%
# =============================================================================
# SECTION 7. BUILD REGIONAL MONTHLY SERIES AND PAIRED METRICS
# =============================================================================

regional_monthly_common = cosine_weighted_regional_series(
    products=common_products,
    region_masks=region_masks_01deg,
)
paired_metrics_common = paired_product_metrics(regional_monthly_common)

regional_monthly_output = TABLE_DIR / (
    f"regional_monthly_all_products_common_{COMMON_PERIOD_TAG}_{RUN_DATE}.csv"
)
paired_metrics_output = TABLE_DIR / (
    f"paired_metrics_against_PMB_common_{COMMON_PERIOD_TAG}_{RUN_DATE}.csv"
)

regional_monthly_common.to_csv(regional_monthly_output, index=False)
paired_metrics_common.to_csv(paired_metrics_output, index=False)

print("Saved:", regional_monthly_output)
print("Saved:", paired_metrics_output)
print(paired_metrics_common.to_string(index=False))


#%%
# =============================================================================
# SECTION 8. PLOT STRICTLY COINCIDENT MONTHLY SERIES
# =============================================================================

monthly_plot_output = PLOT_DIR / (
    f"regional_monthly_all_products_common_{COMMON_PERIOD_TAG}_{RUN_DATE}.png"
)
monthly_figure, monthly_axes = plot_regional_monthly_comparison(
    regional_monthly_common,
    output_path=monthly_plot_output,
)
print("Saved:", monthly_plot_output)


#%%
# =============================================================================
# SECTION 9. PLOT MEAN ANNUAL TOTALS FOR COMPLETE COMMON YEARS
# =============================================================================

annual_plot_output = PLOT_DIR / (
    f"regional_mean_annual_all_products_common_{COMMON_PERIOD_TAG}_{RUN_DATE}.png"
)
annual_figure, annual_axis, annual_values, annual_climatology = (
    plot_regional_mean_annual_comparison(
        regional_monthly_common,
        output_path=annual_plot_output,
    )
)

annual_values.to_csv(
    TABLE_DIR / f"regional_annual_all_products_common_{COMMON_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)
annual_climatology.to_csv(
    TABLE_DIR / f"regional_mean_annual_all_products_common_{COMMON_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)
print("Saved:", annual_plot_output)

#%%
"""Standalone monthly PMB/product comparison including UA-HIPA q75 and AIRS.

Run the numbered cells manually in one Python kernel. This file loads and
prepares every required product itself; it does not depend on variables created
by another analysis script. The final comparison uses exact common monthly
timestamps and complete calendar years only.
"""

#%%
# =============================================================================
# SECTION 1. IMPORTS
# =============================================================================

from datetime import date
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling

from program_utils import *
from Extra_util_functions import *

# This workflow is normally run cell-by-cell in a persistent kernel. Reload
# the edited project modules so saved utility/style updates are used rather
# than an older object retained in Python's import cache.
import program_utile_13Apr2026 as manuscript_utils
import comparative_uahipa_airs_common_period_utils as comparison_utils

importlib.reload(manuscript_utils)
importlib.reload(comparison_utils)

from program_utile_13Apr2026 import *
from comparative_uahipa_airs_common_period_utils import (
    basin_mean_annual_table,
    cosine_weighted_regional_series,
    paired_product_metrics,
    plot_figure4_seasonal,
    plot_regional_mean_annual_comparison,
    plot_regional_monthly_comparison,
    prepare_all_product_common_period,
    seasonal_climatology_with_common_month_fallback,
)

# Fail early rather than silently falling back to Matplotlib's blue/orange
# color cycle when an interactive kernel still holds an older utility module.
_required_extended_colors = {
    "UA-HIPA q75": "magenta",
    "AIRS": "#d62728",
}
_stale_or_missing_styles = {
    product: product_styles_corr.get(product, {}).get("color")
    for product, expected_color in _required_extended_colors.items()
    if product_styles_corr.get(product, {}).get("color") != expected_color
}
if _stale_or_missing_styles:
    raise RuntimeError(
        "The kernel has stale project plotting styles "
        f"({_stale_or_missing_styles}). Restart the kernel and rerun Section 1. "
        "UA-HIPA q75 must be magenta and AIRS must be #d62728 red."
    )
print(
    "Extended-product plot colors:",
    {name: product_styles_corr[name]["color"] for name in _required_extended_colors},
)


#%%
# =============================================================================
# SECTION 2. PATHS AND ISOLATED OUTPUT DIRECTORIES
# =============================================================================

GPM_ROOT = Path("/ra1/pubdat/GPM-Constellation-Satellites_MI_and_Sounders")
GPCP_ROOT = Path(
    "/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/"
    "GPCP_v3_pnt_3_monthly_1983_2024"
)
ERA5_FILE = Path(
    "/ra1/pubdat/GPCP/GPCP_Reproduce_GJ/era5_tp_198001202412_monthly.nc"
)
BASIN_ROOT = Path(
    "/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins"
)
PMB_FILE = BASIN_ROOT / (
    "Monthly_mass_budget_precip_RignotBasin_in_mm_forward_deltaS_"
    "uncorrected_positive_sublimation_loss_GRACE_updated_20260609.nc"
)
PMB_UNCERTAINTY_FILE = BASIN_ROOT / (
    "Monthly_mass_budget_precip_RignotBasin_uncertainty_in_mm_forward_"
    "deltaS_uncorrected_positive_sublimation_loss_GRACE_updated_20260609.nc"
)
MONTHLY_NEW_PRODUCT_ROOT = Path(
    "/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/"
    "data/uahipa_airs_monthly"
)

# These outputs are separate from the submitted-manuscript directories.
ANALYSIS_ROOT = Path(
    "/home/kkumah/Projects/Antarctic_discharge_work/uahipa_airs_comparison"
)
PLOT_DIR = ANALYSIS_ROOT / "plots"
TABLE_DIR = ANALYSIS_ROOT / "tables"
QC_DIR = ANALYSIS_ROOT / "qc"
for directory in [PLOT_DIR, TABLE_DIR, QC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RUN_DATE = date.today().strftime("%Y%m%d")

# Select the calendar year(s) to analyze.
#   None         -> use every complete year common to all six products
#   [2018]       -> analyze 2018 only
#   [2015, 2018] -> analyze those two years only (non-contiguous is allowed)
# Every coincident month is retained. Annual figures subsequently use only
# selected years containing all 12 coincident months.
ANALYSIS_YEARS = None


#%%
# =============================================================================
# SECTION 3. BASIN AND REGION DEFINITIONS
# =============================================================================

WAIS_BASINS = [10, 11, 12, 13, 14, 15, 16, 17]
EAIS_BASINS = [2, 3, 4, 5, 6, 7, 8, 9, 18, 19]
AIS_BASINS = WAIS_BASINS + EAIS_BASINS
REGION_BASINS = {
    "Antarctica": AIS_BASINS,
    "West Antarctica": WAIS_BASINS,
    "East Antarctica": EAIS_BASINS,
}


#%%
# =============================================================================
# SECTION 4. FIND AND INSPECT UA-HIPA q75 AND AIRS MONTHLY FILES
# =============================================================================

def latest_matching_file(directory, pattern):
    matches = sorted(Path(directory).glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")
    return matches[-1]


UAHIPA_MONTHLY_FILE = latest_matching_file(
    MONTHLY_NEW_PRODUCT_ROOT, "uahipa_q75_equal_platform_monthly_*.nc"
)
AIRS_MONTHLY_FILE = latest_matching_file(
    MONTHLY_NEW_PRODUCT_ROOT, "airs_monthly_*.nc"
)

with xr.open_dataset(UAHIPA_MONTHLY_FILE) as dataset:
    uahipa_q75_native = dataset["uahipa_q75_mm_month"].load()
    uahipa_valid_platform_count = dataset["valid_platform_count"].load()
with xr.open_dataset(AIRS_MONTHLY_FILE) as dataset:
    airs_native = dataset["airs_precip_mm_month"].load()
    airs_valid_day_count = dataset["airs_valid_day_count"].load()

uahipa_available_months = pd.DatetimeIndex(
    pd.to_datetime(uahipa_q75_native.time.values)
).to_period("M").to_timestamp()
airs_available_months = pd.DatetimeIndex(
    pd.to_datetime(airs_native.time.values)
).to_period("M").to_timestamp()
CANDIDATE_MONTHS = uahipa_available_months.intersection(airs_available_months)
if len(CANDIDATE_MONTHS) == 0:
    raise RuntimeError("UA-HIPA and AIRS have no common monthly timestamps")

candidate_counts_by_year = (
    pd.Series(CANDIDATE_MONTHS.year, dtype="int64").value_counts().sort_index()
)
print("UA-HIPA/AIRS common-month counts by year:")
print(candidate_counts_by_year.to_string())

if ANALYSIS_YEARS is not None:
    requested_years = sorted({int(year) for year in ANALYSIS_YEARS})
    if not requested_years:
        raise ValueError("ANALYSIS_YEARS cannot be an empty list")
    unavailable_years = [
        year for year in requested_years if year not in candidate_counts_by_year.index
    ]
    if unavailable_years:
        raise ValueError(
            "Requested ANALYSIS_YEARS have no common UA-HIPA/AIRS months: "
            f"{unavailable_years}"
        )
    CANDIDATE_MONTHS = CANDIDATE_MONTHS[
        CANDIDATE_MONTHS.year.astype(int).isin(requested_years)
    ]

CANDIDATE_START = CANDIDATE_MONTHS.min()
CANDIDATE_END = CANDIDATE_MONTHS.max()

print("UA-HIPA file:", UAHIPA_MONTHLY_FILE)
print("AIRS file:    ", AIRS_MONTHLY_FILE)
print(
    "Selected candidate months:",
    [timestamp.strftime("%Y-%m") for timestamp in CANDIDATE_MONTHS],
)


#%%
# =============================================================================
# SECTION 5. COMMON 0.1-DEGREE WGS84 BASIN GRID
# =============================================================================

basins = load_basin_grid(str(BASIN_ROOT), crs_stereo)
print_basin_grid_info(basins)
target_template_01deg = build_target_latlon_template_from_basin_grid(basins)
basin_mask_01deg = reproject_basin_ids_to_target_grid(basins, target_template_01deg)
region_masks_01deg = make_region_masks_from_basin_mask(
    basin_mask_01deg, REGION_BASINS
)
print("Common comparison CRS:", target_template_01deg.rio.crs)
print("Common target dimensions:", target_template_01deg.sizes)


#%%
# =============================================================================
# SECTION 6. SHARED GEOGRAPHIC REMAPPING FUNCTION
# =============================================================================

def remap_geographic_monthly_to_common_grid(
    field,
    target_template,
    basin_mask,
    lat_name="lat",
    lon_name="lon",
    resampling=Resampling.nearest,
):
    """Remap monthly WGS84 lat-lon data to the established 0.1-degree grid."""

    field = field.assign_coords(
        time=pd.to_datetime(field.time.values).to_period("M").to_timestamp()
    )
    field = field.sortby(lon_name).sortby(lat_name, ascending=False)
    field = field.rio.set_spatial_dims(
        x_dim=lon_name, y_dim=lat_name, inplace=False
    )
    field = field.rio.write_crs("EPSG:4326", inplace=False)
    remapped = field.rio.reproject_match(target_template, resampling=resampling)
    rename_map = {}
    if "x" in remapped.dims:
        rename_map["x"] = "lon"
    if "y" in remapped.dims:
        rename_map["y"] = "lat"
    if rename_map:
        remapped = remapped.rename(rename_map)
    remapped = remapped.sortby("lon").sortby("lat", ascending=False)
    remapped = remapped.where(basin_mask.notnull())
    return remapped.where(remapped.lat < -60)


#%%
# =============================================================================
# SECTION 7. LOAD AND PREPARE PMB AND PMB UNCERTAINTY
# =============================================================================

P_mm_mnth = xr.open_dataarray(PMB_FILE)
P_unc_mm_mnth = xr.open_dataarray(PMB_UNCERTAINTY_FILE)
pmb_mon_01 = prepare_pmb_monthly_on_target(
    P_mm_mnth, target_template_01deg
).sel(time=slice(CANDIDATE_START, CANDIDATE_END))
pmb_unc_mon_01 = abs(
    prepare_pmb_monthly_on_target(P_unc_mm_mnth, target_template_01deg)
    .sel(time=slice(CANDIDATE_START, CANDIDATE_END))
)
pmb_mon_01 = pmb_mon_01.where(basin_mask_01deg.notnull())
pmb_unc_mon_01 = pmb_unc_mon_01.where(basin_mask_01deg.notnull())


#%%
# =============================================================================
# SECTION 8. LOAD AND PREPARE GPCP V3.3
# =============================================================================

gpcp_files = []
for path in sorted(GPCP_ROOT.glob("*.nc4")):
    try:
        file_year = int(path.name.split("_")[2][:4])
    except (IndexError, ValueError):
        continue
    if CANDIDATE_START.year <= file_year <= CANDIDATE_END.year:
        gpcp_files.append(str(path))
if not gpcp_files:
    raise FileNotFoundError("No GPCP v3.3 files found for the candidate years")

gpcp_dataset = xr.open_mfdataset(
    gpcp_files, combine="nested", concat_dim="time", coords="minimal",
    compat="override", parallel=True, engine="netcdf4", cache=False,
)
gpcp_dataset = ds_swaplon(gpcp_dataset)
gpcp_monthly = gpcp_dataset["sat_gauge_precip"].copy()
gpcp_monthly = gpcp_monthly.assign_coords(
    time=pd.to_datetime(gpcp_monthly.time.values).to_period("M").to_timestamp()
)
gpcp_days = xr.DataArray(
    pd.to_datetime(gpcp_monthly.time.values).days_in_month,
    dims=["time"], coords={"time": gpcp_monthly.time},
)
gpcp_monthly = (gpcp_monthly * gpcp_days).where(np.isfinite(gpcp_monthly))
gpcp_monthly = gpcp_monthly.sel(time=slice(CANDIDATE_START, CANDIDATE_END))
gpcp_mon_01 = remap_geographic_monthly_to_common_grid(
    gpcp_monthly.sel(lat=slice(-60, -90)),
    target_template_01deg, basin_mask_01deg,
)


#%%
# =============================================================================
# SECTION 9. LOAD AND PREPARE ERA5
# =============================================================================

era5_dataset = xr.open_dataset(ERA5_FILE, engine="netcdf4")[["tp"]]
era5_dataset = ds_swaplon(era5_dataset)
era5_dataset = replace_fill_with_nan(era5_dataset)
if "valid_time" in era5_dataset.dims or "valid_time" in era5_dataset.coords:
    era5_dataset = era5_dataset.rename({"valid_time": "time"})
era5_dataset = era5_dataset.assign_coords(
    time=pd.to_datetime(era5_dataset.time.values).to_period("M").to_timestamp()
)
era5_days = xr.DataArray(
    era5_dataset.time.dt.days_in_month,
    dims=["time"], coords={"time": era5_dataset.time},
)
era5_monthly = era5_dataset["tp"] * 1000.0 * era5_days
era5_monthly = era5_monthly.rename({"latitude": "lat", "longitude": "lon"})
era5_monthly = era5_monthly.sel(time=slice(CANDIDATE_START, CANDIDATE_END))
era5_monthly = era5_monthly.sel(lat=slice(-60, -90))
era5_mnth_01 = remap_geographic_monthly_to_common_grid(
    era5_monthly, target_template_01deg, basin_mask_01deg,
)


#%%
# =============================================================================
# SECTION 10. PREPARE UA-HIPA q75 AND AIRS ON THE COMMON GRID
# =============================================================================

uahipa_q75_native = uahipa_q75_native.rename({"y": "lat", "x": "lon"})
uahipa_q75_mon_01 = remap_geographic_monthly_to_common_grid(
    uahipa_q75_native, target_template_01deg, basin_mask_01deg,
)
airs_mon_01 = remap_geographic_monthly_to_common_grid(
    airs_native, target_template_01deg, basin_mask_01deg,
)


#%%
# =============================================================================
# SECTION 11. BUILD GPM/GPROF V08 PMW FOR THE CANDIDATE PERIOD
# =============================================================================

gpm_time_axis = pd.date_range(CANDIDATE_START, CANDIDATE_END, freq="MS")
(
    gpm_v8_family_monthly_dict,
    gpm_v8_platform_monthly_dict,
    gpm_v8_inventory_df,
) = build_gpm_v8_family_monthly_dict(
    gpm_satellites_path=str(GPM_ROOT),
    target_template_01deg=target_template_01deg,
    basin_mask_01deg=basin_mask_01deg,
    start_time=CANDIDATE_START.strftime("%Y-%m-%d"),
    end_time=CANDIDATE_END.strftime("%Y-%m-%d"),
    convert_rate_to_month=True,
    min_valid_platforms_by_family={
        "ATMS": 1, "MHS": 1, "DMSP-SSMIS": 1, "AMSR2": 1,
    },
    return_platform_dict=True,
)
gpm_pmw_v08_mon_01 = build_gpm_pmw_v8_mean(
    gpm_v8_family_dict=gpm_v8_family_monthly_dict,
    mean_name="GPM PMW V08",
    common_months=gpm_time_axis,
    min_valid_families=4,
)
gpm_v8_inventory_df.to_csv(
    QC_DIR / f"gpm_v8_platform_inventory_{RUN_DATE}.csv", index=False,
)


#%%
# =============================================================================
# SECTION 12. STRICT SIX-PRODUCT TEMPORAL COINCIDENCE
# =============================================================================

comparison_inputs = {
    "pmb": pmb_mon_01,
    "era5": era5_mnth_01,
    "gpcp_v33": gpcp_mon_01,
    "gpm_pmw_v08": gpm_pmw_v08_mon_01,
    "uahipa_q75": uahipa_q75_mon_01,
    "airs": airs_mon_01,
}
# Apply the exact candidate-month selection before the strict six-product
# intersection. This supports non-contiguous selections such as [2015, 2018].
comparison_inputs = {
    name: field.where(field.time.isin(CANDIDATE_MONTHS), drop=True)
    for name, field in comparison_inputs.items()
}
common_products, common_months = prepare_all_product_common_period(
    **comparison_inputs,
    require_complete_calendar_years=False,
)
COMMON_YEARS = sorted({int(year) for year in common_months.year})
if ANALYSIS_YEARS is not None:
    omitted_requested_years = sorted(set(requested_years) - set(COMMON_YEARS))
    if omitted_requested_years:
        raise ValueError(
            "Requested years were removed by the strict six-product monthly "
            f"intersection: {omitted_requested_years}. Check the per-product month "
            "inventory printed above."
        )

# Separate the exact monthly comparison period from annual-capable years.
common_month_counts = pd.Series(common_months.year).value_counts().sort_index()
COMPLETE_COMMON_YEARS = sorted(
    int(year) for year, count in common_month_counts.items() if int(count) == 12
)
if not COMPLETE_COMMON_YEARS:
    raise ValueError(
        "No selected year contains 12 coincident months across all six products. "
        "Monthly and seasonal comparisons are possible, but Figures 2, 3, 5, 6, "
        "and 7 require at least one complete year."
    )
annual_common_months = common_months[
    common_months.year.astype(int).isin(COMPLETE_COMMON_YEARS)
]
annual_products = {
    name: field.sel(time=annual_common_months)
    for name, field in common_products.items()
}

# Explicit month counts avoid implying that non-contiguous years form one range.
COMMON_PERIOD_TAG = "common_" + "_".join(
    f"{int(year)}_{int(count):02d}m" for year, count in common_month_counts.items()
)
ANNUAL_PERIOD_TAG = "annual_years_" + "_".join(
    str(year) for year in COMPLETE_COMMON_YEARS
)
COMMON_YEAR_LABEL = ", ".join(str(year) for year in COMPLETE_COMMON_YEARS)
print("\nExact months used by every product:")
print([timestamp.strftime("%Y-%m") for timestamp in common_months])
print("Coincident month counts by year:", common_month_counts.to_dict())
print("Complete years used by annual figures:", COMPLETE_COMMON_YEARS)
print("Output period tag:", COMMON_PERIOD_TAG)
print("Annual-output tag:", ANNUAL_PERIOD_TAG)


#%%
# =============================================================================
# SECTION 13. MONTHLY COVERAGE AND SAMPLING QC
# =============================================================================

coverage_rows = []
for product, field in common_products.items():
    for timestamp in common_months:
        monthly_field = field.sel(time=timestamp)
        coverage_rows.append({
            "product": product,
            "time": timestamp,
            "valid_pixels": int(monthly_field.notnull().sum()),
            "total_pixels": int(monthly_field.size),
            "valid_fraction": float(monthly_field.notnull().mean()),
        })
coverage_df = pd.DataFrame(coverage_rows)
coverage_df.to_csv(
    QC_DIR / f"all_product_monthly_coverage_{COMMON_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)
uahipa_valid_platform_count.max(("y", "x")).to_dataframe(
    name="maximum_valid_platforms"
).reset_index().to_csv(QC_DIR / f"uahipa_platform_coverage_{RUN_DATE}.csv", index=False)
airs_valid_day_count.min(("lat", "lon")).to_dataframe(
    name="minimum_valid_days"
).reset_index().to_csv(QC_DIR / f"airs_daily_coverage_{RUN_DATE}.csv", index=False)
print(coverage_df.groupby("product")["valid_fraction"].agg(["min", "mean", "max"]))


#%%
# =============================================================================
# SECTION 14. REGIONAL MONTHLY SERIES AND PAIRED METRICS
# =============================================================================

regional_monthly_common = cosine_weighted_regional_series(
    products=common_products, region_masks=region_masks_01deg,
)
regional_metrics_common = paired_product_metrics(regional_monthly_common)
regional_monthly_common.to_csv(
    TABLE_DIR / f"regional_monthly_all_products_{COMMON_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)
regional_metrics_common.to_csv(
    TABLE_DIR / f"regional_metrics_against_PMB_{COMMON_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)
print(regional_metrics_common.to_string(index=False))


#%%
# =============================================================================
# SECTION 15. IMBIE BASIN MONTHLY SERIES AND PAIRED METRICS
# =============================================================================

BASIN_NAMES = {
    2: "A-Ap", 3: "Ap-B", 4: "B-C", 5: "C-Cp", 6: "Cp-D",
    7: "D-Dp", 8: "Dp-E", 9: "E-Ep", 10: "Ep-F", 11: "F-G",
    12: "G-H", 13: "H-Hp", 14: "Hp-I", 15: "I-Ipp", 16: "Ipp-J",
    17: "J-Jpp", 18: "Jpp-K", 19: "K-A",
}
basin_masks_01deg = {
    f"{basin_id:02d} {basin_name}": basin_mask_01deg == basin_id
    for basin_id, basin_name in BASIN_NAMES.items()
}
basin_monthly_common = cosine_weighted_regional_series(
    products=common_products, region_masks=basin_masks_01deg,
)
basin_metrics_common = paired_product_metrics(basin_monthly_common)
basin_monthly_common.to_csv(
    TABLE_DIR / f"imbie_basin_monthly_all_products_{COMMON_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)
basin_metrics_common.to_csv(
    TABLE_DIR / f"imbie_basin_metrics_against_PMB_{COMMON_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)

# Mean-annual basin table shared by manuscript-analogous Figures 2, 6, and 7.
basin_mean_annual_common = basin_mean_annual_table(basin_monthly_common)
basin_mean_annual_common.to_csv(
    TABLE_DIR / f"imbie_basin_mean_annual_all_products_{ANNUAL_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)

# Figure 2 in R1 also shows the individual V08 sensor-family estimates.  Add
# those panels here while retaining the six-product constellation comparison
# used everywhere else.  The family fields are restricted to the same exact
# common months, so they cannot extend the comparison period silently.
figure2_products = dict(common_products)
for family_name, family_field in gpm_v8_family_monthly_dict.items():
    figure2_products[f"{family_name} V08"] = family_field.sel(time=common_months)
figure2_basin_monthly = cosine_weighted_regional_series(
    products=figure2_products, region_masks=basin_masks_01deg,
)
figure2_basin_mean_annual = basin_mean_annual_table(figure2_basin_monthly)
figure2_basin_mean_annual.to_csv(
    TABLE_DIR / f"Figure_2_basin_mean_annual_with_V08_families_{ANNUAL_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)

# Reuse the established PMB uncertainty propagation.  The field is constant
# within each basin, so its pixels are not independent: recover one uncertainty
# per basin and propagate the regional weighted mean in quadrature.
regional_pmb_uncertainty = build_region_monthly_uncertainty_from_basin_painted_field(
    unc_da=pmb_unc_mon_01.sel(time=common_months),
    basin_mask=basin_mask_01deg,
    region_defs=REGION_BASINS,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
    value_name="pmb_uncertainty",
)
regional_pmb_uncertainty.to_csv(
    TABLE_DIR / f"regional_PMB_uncertainty_{COMMON_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)
regional_annual_pmb_uncertainty = compute_annual_uncertainty_from_monthly_unc_df(
    regional_pmb_uncertainty,
    min_months_per_year=12,
)


#%%
# =============================================================================
# SECTION 16. SUPPLEMENTAL MONTHLY COMPARISON DIAGNOSTIC
# =============================================================================

monthly_plot_path = PLOT_DIR / (
    f"regional_monthly_all_products_{COMMON_PERIOD_TAG}_{RUN_DATE}.png"
)
monthly_figure, monthly_axes = plot_regional_monthly_comparison(
    regional_monthly_common, output_path=monthly_plot_path,
)
print("Saved:", monthly_plot_path)


#%%
# =============================================================================
# SECTION 17. FIGURE 2 ANALOGUE: BASIN-SCALE MEAN-ANNUAL MAPS
# =============================================================================

figure2_path = PLOT_DIR / (
    f"Figure_2_basin_mean_annual_all_products_{ANNUAL_PERIOD_TAG}_{RUN_DATE}.png"
)
figure2_fields = [
    (r"P$_{MB}$", annual_products[r"$P_{\mathrm{MB}}$"]),
    ("ERA5", annual_products["ERA5"]),
    ("GPCP V3.3", annual_products["GPCP V3.3"]),
    ("UA-HIPA q75", annual_products["UA-HIPA q75"]),
    ("AIRS", annual_products["AIRS"]),
]
# Preserve the R1 sensor-family order and avoid appending "V08" when it is
# already part of the dictionary key.
for requested_family in ("ATMS", "MHS", "DMSP SSMIS", "AMSR2"):
    matching_keys = [
        key for key in gpm_v8_family_monthly_dict
        if requested_family.replace("-", " ") in key.replace("-", " ")
    ]
    if not matching_keys:
        raise KeyError(f"Missing expected GPM V08 family: {requested_family}")
    family_key = matching_keys[0]
    family_label = family_key if "V08" in family_key else f"{family_key} V08"
    figure2_fields.append(
        (
            family_label,
            gpm_v8_family_monthly_dict[family_key].sel(time=annual_common_months),
        )
    )
figure2_fields.append(("GPM PMW V08", annual_products["GPM PMW V08"]))

figure2_packs = [
    build_basin_mean_plot_product(
        field,
        basin_mask_01deg.where(basin_mask_01deg.isin(AIS_BASINS)),
        AIS_BASINS,
        label,
        year_start=min(COMPLETE_COMMON_YEARS),
        year_end=max(COMPLETE_COMMON_YEARS),
    )
    for label, field in figure2_fields
]
figure2_arr = [
    (pack["product"], pack["plot_grid"], pack["panel_mean"])
    for pack in figure2_packs
]
figure2, figure2_axes, figure2_colorbar = (
    compare_mean_precip_basin_v7_v8_common_cbar(
        arr_lst_mean=figure2_arr,
        basin_mask_latlon=basin_mask_01deg,
        row1_idx=list(range(0, 5)),
        row2_idx=list(range(5, len(figure2_arr))),
        row3_idx=[],
        ncols=5,
        gamma=0.6,
        vmin=0,
        vmax=400,
        cbar_ticks=[0, 25, 50, 100, 200, 300, 400],
        cbar_label=r"Mean annual precipitation (mm yr$^{-1}$)",
        panel_letters=True,
        show_panel_mean=True,
    )
)
figure2.savefig(figure2_path, dpi=150, bbox_inches="tight")
print("Saved:", figure2_path)


#%%
# =============================================================================
# SECTION 18. FIGURE 3 ANALOGUE: REGIONAL MEAN-ANNUAL BARS
# =============================================================================

figure3_path = PLOT_DIR / (
    f"Figure_3_regional_mean_annual_all_products_{ANNUAL_PERIOD_TAG}_{RUN_DATE}.png"
)
regional_annual_r1 = compute_annual_totals_from_regional_series(
    regional_monthly_common.rename(columns={"precipitation_mm_month": "precipitation"})
)
regional_annual_climatology_r1 = compute_regional_mean_annual_precip(
    regional_annual_r1,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=tuple(common_products),
)
regional_annual_values = regional_annual_r1.rename(
    columns={"precipitation": "precipitation_mm_year"}
)
regional_annual_climatology = regional_annual_climatology_r1.rename(
    columns={"precipitation": "precipitation_mm_year"}
)
figure3, figure3_axis = plot_regional_mean_annual_bars(
    regional_annual_climatology_r1,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=tuple(common_products),
    product_colors=product_styles_corr,
    figsize=(10, 6),
    ylabel="[mm/year]",
    # Keep the plotting canvas title-free for slide assembly.  The complete
    # annual years remain explicit in ``figure3_path`` via ANNUAL_PERIOD_TAG.
    title=None,
    annotate=True,
)
figure3.savefig(figure3_path, dpi=150, bbox_inches="tight")
regional_annual_values.to_csv(
    TABLE_DIR / f"regional_annual_all_products_{ANNUAL_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)
regional_annual_climatology.to_csv(
    TABLE_DIR / f"regional_mean_annual_all_products_{ANNUAL_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)
print("Saved:", figure3_path)


#%%
# =============================================================================
# SECTION 19. FIGURE 4 ANALOGUE: REGIONAL SEASONAL CLIMATOLOGY
# =============================================================================

# Preserve the established complete-season calculation whenever complete
# conventional seasons exist.  Only when a named season has no complete case
# (currently DJF) use at least two jointly available months across all products
# and estimate the seasonal total as 3 * their monthly mean.
(
    regional_seasonal_climatology,
    regional_seasonal_pmb_uncertainty,
    regional_seasonal_by_year,
) = seasonal_climatology_with_common_month_fallback(
    regional_monthly=regional_monthly_common,
    regional_pmb_uncertainty=regional_pmb_uncertainty,
    min_common_months=2,
)
regional_monthly_for_r1 = regional_monthly_common.rename(
    columns={"precipitation_mm_month": "precipitation"}
)

figure4_path = PLOT_DIR / (
    f"Figure_4_regional_seasonal_all_products_{COMMON_PERIOD_TAG}_{RUN_DATE}.png"
)
figure4, figure4_axes, regional_seasonal_climatology = plot_figure4_seasonal(
    regional_seasonal_climatology,
    pmb_uncertainty_monthly=regional_seasonal_pmb_uncertainty,
    output_path=figure4_path,
)
regional_seasonal_climatology.to_csv(
    TABLE_DIR / f"regional_seasonal_all_products_{COMMON_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)
regional_seasonal_by_year.to_csv(
    TABLE_DIR / f"regional_seasonal_coverage_details_{COMMON_PERIOD_TAG}_{RUN_DATE}.csv",
    index=False,
)
print("Saved:", figure4_path)


#%%
# =============================================================================
# SECTION 20. FIGURE 5 PLACEHOLDER: REGIONAL INTERANNUAL VARIABILITY
# =============================================================================

figure5_status = "PLACEHOLDER_" if len(COMPLETE_COMMON_YEARS) < 2 else ""
figure5_path = PLOT_DIR / (
    f"Figure_5_{figure5_status}interannual_all_products_{ANNUAL_PERIOD_TAG}_{RUN_DATE}.png"
)
regional_annual_for_figure5 = compute_annual_totals_from_regional_series(
    regional_monthly_for_r1
)
regional_annual_for_figure5 = regional_annual_for_figure5[
    regional_annual_for_figure5["year"].isin(COMPLETE_COMMON_YEARS)
].copy()
figure5, figure5_axes = plot_interannual_variability(
    regional_annual_for_figure5,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=tuple(common_products),
    # Use a single ink color and product-specific line/marker patterns for
    # direct comparison of temporal behaviour and grayscale readability.
    product_styles=product_line_styles_monochrome,
    figsize=(10, 9),
    ylabel="mm/year",
    y_nbins=4,
    legend_ncol=3,
)
for axis, region in zip(figure5_axes, ("Antarctica", "West Antarctica", "East Antarctica")):
    pmb = regional_annual_for_figure5[
        (regional_annual_for_figure5.region == region) &
        (regional_annual_for_figure5["product"] == r"$P_{\mathrm{MB}}$")
    ].set_index("year")
    unc = regional_annual_pmb_uncertainty[
        regional_annual_pmb_uncertainty.region == region
    ].set_index("year")
    years = pmb.index.intersection(unc.index)
    axis.fill_between(years, pmb.loc[years, "precipitation"] - unc.loc[years, "pmb_uncertainty"],
                      pmb.loc[years, "precipitation"] + unc.loc[years, "pmb_uncertainty"],
                      color="0.8", alpha=0.6, linewidth=0, zorder=1)
    if len(COMPLETE_COMMON_YEARS) < 2:
        axis.text(
            0.5, 0.08,
            "PLACEHOLDER: interannual variability requires ≥2 complete years",
            transform=axis.transAxes, ha="center", fontsize=9,
        )
figure5.savefig(figure5_path, dpi=150, bbox_inches="tight")
print("Saved:", figure5_path)


#%%
# =============================================================================
# SECTION 21. FIGURE 6 ANALOGUE: BASIN SCATTERS AGAINST PMB
# =============================================================================

figure6_path = PLOT_DIR / (
    f"Figure_6_basin_scatter_against_PMB_{ANNUAL_PERIOD_TAG}_{RUN_DATE}.png"
)
basin_mean_annual_r1 = basin_mean_annual_common.rename(
    columns={"basin_id": "basin"}
).drop(columns="region")
figure6, figure6_axes, figure6_stats = plot_pmb_scatter_oldstyle(
    df_mean_yr_acc=basin_mean_annual_r1,
    ref=r"$P_{\mathrm{MB}}$",
    products=["ERA5", "GPCP V3.3", "GPM PMW V08", "UA-HIPA q75", "AIRS"],
    high_thresh=500.0,
    scale="log",
    log_min=2,
    log_ticks=(5, 10, 20, 50, 100, 200, 500, 1000, 2000),
    ncols=3,
    figsize_per_col=5.0,
    figsize_per_row=5.0,
    share_axes=False,
    show_ylabel_only_left=False,
)
# Preserve the established function signature; make this figure-specific
# spacing adjustment only after the reusable function returns.
figure6.subplots_adjust(wspace=0.20, hspace=0.38)
figure6.savefig(figure6_path, dpi=150, bbox_inches="tight")
print("Saved:", figure6_path)


#%%
# =============================================================================
# SECTION 22. FIGURE 7 ANALOGUE: BASIN PRODUCT SPREAD
# =============================================================================

figure7_path = PLOT_DIR / (
    f"Figure_7_basin_product_spread_{ANNUAL_PERIOD_TAG}_{RUN_DATE}.png"
)
figure7, figure7_axes, figure7_spreads = plot_basin_spread_points_by_region(
    df=basin_mean_annual_r1,
    basin_col="basin",
    ref_col=r"$P_{\mathrm{MB}}$",
    prod_cols=("ERA5", "GPCP V3.3", "GPM PMW V08", "UA-HIPA q75", "AIRS"),
    product_styles=product_styles_corr,
    non_gpm_group=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP V3.3", "GPM PMW V08", "UA-HIPA q75", "AIRS"),
    gpm_group=(r"$P_{\mathrm{MB}}$", "GPM PMW V08"),
    wais_basins=WAIS_BASINS,
    eais_basins=EAIS_BASINS,
    figsize=(13, 7.2),
    legend_ncol=6,
    place_key=False,
)
figure7_axes[0].text(
    0.01,
    0.96,
    r"% = spread($P_{\mathrm{MB}}$, ERA5, GPCP V3.3, "
    "GPM PMW V08, UA-HIPA q75, AIRS)",
    transform=figure7_axes[0].transAxes,
    ha="left",
    va="top",
    fontsize=14,
    color="black",
    fontweight="bold",
)
figure7.savefig(figure7_path, dpi=150, bbox_inches="tight")
print("Saved:", figure7_path)


#%%
# =============================================================================
# SECTION 23. FINAL RUN SUMMARY
# =============================================================================

print("\nComparison completed.")
print("Coincident monthly/seasonal period:", COMMON_PERIOD_TAG)
print("Complete years used for annual figures:", COMPLETE_COMMON_YEARS)
print("Products:", list(common_products))
print("Plots:", PLOT_DIR)
print("Tables:", TABLE_DIR)
print("QC outputs:", QC_DIR)

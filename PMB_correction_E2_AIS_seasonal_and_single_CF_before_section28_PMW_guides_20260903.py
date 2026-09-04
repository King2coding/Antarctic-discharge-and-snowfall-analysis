#%%
# =============================================================================
# SECTION 1. IMPORTS AND BASIC SETUP
# =============================================================================

from program_utils import *
from Extra_util_functions import *
from program_utile_13Apr2026 import *

#%%
# =============================================================================
# SECTION 2. PATHS AND FILE LISTS
# =============================================================================

# path to put outs e.g. plots, dfs
path_to_plots = r'/home/kkumah/Projects/Antarctic_discharge_work/antarctica_coef_work_results/plots'
path_to_dfs = r'/home/kkumah/Projects/Antarctic_discharge_work/antarctica_coef_work_results/dfs'
gpm_satellites_path = r'/ra1/pubdat/GPM-Constellation-Satellites_MI_and_Sounders'
# --- Precipitation products ---
gpcp_v3pt3_mnthly_ds_path = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_3_monthly_1983_2024'
era5_mnhtly_file = r'/ra1/pubdat/GPCP/GPCP_Reproduce_GJ/era5_tp_198001202412_monthly.nc'


# --- Basin / PMB data ---
# fnme = "Monthly_mass_budget_precip_RignotBasin_in_mm_forward_deltaS_uncorrected_positive_sublimation_loss_20260507.nc"
fnme = "Monthly_mass_budget_precip_RignotBasin_in_mm_forward_deltaS_uncorrected_positive_sublimation_loss_GRACE_updated_20260609.nc"
basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'
Pmb_mm_fle  = os.path.join(
    basins_path, 
    fnme
)
#    'Monthly_mass_budget_precip_RignotBasin_in_mm_20260226.nc')
# fnme = "Monthly_mass_budget_precip_RignotBasin_uncertainty_in_mm_forward_deltaS_uncorrected_positive_sublimation_loss_20260519.nc"
fnme = "Monthly_mass_budget_precip_RignotBasin_uncertainty_in_mm_forward_deltaS_uncorrected_positive_sublimation_loss_GRACE_updated_20260609.nc"
Pmb_unc_mm_fle = os.path.join(
    basins_path,
    fnme
)
# --- File lists: 2013–2020 only ---
all_gpcp_v3pt3_mnthly_files = sorted(
    [os.path.join(gpcp_v3pt3_mnthly_ds_path, f) for f in os.listdir(gpcp_v3pt3_mnthly_ds_path) if f.endswith('.nc4')]
    )

all_gpcp_v3pt3_mnthly_files_2013_2020 = [
    f for f in all_gpcp_v3pt3_mnthly_files
    if 2013 <= int(os.path.basename(f).split('_')[2][:4]) <= 2020
]


# =============================================================================
# SECTION 3. BASIN DEFINITIONS
# =============================================================================

WAIS_BASINS = [10, 11, 12, 13, 14, 15, 16, 17]
EAIS_BASINS = [2, 3, 4, 5, 6, 7, 8, 9, 18, 19]
AIS_BASINS  = WAIS_BASINS + EAIS_BASINS

REGION_BASINS = {
    "Antarctica": AIS_BASINS,
    "West Antarctica": WAIS_BASINS,
    "East Antarctica": EAIS_BASINS,
}

ANNUAL_YEAR_START = 2013
ANNUAL_YEAR_END = 2020
ANNUAL_PERIOD_TAG = f"{ANNUAL_YEAR_START}_{ANNUAL_YEAR_END}"


# =============================================================================
# SECTION 4. LOAD BASIN GRID AND BUILD COMMON 0.1° TARGET GRID
# =============================================================================

basins = load_basin_grid(basins_path, crs_stereo)
print_basin_grid_info(basins)

# Common 0.1° comparison grid in lat-lon, derived from basin geometry
target_template_01deg = build_target_latlon_template_from_basin_grid(basins)

# Basin IDs remapped onto the same common target grid
basin_mask_01deg = reproject_basin_ids_to_target_grid(basins, target_template_01deg)

# Region masks on the same target grid
region_masks_01deg = make_region_masks_from_basin_mask(basin_mask_01deg, REGION_BASINS)

print("✅ Common 0.1° target grid ready")
print("Target dims:", target_template_01deg.dims)
print("Basin-mask dims:", basin_mask_01deg.dims)

# =============================================================================
# LOAD BASIN GRID AND BUILD COMMON 0.1° TARGET GRID
# =============================================================================

CALIBRATION_YEARS = [2013, 2014, 2015, 2016, 2017]
VALIDATION_YEARS  = [2018, 2019, 2020]

# Final deliverable factors are derived only after the independent validation
# experiment has been defined. They use all available complete seasons in the
# full PMB record. These factors are deliverables, not independent-validation
# coefficients, because 2018-2020 contribute to their estimation.
FINAL_FACTOR_YEARS = list(range(2013, 2021))

REFERENCE_NAME = r"$P_{\mathrm{MB}}$"

GPCP_NAME = "GPCP V3.3"
GPCP_CORR_NAME = "GPCP V3.3 corrected"
GPCP_DEADJ_NAME = "GPCP V3.3 / 1.4"

PMW8_NAME = "GPM PMW V08"
PMW8_CORR_NAME = "GPM PMW V08 corrected"

GPCP_SINGLE_CORR_NAME = "GPCP V3.3 single-CF corrected"
PMW8_SINGLE_CORR_NAME = "GPM PMW V08 single-CF corrected"

ERA5_NAME = "ERA5"

#%%
# =============================================================================
# SECTION 5. LOAD RAW PRODUCT DATA
# =============================================================================

print("Loading GPCP monthly dataset ...")


# -------------------------------------------------------------------------
# Build the expected monthly time coordinate directly from the filenames.
#
# Example:
#   GPCPMON_L3_201710_V3.3.nc4
#                    ^^^^^^
#                    YYYYMM
#
# We use the filename as the authoritative monthly timestamp because
# open_mfdataset() was found to introduce one duplicate timestamp
# (2018-11) and lose 2017-10 even though the individual source files
# themselves contain the correct timestamps.
# -------------------------------------------------------------------------

gpcp_time_from_files = pd.DatetimeIndex([
    pd.to_datetime(
        os.path.basename(f).split("_")[2],
        format="%Y%m"
    )
    for f in all_gpcp_v3pt3_mnthly_files_2013_2020
])


# -------------------------------------------------------------------------
# Load the monthly GPCP files in their sorted filename order.
# -------------------------------------------------------------------------

gpcp_ds_v3pt3 = xr.open_mfdataset(
    all_gpcp_v3pt3_mnthly_files_2013_2020,
    combine="nested",
    concat_dim="time",
    coords="minimal",
    compat="override",
    parallel=True,
    engine="netcdf4",
    chunks={"time": 120, "lat": 180, "lon": 360},
    cache=False,
)


# -------------------------------------------------------------------------
# Explicitly restore the correct monthly chronology using the filenames.
#
# Because combine="nested" concatenates according to the supplied file-list
# order, each time index corresponds directly to the same-index filename.
# -------------------------------------------------------------------------

gpcp_ds_v3pt3 = gpcp_ds_v3pt3.assign_coords(
    time=gpcp_time_from_files
)


# Longitude conversion can now proceed normally.
gpcp_ds_v3pt3 = ds_swaplon(gpcp_ds_v3pt3)


# Keep monthly precipitation variable.
gpcp_mnth = gpcp_ds_v3pt3["sat_gauge_precip"].copy()


# Convert from mm/day to mm/month.
days_in_month = xr.DataArray(
    gpcp_mnth["time"].dt.days_in_month,
    dims=["time"],
    coords={"time": gpcp_mnth["time"]}
)

gpcp_mnth = gpcp_mnth * days_in_month
gpcp_mnth.name = "gpcp_mm_month"


# Replace fill/missing values with NaN if needed.
fillv = gpcp_mnth.attrs.get("_FillValue", None)

if fillv is not None:
    gpcp_mnth = gpcp_mnth.where(
        gpcp_mnth != fillv
    )

gpcp_mnth = gpcp_mnth.where(
    np.isfinite(gpcp_mnth)
)


# Subset Antarctica.
gpcp_mnth = gpcp_mnth.sel(
    lat=slice(-60, -90)
)
#----------------------------------------------------------------------------


print("Loading ERA5 monthly dataset ...")

era5_mnth_ds = xr.open_dataset(era5_mnhtly_file, engine="netcdf4")[["tp"]]

# Standardize longitude to -180..180 if needed
era5_mnth_ds = ds_swaplon(era5_mnth_ds)
era5_mnth_ds = replace_fill_with_nan(era5_mnth_ds)
# Rename valid_time -> time if needed
if "valid_time" in era5_mnth_ds.dims or "valid_time" in era5_mnth_ds.coords:
    era5_mnth_ds = era5_mnth_ds.rename({"valid_time": "time"})

# Sort coordinates
era5_mnth_ds = era5_mnth_ds.sortby("longitude")
era5_mnth_ds = era5_mnth_ds.sortby("latitude", ascending=False)

# Drop bookkeeping coordinates that are not needed downstream
drop_coords = [c for c in ["expver", "number"] if c in era5_mnth_ds.coords]
era5_mnth_ds = era5_mnth_ds.drop_vars(drop_coords, errors="ignore")

# Normalize monthly timestamps to clean month-start values
era5_mnth_ds = era5_mnth_ds.assign_coords(
    time=pd.to_datetime(era5_mnth_ds["time"].values).to_period("M").to_timestamp()
)

# Convert ERA5 monthly tp to mm/month
# Assumption for this monthly file:
# tp is monthly mean daily precipitation in meters/day
days_in_month = xr.DataArray(
    era5_mnth_ds["time"].dt.days_in_month,
    dims=["time"],
    coords={"time": era5_mnth_ds["time"]}
)

era5_mnth_ds["tp_mm_month"] = era5_mnth_ds["tp"] * 1000.0 * days_in_month

# Keep processed variable only
era5_mnth = era5_mnth_ds["tp_mm_month"]

# Subset study period
era5_mnth = era5_mnth.sel(time=slice("2013-01-01", "2020-12-31"))

# Subset Antarctica
era5_mnth = era5_mnth.sel(latitude=slice(-60, -90))

# Rename spatial dims to standard names
era5_mnth = era5_mnth.rename({"latitude": "lat", "longitude": "lon"})

#----------------------------------------------------------------------------
print("Loading PMB monthly dataset ...")
P_mm_mnth = xr.open_dataarray(Pmb_mm_fle)

print("Loading PMB monthly uncertainty dataset ...")
P_unc_mm_mnth = xr.open_dataarray(Pmb_unc_mm_fle)

gc.collect()

#%%

# =============================================================================
# SECTION 6. REMAP ALL PRODUCTS TO THE COMMON 0.1° TARGET GRID
# =============================================================================

print("Reprojecting GPCP monthly to common 0.1° grid ...")
# Attach CRS and spatial dims
gpcp_mnth = gpcp_mnth.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
gpcp_mnth = gpcp_mnth.rio.write_crs("EPSG:4326", inplace=False)

# Reproject to the common 0.1° target grid using nearest neighbor
gpcp_mon_01 = gpcp_mnth.rio.reproject_match(
    target_template_01deg,
    resampling=Resampling.nearest
)

# Rename x/y back if needed
rename_map = {}
if "x" in gpcp_mon_01.dims:
    rename_map["x"] = "lon"
if "y" in gpcp_mon_01.dims:
    rename_map["y"] = "lat"
if rename_map:
    gpcp_mon_01 = gpcp_mon_01.rename(rename_map)

gpcp_mon_01 = gpcp_mon_01.sortby("lon")
gpcp_mon_01 = gpcp_mon_01.sortby("lat", ascending=False)

# Apply valid basin-analysis mask
gpcp_mon_01 = gpcp_mon_01.where(basin_mask_01deg.notnull())
gpcp_mon_01 = gpcp_mon_01.where(gpcp_mon_01["lat"] < -60)

# =============================================================================
# GPCP ANTARCTIC LEGACY ADJUSTMENT REMOVAL
#
# GPCP applies an approximately 1.4 adjustment to its AIRS/TOVS-based
# Antarctic precipitation estimate.
#
# For the PMB experiment, remove that legacy multiplicative adjustment first
# so that the PMB-derived correction replaces, rather than compounds, it.
# =============================================================================

GPCP_ANTARCTIC_LEGACY_FACTOR = 1.4

gpcp_mon_01_deadjusted = (
    gpcp_mon_01
    / GPCP_ANTARCTIC_LEGACY_FACTOR
)

gpcp_mon_01_deadjusted.name = (
    "GPCP V3.3 legacy-adjustment removed"
)

gpcp_mon_01_deadjusted.attrs.update(
    gpcp_mon_01.attrs
)

gpcp_mon_01_deadjusted.attrs[
    "Antarctic_legacy_adjustment_removed"
] = GPCP_ANTARCTIC_LEGACY_FACTOR

#----------------------------------------------------------------------------

print("Reprojecting PMB monthly to common 0.1° grid ...")
pmb_mon_01 = prepare_pmb_monthly_on_target(P_mm_mnth, target_template_01deg)
pmb_mon_01 = subset_common_period(pmb_mon_01)

print("Reprojecting PMB monthly uncertainty to common 0.1° grid ...")
pmb_unc_mon_01 = prepare_pmb_monthly_on_target(P_unc_mm_mnth, target_template_01deg)
pmb_unc_mon_01 = subset_common_period(pmb_unc_mon_01)

# Ensure uncertainty is positive
pmb_unc_mon_01 = abs(pmb_unc_mon_01)

#----------------------------------------------------------------------------
# Attach CRS metadata
era5_mnth = era5_mnth.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
era5_mnth = era5_mnth.rio.write_crs("EPSG:4326", inplace=False)

# Remap to common 0.1° target grid using nearest neighbor
era5_mnth_01 = era5_mnth.rio.reproject_match(
    target_template_01deg,
    resampling=Resampling.nearest
)
era5_mnth_01 = replace_fill_with_nan(era5_mnth_01)
# Rename x/y back to lon/lat if needed
rename_map = {}
if "x" in era5_mnth_01.dims:
    rename_map["x"] = "lon"
if "y" in era5_mnth_01.dims:
    rename_map["y"] = "lat"
if rename_map:
    era5_mnth_01 = era5_mnth_01.rename(rename_map)

era5_mnth_01 = era5_mnth_01.sortby("lon")
era5_mnth_01 = era5_mnth_01.sortby("lat", ascending=False)

# Apply valid basin-analysis mask
era5_mnth_01 = era5_mnth_01.where(basin_mask_01deg.notnull())
era5_mnth_01 = era5_mnth_01.where(era5_mnth_01["lat"] < -60)
#----------------------------------------------------------------------------


common_time_main = pd.date_range("2013-01-01", "2020-12-01", freq="MS")

gpm_v8_family_monthly_dict, gpm_v8_platform_monthly_dict, gpm_v8_inventory_df = (
    build_gpm_v8_family_monthly_dict(
        gpm_satellites_path=gpm_satellites_path,
        target_template_01deg=target_template_01deg,
        basin_mask_01deg=basin_mask_01deg,
        start_time="2013-01-01",
        end_time="2020-12-31",
        convert_rate_to_month=True,
        min_valid_platforms_by_family={
            "ATMS": 1,
            "MHS": 1,
            "DMSP-SSMIS": 1,
            "AMSR2": 1,
        },
        return_platform_dict=True,
    )
)


gpm_pmw_v08_mon_01 = build_gpm_pmw_v8_mean(
    gpm_v8_family_dict=gpm_v8_family_monthly_dict,
    mean_name="GPM PMW V08",
    common_months=common_time_main,
    min_valid_families=4,
)

#%%
# =============================================================================
# SECTION X. BUILD COMMON MONTHLY TIME AXES
#
# These common time axes ensure that products are always compared over
# identical months.
#
# common_time_main
#     Entire PMB study period (expected: Feb 2013 – Nov 2020)
#
# common_time_validation
#     Validation-period intersection only
#
# =============================================================================

common_time_main = sorted(
    set(pd.to_datetime(pmb_mon_01.time.values))
    &
    set(pd.to_datetime(era5_mnth_01.time.values))
    &
    set(pd.to_datetime(gpcp_mon_01.time.values))
    &
    set(pd.to_datetime(gpcp_mon_01_deadjusted.time.values))
    &
    set(pd.to_datetime(gpm_pmw_v08_mon_01.time.values))
)

common_time_main = pd.DatetimeIndex(common_time_main)


validation_mask = (
    common_time_main.year.isin(VALIDATION_YEARS)
)

common_time_validation = common_time_main[
    validation_mask
]


print("Common study months      :", len(common_time_main))
print("Common validation months :", len(common_time_validation))

print(
    common_time_main.min(),
    "to",
    common_time_main.max()
)

gc.collect()

#%%

# =============================================================================
# SECTION 7. APPLY BASIN MASK DOMAIN
# =============================================================================

# Keep only cells that belong to Antarctica basins included in the study
valid_basin_mask = basin_mask_01deg.notnull()

gpcp_mon_01 = gpcp_mon_01.where(valid_basin_mask)
gpcp_mon_01_deadjusted = gpcp_mon_01_deadjusted.where(valid_basin_mask)
era5_mon_01 = era5_mnth_01.where(valid_basin_mask)
pmb_mon_01 = pmb_mon_01.where(valid_basin_mask)

print("✅ Common masked monthly fields ready")
print("GPCP  :", gpcp_mon_01.shape)
print("GPCP deadjusted :", gpcp_mon_01_deadjusted.shape)
print("ERA5  :", era5_mon_01.shape)
print("PMB   :", pmb_mon_01.shape)

# =============================================================================
# Coincident monthly datasets
# =============================================================================

pmb_mon_common = pmb_mon_01.sel(time=common_time_main)

era5_mon_common = era5_mon_01.sel(time=common_time_main)

gpcp_mon_common = gpcp_mon_01.sel(time=common_time_main)

gpcp_mon_common_deadjusted = gpcp_mon_01_deadjusted.sel(time=common_time_main)

gpm_pmw_v08_mon_common = gpm_pmw_v08_mon_01.sel(time=common_time_main)

# =============================================================================
# Validation-period monthly datasets
# =============================================================================

pmb_mon_validation = pmb_mon_01.sel(
    time=common_time_validation
)

era5_mon_validation = era5_mon_01.sel(
    time=common_time_validation
)

gpcp_mon_validation = gpcp_mon_01.sel(
    time=common_time_validation
)

gpcp_mon_validation_deadjusted = gpcp_mon_01_deadjusted.sel(
    time=common_time_validation
)

gpm_pmw_v08_mon_validation = gpm_pmw_v08_mon_01.sel(
    time=common_time_validation
)

# =============================================================================
# SECTION 8. QUICK SANITY CHECKS
# =============================================================================

print("\n--- Sanity checks ---")
print("Target grid CRS:", target_template_01deg.rio.crs)
print("Basin mask CRS :", basin_mask_01deg.rio.crs)

print("GPCP time range:", str(gpcp_mon_common.time.min().values), "->", str(gpcp_mon_common.time.max().values))
print("GPCP deadjusted time range:", str(gpcp_mon_common_deadjusted.time.min().values), "->", str(gpcp_mon_common_deadjusted.time.max().values))
print("ERA5 time range:", str(era5_mon_common.time.min().values), "->", str(era5_mon_common.time.max().values))
print("PMB time range :", str(pmb_mon_common.time.min().values),  "->", str(pmb_mon_common.time.max().values))
print("GPM PMW V08 time range:", str(gpm_pmw_v08_mon_common.time.min().values), "->", str(gpm_pmw_v08_mon_common.time.max().values))
#%%
# =============================================================================
# SECTION 9. E2-A. BUILD REGIONAL MONTHLY SERIES FROM UNCORRECTED DATA
#
# These are the ORIGINAL fields.
# No correction has yet been applied.
# =============================================================================


e2_uncorrected_product_dict = {
    REFERENCE_NAME: pmb_mon_common,
    ERA5_NAME: era5_mon_common,
    GPCP_DEADJ_NAME: gpcp_mon_common_deadjusted,
    PMW8_NAME: gpm_pmw_v08_mon_common,
}


e2_regional_monthly_uncorrected = (
    build_all_region_monthly_series_cosine(
        product_dict=e2_uncorrected_product_dict,
        region_masks=region_masks_01deg,
        lat_name="lat",
        lon_name="lon",
        time_name="time",
    )
)


print(e2_regional_monthly_uncorrected.head())

print(
    "\nAvailable products:",
    e2_regional_monthly_uncorrected["product"].unique()
)

print(
    "\nTime coverage:",
    e2_regional_monthly_uncorrected["time"].min(),
    "to",
    e2_regional_monthly_uncorrected["time"].max()
)


#%%
# =============================================================================
# SECTION 10. E2-B. BUILD COMPLETE METEOROLOGICAL SEASONS
#
# Critical:
# December belongs to the following season-year.
#
# Dec 2017 + Jan 2018 + Feb 2018 -> DJF 2018.
# =============================================================================


e2_seasonal_uncorrected = (
    build_complete_seasonal_totals_for_correction(
        monthly_region_df=e2_regional_monthly_uncorrected,
        region_col="region",
        product_col="product",
        time_col="time",
        value_col="precipitation",
        require_complete_season=True,
    )
)


# print(
#     e2_seasonal_uncorrected
#     .sort_values(
#         ["region", "product", "season_year", "season"]
#     )
#     .head(30)
# )

#%%
# =============================================================================
# SECTION 10_A. CHECK NUMBER OF COMPLETE SEASONS AVAILABLE
# =============================================================================


season_counts = (
    e2_seasonal_uncorrected
    .groupby(
        ["region", "product", "season"]
    )["season_year"]
    .nunique()
    .unstack("season")
)


# print("\nNumber of complete seasons:")
# print(season_counts)


#%%
# =============================================================================
# SECTION 11. E2-C. SPLIT CALIBRATION AND VALIDATION USING SEASON_YEAR
# =============================================================================


e2_calibration_seasonal, e2_validation_seasonal_uncorrected = (
    split_seasonal_calibration_validation(
        seasonal_df=e2_seasonal_uncorrected,
        calibration_years=CALIBRATION_YEARS,
        validation_years=VALIDATION_YEARS,
    )
)


print("\nCalibration seasonal records:")
print(
    e2_calibration_seasonal
    .groupby(["region", "product", "season"])
    .size()
    .unstack("season")
)


print("\nValidation seasonal records:")
print(
    e2_validation_seasonal_uncorrected
    .groupby(["region", "product", "season"])
    .size()
    .unstack("season")
)

#%%
# =============================================================================
# SECTION 12. E2-D. DERIVE AIS-WIDE SEASONAL CORRECTION COEFFICIENTS
#
# CF = calibration-period long-term mean PMB
#      -------------------------------------
#      calibration-period long-term mean satellite product
#
# Separate coefficient for:
#   DJF
#   MAM
#   JJA
#   SON
#
# Separate coefficient for:
#   GPCP V3.3
#   GPM PMW V08
#
# Spatially:
#   only ONE AIS-wide coefficient per product/season.
# =============================================================================


e2_correction_factors = derive_seasonal_correction_factors(
    calibration_seasonal_df=e2_calibration_seasonal,

    reference_product=REFERENCE_NAME,

    target_products=(
        GPCP_DEADJ_NAME,
        PMW8_NAME,
    ),

    regions=(
        "Antarctica",
    ),

    seasons=(
        "DJF",
        "MAM",
        "JJA",
        "SON",
    ),
)


print_correction_factor_summary(
    e2_correction_factors
)

#%%
# =============================================================================
# SECTION 12_A. SAVE E2 CORRECTION FACTORS
# =============================================================================


e2_cf_file = os.path.join(
    path_to_dfs,
    f"E2_seasonal_AIS_PMB_correction_factors_"
    f"cal2013_2017_val2018_2020_{cde_run_dte}.csv"
)


e2_correction_factors.to_csv(
    e2_cf_file,
    index=False,
)


print("Saved:")
print(e2_cf_file)

#%%
# =============================================================================
# SECTION 12_B. QUICK DIAGNOSTIC: E2 CORRECTION FACTORS
# =============================================================================


fig, ax = plt.subplots(
    figsize=(8, 5),
    dpi=150
)


season_order = ["DJF", "MAM", "JJA", "SON"]


for product in [GPCP_DEADJ_NAME, PMW8_NAME]:

    sub = (
        e2_correction_factors[
            e2_correction_factors["product"] == product
        ]
        .set_index("season")
        .reindex(season_order)
    )

    ax.plot(
        season_order,
        sub["correction_factor"],
        marker="o",
        linewidth=2.5,
        label=product,
    )


# CF = 1 means no correction required.
ax.axhline(
    1.0,
    color="k",
    linestyle="--",
    linewidth=1.2,
)


ax.set_ylabel(
    "PMB correction factor",
    fontsize=13,
    fontweight="bold",
)

ax.set_xlabel(
    "Season",
    fontsize=13,
    fontweight="bold",
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend(
    frameon=False,
    fontsize=12,
)
svnme = os.path.join(
    path_to_plots,
    f" E2_seasonal_AIS_PMB_correction_factors_"
    f"cal2013_2017_val2018_2020_{cde_run_dte}.png"
)
fig.savefig(svnme, dpi=150)

plt.tight_layout()
# plt.show()

#%%
# =============================================================================
# SECTION 13. E2-F. APPLY SEASONAL AIS CORRECTION TO THE MONTHLY GRIDDED PRODUCTS
# =============================================================================


AIS_MASK_01 = basin_mask_01deg.isin(AIS_BASINS)


gpcp_mon_01_e2_corr = apply_e2_seasonal_ais_correction(
    da_monthly=gpcp_mon_01_deadjusted,

    correction_factor_df=e2_correction_factors,

    product_name=GPCP_DEADJ_NAME,

    ais_mask=AIS_MASK_01,

    corrected_name=GPCP_CORR_NAME,
)


gpm_pmw_v08_mon_01_e2_corr = apply_e2_seasonal_ais_correction(
    da_monthly=gpm_pmw_v08_mon_01,

    correction_factor_df=e2_correction_factors,

    product_name=PMW8_NAME,

    ais_mask=AIS_MASK_01,

    corrected_name=PMW8_CORR_NAME,
)


# print(gpcp_mon_01_e2_corr)
# print(gpm_pmw_v08_mon_01_e2_corr)

#%%
# =============================================================================
# SECTION 13_A. QUICK DIAGNOSTIC: E2 GRID-LEVEL SANITY CHECK
# =============================================================================


test_time = "2019-07-01"


gpcp_ratio_test = (
    gpcp_mon_01_e2_corr.sel(time=test_time)
    /
    gpcp_mon_01_deadjusted.sel(time=test_time)
)


pmw8_ratio_test = (
    gpm_pmw_v08_mon_01_e2_corr.sel(time=test_time)
    /
    gpm_pmw_v08_mon_01.sel(time=test_time)
)


# print(
#     "GPCP corrected/original ratio range:",
#     float(gpcp_ratio_test.min(skipna=True)),
#     float(gpcp_ratio_test.max(skipna=True)),
# )


# print(
#     "PMW8 corrected/original ratio range:",
#     float(pmw8_ratio_test.min(skipna=True)),
#     float(pmw8_ratio_test.max(skipna=True)),
# )

gpcp_net_ratio_test = (
    gpcp_mon_01_e2_corr.sel(time=test_time)
    /
    gpcp_mon_01.sel(time=test_time)
)

print(
    "GPCP PMB-corrected / de-adjusted ratio:",
    float(gpcp_ratio_test.min(skipna=True)),
    float(gpcp_ratio_test.max(skipna=True)),
)

print(
    "GPCP PMB-corrected / published GPCP ratio:",
    float(gpcp_net_ratio_test.min(skipna=True)),
    float(gpcp_net_ratio_test.max(skipna=True)),
)

#%%
# =============================================================================
# SECTION 13_B. COMMON-TIME CORRECTED PRODUCTS
# =============================================================================

gpcp_mon_e2_corr_common = gpcp_mon_01_e2_corr.sel(
    time=common_time_main
)

gpm_pmw_v08_mon_e2_corr_common = gpm_pmw_v08_mon_01_e2_corr.sel(
    time=common_time_main
)


gpcp_mon_e2_corr_validation = gpcp_mon_01_e2_corr.sel(
    time=common_time_validation
)

gpm_pmw_v08_mon_e2_corr_validation = gpm_pmw_v08_mon_01_e2_corr.sel(
    time=common_time_validation
)

#%%
# =============================================================================
# SECTION 14. E2-H. MASTER PRODUCT DICTIONARY
#
# This becomes the common input for nearly all downstream figures.
# =============================================================================


e2_product_dict = {
    REFERENCE_NAME: pmb_mon_01,
    ERA5_NAME: era5_mon_01,
    GPCP_NAME: gpcp_mon_01,
    GPCP_CORR_NAME: gpcp_mon_01_e2_corr,
    PMW8_NAME: gpm_pmw_v08_mon_01,
    PMW8_CORR_NAME: gpm_pmw_v08_mon_01_e2_corr,
}


print(e2_product_dict.keys())

#%%
# =============================================================================
# SECTION 15. E2 VALIDATION MONTHLY CLIMATOLOGY
#
# Validation only: 2018-2020
#
# Products:
#   ERA5
#   GPCP V3.3 uncorrected
#   GPCP V3.3 corrected
#   GPM PMW V08 uncorrected
#   GPM PMW V08 corrected
# =============================================================================


e2_monthly_clim_products = {
    ERA5_NAME: era5_mon_01,

    GPCP_NAME: gpcp_mon_01,
    GPCP_CORR_NAME: gpcp_mon_01_e2_corr,

    PMW8_NAME: gpm_pmw_v08_mon_01,
    PMW8_CORR_NAME: gpm_pmw_v08_mon_01_e2_corr,
}


(
    e2_validation_monthly_regional_df,
    e2_validation_monthly_clim_df,
) = validation_monthly_climatology_from_fields(
    product_dict=e2_monthly_clim_products,

    region_masks=region_masks_01deg,

    validation_years=VALIDATION_YEARS,

    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

# Common validation-month dictionary for annual comparisons only
e2_product_dict_validation_common = {
    REFERENCE_NAME: pmb_mon_validation,
    ERA5_NAME: era5_mon_validation,
    GPCP_NAME: gpcp_mon_validation,
    GPCP_CORR_NAME: gpcp_mon_e2_corr_validation,
    PMW8_NAME: gpm_pmw_v08_mon_validation,
    PMW8_CORR_NAME: gpm_pmw_v08_mon_e2_corr_validation,
}


# print(e2_validation_monthly_clim_df)

#%% SECTION 15.A. PLOT VALIDATION MONTHLY CLIMATOLOGY FOR E2
product_styles_e2 = {

    ERA5_NAME: {
        "color": "blue",
        "marker": "s",
        "lw": 2.5,
    },

    GPCP_NAME: {
        "color": "orange",
        "marker": "D",
        "lw": 2.2,
        "linestyle": "--",
    },

    GPCP_CORR_NAME: {
        "color": "orange",
        "marker": "o",
        "lw": 3.0,
        "linestyle": "-",
    },

    PMW8_NAME: {
        "color": "green",
        "marker": "s",
        "lw": 2.2,
        "linestyle": "--",
    },

    PMW8_CORR_NAME: {
        "color": "green",
        "marker": "o",
        "lw": 3.0,
        "linestyle": "-",
    },
}


fig, axes = plot_validation_monthly_climatology_numeric_months(
    clim_df=e2_validation_monthly_clim_df,

    region_order=(
        "Antarctica",
        "West Antarctica",
        "East Antarctica",
    ),

    product_order=(
        ERA5_NAME,
        GPCP_NAME,
        GPCP_CORR_NAME,
        PMW8_NAME,
        PMW8_CORR_NAME,
    ),

    product_styles=product_styles_e2,

    figsize=(10, 9),

    ylabel="mm/month",
)

svnme = os.path.join(
    path_to_plots,
    f"e2_validation_monthly_climatology_{cde_run_dte}.png",
)

plt.savefig(svnme, dpi=150, bbox_inches="tight")
# plt.show()

#%%
# =============================================================================
# SECTION 16. E2 VALIDATION SEASONAL CLIMATOLOGY
# =============================================================================


e2_regional_monthly_all = (
    build_all_region_monthly_series_cosine(
        product_dict=e2_product_dict,

        region_masks=region_masks_01deg,

        lat_name="lat",
        lon_name="lon",
        time_name="time",
    )
)


(
    e2_validation_seasonal_df,
    e2_validation_seasonal_clim_df,
) = validation_seasonal_climatology_from_monthly_df(
    full_monthly_region_df=e2_regional_monthly_all,

    validation_years=VALIDATION_YEARS,

    require_complete_season=True,
)


# print(e2_validation_seasonal_clim_df)

#%% SECTION 16.A. PLOT VALIDATION SEASONAL CLIMATOLOGY FOR E2
fig, axes = plot_seasonal_climatology(
    clim_df=e2_validation_seasonal_clim_df,

    region_order=(
        "Antarctica",
        "West Antarctica",
        "East Antarctica",
    ),

    product_order=(
        REFERENCE_NAME,
        ERA5_NAME,
        GPCP_NAME,
        GPCP_CORR_NAME,
        PMW8_NAME,
        PMW8_CORR_NAME,
    ),

    product_styles=product_styles_e2,

    figsize=(10, 9),

    ylabel="mm/season",

    y_nbins=4,

    legend_ncol=3,
)

svnme = os.path.join(
    path_to_plots,
    f"e2_validation_seasonal_climatology_{cde_run_dte}.png",
)

plt.savefig(svnme, dpi=150, bbox_inches="tight")
# plt.show()
gc.collect()
#%%
# =============================================================================
# SECTION 17. E2 VALIDATION REGIONAL MEAN ANNUAL PRECIPITATION
# =============================================================================

(
    e2_validation_annual_regional_df,
    e2_validation_mean_annual_regional_df,
) = validation_regional_annual_dataframe(
    product_dict=e2_product_dict_validation_common,

    region_masks=region_masks_01deg,

    validation_years=VALIDATION_YEARS,
)

#%% SECTION 17.A. PLOT VALIDATION REGIONAL MEAN ANNUAL PRECIPITATION FOR E2
product_colors_e2 = {

    REFERENCE_NAME: {
        "color": "black",
    },

    ERA5_NAME: {
        "color": "blue",
    },

    GPCP_NAME: {
        "color": "orange",
        "alpha": 0.50,
    },

    GPCP_CORR_NAME: {
        "color": "orange",
        "alpha": 1.00,
    },

    PMW8_NAME: {
        "color": "green",
        "alpha": 0.50,
    },

    PMW8_CORR_NAME: {
        "color": "green",
        "alpha": 1.00,
    },
}


fig, ax = plot_regional_mean_annual_bars(
    df_mean_regional=e2_validation_mean_annual_regional_df,

    region_order=(
        "Antarctica",
        "West Antarctica",
        "East Antarctica",
    ),

    product_order=(
        REFERENCE_NAME,
        ERA5_NAME,
        GPCP_NAME,
        GPCP_CORR_NAME,
        PMW8_NAME,
        PMW8_CORR_NAME,
    ),

    product_colors=product_colors_e2,

    ylabel="[mm/year]",

    title="",

    annotate=True,

    legend_ncol=2,
)
svnme = os.path.join(
    path_to_plots,
    f"e2_validation_annual_regional_{cde_run_dte}.png",
)

plt.savefig(svnme, dpi=150, bbox_inches="tight")

# plt.show()
gc.collect()

#%%
# =============================================================================
# SECTION 18. E2 VALIDATION PIXEL-LEVEL MEAN ANNUAL FIELDS
# =============================================================================
e2_pixel_map_products = {

    ERA5_NAME: era5_mon_validation,

    GPCP_NAME: gpcp_mon_validation,

    GPCP_CORR_NAME: gpcp_mon_e2_corr_validation,

    PMW8_NAME: gpm_pmw_v08_mon_validation,

    PMW8_CORR_NAME: gpm_pmw_v08_mon_e2_corr_validation,
}


e2_validation_annual_mean_fields = (
    build_validation_annual_mean_fields(
        product_dict=e2_pixel_map_products,

        validation_years=VALIDATION_YEARS,
    )
)


for name, field in e2_validation_annual_mean_fields.items():

    print(
        name,
        field.shape,
        float(field.mean(skipna=True)),
    )


#%% SECTION 18.A. PLOT VALIDATION PIXEL-LEVEL MEAN ANNUAL FIELDS FOR E2

BASIN_IDS = sorted(AIS_BASINS)

basin_mask_01deg_clean = basin_mask_01deg.where(basin_mask_01deg.isin(BASIN_IDS))


e2_pixel_arr_lst = [

    (
        ERA5_NAME,
        e2_validation_annual_mean_fields[
            ERA5_NAME
        ]
    ),

    (
        GPCP_NAME,
        e2_validation_annual_mean_fields[
            GPCP_NAME
        ]
    ),

    (
        GPCP_CORR_NAME,
        e2_validation_annual_mean_fields[
            GPCP_CORR_NAME
        ]
    ),

    (
        PMW8_NAME,
        e2_validation_annual_mean_fields[
            PMW8_NAME
        ]
    ),

    (
        PMW8_CORR_NAME,
        e2_validation_annual_mean_fields[
            PMW8_CORR_NAME
        ]
    ),
]

# =============================================================================
# AIS MEAN VALUES FOR PIXEL-WISE PANELS
# =============================================================================

AIS_MASK_01 = basin_mask_01deg.isin(
    AIS_BASINS
)


e2_pixel_mean_vals = {}


for product_name, field in e2_pixel_arr_lst:

    mean_val = cosine_weighted_mean_masked(
        da_2d=field,
        region_mask=AIS_MASK_01,
        lat_name="lat",
        lon_name="lon"
    )

    e2_pixel_mean_vals[
        product_name
    ] = float(
        mean_val.values
    )


print(
    e2_pixel_mean_vals
)

fig, axes, cb = plot_annual_comparison_multi_row_grid_spec(

    arr_lst_mean=e2_pixel_arr_lst,

    mean_vals=e2_pixel_mean_vals,

    vmin=0,
    vmax=400,

    smooth=False,

    cbr_lbl=r"Precipitation [mm yr$^{-1}$]",

    extent_plt=[
        -180,
        180,
        -90,
        -60
    ],

    hem="SH",

    ncols=3,

    figsize_per_row=(
        22,
        7
    ),

    cbar_ticks=[
        0,
        25,
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400
    ],

    panel_letters=True,

    show_mean=True,
)
svnme = os.path.join(
    path_to_plots,
    f"e2_validation_pixel_annual_{cde_run_dte}.png",
)

plt.savefig(
    svnme,
    dpi=150,
    bbox_inches="tight"
)

# plt.show()
#%%
# =============================================================================
# SECTION 19. E2 VALIDATION BASIN-PAINTED MEAN ANNUAL FIELDS
# =============================================================================

e2_validation_all_annual_mean_fields = (
    build_validation_annual_mean_fields(
        product_dict=e2_product_dict_validation_common,

        validation_years=VALIDATION_YEARS,
    )
)


e2_validation_basin_plot_list = (
    build_validation_basin_plot_products(
        annual_mean_field_dict=e2_validation_all_annual_mean_fields,

        basin_mask_2d=basin_mask_01deg_clean,

        basin_ids=BASIN_IDS,
    )
)


for item in e2_validation_basin_plot_list:
    print(
        item[0],
        "AIS panel mean =",
        round(item[2], 2)
    )


#%% SECTION 19.A. PLOT E2 VALIDATION BASIN-PAINTED MEAN ANNUAL FIELDS
fig, axes, cb = compare_mean_precip_basin_2x3_common_cbar(
    arr_lst_mean=e2_validation_basin_plot_list,

    basin_mask_latlon=basin_mask_01deg_clean,

    figsize=(14, 9),

    gamma=0.6,

    vmin=0,

    vmax=400,

    cbar_ticks=[
        0,
        25,
        50,
        100,
        200,
        300,
        400,
    ],

    cbar_label=r"Precipitation [mm yr$^{-1}$]",

    panel_letters=True,

    show_panel_mean=True,
)

svneme = os.path.join(
    path_to_plots,
    f"e2_validation_basin_annual_{cde_run_dte}.png",
)
plt.savefig(
    svneme,
    dpi=150,
    bbox_inches="tight"
)
# plt.show()
gc.collect()

#%%
# =============================================================================
# SECTION 20. E2 ANNUAL DIFFERENCE MAPS
# =============================================================================

gpcp_corr_minus_uncorr = (
    e2_validation_annual_mean_fields[GPCP_CORR_NAME]
    -
    e2_validation_annual_mean_fields[GPCP_NAME]
)

pmw8_corr_minus_uncorr = (
    e2_validation_annual_mean_fields[PMW8_CORR_NAME]
    -
    e2_validation_annual_mean_fields[PMW8_NAME]
)


# -------------------------------------------------------------------------
# Residual relative to ERA5:
# corrected product - ERA5
# -------------------------------------------------------------------------

gpcp_corr_minus_era5 = (
    e2_validation_annual_mean_fields[GPCP_CORR_NAME]
    -
    e2_validation_annual_mean_fields[ERA5_NAME]
)

pmw8_corr_minus_era5 = (
    e2_validation_annual_mean_fields[PMW8_CORR_NAME]
    -
    e2_validation_annual_mean_fields[ERA5_NAME]
)


# -------------------------------------------------------------------------
# Row 1: what did the PMB adjustment actually change?
# -------------------------------------------------------------------------

correction_difference_data = [

    (
        "GPCP V3.3 corrected - GPCP V3.3",
        gpcp_corr_minus_uncorr,
    ),

    (
        "GPM PMW V08 corrected - GPM PMW V08",
        pmw8_corr_minus_uncorr,
    ),
]


# -------------------------------------------------------------------------
# Row 2: what discrepancy remains relative to ERA5?
# -------------------------------------------------------------------------

era5_residual_difference_data = [

    (
        "GPCP V3.3 corrected - ERA5",
        gpcp_corr_minus_era5,
    ),

    (
        "GPM PMW V08 corrected - ERA5",
        pmw8_corr_minus_era5,
    ),
]


fig, axes = plot_antarctic_difference_maps_2x2(

    correction_data=correction_difference_data,

    residual_data=era5_residual_difference_data,

    extent_plt=(-180, 180, -90, -60),

    gpcp_vmin=-5,
    gpcp_vmax=50,
    gpcp_step=5,

    pmw_vmin=-2,
    pmw_vmax=150,
    pmw_step=10,

    residual_vmin=-200,
    residual_vmax=200,
    residual_step=25,

    cmap_name="RdBu_r",

    figsize=(15, 11),
)

gc.collect()


#%% AIS Specific Analysis For GPCP Meeting
#SECTION 15.B. AIS-ONLY MONTHLY CLIMATOLOGY FOR PRESENTATION

ais_monthly_df = (
    e2_validation_monthly_clim_df[
        e2_validation_monthly_clim_df["region"] == "Antarctica"
    ]
    .copy()
)

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

for product in (
    ERA5_NAME,
    GPCP_NAME,
    GPCP_CORR_NAME,
    PMW8_NAME,
    PMW8_CORR_NAME,
):

    ss = (
        ais_monthly_df[
            ais_monthly_df["product"] == product
        ]
        .sort_values("month")
    )

    style = product_styles_e2[product].copy()

    ax.plot(
        ss["month"],
        ss["precipitation"],
        label=product,
        **style,
    )

ax.set_xticks(np.arange(1, 13))

ax.set_xlabel(
    "Month",
    fontsize=14,
    fontweight="bold",
)

ax.set_ylabel(
    "Precipitation [mm month$^{-1}$]",
    fontsize=14,
    fontweight="bold",
)

ax.set_title(
    "AIS Monthly Climatology:\nIndependent Validation Period (2018–2020)",
    fontsize=16,
    fontweight="bold",
)

ax.grid(True, alpha=0.25)

# ------------------------------------------------------------------
# Custom legend order
# ------------------------------------------------------------------
handles, labels = ax.get_legend_handles_labels()

desired_order = [
    GPCP_NAME,
    GPCP_CORR_NAME,
    PMW8_NAME,
    PMW8_CORR_NAME,
    ERA5_NAME,
]

handle_dict = dict(zip(labels, handles))

ax.legend(
    [handle_dict[l] for l in desired_order],
    desired_order,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=3,
    frameon=False,
    fontsize=11.5,
    handlelength=4.0,
    handletextpad=0.8,
    columnspacing=2.0,
)

plt.tight_layout()

svnme = os.path.join(
    path_to_plots,
    f"e2_AIS_validation_monthly_climatology_{cde_run_dte}.png",
)

plt.savefig(
    svnme,
    dpi=150,
    bbox_inches="tight",
)

# plt.show()
gc.collect()

#=============================================================================
# SECTION 16.B. AIS-ONLY SEASONAL CLIMATOLOGY FOR PRESENTATION

ais_seasonal_df = (
    e2_validation_seasonal_clim_df[
        e2_validation_seasonal_clim_df["region"] == "Antarctica"
    ]
    .copy()
)

season_order = ["DJF", "MAM", "JJA", "SON"]

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

for product in (
    # REFERENCE_NAME,
    ERA5_NAME,
    GPCP_NAME,
    GPCP_CORR_NAME,
    PMW8_NAME,
    PMW8_CORR_NAME,
):

    ss = (
        ais_seasonal_df[
            ais_seasonal_df["product"] == product
        ]
        .set_index("season")
        .reindex(season_order)
    )

    if product == REFERENCE_NAME:
        style = {
            "color": "black",
            "marker": "o",
            "lw": 2.8,
        }
    else:
        style = product_styles_e2[product].copy()

    ax.plot(
        season_order,
        ss["precipitation"],
        label=product,
        **style,
    )

ax.set_xlabel(
    "Season",
    fontsize=14,
    fontweight="bold",
)

ax.set_ylabel(
    "Precipitation [mm season$^{-1}$]",
    fontsize=14,
    fontweight="bold",
)

ax.set_title(
    "AIS Seasonal Climatology:\nIndependent Validation Period (2018–2020)",
    fontsize=16,
    fontweight="bold",
)

ax.grid(True, alpha=0.25)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=3,
    frameon=False,
    fontsize=11.5,
    handlelength=4.0,   # longer legend line
)

plt.tight_layout()

svnme = os.path.join(
    path_to_plots,
    f"e2_AIS_validation_seasonal_climatology_{cde_run_dte}.png",
)

plt.savefig(
    svnme,
    dpi=200,
    bbox_inches="tight",
)

# plt.show()
gc.collect()
#============================================================================
# SECTION 17.B. AIS-ONLY MEAN ANNUAL PRECIPITATION FOR PRESENTATION

ais_annual_df = (
    e2_validation_mean_annual_regional_df[
        e2_validation_mean_annual_regional_df["region"] == "Antarctica"
    ]
    .copy()
)


product_order_ais = [
    # REFERENCE_NAME,
    ERA5_NAME,
    GPCP_NAME,
    GPCP_CORR_NAME,
    PMW8_NAME,
    PMW8_CORR_NAME,
]

ais_annual_df["product"] = pd.Categorical(
    ais_annual_df["product"],
    categories=product_order_ais,
    ordered=True,
)

ais_annual_df = (
    ais_annual_df[
        ais_annual_df["product"].isin(product_order_ais)
    ]
    .sort_values("product")
    .reset_index(drop=True)
)

ais_annual_df = ais_annual_df.sort_values("product")

colors = [
    # "black",
    "blue",
    "orange",
    "orange",
    "green",
    "green",
]

alphas = [
    # 1.0,
    1.0,
    0.50,
    1.0,
    0.50,
    1.0,
]

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

bars = ax.bar(
    np.arange(len(ais_annual_df)),
    ais_annual_df["precipitation"],
    color=colors,
)

for bar, alpha in zip(bars, alphas):
    bar.set_alpha(alpha)

ax.set_xticks(
    np.arange(len(ais_annual_df))
)

ax.set_xticklabels(
    [
        # r"$P_{MB}$",
        "ERA5",
        "GPCP\nV3.3",
        "GPCP V3.3\ncorrected",
        "PMW\nV08",
        "PMW V08\ncorrected",
    ],
    fontsize=11,
)

ax.set_ylabel(
    "Precipitation [mm yr$^{-1}$]",
    fontsize=14,
    fontweight="bold",
)

ax.set_title(
    "AIS Mean Annual Precipitation:\nIndependent Validation (2018–2020)",
    fontsize=16,
    fontweight="bold",
)

ax.grid(
    axis="y",
    alpha=0.25,
)

for bar, value in zip(
    bars,
    ais_annual_df["precipitation"]
):

    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 2,
        f"{value:.0f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()

svnme = os.path.join(
    path_to_plots,
    f"e2_AIS_validation_annual_mean_{cde_run_dte}.png",
)

plt.savefig(
    svnme,
    dpi=200,
    bbox_inches="tight",
)

# plt.show()
gc.collect()

#=============================================================================

#%%
# =============================================================================
# SECTION 21. FINAL DELIVERABLE SEASONAL FACTORS FROM 2013-2020
# =============================================================================
# The 2013-2017 factors above remain the only factors used for independent
# validation on 2018-2020. This section refits the same seasonal model using all
# available complete seasons in 2013-2020 to produce the final coefficients
# intended for delivery.
#
# Because PMB begins/ends at the edges of calendar years, the number of complete
# DJF seasons can differ from MAM/JJA/SON. The output table retains n_reference
# and n_product so that this sampling is explicit.
# =============================================================================

e2_final_period_seasonal = e2_seasonal_uncorrected[
    e2_seasonal_uncorrected["season_year"].isin(FINAL_FACTOR_YEARS)
].copy()

e2_final_seasonal_correction_factors = derive_seasonal_correction_factors(
    calibration_seasonal_df=e2_final_period_seasonal,
    reference_product=REFERENCE_NAME,
    target_products=(GPCP_DEADJ_NAME, PMW8_NAME),
    regions=("Antarctica",),
    seasons=("DJF", "MAM", "JJA", "SON"),
)

# Period labels prevent the final factors from being confused with the
# calibration-only factors in downstream tables or plots.
e2_correction_factors_labeled = e2_correction_factors.assign(
    derivation_period="2013-2017 calibration"
)
e2_final_seasonal_correction_factors = (
    e2_final_seasonal_correction_factors.assign(
        derivation_period="2013-2020 final deliverable"
    )
)

e2_seasonal_cf_period_comparison = pd.concat(
    [e2_correction_factors_labeled, e2_final_seasonal_correction_factors],
    ignore_index=True,
)

final_seasonal_cf_file = os.path.join(
    path_to_dfs,
    f"E2_seasonal_AIS_PMB_correction_factors_2013_2020_FINAL_{cde_run_dte}.csv",
)
e2_final_seasonal_correction_factors.to_csv(
    final_seasonal_cf_file,
    index=False,
)

comparison_seasonal_cf_file = os.path.join(
    path_to_dfs,
    f"E2_seasonal_AIS_PMB_correction_factors_cal_vs_final_{cde_run_dte}.csv",
)
e2_seasonal_cf_period_comparison.to_csv(
    comparison_seasonal_cf_file,
    index=False,
)

print("\nFinal 2013-2020 seasonal correction factors:")
print_correction_factor_summary(e2_final_seasonal_correction_factors)
print("Saved final factors:", final_seasonal_cf_file)

# fig, ax = plot_e2_seasonal_cf_period_comparison(
#     calibration_cf_df=e2_correction_factors_labeled,
#     final_cf_df=e2_final_seasonal_correction_factors,
#     product_order=(GPCP_DEADJ_NAME, PMW8_NAME),
#     product_colors={
#         GPCP_DEADJ_NAME: "tab:orange",
#         PMW8_NAME: "tab:green",
#     },
# )

# seasonal_cf_comparison_plot = os.path.join(
#     path_to_plots,
#     f"E2_seasonal_CF_2013_2017_vs_2013_2020_{cde_run_dte}.png",
# )
# fig.savefig(seasonal_cf_comparison_plot, dpi=200, bbox_inches="tight")
# # plt.close(fig)

#%%
# =============================================================================
# SECTION 21_A. SINGLE YEAR-ROUND FACTORS FROM THE FULL 2013-2020 PERIOD
# =============================================================================
# These are final-period diagnostic/deliverable alternatives. They are not used
# in the independent 2018-2020 validation because the validation years
# contribute to their estimation.
# =============================================================================

e2_final_single_correction_factors = (
    derive_single_ais_correction_factors(
        monthly_region_df=e2_regional_monthly_uncorrected,
        period_years=FINAL_FACTOR_YEARS,
        reference_product=REFERENCE_NAME,
        target_products=(
            GPCP_DEADJ_NAME,
            PMW8_NAME,
        ),
        regions=(
            "Antarctica",
        ),
    )
)

final_single_cf_file = os.path.join(
    path_to_dfs,
    f"E2_single_AIS_PMB_correction_factors_2013_2020_FINAL_"
    f"{cde_run_dte}.csv",
)

e2_final_single_correction_factors.to_csv(
    final_single_cf_file,
    index=False,
)

print("\nFinal single year-round correction factors from 2013-2020:")
print(
    e2_final_single_correction_factors.to_string(
        index=False
    )
)

print(
    "Saved final single factors:",
    final_single_cf_file,
)

#%%
# =============================================================================
# SECTION 22. SINGLE YEAR-ROUND CF: DERIVE ON 2013-2017 ONLY
# =============================================================================
# This is the second independent-validation experiment. One constant factor is
# estimated per product from exact PMB/product monthly pairs in 2013-2017.
# The factors are then applied unchanged to every validation month in 2018-2020.
# =============================================================================

e2_single_correction_factors = derive_single_ais_correction_factors(
    monthly_region_df=e2_regional_monthly_uncorrected,
    period_years=CALIBRATION_YEARS,
    reference_product=REFERENCE_NAME,
    target_products=(GPCP_DEADJ_NAME, PMW8_NAME),
    regions=("Antarctica",),
)

single_cf_file = os.path.join(
    path_to_dfs,
    f"E2_single_AIS_PMB_correction_factors_cal2013_2017_{cde_run_dte}.csv",
)
e2_single_correction_factors.to_csv(single_cf_file, index=False)

print("\nSingle year-round correction factors derived from 2013-2017:")
print(e2_single_correction_factors.to_string(index=False))
print("Saved single factors:", single_cf_file)


#%%
# =============================================================================
# SECTION 22_A. PLOT ALL EIGHT CORRECTION-FACTOR SCENARIOS
# =============================================================================
# Seasonal curves:
#   - 2013-2017 calibration factors used for seasonal-CF validation
#   - 2013-2020 final deliverable factors
#
# Horizontal lines:
#   - 2013-2017 single factors used for single-CF validation
#   - 2013-2020 final single-factor alternatives
# =============================================================================

fig, ax = plot_e2_seasonal_cf_period_comparison(
    calibration_cf_df=e2_correction_factors_labeled,

    final_cf_df=e2_final_seasonal_correction_factors,

    calibration_single_cf_df=e2_single_correction_factors,

    final_single_cf_df=e2_final_single_correction_factors,

    product_order=(
        GPCP_DEADJ_NAME,
        PMW8_NAME,
    ),

    product_colors={
        GPCP_DEADJ_NAME: "tab:orange",
        PMW8_NAME: "tab:green",
    },

    calibration_label="2013-2017 calibration",

    final_label="2013-2020 final",
)

seasonal_cf_comparison_plot = os.path.join(
    path_to_plots,
    f"E2_all_CF_scenarios_2013_2017_vs_2013_2020_"
    f"{cde_run_dte}.png",
)

fig.savefig(
    seasonal_cf_comparison_plot,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print(
    "Saved eight-scenario CF plot:",
    seasonal_cf_comparison_plot,
)
#%%
# =============================================================================
# SECTION 23. APPLY SINGLE CF TO THE MONTHLY GRIDDED PRODUCTS
# =============================================================================
# The GPCP factor is applied to the de-adjusted GPCP field (published GPCP / 1.4),
# matching the input used for the seasonal-factor experiment.
# =============================================================================

gpcp_mon_01_e2_single_corr = apply_e2_single_ais_correction(
    da_monthly=gpcp_mon_01_deadjusted,
    correction_factor_df=e2_single_correction_factors,
    product_name=GPCP_DEADJ_NAME,
    ais_mask=AIS_MASK_01,
    corrected_name=GPCP_SINGLE_CORR_NAME,
)

gpm_pmw_v08_mon_01_e2_single_corr = apply_e2_single_ais_correction(
    da_monthly=gpm_pmw_v08_mon_01,
    correction_factor_df=e2_single_correction_factors,
    product_name=PMW8_NAME,
    ais_mask=AIS_MASK_01,
    corrected_name=PMW8_SINGLE_CORR_NAME,
)

#%%
# =============================================================================
# SECTION 24. MONTHLY EFFECT OF SINGLE-CF VERSUS SEASONAL-CF CORRECTION
# =============================================================================
# Validation uses 2018-2020 only. PMB is intentionally absent from this plot:
# it supplied the calibration factors, while ERA5 is shown as the independent
# comparison dataset. Separate product panels prevent seven overlapping curves
# from obscuring the change in monthly pattern.
# =============================================================================

e2_cf_method_comparison_products = {
    ERA5_NAME: era5_mon_01,
    GPCP_NAME: gpcp_mon_01,
    GPCP_SINGLE_CORR_NAME: gpcp_mon_01_e2_single_corr,
    GPCP_CORR_NAME: gpcp_mon_01_e2_corr,
    PMW8_NAME: gpm_pmw_v08_mon_01,
    PMW8_SINGLE_CORR_NAME: gpm_pmw_v08_mon_01_e2_single_corr,
    PMW8_CORR_NAME: gpm_pmw_v08_mon_01_e2_corr,
}

(
    e2_cf_method_validation_monthly_df,
    e2_cf_method_validation_monthly_clim_df,
) = validation_monthly_climatology_from_fields(
    product_dict=e2_cf_method_comparison_products,
    region_masks=region_masks_01deg,
    validation_years=VALIDATION_YEARS,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

monthly_comparison_csv = os.path.join(
    path_to_dfs,
    f"E2_single_vs_seasonal_CF_monthly_validation_2018_2020_{cde_run_dte}.csv",
)
e2_cf_method_validation_monthly_clim_df.to_csv(
    monthly_comparison_csv,
    index=False,
)

fig, axes = plot_e2_single_vs_seasonal_monthly_climatology(
    monthly_clim_df=e2_cf_method_validation_monthly_clim_df,
    product_panels={
        "GPCP V3.3": {
            "original": GPCP_NAME,
            "single": GPCP_SINGLE_CORR_NAME,
            "seasonal": GPCP_CORR_NAME,
        },
        "GPM PMW V08": {
            "original": PMW8_NAME,
            "single": PMW8_SINGLE_CORR_NAME,
            "seasonal": PMW8_CORR_NAME,
        },
    },
    era5_product=ERA5_NAME,
    region="Antarctica",
)

monthly_comparison_plot = os.path.join(
    path_to_plots,
    f"E2_single_vs_seasonal_CF_monthly_validation_2018_2020_{cde_run_dte}.png",
)
fig.savefig(monthly_comparison_plot, dpi=200, bbox_inches="tight")
# plt.close(fig)

print("\nSaved new E2 outputs:")
print("  Seasonal-period comparison:", seasonal_cf_comparison_plot)
print("  Single-vs-seasonal monthly comparison:", monthly_comparison_plot)
print("  Monthly comparison values:", monthly_comparison_csv)

#%%
# =============================================================================
# SECTION 25. MONTHLY-CLIMATOLOGICAL CF SCENARIO: DERIVE AND SAVE FACTORS
# =============================================================================
# Scenario definition:
#   - one AIS-wide correction factor for each calendar month and product;
#   - 2013-2017 factors are frozen parameters for independent validation;
#   - 2013-2020 factors are the final operational values for delivery;
#   - PMB is used only to derive the factors and is not treated as a validation
#     product.
#
# Exact PMB/product timestamps are paired inside the reusable derivation
# function before monthly means and correction factors are calculated.
# =============================================================================

e2_monthly_correction_factors = (
    derive_monthly_climatological_correction_factors(
        monthly_region_df=e2_regional_monthly_uncorrected,
        period_years=CALIBRATION_YEARS,
        reference_product=REFERENCE_NAME,
        target_products=(
            GPCP_DEADJ_NAME,
            PMW8_NAME,
        ),
        regions=("Antarctica",),
        months=tuple(range(1, 13)),
    )
)

e2_final_monthly_correction_factors = (
    derive_monthly_climatological_correction_factors(
        monthly_region_df=e2_regional_monthly_uncorrected,
        period_years=FINAL_FACTOR_YEARS,
        reference_product=REFERENCE_NAME,
        target_products=(
            GPCP_DEADJ_NAME,
            PMW8_NAME,
        ),
        regions=("Antarctica",),
        months=tuple(range(1, 13)),
    )
)

# Fail visibly in the interactive session if either table is incomplete. Each
# period must contain exactly one finite, positive factor for all 12 months of
# both products before results are saved or used downstream.
for period_label, factor_table in (
    ("2013-2017 validation", e2_monthly_correction_factors),
    ("2013-2020 operational", e2_final_monthly_correction_factors),
):
    for product_name in (GPCP_DEADJ_NAME, PMW8_NAME):
        product_factors = factor_table[
            (factor_table["region"] == "Antarctica")
            & (factor_table["product"] == product_name)
        ]
        available_months = set(product_factors["month"].astype(int))
        expected_months = set(range(1, 13))
        valid_factors = (
            len(product_factors) == 12
            and available_months == expected_months
            and np.isfinite(product_factors["correction_factor"]).all()
            and (product_factors["correction_factor"] > 0).all()
        )
        if not valid_factors:
            raise ValueError(
                "Incomplete or invalid monthly correction factors for "
                f"{product_name!r}, {period_label!r}"
            )

# Add explicit labels so saved tables cannot be confused downstream.
e2_monthly_correction_factors = e2_monthly_correction_factors.assign(
    derivation_period="2013-2017 validation parameters",
    intended_use="frozen for independent validation on 2018-2020",
)
e2_final_monthly_correction_factors = (
    e2_final_monthly_correction_factors.assign(
        derivation_period="2013-2020 final operational",
        intended_use="science-team operational delivery",
    )
)

monthly_validation_cf_file = os.path.join(
    path_to_dfs,
    f"E2_monthly_AIS_PMB_correction_factors_cal2013_2017_"
    f"{cde_run_dte}.csv",
)
monthly_operational_cf_file = os.path.join(
    path_to_dfs,
    f"E2_monthly_AIS_PMB_correction_factors_2013_2020_FINAL_"
    f"{cde_run_dte}.csv",
)

e2_monthly_correction_factors.to_csv(
    monthly_validation_cf_file,
    index=False,
)
e2_final_monthly_correction_factors.to_csv(
    monthly_operational_cf_file,
    index=False,
)

print("\nMonthly CFs derived successfully (rows per product):")
print(
    e2_monthly_correction_factors
    .groupby("product")["month"]
    .nunique()
    .rename("2013-2017 month count")
)
print(
    e2_final_monthly_correction_factors
    .groupby("product")["month"]
    .nunique()
    .rename("2013-2020 month count")
)
print("Saved validation factors :", monthly_validation_cf_file)
print("Saved operational factors:", monthly_operational_cf_file)

#%%
# =============================================================================
# SECTION 25_A. PLOT MONTHLY-CLIMATOLOGICAL CORRECTION FACTORS
# =============================================================================
# Both panels show the same type of quantity: one PMB-derived factor for each
# calendar month. A shared y-axis is therefore appropriate and allows direct
# visual comparison of GPCP and PMW correction-factor magnitudes.
#
# Dashed circles: 2013-2017 factors used only for independent validation.
# Solid squares : 2013-2020 final operational factors for science-team use.
# The numerical annotations report the final operational values.
# =============================================================================

fig, axes = plot_monthly_climatological_cf_comparison(
    calibration_cf_df=e2_monthly_correction_factors,
    operational_cf_df=e2_final_monthly_correction_factors,
    product_order=(
        GPCP_DEADJ_NAME,
        PMW8_NAME,
    ),
    product_colors={
        GPCP_DEADJ_NAME: "tab:orange",
        PMW8_NAME: "tab:green",
    },
    calibration_label="2013-2017 validation factors",
    operational_label="2013-2020 operational factors",
    region="Antarctica",
    annotation_decimals=2,
)

monthly_cf_comparison_plot = os.path.join(
    path_to_plots,
    f"E2_monthly_AIS_PMB_correction_factors_"
    f"2013_2017_vs_2013_2020_{cde_run_dte}.png",
)

fig.savefig(
    monthly_cf_comparison_plot,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print(
    "Saved monthly correction-factor comparison:",
    monthly_cf_comparison_plot,
)

#%%
# =============================================================================
# SECTION 26. APPLY 2013-2017 MONTHLY CFs TO GPCP AND PMW
# =============================================================================
# Independent-validation rule:
#   Only the 12 factors derived from the 2013-2017 calibration period are
#   applied here. The 2013-2020 operational factors produced in Section 25 are
#   deliverables and must not be used to assess performance during 2018-2020.
#
# The factors are applied to the full monthly grids so that one reusable field
# is available for downstream summaries. Validation cells must select only
# common_time_validation / VALIDATION_YEARS (2018-2020).
# =============================================================================

GPCP_MONTHLY_CORR_NAME = "GPCP V3.3 monthly-CF corrected"
PMW8_MONTHLY_CORR_NAME = "GPM PMW V08 monthly-CF corrected"

gpcp_mon_01_e2_monthly_corr = apply_monthly_climatological_correction(
    da_monthly=gpcp_mon_01_deadjusted,
    correction_factor_df=e2_monthly_correction_factors,
    product_name=GPCP_DEADJ_NAME,
    ais_mask=AIS_MASK_01,
    region="Antarctica",
    time_name="time",
    lat_name="lat",
    lon_name="lon",
    corrected_name=GPCP_MONTHLY_CORR_NAME,
)

gpm_pmw_v08_mon_01_e2_monthly_corr = (
    apply_monthly_climatological_correction(
        da_monthly=gpm_pmw_v08_mon_01,
        correction_factor_df=e2_monthly_correction_factors,
        product_name=PMW8_NAME,
        ais_mask=AIS_MASK_01,
        region="Antarctica",
        time_name="time",
        lat_name="lat",
        lon_name="lon",
        corrected_name=PMW8_MONTHLY_CORR_NAME,
    )
)

# Guard against accidental leakage from the 2018-2020 validation years.
for corrected_field in (
    gpcp_mon_01_e2_monthly_corr,
    gpm_pmw_v08_mon_01_e2_monthly_corr,
):
    derivation_start = corrected_field.attrs.get(
        "correction_period_start_year"
    )
    derivation_end = corrected_field.attrs.get(
        "correction_period_end_year"
    )
    if (derivation_start, derivation_end) != (2013, 2017):
        raise ValueError(
            "Independent-validation monthly correction must use factors "
            "derived only from 2013-2017; found "
            f"{derivation_start}-{derivation_end} for "
            f"{corrected_field.name!r}"
        )

print("\nMonthly-CF corrected gridded products are ready:")
print(" ", gpcp_mon_01_e2_monthly_corr.name)
print(" ", gpm_pmw_v08_mon_01_e2_monthly_corr.name)
print("Factor derivation period: 2013-2017")
print("Validation period to evaluate: 2018-2020")

#%%
# =============================================================================
# SECTION 27. MONTHLY-CF SCENARIO: INDEPENDENT VALIDATION, 2018-2020
# =============================================================================
# Compare ERA5, the original satellite products, and the versions corrected
# with the frozen 2013-2017 monthly factors. PMB is absent from the validation
# plots because it was used to estimate the correction parameters.
#
# Regional values are retained in the saved tables, while the figures show the
# AIS-wide (Antarctica) result used in the E2 presentation.
# =============================================================================

e2_monthly_cf_validation_products = {
    ERA5_NAME: era5_mon_01,
    GPCP_NAME: gpcp_mon_01,
    GPCP_MONTHLY_CORR_NAME: gpcp_mon_01_e2_monthly_corr,
    PMW8_NAME: gpm_pmw_v08_mon_01,
    PMW8_MONTHLY_CORR_NAME: gpm_pmw_v08_mon_01_e2_monthly_corr,
}

e2_monthly_cf_product_order = (
    ERA5_NAME,
    GPCP_NAME,
    GPCP_MONTHLY_CORR_NAME,
    PMW8_NAME,
    PMW8_MONTHLY_CORR_NAME,
)

e2_monthly_cf_styles = {
    ERA5_NAME: {
        "color": "blue", "marker": "s", "lw": 2.5,
        "linestyle": "-",
    },
    GPCP_NAME: {
        "color": "tab:orange", "marker": "D", "lw": 2.0,
        "linestyle": "--", "alpha": 0.60,
    },
    GPCP_MONTHLY_CORR_NAME: {
        "color": "tab:orange", "marker": "o", "lw": 3.0,
        "linestyle": "-",
    },
    PMW8_NAME: {
        "color": "tab:green", "marker": "D", "lw": 2.0,
        "linestyle": "--", "alpha": 0.60,
    },
    PMW8_MONTHLY_CORR_NAME: {
        "color": "tab:green", "marker": "o", "lw": 3.0,
        "linestyle": "-",
    },
}

# -----------------------------------------------------------------------------
# 27.1 Monthly climatology
# -----------------------------------------------------------------------------

(
    e2_monthly_cf_validation_monthly_df,
    e2_monthly_cf_validation_monthly_clim_df,
) = validation_monthly_climatology_from_fields(
    product_dict=e2_monthly_cf_validation_products,
    region_masks=region_masks_01deg,
    validation_years=VALIDATION_YEARS,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

monthly_cf_validation_csv = os.path.join(
    path_to_dfs,
    f"E2_monthly_CF_validation_monthly_climatology_2018_2020_"
    f"{cde_run_dte}.csv",
)
e2_monthly_cf_validation_monthly_clim_df.to_csv(
    monthly_cf_validation_csv,
    index=False,
)

fig, axes = plot_validation_monthly_climatology_numeric_months(
    clim_df=e2_monthly_cf_validation_monthly_clim_df,
    region_order=("Antarctica",),
    product_order=e2_monthly_cf_product_order,
    product_styles=e2_monthly_cf_styles,
    figsize=(10, 5.5),
    ylabel="mm/month",
)

monthly_cf_validation_plot = os.path.join(
    path_to_plots,
    f"E2_monthly_CF_validation_monthly_climatology_2018_2020_"
    f"{cde_run_dte}.png",
)
fig.savefig(monthly_cf_validation_plot, dpi=200, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# 27.2 Seasonal climatology
# -----------------------------------------------------------------------------

e2_monthly_cf_regional_monthly_df = (
    build_all_region_monthly_series_cosine(
        product_dict=e2_monthly_cf_validation_products,
        region_masks=region_masks_01deg,
        lat_name="lat",
        lon_name="lon",
        time_name="time",
    )
)

(
    e2_monthly_cf_validation_seasonal_df,
    e2_monthly_cf_validation_seasonal_clim_df,
) = validation_seasonal_climatology_from_monthly_df(
    full_monthly_region_df=e2_monthly_cf_regional_monthly_df,
    validation_years=VALIDATION_YEARS,
    require_complete_season=True,
)

seasonal_cf_validation_csv = os.path.join(
    path_to_dfs,
    f"E2_monthly_CF_validation_seasonal_climatology_2018_2020_"
    f"{cde_run_dte}.csv",
)
e2_monthly_cf_validation_seasonal_clim_df.to_csv(
    seasonal_cf_validation_csv,
    index=False,
)

fig, axes = plot_seasonal_climatology(
    clim_df=e2_monthly_cf_validation_seasonal_clim_df,
    region_order=("Antarctica",),
    product_order=e2_monthly_cf_product_order,
    product_styles=e2_monthly_cf_styles,
    figsize=(10, 5.5),
    ylabel="mm/season",
    y_nbins=4,
    legend_ncol=3,
)

seasonal_cf_validation_plot = os.path.join(
    path_to_plots,
    f"E2_monthly_CF_validation_seasonal_climatology_2018_2020_"
    f"{cde_run_dte}.png",
)
fig.savefig(seasonal_cf_validation_plot, dpi=200, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# 27.3 Mean annual precipitation
# -----------------------------------------------------------------------------

e2_monthly_cf_validation_common_products = {
    product_name: product_field.sel(time=common_time_validation)
    for product_name, product_field
    in e2_monthly_cf_validation_products.items()
}

(
    e2_monthly_cf_validation_annual_df,
    e2_monthly_cf_validation_mean_annual_df,
) = validation_regional_annual_dataframe(
    product_dict=e2_monthly_cf_validation_common_products,
    region_masks=region_masks_01deg,
    validation_years=VALIDATION_YEARS,
)

annual_cf_validation_csv = os.path.join(
    path_to_dfs,
    f"E2_monthly_CF_validation_mean_annual_2018_2020_"
    f"{cde_run_dte}.csv",
)
e2_monthly_cf_validation_mean_annual_df.to_csv(
    annual_cf_validation_csv,
    index=False,
)

e2_monthly_cf_bar_colors = {
    ERA5_NAME: {"color": "blue", "alpha": 1.00},
    GPCP_NAME: {"color": "tab:orange", "alpha": 0.50},
    GPCP_MONTHLY_CORR_NAME: {"color": "tab:orange", "alpha": 1.00},
    PMW8_NAME: {"color": "tab:green", "alpha": 0.50},
    PMW8_MONTHLY_CORR_NAME: {"color": "tab:green", "alpha": 1.00},
}

fig, ax = plot_regional_mean_annual_bars(
    df_mean_regional=e2_monthly_cf_validation_mean_annual_df,
    region_order=("Antarctica",),
    product_order=e2_monthly_cf_product_order,
    product_colors=e2_monthly_cf_bar_colors,
    ylabel="[mm/year]",
    title="Monthly-CF Independent Validation (2018-2020)",
    annotate=True,
    legend_ncol=2,
)

annual_cf_validation_plot = os.path.join(
    path_to_plots,
    f"E2_monthly_CF_validation_mean_annual_2018_2020_"
    f"{cde_run_dte}.png",
)
fig.savefig(annual_cf_validation_plot, dpi=200, bbox_inches="tight")
plt.show()

print("\nSaved monthly-CF validation outputs:")
print(" Monthly plot :", monthly_cf_validation_plot)
print(" Seasonal plot:", seasonal_cf_validation_plot)
print(" Annual plot  :", annual_cf_validation_plot)
print(" Monthly data :", monthly_cf_validation_csv)
print(" Seasonal data:", seasonal_cf_validation_csv)
print(" Annual data  :", annual_cf_validation_csv)

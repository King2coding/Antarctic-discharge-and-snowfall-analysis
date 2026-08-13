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

REFERENCE_NAME = r"$P_{\mathrm{MB}}$"

GPCP_NAME = "GPCP V3.3"
GPCP_CORR_NAME = "GPCP V3.3 corrected"

PMW8_NAME = "GPM PMW V08"
PMW8_CORR_NAME = "GPM PMW V08 corrected"

ERA5_NAME = "ERA5"

#%%
# =============================================================================
# SECTION 5. LOAD RAW PRODUCT DATA
# =============================================================================

print("Loading GPCP monthly dataset ...")

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

gpcp_ds_v3pt3 = ds_swaplon(gpcp_ds_v3pt3)

# Keep the monthly precipitation variable
# Your file shows the variable name is sat_gauge_precip
gpcp_mnth = gpcp_ds_v3pt3["sat_gauge_precip"].copy()

# Normalize monthly timestamps to month-start
gpcp_mnth = gpcp_mnth.assign_coords(
    time=pd.to_datetime(gpcp_mnth["time"].values).to_period("M").to_timestamp()
)

# Convert from mm/day to mm/month
days_in_month = xr.DataArray(
    pd.to_datetime(gpcp_mnth["time"].values).days_in_month,
    dims=["time"],
    coords={"time": gpcp_mnth["time"]}
)

gpcp_mnth = gpcp_mnth * days_in_month
gpcp_mnth.name = "gpcp_mm_month"

# Replace fill/missing with NaN if needed
fillv = gpcp_mnth.attrs.get("_FillValue", None)
if fillv is not None:
    gpcp_mnth = gpcp_mnth.where(gpcp_mnth != fillv)
gpcp_mnth = gpcp_mnth.where(np.isfinite(gpcp_mnth))

# Subset Antarctica
gpcp_mnth = gpcp_mnth.sel(lat=slice(-60, -90))

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

gc.collect()

#%%

# =============================================================================
# SECTION 7. APPLY BASIN MASK DOMAIN
# =============================================================================

# Keep only cells that belong to Antarctica basins included in the study
valid_basin_mask = basin_mask_01deg.notnull()

gpcp_mon_01 = gpcp_mon_01.where(valid_basin_mask)
era5_mon_01 = era5_mnth_01.where(valid_basin_mask)
pmb_mon_01 = pmb_mon_01.where(valid_basin_mask)
pmb_unc_mon_01 = pmb_unc_mon_01.where(valid_basin_mask)

print("✅ Common masked monthly fields ready")
print("GPCP  :", gpcp_mon_01.shape)
print("ERA5  :", era5_mon_01.shape)
print("PMB   :", pmb_mon_01.shape)
print("PMB unc:", pmb_unc_mon_01.shape)


# =============================================================================
# SECTION 8. QUICK SANITY CHECKS
# =============================================================================

print("\n--- Sanity checks ---")
print("Target grid CRS:", target_template_01deg.rio.crs)
print("Basin mask CRS :", basin_mask_01deg.rio.crs)

print("GPCP time range:", str(gpcp_mon_01.time.min().values), "->", str(gpcp_mon_01.time.max().values))
print("ERA5 time range:", str(era5_mon_01.time.min().values), "->", str(era5_mon_01.time.max().values))
print("PMB time range :", str(pmb_mon_01.time.min().values),  "->", str(pmb_mon_01.time.max().values))
print("PMB uncertainty time range :", str(pmb_unc_mon_01.time.min().values), "->", str(pmb_unc_mon_01.time.max().values))

#%%
# =============================================================================
# SECTION 9. E2-A. BUILD REGIONAL MONTHLY SERIES FROM UNCORRECTED DATA
#
# These are the ORIGINAL fields.
# No correction has yet been applied.
# =============================================================================


e2_uncorrected_product_dict = {
    REFERENCE_NAME: pmb_mon_01,
    ERA5_NAME: era5_mon_01,
    GPCP_NAME: gpcp_mon_01,
    PMW8_NAME: gpm_pmw_v08_mon_01,
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


print(
    e2_seasonal_uncorrected
    .sort_values(
        ["region", "product", "season_year", "season"]
    )
    .head(30)
)

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


print("\nNumber of complete seasons:")
print(season_counts)


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
        GPCP_NAME,
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
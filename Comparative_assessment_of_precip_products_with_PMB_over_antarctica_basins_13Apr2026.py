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
path_to_plots = r'/home/kkumah/Projects/Antarctic_discharge_work/plots'
path_to_dfs = r'/home/kkumah/Projects/Antarctic_discharge_work/dfs'
gpm_satellites_path = r'/ra1/pubdat/GPM-Constellation-Satellites_MI_and_Sounders'

# --- Precipitation products ---
gpcp_v3pt3_mnthly_ds_path = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_3_monthly_1983_2024'
era5_mnhtly_file = r'/ra1/pubdat/GPCP/GPCP_Reproduce_GJ/era5_tp_198001202412_monthly.nc'

# --- Basin / PMB data ---
basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'
Pmb_mm_fle  = os.path.join(basins_path, 'Monthly_mass_budget_precip_RignotBasin_in_mm_20260226.nc')

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

#%%

# =============================================================================
# SECTION 5. LOAD BASIN GRID AND BUILD COMMON 0.1° TARGET GRID
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
# floating variables

#%%
# =============================================================================
# SECTION 6. LOAD RAW PRODUCT DATA
# =============================================================================

print("Loading GPCP monthly dataset ...")
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


gc.collect()

#%%

# =============================================================================
# SECTION 7. REMAP ALL PRODUCTS TO THE COMMON 0.1° TARGET GRID
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


gc.collect()

#%%

# =============================================================================
# SECTION 8. APPLY BASIN MASK DOMAIN
# =============================================================================

# Keep only cells that belong to Antarctica basins included in the study
valid_basin_mask = basin_mask_01deg.notnull()

gpcp_mon_01 = gpcp_mon_01.where(valid_basin_mask)
era5_mon_01 = era5_mnth_01.where(valid_basin_mask)
pmb_mon_01  = pmb_mon_01.where(valid_basin_mask)

print("✅ Common masked monthly fields ready")
print("GPCP  :", gpcp_mon_01.shape)
print("ERA5  :", era5_mon_01.shape)
print("PMB   :", pmb_mon_01.shape)


# =============================================================================
# SECTION 9. QUICK SANITY CHECKS
# =============================================================================

print("\n--- Sanity checks ---")
print("Target grid CRS:", target_template_01deg.rio.crs)
print("Basin mask CRS :", basin_mask_01deg.rio.crs)

print("GPCP time range:", str(gpcp_mon_01.time.min().values), "->", str(gpcp_mon_01.time.max().values))
print("ERA5 time range:", str(era5_mon_01.time.min().values), "->", str(era5_mon_01.time.max().values))
print("PMB time range :", str(pmb_mon_01.time.min().values),  "->", str(pmb_mon_01.time.max().values))


#%% Build GMP Data Series
# =============================================================================
# SECTION 1D. GPM MICROWAVE MONTHLY INPUT PREPARATION
# =============================================================================
gpm_family_monthly_dict = build_gpm_family_monthly_dict(
    gpm_satellites_path=gpm_satellites_path,
    preprocess_func=preprocess,
    target_template_01deg=target_template_01deg,
    basin_mask_01deg=basin_mask_01deg,
)

print(gpm_family_monthly_dict.keys())

gpm_pmw_v07_mon_01 = build_gpm_pmw_mean(
    gpm_family_dict=gpm_family_monthly_dict,
    mean_name="GPM PMW V07"
)

# print(gpm_pmw_v07_mon_01)
print(gpm_pmw_v07_mon_01.shape)
#%%
# =============================================================================
# SECTION 2B. BUILD MONTHLY REGIONAL SERIES FOR PMB, ERA5, GPCP
# =============================================================================

product_monthly_dict = {
    r"$P_{\mathrm{MB}}$": pmb_mon_01,
    "ERA5": era5_mnth_01,
    "GPCP V3.3": gpcp_mon_01,
    "GPM PMW V07": gpm_pmw_v07_mon_01
}

regional_monthly_cos_df = build_all_region_monthly_series_cosine(
    product_dict=product_monthly_dict,
    region_masks=region_masks_01deg,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

# print(regional_monthly_cos_df.head())
# print(regional_monthly_cos_df.tail())
# print(regional_monthly_cos_df.groupby(["region", "product"]).size())


#%% Monthly Climatology

region_monthly_clim_cos = compute_monthly_climatology_from_regional_series(
    regional_monthly_cos_df
)

region_monthly_wide_dict = regional_monthly_tidy_to_region_dict(
    regional_monthly_cos_df,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
)

svnme = os.path.join(path_to_plots, f'monthly_climatology_precip_over_imbie_basins_{cde_run_dte}.png')
fig, axes = plot_monthly_climatology(
    region_monthly_clim_cos,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP V3.3", "GPM PMW V07"),
    product_styles=product_styles_corr,
    figsize=(10, 9),
    ylabel="mm/month",
    y_nbins=4,
    legend_ncol=3,
)

fig.savefig(svnme, dpi=300)
plt.show()
gc.collect()

#%% Seasonal Climatology
region_seasonal_clim_cos = compute_seasonal_climatology_from_regional_series(
    regional_monthly_cos_df,
    drop_incomplete=True
)

# region_seasonal_clim_cos should already exist
# and should include raw GPM PMW V07

pmw_scalar_factors_df = compute_scalar_pmw_factors_from_seasonal_clim_df(
    region_seasonal_clim_df=region_seasonal_clim_cos,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    reference_col=r"$P_{\mathrm{MB}}$",
    pmw_col="GPM PMW V07",
)

region_seasonal_clim_corr = add_scalar_corrected_pmw_to_seasonal_clim_df(
    region_seasonal_clim_df=region_seasonal_clim_cos,
    scalar_factors_df=pmw_scalar_factors_df,
    pmw_col="GPM PMW V07",
    corrected_col="GPM PMW V07 (corr.)",
)

fig, axes = plot_seasonal_climatology(
    region_seasonal_clim_corr,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(
        r"$P_{\mathrm{MB}}$",
        "ERA5",
        "GPCP V3.3",
        "GPM PMW V07",
        "GPM PMW V07 (corr.)",
    ),
    product_styles=product_styles_corr,
    figsize=(10, 8),
    ylabel="mm/season",
    y_nbins=4,
    legend_ncol=3,
)

svnme = os.path.join(
    path_to_plots,
    f"seasonal_climatology_precip_over_imbie_basins_{cde_run_dte}.png"
)
plt.savefig(svnme, dpi=500, bbox_inches="tight")
# plt.show()
gc.collect()

#%% Interannual Variability
region_annual_cos = compute_annual_totals_from_regional_series(
    regional_monthly_cos_df
)

region_annual_corr = add_scalar_corrected_pmw_to_annual_df(
    region_annual_df=region_annual_cos,
    scalar_factors_df=pmw_scalar_factors_df,
    pmw_col="GPM PMW V07",
    corrected_col="GPM PMW V07 (corr.)",
)

svnme = os.path.join(path_to_plots, f'interannual_variability_precip_over_imbie_basins_{cde_run_dte}.png')
fig, axes = plot_interannual_variability(
    region_annual_corr,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP V3.3", "GPM PMW V07", "GPM PMW V07 (corr.)"),
    product_styles=product_styles_corr,
    figsize=(10, 9),
    ylabel="mm/year",
    y_nbins=4,
    legend_ncol=3,
)

fig.savefig(svnme, dpi=300)
plt.show()
gc.collect()


#%% Seasonal Anomalies
# =============================================================================
#BUILD REGION-WISE SEASONAL SERIES AND ANOMALIES
# =============================================================================

region_monthly_wide_dict = regional_monthly_tidy_to_region_dict(
    regional_monthly_cos_df,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
)

ts_seasonal_dict = {}
ts_seasonal_anom_dict = {}

for region, wide_df in region_monthly_wide_dict.items():
    ts_seasonal = build_conventional_seasonal_series_from_region_monthly(
        ts_region_monthly=wide_df,
        seasonal_mode="mean",
        drop_incomplete=True,
    )
    ts_anom = deseasonalize_seasonal_series(ts_seasonal)

    ts_seasonal_dict[region] = ts_seasonal
    ts_seasonal_anom_dict[region] = ts_anom

# print(ts_seasonal_dict["Antarctica"].head())
# print(ts_seasonal_anom_dict["Antarctica"].head())

yticks_by_region = {
    "Antarctica": [-4, 0, 4],
    "West Antarctica": [-20, 0, 20],
    "East Antarctica": [-4, 0, 4],
}
svnme = os.path.join(path_to_plots, f'seasonal_anomalies_precip_over_imbie_basins_{cde_run_dte}.png')

fig, axes = plot_seasonal_anomaly_timeseries_regions_3x1(
    ts_seasonal_anom_dict=ts_seasonal_anom_dict,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    ref_col=r"$P_{\mathrm{MB}}$",
    target_cols=("ERA5", "GPCP V3.3", "GPM PMW V07"),
    product_styles=product_styles_corr,
    figsize=(10, 9),
    ylabel="mm/season",
    legend_ncol=4,  
    yticks_by_region=yticks_by_region,
)

fig.savefig(svnme, dpi=300)
plt.show()
gc.collect()

#----------------------------------------------------------------------------------
lims_by_region = {
    "Antarctica": (-3, 3),
    "West Antarctica": (-10, 10),
    "East Antarctica": (-3, 3),
}
svnme = os.path.join(path_to_plots, f'seasonal_anomaly_scatter_precip_over_imbie_basins_{cde_run_dte}.png')

fig_sc, axes_sc, stats_sc = plot_seasonal_anomaly_scatter_regions_3x3(
    region_anom_dict=ts_seasonal_anom_dict,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    ref_col=r"$P_{\mathrm{MB}}$",
    target_cols=("ERA5", "GPCP V3.3", "GPM PMW V07"),
    figsize=(12.0, 11.0),
    share_lims=False,
    lims=lims_by_region,
    equal_axes=True,
    point_size=58,
)

fig_sc.savefig(svnme, dpi=300)
plt.show()
gc.collect()

stats_table_wide = seasonal_scatter_stats_to_wide_table(
    stats_out=stats_sc,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    target_cols=("ERA5", "GPCP V3.3", "GPM PMW V07"),
    ref_col=r"$P_{\mathrm{MB}}$",
)

print(stats_table_wide)

#%% ANNUAL TOTALS AND 2013–2020 MEAN ANNUAL FIELDS

# Build annual fields [mm/year]
pmb_annual_01  = monthly_to_annual_totals_field(pmb_mon_01)
era5_annual_01 = monthly_to_annual_totals_field(era5_mnth_01)
gpcp_annual_01 = monthly_to_annual_totals_field(gpcp_mon_01)
gpm_pmw_v07_01 = monthly_to_annual_totals_field(gpm_pmw_v07_mon_01)
# Build 2013–2020 mean annual fields [mm/year]
pmb_annual_mean_01  = annual_to_multiyear_mean_field(pmb_annual_01,  2013, 2020)
era5_annual_mean_01 = annual_to_multiyear_mean_field(era5_annual_01, 2013, 2020)
gpcp_annual_mean_01 = annual_to_multiyear_mean_field(gpcp_annual_01, 2013, 2020)
gpm_pmw_v07_01 = annual_to_multiyear_mean_field(gpm_pmw_v07_01, 2013, 2020)

print(pmb_annual_mean_01.shape, era5_annual_mean_01.shape, gpcp_annual_mean_01.shape)


# =============================================================================
# SECTION 10.3. BUILD BASIN MULTI-YEAR MEAN ANNUAL DATAFRAME
# =============================================================================

BASIN_IDS = sorted(AIS_BASINS)

basin_mask_01deg_clean = basin_mask_01deg.where(basin_mask_01deg.isin(BASIN_IDS))

df_pmb_basin = compute_basin_cosine_weighted_means_from_field(
    da_2d=pmb_annual_mean_01,
    basin_mask_2d=basin_mask_01deg_clean,
    basin_ids=BASIN_IDS,
    value_name=r"$P_{\mathrm{MB}}$",
)

df_era5_basin = compute_basin_cosine_weighted_means_from_field(
    da_2d=era5_annual_mean_01,
    basin_mask_2d=basin_mask_01deg_clean,
    basin_ids=BASIN_IDS,
    value_name="ERA5",
)

df_gpcp_basin = compute_basin_cosine_weighted_means_from_field(
    da_2d=gpcp_annual_mean_01,
    basin_mask_2d=basin_mask_01deg_clean,
    basin_ids=BASIN_IDS,
    value_name="GPCP V3.3",
)

df_gpm_pmw_v07 = compute_basin_cosine_weighted_means_from_field(
    da_2d=gpm_pmw_v07_01,
    basin_mask_2d=basin_mask_01deg_clean,
    basin_ids=BASIN_IDS,
    value_name="GPM PMW V07",
)

# Merge
df_basin_mean_annual = (
    df_pmb_basin
    .merge(df_era5_basin, on="basin", how="outer")
    .merge(df_gpcp_basin, on="basin", how="outer")
    .merge(df_gpm_pmw_v07, on="basin", how="outer")
    .sort_values("basin")
    .reset_index(drop=True)
)

# print(df_basin_mean_annual)
svnme = os.path.join(path_to_plots, f'annual_precip_basin_mean_scatter_{cde_run_dte}.png')
fig_sc_basin, axes_sc_basin, stats_sc_basin = plot_pmb_scatter_oldstyle(
    df_mean_yr_acc=df_basin_mean_annual,
    ref=r"$P_{\mathrm{MB}}$",
    products=["ERA5", "GPCP V3.3", "GPM PMW V07"],
    high_thresh=500.0,
    scale="log",
    log_min=2,
    log_ticks=(5, 10, 20, 50, 100, 200, 500, 1000, 2000),
    ncols=3,
    figsize_per_col=4.8,
    figsize_per_row=4.5,
    share_axes=False,
    show_ylabel_only_left=False,
)

fig_sc_basin.savefig(svnme, dpi=300)
plt.show()

print(stats_sc_basin)
gc.collect()

#----------------------------------------------------------------------------------
svnme = os.path.join(path_to_plots, f'basin_spread_points_precip_over_imbie_basins_{cde_run_dte}.png')
fig_spread, ax_spread, spread_non_gpm, spread_gpm = plot_basin_spread_points_dual(
    df=df_basin_mean_annual,
    basin_col="basin",
    ref_col=r"$P_{\mathrm{MB}}$",
    prod_cols=["ERA5", "GPCP V3.3", "GPM PMW V07"],   # GPM placeholder okay if absent
    prod_labels=None,
    product_styles=product_styles_corr,
    non_gpm_group=[r"$P_{\mathrm{MB}}$", "ERA5", "GPCP V3.3"],
    gpm_group=[r"$P_{\mathrm{MB}}$", "GPM PMW V07"],
    figsize=(13, 5.2),
    log_scale=True,
    ylim=(2, 2000),
    legend_ncol=4,
    place_key=True,
)

fig_spread.savefig(svnme, dpi=300)

plt.show()
gc.collect()

#%% ANnual Mean 
BASIN_IDS = sorted(AIS_BASINS)
basin_mask_01deg_clean = basin_mask_01deg.where(basin_mask_01deg < 1e10)
basin_mask_01deg_clean = basin_mask_01deg_clean.where(basin_mask_01deg_clean.isin(BASIN_IDS))

pmb_pack = build_basin_mean_plot_product(
    pmb_mon_01, basin_mask_01deg_clean, BASIN_IDS, r"P$_{MB}$"
)
era5_pack = build_basin_mean_plot_product(
    era5_mnth_01, basin_mask_01deg_clean, BASIN_IDS, "ERA5"
)
gpcp_pack = build_basin_mean_plot_product(
    gpcp_mon_01, basin_mask_01deg_clean, BASIN_IDS, "GPCP V3.3"
)
# =============================================================================
# BUILD GPM MONTHLY 0.1° FIELDS, BASIN PACKS, AND FINAL DUAL-CBAR MAP
# =============================================================================

# -------------------------------------------------------------------------
# 1. Pull GPM-family monthly 0.1° fields from the prepared dictionary
# -------------------------------------------------------------------------
dmsp_mon_01 = gpm_family_monthly_dict["DMSP SSMIS"]
atms_mon_01 = gpm_family_monthly_dict["ATMS"]
mhs_mon_01  = gpm_family_monthly_dict["MHS"]
amsr2_mon_01 = gpm_family_monthly_dict["AMSR2"]

# -------------------------------------------------------------------------
# 2. Build overall GPM PMW V07 mean monthly 0.1° field
# -------------------------------------------------------------------------
gpm_pmw_mon_01 = build_gpm_pmw_mean(
    gpm_family_dict=gpm_family_monthly_dict,
    mean_name="GPM PMW V07"
)

print("GPM PMW V07:", gpm_pmw_mon_01.shape, gpm_pmw_mon_01.name)

# -------------------------------------------------------------------------
# 3. Build basin-aggregated annual-mean packs
# -------------------------------------------------------------------------
atms_pack = build_basin_mean_plot_product(
    atms_mon_01,
    basin_mask_01deg_clean,
    BASIN_IDS,
    "ATMS"
)

mhs_pack = build_basin_mean_plot_product(
    mhs_mon_01,
    basin_mask_01deg_clean,
    BASIN_IDS,
    "MHS"
)

dmsp_pack = build_basin_mean_plot_product(
    dmsp_mon_01,
    basin_mask_01deg_clean,
    BASIN_IDS,
    "DMSP-SSMIS"
)

amsr2_pack = build_basin_mean_plot_product(
    amsr2_mon_01,
    basin_mask_01deg_clean,
    BASIN_IDS,
    "AMSR2"
)

gpm_pmw_pack = build_basin_mean_plot_product(
    gpm_pmw_mon_01,
    basin_mask_01deg_clean,
    BASIN_IDS,
    "GPM PMW V07"
)

# -------------------------------------------------------------------------
# 4. Assemble the final list for plotting
# -------------------------------------------------------------------------
arr_lst_mean = [
    (pmb_pack["product"],      pmb_pack["plot_grid"],      pmb_pack["panel_mean"]),
    (era5_pack["product"],     era5_pack["plot_grid"],     era5_pack["panel_mean"]),
    (gpcp_pack["product"],     gpcp_pack["plot_grid"],     gpcp_pack["panel_mean"]),
    (atms_pack["product"],     atms_pack["plot_grid"],     atms_pack["panel_mean"]),
    (mhs_pack["product"],      mhs_pack["plot_grid"],      mhs_pack["panel_mean"]),
    (dmsp_pack["product"],     dmsp_pack["plot_grid"],     dmsp_pack["panel_mean"]),
    (amsr2_pack["product"],    amsr2_pack["plot_grid"],    amsr2_pack["panel_mean"]),
    (gpm_pmw_pack["product"],  gpm_pmw_pack["plot_grid"],  gpm_pmw_pack["panel_mean"]),
]


svnme = os.path.join(path_to_plots, f'basin_mean_annual_precip_over_imbie_basins_{cde_run_dte}.png')
fig, axes, cb1, cb2 = compare_mean_precip_basin_dual_cbar(
    arr_lst_mean=arr_lst_mean,
    basin_mask_latlon=basin_mask_01deg_clean,
    group1_idx=[0, 1, 2],
    group2_idx=[3, 4, 5, 6, 7],
    ncols=4,
    gamma1=0.6,
    vmin1=0,
    vmax1=400,
    cbar_tcks1=[0, 25, 50, 100, 200, 300, 400],
    cbar_label1="axes (a, b, c)",
    gamma2=0.6,
    vmin2=0,
    vmax2=80,
    cbar_tcks2=[0, 5, 10, 20, 40, 60, 80],
    cbar_label2="axes (d, e, f, g, h)",
    panel_letters=True,
    show_panel_mean=True,
)
plt.show()

fig.savefig(svnme, dpi=300)

plt.show()
gc.collect()

#==============================================================================
df_regional_mean_annual = compute_regional_mean_annual_precip(
    region_annual_cos,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP V3.3", "GPM PMW V07"),
)
svnme = os.path.join(path_to_plots, f'regional_mean_annual_precip_over_imbie_basins_{cde_run_dte}.png')
fig, ax = plot_regional_mean_annual_bars(
    df_regional_mean_annual,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP V3.3", "GPM PMW V07"),
    product_colors=product_styles_corr,
    figsize=(9, 6),
    ylabel="[mm/year]",
    title="2013–2020 mean annual precipitation",
    annotate=True,
)

fig.savefig(svnme, dpi=300)

plt.show()
gc.collect()

#%% CloudSat
file_cs = r'/ra1/pubdat/Reza_archive/CS_2022_Maps/Monthly'
all_cs_files = [os.path.join(file_cs, f) for f in os.listdir(file_cs) if f.endswith('.h5')]

import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = all_cs_files[0]


with h5py.File(file_path, "r") as f:
    x = f["X"][...]
    y = f["Y"][...]
    snow = f["surface_snowfall_rate"][...]

print("X shape:", x.shape)
print("Y shape:", y.shape)
print("snow shape:", snow.shape)
print("snow min/max:", np.nanmin(snow), np.nanmax(snow))

plt.figure(figsize=(8, 8))
plt.pcolormesh(x, y, snow, shading="auto")
plt.colorbar(label="surface_snowfall_rate")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Surface snowfall rate")
plt.gca().set_aspect("equal")
plt.show()


csfilept = r'/ra1/pubdat/AVHRR_CloudSat_proj/CS_Antartica_analysis_kkk/CS-Antarctica_maps'
cs_ant_mnthly = os.path.join(csfilept, 'CS-Antarctica_monthly_climatology_2007-2010.nc')
xr.open_dataarray(cs_ant_mnthly)[0].plot()
#%%
crs = "+proj=longlat +datum=WGS84 +no_defs"  
crs_format = 'proj4' 

batch_size = 10

cde_run_dte = str(date.today().strftime('%Y%m%d'))


#----------------------------------------------------------------------------------

# id2name must already exist, e.g.
# id2name = {1: "F-G", 2: "A-Ap", 3: "Ap-B", 4: "B-C", ...}

#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# basin grid

basins  = xr.open_dataarray(os.path.join(basins_path,'bedmap3_basins_0.1deg.tif'))
# Mask out invalid values (0 or NaN)
# zwally_data = basins_zwally.where((basins_zwally > 0) & (basins_zwally.notnull()))
basins = basins.where((basins > 1) & (basins.notnull()))
if not basins.rio.crs:
    basins = basins.rio.write_crs(CRS.from_proj4(crs_stereo))
basin_transform = basins.rio.transform()
height, width = basins.data.shape[1:]
xmin, ymax = basin_transform.c, basin_transform.f
xres, yres = basin_transform.a, -basin_transform.e
xmax = xmin + width * xres
ymin = ymax - height * yres
print(f"Basin grid: width={width}, height={height}, xres={xres}, yres={yres}")
basin_bounds = (xmin, xmax, ymin, ymax)  # (minx, maxx, miny, maxy)
print(f"Basin bounds: {basin_bounds}")

# basins_zwally = basins['zwally']

# basins_imbie = basins['imbie']

# %%
colors = plt.cm.gist_ncar(np.linspace(0, 1, 19))

# Give Basin 19 a unique neutral color not used elsewhere
colors[-1] = np.array([0.60, 0.60, 0.60, 1.0])   # medium gray

cmap = mcolors.ListedColormap(colors)
cmap.set_bad(color='white')

vmin, vmax = 1, 19
levels = np.linspace(vmin, vmax, vmax - vmin + 2)
norm = mcolors.BoundaryNorm(levels, cmap.N)

# work with a 2D slice (drop the band dimension)
da = basins.isel(band=0)

proj = ccrs.SouthPolarStereo()
fig, ax = plt.subplots(figsize=(12, 8), dpi=300,
                       subplot_kw={'projection': proj})
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

p = da.plot(
    ax=ax,
    transform=proj,
    cmap=cmap,
    norm=norm,
    add_colorbar=False,
    add_labels=False,
)

ax.set_facecolor('white')

# --- boundaries: use a copy with ocean=0 instead of NaN ---
da_for_contour = da.fillna(0)          # 0 outside basins, 1..19 inside

# coastline (0–1) + internal boundaries (1–19)
boundary_levels = np.arange(0.5, 19.5, 1.0)

ax.contour(
    da["x"].values,
    da["y"].values,
    da_for_contour.values,   # 2D (y, x) with 0/1..19
    levels=boundary_levels,
    colors="k",
    linewidths=0.8,
    transform=proj,
    zorder=5,
)

# --- label offsets for small WAIS basins (same as your name-plot) ---
small_id_offsets = {
    11: (-4.0e5, -1.0e5),  # F-G
    13: (-4.8e5,  1.3e5),  # H-Hp
    14: (-8.4e5,  1.9e5),  # Hp-I
    15: (-4.8e5,  2.5e5),  # I-Ipp
    16: (-0.8e5,  3.7e5),  # Ipp-J
    17: (-0.5e5,  4.2e5),  # J-Jpp
    # add 12 or others if you want them outside too
}

# Annotate each basin
for basin_id in range(1, 20):
    mask = (da == basin_id)
    y, x = np.where(mask.values)
    if len(x) == 0:
        continue

    cx = da["x"].values[x].mean()
    cy = da["y"].values[y].mean()
    label = str(basin_id)

    if basin_id in small_id_offsets:
        # place label outside with a leader line
        dx, dy = small_id_offsets[basin_id]
        lx, ly = cx + dx, cy + dy

        ax.annotate(
            label,
            xy=(cx, cy),      # centroid (tail of line)
            xytext=(lx, ly),  # label position
            textcoords='data',
            xycoords='data',
            ha='center',
            va='center',
            fontsize=15,
            transform=proj,
            arrowprops=dict(
                arrowstyle="-",   # simple line
                lw=0.8,
                color="k"
            ),
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.7
            ),
        )
    else:
        # “normal” in-basin label
        ax.text(
            cx, cy, label,
            color="black",
            fontsize=15,
            ha="center",
            va="center",
            transform=proj,
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.6
            ),
        )

ax.axis("off")
plt.tight_layout()
# plt.show()
# Save the imbie basin plot
output_path = os.path.join(path_to_plots, 'imbie_basins_with_ids.png')
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
gc.collect()

all_basin_ids = sorted({
    bid
    for _, ids in REGION_DEFS
    for bid in ids
})

basin_weights = compute_basin_area_weights_from_mask(
    basins,
    basin_ids=all_basin_ids
)


basin_weights_ = dict(zip(basin_weights["basin"].astype(int),
                         basin_weights["weight_global"].astype(float)))
#%%
Pmb_mm_fle = os.path.join(basins_path, 'Monthly_mass_budget_precip_RignotBasin_in_mm_20260226.nc')
# 'Monthly_mass_budget_precip_RignotBasin_in_mm.nc'

P_mm_mnth = xr.open_dataarray(Pmb_mm_fle)
p_mm_df = P_mm_mnth.to_dataframe().reset_index().dropna(axis=0)
p_mm_df = p_mm_df.dropna(axis=0, subset=["precip_mm_per_month"])
p_mm_df = p_mm_df[['date','basin_id','precip_mm_per_month']].copy()
p_mm_df['year'] = p_mm_df['date'].dt.year
p_mm_df['month'] = p_mm_df['date'].dt.month 
p_mm_mean_df = p_mm_df.groupby(['year','month','basin_id'])['precip_mm_per_month'].mean().reset_index()
p_mm_mean_df['time'] = pd.to_datetime(dict(year=p_mm_mean_df['year'], month=p_mm_mean_df['month'], day=1))
p_mm_mean_df['basin_id'] = p_mm_mean_df['basin_id'].astype(int)
p_mnth_mean_df = p_mm_mean_df[p_mm_mean_df['year'].isin(YEARS)].copy()
p_mnth_mean_df.rename(columns={'basin_id': 'basin',
                               'precip_mm_per_month': 'precipitation'}, 
                               inplace=True)
Pmb_annual = xr.open_dataarray(os.path.join(basins_path, "Pmb_annual_2013_2022_mm.nc"))
Pmb_seasonal = xr.open_dataarray(os.path.join(basins_path, "Pmb_seasonal_mm_2013_2022.nc"))
img_fle_lst = sorted([os.path.join(imerg_basin_path, x) for x in os.listdir(imerg_basin_path) if 'imbie_basin' in x])
era5_fle_lst = sorted([os.path.join(era5_basin_path, x) for x in os.listdir(era5_basin_path) if 'imbie_basin' in x])
gpcpv3pt3_fle_lst = sorted([os.path.join(gpcpv3pt3_basin_path, x) for x in os.listdir(gpcpv3pt3_basin_path) if 'imbie_basin' in x])
racmo_pr = xr.open_dataarray(os.path.join(racmo_path,'pr_monthlyS_ANT11_RACMO2.4p1_ERA5_2013_2022.nc'))

atms_fle_lst = sorted([os.path.join(atms_basin_path, x) for x in os.listdir(atms_basin_path) if x.endswith('.nc')])
dmsp_ssmis_fle_lst = sorted([os.path.join(dmsp_ssmis_basin_path, x) for x in os.listdir(dmsp_ssmis_basin_path) if x.endswith('.nc')])
amsr2_fle_lst = sorted([os.path.join(amsr2_basin_path, x) for x in os.listdir(amsr2_basin_path) if x.endswith('.nc')])
mhs_fle_lst = sorted([os.path.join(mhs_basin_path, x) for x in os.listdir(mhs_basin_path) if x.endswith('.nc')])
all_gpm_fle = os.path.join(all_gpm_basin_path, "all_gpm_satellite_precip_mean_20260304.nc")

# 1) drop the dummy band dimension
racmo_pr = racmo_pr.squeeze("band", drop=True)          # now (time, y, x)

# 2) remove non-essential coordinates that are hitchhiking on the array
#    (keep x,y,time; drop the CRS helpers and the basin mask)
racmo_pr = racmo_pr.reset_coords(names=["mapping", "basin_id"], drop=True)

# 3) tidy attributes / name
racmo_pr = racmo_pr.assign_attrs({
    "standard_name": "precipitation_flux",
    "long_name": "Precipitation",
    "units": "kg m-2"     # 1 kg m-2 == 1 mm w.e.
}).rename("precipitation")

gc.collect()
#%%
# read and process satellite precipitation data

# read and process era5 data
print('Processing ERA5 data')

era5_annual_mean, era5_seasonal_mean,era5_b_mean = process_precipitation_data(era5_fle_lst, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)
era5_annual_mean = era5_annual_mean * 365

era5_basin_mean = era5_b_mean.to_dataframe().reset_index()
era5_basin_mean['year'] = era5_basin_mean['time'].dt.year
era5_basin_mean['month'] = era5_basin_mean['time'].dt.month

era5_basin_mnth_mean = era5_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
# era5_basin_mnth_mean = era5_basin_mean.groupby(['year','month','basin'])['precipitation'].mean().reset_index()

era5_basin_mnth_mean['time'] = pd.to_datetime(dict(year=era5_basin_mnth_mean['year'], month=era5_basin_mnth_mean['month'], day=1))
era5_mnth_mean = era5_basin_mnth_mean[era5_basin_mnth_mean['year'].isin(YEARS)].copy()
# era5_mnth_mean['precipitation'] = era5_mnth_mean['precipitation'] * 30
gc.collect()


#----------------------------------------------------------------------------------

print('Processing GPCP v3.3 data')

gpcpv3pt3_annual_mean, gpcpv3pt3_seasonal_mean, gpcpv3pt3_b_mean = process_precipitation_data(gpcpv3pt3_fle_lst, 
                                                                            basins,
                                                                            'precipitation',
                                                                            False,)
gpcpv3pt3_annual_mean = gpcpv3pt3_annual_mean * 365

gpcpv3pt3_basin_mean = gpcpv3pt3_b_mean.to_dataframe().reset_index()
gpcpv3pt3_basin_mean['year'] = gpcpv3pt3_basin_mean['time'].dt.year
gpcpv3pt3_basin_mean['month'] = gpcpv3pt3_basin_mean['time'].dt.month

gpcpv3pt3_basin_mnth_mean = gpcpv3pt3_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
gpcpv3pt3_basin_mnth_mean['time'] = pd.to_datetime(dict(year=gpcpv3pt3_basin_mnth_mean['year'], month=gpcpv3pt3_basin_mnth_mean['month'], day=1))
gpcpv3pt3_mnth_mean = gpcpv3pt3_basin_mnth_mean[gpcpv3pt3_basin_mnth_mean['year'].isin(YEARS)].copy()

gc.collect()

#----------------------------------------------------------------------------------
print('processing RACMO pr data')
racmo_pr_annual_mean, racmo_pr_seasonal_mean, racmo_pr_b_mean = process_precipitation_data(racmo_pr, 
                                                                          basins, 
                                                                          'pr',
                                                                          True,)
racmo_basin_mnth_mean = racmo_pr_b_mean.to_dataframe().reset_index()
racmo_basin_mnth_mean['year'] = racmo_basin_mnth_mean['time'].dt.year
racmo_basin_mnth_mean['month'] = racmo_basin_mnth_mean['time'].dt.month
racmo_basin_mnth_mean = racmo_basin_mnth_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
racmo_basin_mnth_mean['time'] = pd.to_datetime(dict(year=racmo_basin_mnth_mean['year'], month=racmo_basin_mnth_mean['month'], day=1))
racmo_mnth_mean = racmo_basin_mnth_mean[racmo_basin_mnth_mean['year'].isin(YEARS)].copy()
# racmo_pr_annual_mean = racmo_pr_annual_mean * 365
gc.collect()

#----------------------------------------------------------------------------------

print('Processing GPM Constellation data')
print('processing ATMS data')
atms_annual_mean, atms_seasonal_mean,atms_b_mean = process_precipitation_data(atms_fle_lst, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)
atms_annual_mean = (atms_annual_mean * 24) * 365

atms_basin_mean = atms_b_mean.to_dataframe().reset_index()
atms_basin_mean['year'] = atms_basin_mean['time'].dt.year
atms_basin_mean['month'] = atms_basin_mean['time'].dt.month

atms_basin_mnth_mean = atms_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
atms_basin_mnth_mean['time'] = pd.to_datetime(dict(year=atms_basin_mnth_mean['year'], month=atms_basin_mnth_mean['month'], day=1))
atms_mnth_mean = atms_basin_mnth_mean[atms_basin_mnth_mean['year'].isin(YEARS)].copy()
gc.collect()

print('Processing MHS data')
mhs_annual_mean, mhs_seasonal_mean,mhs_b_mean = process_precipitation_data(mhs_fle_lst, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)
mhs_annual_mean = (mhs_annual_mean * 24) * 365

mhs_basin_mean = mhs_b_mean.to_dataframe().reset_index()
mhs_basin_mean['year'] = mhs_basin_mean['time'].dt.year
mhs_basin_mean['month'] = mhs_basin_mean['time'].dt.month

mhs_basin_mnth_mean = mhs_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
mhs_basin_mnth_mean['time'] = pd.to_datetime(dict(year=mhs_basin_mnth_mean['year'], month=mhs_basin_mnth_mean['month'], day=1))
mhs_mnth_mean = mhs_basin_mnth_mean[mhs_basin_mnth_mean['year'].isin(YEARS)].copy()
gc.collect()

print('Processing DMSP-SSMIS data')
dmsp_ssmis_annual_mean, dmsp_ssmis_seasonal_mean,dmsp_ssmis_b_mean = process_precipitation_data(dmsp_ssmis_fle_lst, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)
dmsp_ssmis_annual_mean = (dmsp_ssmis_annual_mean * 24) * 365

dmsp_ssmis_basin_mean = dmsp_ssmis_b_mean.to_dataframe().reset_index()
dmsp_ssmis_basin_mean['year'] = dmsp_ssmis_basin_mean['time'].dt.year
dmsp_ssmis_basin_mean['month'] = dmsp_ssmis_basin_mean['time'].dt.month

dmsp_ssmis_basin_mnth_mean = dmsp_ssmis_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
dmsp_ssmis_basin_mnth_mean['time'] = pd.to_datetime(dict(year=dmsp_ssmis_basin_mnth_mean['year'], month=dmsp_ssmis_basin_mnth_mean['month'], day=1))
dmsp_ssmis_mnth_mean = dmsp_ssmis_basin_mnth_mean[dmsp_ssmis_basin_mnth_mean['year'].isin(YEARS)].copy()
gc.collect()

print('Processing AMSR2 data')
amsr2_annual_mean, amsr2_seasonal_mean,amsr2_b_mean = process_precipitation_data(amsr2_fle_lst, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)
amsr2_annual_mean = (amsr2_annual_mean * 24) * 365

amsr2_basin_mean = amsr2_b_mean.to_dataframe().reset_index()
amsr2_basin_mean['year'] = amsr2_basin_mean['time'].dt.year
amsr2_basin_mean['month'] = amsr2_basin_mean['time'].dt.month

amsr2_basin_mnth_mean = amsr2_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
amsr2_basin_mnth_mean['time'] = pd.to_datetime(dict(year=amsr2_basin_mnth_mean['year'], month=amsr2_basin_mnth_mean['month'], day=1))
amsr2_mnth_mean = amsr2_basin_mnth_mean[amsr2_basin_mnth_mean['year'].isin(YEARS)].copy()
gc.collect()

print('Processing Mean GPM Satellite Data')
gpm_sat_annual_mean, gpm_sat_seasonal_mean,gpm_sat_b_mean = process_precipitation_data(all_gpm_fle, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)

gpm_sat_annual_mean = (gpm_sat_annual_mean * 24) * 365

gpm_sat_basin_mean = gpm_sat_b_mean.to_dataframe().reset_index()
gpm_sat_basin_mean['year'] = gpm_sat_basin_mean['time'].dt.year
gpm_sat_basin_mean['month'] = gpm_sat_basin_mean['time'].dt.month

gpm_sat_basin_mnth_mean = gpm_sat_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
gpm_sat_basin_mnth_mean['time'] = pd.to_datetime(dict(year=gpm_sat_basin_mnth_mean['year'], month=gpm_sat_basin_mnth_mean['month'], day=1))
gpm_sat_mnth_mean = gpm_sat_basin_mnth_mean[gpm_sat_basin_mnth_mean['year'].isin(YEARS)].copy()
gc.collect()


#----------------------------------------------------------------------------------
Pmb_annual_mean = Pmb_annual.sel(year=slice(2013,2020)).copy()
Pmb_annual_mean = Pmb_annual_mean.mean(dim='year')

era5_annual_mean_mean = era5_annual_mean.sel(year=slice(2013,2020)).copy()
era5_annual_mean_mean = era5_annual_mean_mean.mean(dim='year')

gpcpv3pt3_annual_mean_mean = gpcpv3pt3_annual_mean.sel(year=slice(2013,2020)).copy()
gpcpv3pt3_annual_mean_mean = gpcpv3pt3_annual_mean_mean.mean(dim='year')

racmo_pr_annual_mean_mean = racmo_pr_annual_mean.sel(year=slice(2013,2020)).copy()
racmo_pr_annual_mean_mean = racmo_pr_annual_mean_mean.mean(dim='year')

atms_annual_mean_mean = atms_annual_mean.sel(year=slice(2013,2020)).copy()
atms_annual_mean_mean = atms_annual_mean_mean.mean(dim='year')

mhs_annual_mean_mean = mhs_annual_mean.sel(year=slice(2013,2020)).copy()
mhs_annual_mean_mean = mhs_annual_mean_mean.mean(dim='year')

dmsp_ssmis_annual_mean_mean = dmsp_ssmis_annual_mean.sel(year=slice(2013,2020)).copy()
dmsp_ssmis_annual_mean_mean = dmsp_ssmis_annual_mean_mean.mean(dim='year')

amsr2_annual_mean_mean = amsr2_annual_mean.sel(year=slice(2013,2020)).copy()
amsr2_annual_mean_mean = amsr2_annual_mean_mean.mean(dim='year')

gpm_sat_annual_mean_mean = gpm_sat_annual_mean.sel(year=slice(2013,2020)).copy()
gpm_sat_annual_mean_mean = gpm_sat_annual_mean_mean.mean(dim='year')

monthly_df_data_mmmonth = {
    r"$P_{\mathrm{MB}}$": p_mnth_mean_df,
    "ERA5": era5_mnth_mean,
    "GPCP v3.3": gpcpv3pt3_mnth_mean,
    "ATMS": atms_mnth_mean,
    "MHS": mhs_mnth_mean,
    "DMSP SSMIS": dmsp_ssmis_mnth_mean,
    "AMSR2": amsr2_mnth_mean,
    "GPM PMW V07": gpm_sat_mnth_mean,
}

# ============================================================
# STEP 1: build raw monthly regional series once per region
# ============================================================

ts_eais_raw = build_region_monthly_series(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="East Antarctica",
)

ts_wais_raw = build_region_monthly_series(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="West Antarctica",
)

ts_ais_raw = build_region_monthly_series(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="Antarctica",
)

#%% Annual Means - Spatial
# calculate and plot the mean across years
mean_annual_plot_arrs = [
                    (r"P$_{MB}$", Pmb_annual_mean),
                    (f'ERA5', era5_annual_mean_mean), 
                    (f'GPCP V3.3', gpcpv3pt3_annual_mean_mean),
                    # (f'RACMO 2.4p1', racmo_pr_annual_mean_mean),
                    (f'ATMS', atms_annual_mean_mean),
                    (f'MHS', mhs_annual_mean_mean),
                    (f'DMSP-SSMIS', dmsp_ssmis_annual_mean_mean),
                    (f'AMSR2', amsr2_annual_mean_mean),                    
                   (f'GPM PMW V07', gpm_sat_annual_mean_mean),
                   ]


fig, axes = compare_mean_precip_grid_power(
    mean_annual_plot_arrs,
    ncols=4,
    gamma=0.6,
    vmin=0,
    vmax=400,
    cbar_tcks=[0, 25, 50, 100,  200, 300, 400],
    cbar_label="Precipitation [mm/year]",
    panel_letters=True,
)
svnme = f'annual_snowfall_accumulation_over_imbie_basins_{cde_run_dte}_sharedplot.png'
# save plot to disk
svnme = os.path.join(path_to_plots, svnme)
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()

#----------------------------------------------------------------------------------

fig, axes, cb1, cb2 = compare_mean_precip_grid_power_dual_cbar(
    mean_annual_plot_arrs,
    group1_idx=[0, 1, 2,],
    group2_idx=[3, 4, 5, 6, 7],
    cbar_tcks1=[0, 25, 50, 100, 200, 300, 400],
    cbar_tcks2=[0, 5, 10, 20, 40, 60, 80],
    ncols=4,
    panel_letters=True,    
    mean_fmt="Mean: {:d}"
)

svnme = os.path.join(path_to_plots, 
                     f'annual_snowfall_accumulation_over_imbie_basins_log_{cde_run_dte}_diff_cbar.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()


#%% COMPUTE AND PLOT AREA WEIGHTED MEAN MONTHLY CYCLES
# monthly_df_data = {
#     r"$P_{\mathrm{MB}}$": p_mm_mean_df.rename(columns={"basin_id": "basin",
#                                                        "precip_mm_per_month": "precipitation"}),
#     "ERA5": era5_basin_mean,
#     "GPCP v3.3": gpcpv3pt3_basin_mean,
#     # "RACMO 2.4p1": racmo_basin_mnth_mean,
#     "ATMS": atms_basin_mean,
#     "MHS": mhs_basin_mean,
#     "DMSP SSMIS": dmsp_ssmis_basin_mean,
#     "AMSR2": amsr2_basin_mean,
#     "GPM PMW V07": gpm_sat_basin_mean,
# }

region_monthly_clim = compute_weighted_region_monthly_climatologies(
    monthly_df_data=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights,
    basin_col="basin",
    value_col="precipitation",
    time_col="time",
)

product_order = [
    r"$P_{\mathrm{MB}}$",
    "ERA5",
    "GPCP v3.3",
    # "RACMO 2.4p1",
    # "ATMS",
    # "MHS",
    # "DMSP SSMIS",
    # "AMSR2",
    "GPM PMW V07",
]

product_styles = {
    r"$P_{\mathrm{MB}}$": {"color": "k", "marker": "o", "lw": 3.5},
    "ERA5": {"color": "tab:blue", "marker": "s", "lw": 3.5},
    "GPCP v3.3": {"color": "tab:orange", "marker": "D", "lw": 3.5},
    # "RACMO 2.4p1": {"color": "tab:green", "marker": "^", "lw": 1.8},
    # "ATMS": {"lw": 1.5},
    # "MHS": {"lw": 1.5},
    # "DMSP SSMIS": {"lw": 1.5},
    # "AMSR2": {"lw": 1.5},
    "GPM PMW V07": {"lw": 3.5},
}

fig, axes = plot_weighted_region_monthly_climatology(
    region_monthly_clim,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=product_order,
    product_styles=product_styles_corr,
    ylabel="Precipitation [mm/month]",
    figsize=(10, 10),
)
#-----------------------------------------------------------------------------------

region_monthly_clim_corr, correction_factors = add_scalar_bias_corrected_products_to_region_clim(
    region_monthly_clim,
    reference_col=r"$P_{\mathrm{MB}}$",
    target_products=corr_targets,
    suffix=" (corr.)",
    clip_factor=None,   # or e.g. (0.25, 10.0)
)


fig, axes = plot_weighted_region_monthly_climatology(
    region_monthly_clim_corr,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=product_order_corr,
    product_styles=product_styles_corr,
    ylabel="[mm/month]",
    figsize=(11, 10),
)
# plt.show()
# Save the plot to disk
svnme = os.path.join(path_to_plots, f'basin_area_weighted_monthly_cycles_precip_over_imbie_basins_regions_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()

#%% SEASONAL CYCLES - Basin Area Weighted
region_seasonal_clim = compute_weighted_region_seasonal_climatologies(
    monthly_df_data=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights,
    basin_col="basin",
    value_col="precipitation",
    time_col="time",
    seasonal_mode="sum",   # gives mm/season
)

fig, axes = plot_weighted_region_seasonal_climatology(
    region_seasonal_clim,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=product_order,
    product_styles=product_styles,
    ylabel="[mm/season]",
    figsize=(10, 8),
)


#----------------------------------------------------------------------------------
region_seasonal_clim_corr, seasonal_corr_factors = add_scalar_bias_corrected_products_to_region_clim(
    region_seasonal_clim,
    reference_col=r"$P_{\mathrm{MB}}$",
    target_products=corr_targets,
    suffix=" (corr.)",
    clip_factor=None,   # or e.g. (0.25, 10.0) if you want to prevent extreme scaling
)

fig, axes = plot_weighted_region_seasonal_climatology(
    region_seasonal_clim_corr,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=product_order_corr,
    product_styles=product_styles_corr,
    ylabel="[mm/season]",
    figsize=(10, 9),
)

svnme = os.path.join(path_to_plots, f'basin_area_weighted_seasonal_cycles_precip_over_imbie_basins_regions_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')

gc.collect()

#%% Year to Year Variability - Basin Area Weighted
monthly_df_data_mmmonth = {
    r"$P_{\mathrm{MB}}$": p_mnth_mean_df,
    "ERA5": era5_mnth_mean,
    "GPCP v3.3": gpcpv3pt3_mnth_mean,
    # "ATMS": atms_mnth_mean,
    # "MHS": mhs_mnth_mean,
    # "DMSP SSMIS": dmsp_ssmis_mnth_mean,
    # "AMSR2": amsr2_mnth_mean,
    # "GPM PMW V07": gpm_sat_mnth_mean,
}
region_annual = compute_weighted_region_annual_totals(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights,
    annual_mode="sum",  # mm/year
)

fig, axes = plot_weighted_region_interannual(
    region_annual,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=product_order_corr,
    product_styles=product_styles_corr,
    ylabel="[mm/year]",
    figsize=(11, 10),
)

#----------------------------------------------------------------------------------
region_annual_corr, annual_corr_factors = add_scalar_bias_corrected_products_to_region_annual(
    region_annual,
    reference_col=r"$P_{\mathrm{MB}}$",
    target_products=corr_targets,
    suffix=" (corr.)",
    clip_factor=None,   # or e.g. (0.25, 10.0)
)

fig, axes = plot_weighted_region_interannual(
    region_annual_corr,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=product_order_corr,
    product_styles=product_styles_corr,
    ylabel="[mm/year]",
    figsize=(11, 10),
)
svnme = os.path.join(path_to_plots, f'basin_area_weighted_year_to_year_variability_precip_over_imbie_basins_regions_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()
# For the continent we’ll just use “all basins we see in the dataframe”
#%% BAR CHART WITH SPREAD

# --- Convert each product to tidy DF ---
df_pmb   = to_df(Pmb_annual.sel(year=slice(2013,2020)).copy())
df_pmb.rename(columns={'basin_id': 'basin',
                       'precip_mm_per_month': 'Pmb'}, inplace=True)
df_era5  = to_df(era5_annual_mean.sel(year=slice(2013,2020)).copy())
df_era5.rename(columns={'precipitation_annual': 'ERA5'}, inplace=True)
df_gpcp  = to_df(gpcpv3pt3_annual_mean.sel(year=slice(2013,2020)).copy())
df_gpcp.rename(columns={'precipitation_annual': 'GPCP v3.3'}, inplace=True)
# df_racmo = to_df(racmo_pr_annual_mean)
# df_racmo.rename(columns={'pr_annual': 'RACMO'}, inplace=True)
df_atms = to_df(atms_annual_mean.sel(year=slice(2013,2020)).copy())
df_atms.rename(columns={'precipitation_annual': 'ATMS'}, inplace=True)
df_mhs = to_df(mhs_annual_mean.sel(year=slice(2013,2020)).copy())
df_mhs.rename(columns={'precipitation_annual': 'MHS'}, inplace=True)
df_dmsp_ssmis = to_df(dmsp_ssmis_annual_mean.sel(year=slice(2013,2020)).copy())
df_dmsp_ssmis.rename(columns={'precipitation_annual': 'DMSP SSMIS'}, inplace=True)
df_amsr2 = to_df(amsr2_annual_mean.sel(year=slice(2013,2020)).copy())
df_amsr2.rename(columns={'precipitation_annual': 'AMSR2'}, inplace=True)

df_gpm_sat = to_df(gpm_sat_annual_mean.sel(year=slice(2013,2020)).copy())
df_gpm_sat.rename(columns={'precipitation_annual': 'GPM PMW V07'}, inplace=True)

# --- Merge all together on (year, basin) ---
df = df_pmb.merge(df_era5, on=["year","basin"])
df = df.merge(df_gpcp, on=["year","basin"])
# df = df.merge(df_racmo, on=["year","basin"])
df = df.merge(df_atms, on=["year","basin"])
df = df.merge(df_mhs, on=["year","basin"])
df = df.merge(df_dmsp_ssmis, on=["year","basin"])
df = df.merge(df_amsr2, on=["year","basin"])
df = df.merge(df_gpm_sat, on=["year","basin"])

# add a "year-basin" key if you like
# "ATMS", "MHS", "DMSP SSMIS", "AMSR2",
df["year_basin"] = df["year"].astype(str) + "-" + df["basin"].astype(str)
cols = ["Pmb", "ERA5", "GPCP v3.3",  "GPM PMW V07"]
df_mean_yr_acc = df.groupby("basin")[cols].mean().reset_index()

fig, ax = plot_basin_ranked_bar_overlay(
    df_mean_yr_acc,
    basin_col="basin",
    ref_col="Pmb",
    prod_cols=cols,
    prod_labels=cols,
    figsize=(12, 5),
)
plt.show()

#----------------------------------------------------------------------------------
# cols = ["ERA5", "GPCP v3.3", "ATMS", "MHS", "DMSP SSMIS", "AMSR2", "GPM PMW V07"]
cols = ["ERA5", "GPCP v3.3", "GPM PMW V07"]

non_gpm_group = ["Pmb", "ERA5", "GPCP v3.3"]
gpm_group     = ["Pmb", "GPM PMW V07"]
#  "ATMS", "MHS", "DMSP SSMIS", "AMSR2"]  # exclude  from spread

fig, ax, spread_non_gpm, spread_gpm = plot_basin_spread_points_dual(
    df_mean_yr_acc,
    basin_col="basin",
    ref_col="Pmb",
    prod_cols=cols,
    prod_labels=cols,                 # or omit; it will auto-use col names
    product_styles=product_styles_corr,
    non_gpm_group=non_gpm_group,
    gpm_group=gpm_group,
    log_scale=True,
    ylim=(2, 2000),
    annotate_non_gpm_color="black",
    annotate_gpm_color="dimgray",
    place_key=True,
)
svnme = os.path.join(path_to_plots, f'basin_spread_points_precip_over_imbie_basins_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()

#%%  ----------- Scatter plot -------------

# Example usage:
products = ["ERA5", "GPCP v3.3",  "GPM PMW V07"]
# "ATMS", "MHS", "DMSP SSMIS", "AMSR2",
fig, axes = plot_pmb_scatter(
    df_mean_yr_acc,
    "Pmb",
    products,
    high_thresh=500.0,
    scale="log",
    log_min=2,                   # <-- show low GPM values
    ncols=3                     # 2 rows for 7 products
)

svnme = os.path.join(path_to_plots, f'log_scatterplot_precip_over_imbie_basins_{cde_run_dte}.png')
plt.savefig(svnme, dpi=500, bbox_inches="tight")
# plt.close(fig)
gc.collect()


#%% Anomal Time Series

# APPROACH A: conventional seasonal anomaly
# ============================================================

# EAIS
ts_eais_seasonal_conv = build_conventional_seasonal_series_from_region_monthly(
    ts_region_monthly=ts_eais_raw,
    seasonal_mode="mean",
    drop_incomplete=True,
)

ts_eais_seasonal_conv_anom = deseasonalize_seasonal_series(ts_eais_seasonal_conv)

#-------------------------------------------------------------------------------------------------------
# WAIS
ts_wais_seasonal_conv = build_conventional_seasonal_series_from_region_monthly(
    ts_region_monthly=ts_wais_raw,
    seasonal_mode="mean",
    drop_incomplete=True,
)

ts_wais_seasonal_conv_anom = deseasonalize_seasonal_series(ts_wais_seasonal_conv)
#-------------------------------------------------------------------------------------------------------
# AIS
ts_ais_seasonal_conv = build_conventional_seasonal_series_from_region_monthly(
    ts_region_monthly=ts_ais_raw,
    seasonal_mode="mean",
    drop_incomplete=True,
)
ts_ais_seasonal_conv_anom = deseasonalize_seasonal_series(ts_ais_seasonal_conv)

# the plots
ts_seasonal_dict = {
    "Antarctica": ts_ais_seasonal_conv,
    "West Antarctica": ts_wais_seasonal_conv,
    "East Antarctica": ts_eais_seasonal_conv,
}

ts_seasonal_anom_dict = {
    "Antarctica": ts_ais_seasonal_conv_anom,
    "West Antarctica": ts_wais_seasonal_conv_anom,
    "East Antarctica": ts_eais_seasonal_conv_anom,
}

# 3by1 anomaly ts
fig_ts, axes_ts, anom_dict_used = plot_seasonal_anomaly_timeseries_regions_3x1(
    ts_seasonal_dict=ts_seasonal_dict,
    method="conventional",
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    ref_col=r"$P_{\mathrm{MB}}$",
    target_cols=("ERA5", "GPCP v3.3", "GPM PMW V07"),
    product_styles=product_styles_corr,
    figsize=(10, 10),
    y_nbins=3,
    ylabel="[mm/season]",
    legend_ncol=4,
)

# 3by3 anomaly scatter
lims_by_region = {
    "Antarctica": (-3, 3),
    "West Antarctica": (-10, 10),
    "East Antarctica": (-3, 3),
}

fig_sc, axes_sc, stats_sc = plot_seasonal_anomaly_scatter_regions_3x3(
    region_anom_dict=ts_seasonal_anom_dict,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    ref_col=r"$P_{\mathrm{MB}}$",
    target_cols=("ERA5", "GPCP v3.3", "GPM PMW V07"),
    figsize=(13.5, 12),
    share_lims=False,
    lims=lims_by_region,
    equal_axes=True,
    point_size=70,
)

stats_table_wide = seasonal_scatter_stats_to_wide_table(
    stats_out=stats_sc,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    target_cols=("ERA5", "GPCP v3.3", "GPM PMW V07"),
    ref_col=r"$P_{\mathrm{MB}}$",
)

print(stats_table_wide)
#%% TREND ANALYSIS
# if basin_weights is a pandas Series with basin IDs as index:


region_ts_wais = region_monthly_series_from_dict(
    monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="West Antarctica",
)

region_ts_eais = region_monthly_series_from_dict(
    monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="East Antarctica",
)

region_ts_ais = region_monthly_series_from_dict(
    monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="Antarctica",
)

product_order = [
    r"$P_{\mathrm{MB}}$",
    "ERA5",
    "GPCP v3.3",
    "ATMS",
    "MHS",
    "DMSP SSMIS",
    "AMSR2",
    "GPM PMW V07",
]

fig, axes = plot_region_trend_panels(
    monthly_df_data_mmmonth,
    REGION_DEFS,
    basin_weights_,
    product_order=product_order,
    product_styles=product_styles_corr,
    use_running_mean=True,   # 13-mo RM for clean plot
    show_pmb_trend_only=True
)

#%% A DIAGNOSIS OF EAIS PRECIP
from Extra_util_functions import *
REGION_DEFS_EAIS = [('Interior EAIS', [8, 9, 18, 19]),
                    ('Coastal EAIS', [2, 3, 4, 5, 6, 7])]

monthly_df_data_mmmonth_fixed = {
    name: ensure_monthly_basin_totals(df)
    for name, df in monthly_df_data_mmmonth.items()
}

region_annual = compute_weighted_region_annual_totals(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS_EAIS,
    basin_weights=basin_weights,
    annual_mode="sum",  # mm/year
)

fig, axes = plot_weighted_region_interannual(
    region_annual,
    region_order=("Interior EAIS", "Coastal EAIS"),
    product_order=product_order_corr,
    product_styles=product_styles_corr,
    ylabel="[mm/year]",
    figsize=(11, 10),
)

#----------------------------------------------------------------------------------
region_annual_corr, annual_corr_factors = add_scalar_bias_corrected_products_to_region_annual(
    region_annual,
    reference_col=r"$P_{\mathrm{MB}}$",
    target_products=corr_targets,
    suffix=" (corr.)",
    clip_factor=None,   # or e.g. (0.25, 10.0)
)

fig, axes = plot_weighted_region_interannual(
    region_annual_corr,
    region_order=("Interior EAIS", "Coastal EAIS"),
    product_order=product_order_corr,
    product_styles=product_styles_corr,
    ylabel="[mm/year]",
    figsize=(11, 10),
)

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------basin by basin annual plot------------------------------------------------
basins_eais = [2, 3, 4, 5, 6, 7, 8, 9, 18, 19]
products_to_plot = [r"$P_{\mathrm{MB}}$", "ERA5", "GPCP v3.3", "GPM PMW V07"]

fig, axes = plot_eais_basin_interannual(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    basin_list=basins_eais,
    products_to_plot=products_to_plot,
    basin_weights=basin_weights,   # optional
    ncols=2,
    figsize=(14, 20),
    ylabel="[mm/year]",
)
plt.show()


#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
bias_tab = basin_mean_bias_table(
    monthly_df_data_mmmonth,
    ref_name=r"$P_{\mathrm{MB}}$",
    test_name="ERA5",
    basin_list=basins_eais
)

print(bias_tab)

interior_basins = [8, 9, 18, 19]
coastal_basins = [2, 3, 4, 5, 6, 7]

fig, ax = plot_basin_bias_ranked(
    bias_tab,
    interior_basins=interior_basins,
    coastal_basins=coastal_basins,
    figsize=(8, 6)
)
plt.show()

eais_basins = [2, 3, 4, 5, 6, 7, 8, 9, 18, 19]

eais_weights = (
    basin_weights[basin_weights["basin"].isin(eais_basins)][["basin", "area_km2"]]
    .copy()
)

eais_weights["weight_eais"] = eais_weights["area_km2"] / eais_weights["area_km2"].sum()

bias_tab_w = bias_tab.merge(
    eais_weights[["basin", "weight_eais"]],
    on="basin",
    how="left"
)

bias_tab_w["weighted_contribution"] = (
    bias_tab_w["mean_bias"] * bias_tab_w["weight_eais"]
)

bias_tab_w = bias_tab_w.sort_values("weighted_contribution", ascending=False)
bias_tab_w

fig, ax = plot_basin_bias_ranked(
    bias_tab_w,
    interior_basins=[8, 9, 18, 19],
    coastal_basins=[2, 3, 4, 5, 6, 7],
    figsize=(8, 6),
    title="Weighted basin contribution to EAIS ERA5 bias",
    xlabel="Weighted contribution [mm/year]",
)
plt.show()


fig, ax = plot_basin_metric_ranked(
    bias_tab_w,
    metric_col="mean_bias",
    interior_basins=[8, 9, 18, 19],
    coastal_basins=[2, 3, 4, 5, 6, 7],
    title="ERA5 - P_MB mean annual bias by basin",
    xlabel="Bias [mm/year]",
)
plt.show()
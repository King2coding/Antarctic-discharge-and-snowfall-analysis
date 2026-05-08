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

# Original / previous PMB used in old slides
Pmb_mm_fle_old = os.path.join(
    basins_path,
    "Monthly_mass_budget_precip_RignotBasin_in_mm_20260226.nc"
)

# New PMB generated using GRACE-derived ΔS monthly climatology correction
Pmb_mm_fle_corr = os.path.join(
    basins_path,
    "Monthly_mass_budget_precip_RignotBasin_in_mm_forward_deltaS_Variant_B_negative_deltaS_only_20260507.nc"
)

PMB_VERSION_LABEL_OLD = r"$P_{\mathrm{MB}}$ old"
PMB_VERSION_LABEL_CORR = r"$P_{\mathrm{MB}}$ $\Delta S$-corr."

# --- File lists: 2013–2020 only ---
all_gpcp_v3pt3_mnthly_files = sorted(
    [os.path.join(gpcp_v3pt3_mnthly_ds_path, f) for f in os.listdir(gpcp_v3pt3_mnthly_ds_path) if f.endswith('.nc4')]
    )

all_gpcp_v3pt3_mnthly_files_2013_2020 = [
    f for f in all_gpcp_v3pt3_mnthly_files
    if 2013 <= int(os.path.basename(f).split('_')[2][:4]) <= 2020
]

ANNUAL_YEAR_START = 2014
ANNUAL_YEAR_END = 2020
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

print("Loading PMB monthly datasets ...")

P_mm_mnth_old = xr.open_dataarray(Pmb_mm_fle_old)
P_mm_mnth_corr = xr.open_dataarray(Pmb_mm_fle_corr)


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
print("Reprojecting old PMB monthly to common 0.1° grid ...")
pmb_mon_01_old = prepare_pmb_monthly_on_target(
    P_mm_mnth_old,
    target_template_01deg
)
pmb_mon_01_old = subset_common_period(pmb_mon_01_old)

print("Reprojecting corrected PMB monthly to common 0.1° grid ...")
pmb_mon_01_corr = prepare_pmb_monthly_on_target(
    P_mm_mnth_corr,
    target_template_01deg
)
pmb_mon_01_corr = subset_common_period(pmb_mon_01_corr)
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
pmb_mon_01_old  = pmb_mon_01_old.where(valid_basin_mask)
pmb_mon_01_corr = pmb_mon_01_corr.where(valid_basin_mask)

# Main PMB version used for product comparison
pmb_mon_01 = pmb_mon_01_corr.copy()

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

#%% PMB OLD VS ΔS-CORRECTED DIAGNOSTIC
# =============================================================================
# Purpose:
# Compare the old PMB and the new ΔS-corrected PMB before any negative-value
# masking/filtering is applied.
#
# This directly tests whether the GRACE-derived ΔS climatology correction reduced
# the problematic PMB months/seasons identified in the diagnostic slides:
#   - WAIS 2014 SON, especially Nov: G-H, Ep-F, I-Ipp
#   - EAIS 2017 DJF, especially Jan/Feb: Dp-E, E-Ep
# =============================================================================

BASIN_IDS = sorted(AIS_BASINS)
BASIN_ID_TO_NAME = {v: k for k, v in BASIN_NAME_TO_ID.items()}

pmb_old_basin_month_df = compute_basin_month_series_from_grid(
    da=pmb_mon_01_old,
    basin_mask=basin_mask_01deg,
    basin_ids=BASIN_IDS,
    value_name="pmb_old_mm_month",
    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

pmb_corr_basin_month_df = compute_basin_month_series_from_grid(
    da=pmb_mon_01_corr,
    basin_mask=basin_mask_01deg,
    basin_ids=BASIN_IDS,
    value_name="pmb_corr_mm_month",
    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

pmb_compare_basin_month_df = (
    pmb_old_basin_month_df
    .merge(
        pmb_corr_basin_month_df,
        on=["time", "year", "month", "basin"],
        how="outer"
    )
    .sort_values(["time", "basin"])
    .reset_index(drop=True)
)

pmb_compare_basin_month_df["basin_name"] = (
    pmb_compare_basin_month_df["basin"].map(BASIN_ID_TO_NAME)
)

pmb_compare_basin_month_df["pmb_change_mm_month"] = (
    pmb_compare_basin_month_df["pmb_corr_mm_month"] -
    pmb_compare_basin_month_df["pmb_old_mm_month"]
)

print("\nPMB old-vs-corrected basin-month comparison:")
print(pmb_compare_basin_month_df.head())

# =============================================================================
# Targeted PMB comparison for diagnostic-slide problem months
# =============================================================================
# Purpose:
# Extract the basin/months identified in the old diagnostic slides and compare
# old PMB versus ΔS-corrected PMB directly.
#
# Key targets:
#   - WAIS 2014 SON issue: November 2014, basins G-H, Ep-F, I-Ipp
#   - EAIS 2017 DJF issue: January/February 2017, basins Dp-E, E-Ep
# =============================================================================

TARGETED_PROBLEM_MONTHS_BY_NAME = [
    {"year": 2014, "month": 11, "basins": ["G-H", "Ep-F", "I-Ipp"]},
    {"year": 2017, "month": 1,  "basins": ["Dp-E", "E-Ep"]},
    {"year": 2017, "month": 2,  "basins": ["Dp-E", "E-Ep"]},
]

target_rows = []

for item in TARGETED_PROBLEM_MONTHS_BY_NAME:
    yy = int(item["year"])
    mm = int(item["month"])

    for basin_name in item["basins"]:

        if basin_name not in BASIN_NAME_TO_ID:
            raise ValueError(f"Basin name not found in BASIN_NAME_TO_ID: {basin_name}")

        basin_id = BASIN_NAME_TO_ID[basin_name]

        sub = pmb_compare_basin_month_df[
            (pmb_compare_basin_month_df["year"] == yy) &
            (pmb_compare_basin_month_df["month"] == mm) &
            (pmb_compare_basin_month_df["basin"] == basin_id)
        ].copy()

        if sub.empty:
            print(f"Warning: no PMB comparison row found for {yy}-{mm:02d}, {basin_name}")
            continue

        sub["target_basin_name"] = basin_name
        target_rows.append(sub)

targeted_pmb_compare = (
    pd.concat(target_rows, ignore_index=True)
    if target_rows else pd.DataFrame()
)

targeted_cols = [
    "time",
    "basin",
    "target_basin_name",
    "pmb_old_mm_month",
    "pmb_corr_mm_month",
    "pmb_change_mm_month",
]

print("\nTargeted old-vs-ΔS-corrected PMB comparison:")
print(targeted_pmb_compare[targeted_cols])

targeted_pmb_compare.to_csv(
    os.path.join(
        path_to_dfs,
        f"targeted_old_vs_deltaS_corrected_PMB_{cde_run_dte}.csv"
    ),
    index=False
)

# =============================================================================
# Regional seasonal comparison: old PMB vs ΔS-corrected PMB
# =============================================================================
# Purpose:
# Compare the affected seasons at regional scale before any masking:
#   - WAIS SON 2014
#   - EAIS DJF 2017
# =============================================================================

product_monthly_dict_pmb_compare = {
    PMB_VERSION_LABEL_OLD: pmb_mon_01_old,
    PMB_VERSION_LABEL_CORR: pmb_mon_01_corr,
}

regional_monthly_pmb_compare_df = build_all_region_monthly_series_cosine(
    product_dict=product_monthly_dict_pmb_compare,
    region_masks=region_masks_01deg,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

seasonal_pmb_compare_df = monthly_regional_df_to_conventional_seasonal(
    regional_monthly_pmb_compare_df,
    time_col="time",
    region_col="region",
    product_col="product",
    value_col="precipitation",
    require_complete_season=True,
)

pmb_old_corr_seasonal_wide = (
    seasonal_pmb_compare_df
    .pivot_table(
        index=["region", "season_year", "season", "time"],
        columns="product",
        values="precipitation"
    )
    .reset_index()
)

pmb_old_corr_seasonal_wide["corr_minus_old"] = (
    pmb_old_corr_seasonal_wide[PMB_VERSION_LABEL_CORR] -
    pmb_old_corr_seasonal_wide[PMB_VERSION_LABEL_OLD]
)

problem_season_compare = pmb_old_corr_seasonal_wide[
    (
        (pmb_old_corr_seasonal_wide["region"] == "West Antarctica") &
        (pmb_old_corr_seasonal_wide["season_year"] == 2014) &
        (pmb_old_corr_seasonal_wide["season"] == "SON")
    )
    |
    (
        (pmb_old_corr_seasonal_wide["region"] == "East Antarctica") &
        (pmb_old_corr_seasonal_wide["season_year"] == 2017) &
        (pmb_old_corr_seasonal_wide["season"] == "DJF")
    )
].copy()

print("\nProblem-season old-vs-ΔS-corrected PMB comparison:")
print(problem_season_compare)

problem_season_compare.to_csv(
    os.path.join(
        path_to_dfs,
        f"problem_season_old_vs_deltaS_corrected_PMB_{cde_run_dte}.csv"
    ),
    index=False
)
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

#%% OPTIONAL COMMON PMB-NEGATIVE BASIN/MONTH MASK
# =============================================================================
# Purpose:
# Old workflow masked extreme negative PMB basin-months from all products.
# New workflow first evaluates the ΔS-corrected PMB without this mask.
#
# Set APPLY_NEGATIVE_PMB_MASK_FOR_PRODUCT_COMPARISON = False for the main
# ΔS-corrected PMB evaluation.
# Set it to True only for a legacy/sensitivity test.
# =============================================================================

APPLY_NEGATIVE_PMB_MASK_FOR_PRODUCT_COMPARISON = False

BASIN_IDS = sorted(AIS_BASINS)

# Compute corrected PMB basin/month table for diagnostics either way
pmb_basin_month_df = compute_basin_month_series_from_grid(
    da=pmb_mon_01_corr,
    basin_mask=basin_mask_01deg,
    basin_ids=BASIN_IDS,
    value_name="pmb_mm_month",
    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

negative_pmb_basin_months = (
    pmb_basin_month_df[pmb_basin_month_df["pmb_mm_month"] < 0]
    .copy()
    .sort_values(["time", "basin"])
    .reset_index(drop=True)
)

print("Number of negative corrected-PMB basin/months:", len(negative_pmb_basin_months))
print(negative_pmb_basin_months.head(30))

extreme_negative_pmb_basin_months_region_specific = build_region_specific_negative_pmb_table(
    pmb_basin_month_df=pmb_basin_month_df,
    region_basins=REGION_BASINS,
    region_thresholds=REGION_NEG_THRESHOLDS,
    value_col="pmb_mm_month",
)

print(
    "Number of region-specific extreme negative corrected-PMB basin/months:",
    len(extreme_negative_pmb_basin_months_region_specific)
)

if len(extreme_negative_pmb_basin_months_region_specific) > 0:
    print(
        extreme_negative_pmb_basin_months_region_specific
        .groupby(["region", "threshold_mm_month"])
        .size()
        .reset_index(name="n_removed")
    )

print(extreme_negative_pmb_basin_months_region_specific.head(50))


# -----------------------------------------------------------------------------
# Main branch
# -----------------------------------------------------------------------------
if APPLY_NEGATIVE_PMB_MASK_FOR_PRODUCT_COMPARISON:

    print("\nApplying legacy/common PMB-negative mask to all products...")

    pmb_mon_01_filtered = mask_basin_months_from_negative_pmb_table(
        da=pmb_mon_01_corr,
        basin_mask=basin_mask_01deg,
        negative_table=extreme_negative_pmb_basin_months_region_specific,
        time_name="time",
    )

    era5_mnth_01_filtered = mask_basin_months_from_negative_pmb_table(
        da=era5_mnth_01,
        basin_mask=basin_mask_01deg,
        negative_table=extreme_negative_pmb_basin_months_region_specific,
        time_name="time",
    )

    gpcp_mon_01_filtered = mask_basin_months_from_negative_pmb_table(
        da=gpcp_mon_01,
        basin_mask=basin_mask_01deg,
        negative_table=extreme_negative_pmb_basin_months_region_specific,
        time_name="time",
    )

    gpm_family_monthly_dict_filtered = {}

    for name, da in gpm_family_monthly_dict.items():
        gpm_family_monthly_dict_filtered[name] = mask_basin_months_from_negative_pmb_table(
            da=da,
            basin_mask=basin_mask_01deg,
            negative_table=extreme_negative_pmb_basin_months_region_specific,
            time_name="time",
        )

    mask_tag = "with_negative_PMB_mask"

else:

    print("\nNegative-PMB mask is OFF. Using ΔS-corrected PMB directly.")

    pmb_mon_01_filtered = pmb_mon_01_corr.copy()
    era5_mnth_01_filtered = era5_mnth_01.copy()
    gpcp_mon_01_filtered = gpcp_mon_01.copy()

    gpm_family_monthly_dict_filtered = {
        name: da.copy() for name, da in gpm_family_monthly_dict.items()
    }

    mask_tag = "no_negative_PMB_mask"


gpm_pmw_v07_mon_01_filtered = build_gpm_pmw_mean(
    gpm_family_dict=gpm_family_monthly_dict_filtered,
    mean_name="GPM PMW V07"
)

# Summary table for optional mask diagnostic
if len(extreme_negative_pmb_basin_months_region_specific) > 0:
    region_specific_threshold_summary = (
        extreme_negative_pmb_basin_months_region_specific
        .groupby("region")
        .agg(
            n_removed=("pmb_mm_month", "size"),
            min_removed=("pmb_mm_month", "min"),
            max_removed=("pmb_mm_month", "max"),
            threshold=("threshold_mm_month", "first"),
        )
        .reset_index()
    )

    region_specific_threshold_summary["n_possible"] = region_specific_threshold_summary["region"].map(
        {
            region: len(basin_ids) * pmb_basin_month_df["time"].nunique()
            for region, basin_ids in REGION_BASINS.items()
        }
    )

    region_specific_threshold_summary["percent_removed"] = (
        100 * region_specific_threshold_summary["n_removed"] /
        region_specific_threshold_summary["n_possible"]
    )

    print(region_specific_threshold_summary)
else:
    region_specific_threshold_summary = pd.DataFrame()
    print("No extreme negative corrected-PMB basin/months found using current thresholds.")
#%%
# =============================================================================
# FORCE COMMON MONTHLY TIME AXIS FOR MAIN PRODUCT COMPARISON
# =============================================================================
# Corrected PMB starts in March 2013 because ΔS requires a previous monthly
# storage anomaly. Therefore, ERA5/GPCP/GPM must be restricted to the same
# PMB-valid months before monthly climatology, seasonal climatology, seasonal
# time series, seasonal anomalies, and annual regional series are built.
#
# This avoids Jan/Feb climatologies being based on 8 years for ERA5/GPCP/GPM
# but only 7 years for PMB.
# =============================================================================

common_time = pd.DatetimeIndex(pmb_mon_01_filtered["time"].values)

era5_mnth_01_filtered = era5_mnth_01_filtered.sel(time=common_time)
gpcp_mon_01_filtered = gpcp_mon_01_filtered.sel(time=common_time)
gpm_pmw_v07_mon_01_filtered = gpm_pmw_v07_mon_01_filtered.sel(time=common_time)

gpm_family_monthly_dict_filtered = {
    name: da.sel(time=common_time)
    for name, da in gpm_family_monthly_dict_filtered.items()
}

print("\n✅ Common monthly time axis enforced for main product comparison")
print("PMB :", str(pmb_mon_01_filtered.time.min().values), "->", str(pmb_mon_01_filtered.time.max().values), pmb_mon_01_filtered.sizes["time"])
print("ERA5:", str(era5_mnth_01_filtered.time.min().values), "->", str(era5_mnth_01_filtered.time.max().values), era5_mnth_01_filtered.sizes["time"])
print("GPCP:", str(gpcp_mon_01_filtered.time.min().values), "->", str(gpcp_mon_01_filtered.time.max().values), gpcp_mon_01_filtered.sizes["time"])
print("GPM :", str(gpm_pmw_v07_mon_01_filtered.time.min().values), "->", str(gpm_pmw_v07_mon_01_filtered.time.max().values), gpm_pmw_v07_mon_01_filtered.sizes["time"])
# =============================================================================
# SECTION 2B. BUILD MONTHLY REGIONAL SERIES FOR PMB, ERA5, GPCP
# =============================================================================

product_monthly_dict_filtered = {
    r"$P_{\mathrm{MB}}$": pmb_mon_01_filtered,
    "ERA5": era5_mnth_01_filtered,
    "GPCP V3.3": gpcp_mon_01_filtered,
    "GPM PMW V07": gpm_pmw_v07_mon_01_filtered,
}

regional_monthly_cos_df_filtered = build_all_region_monthly_series_cosine(
    product_dict=product_monthly_dict_filtered,
    region_masks=region_masks_01deg,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

# print(regional_monthly_cos_df_valid.head())
# print(regional_monthly_cos_df_valid.tail())
# print(regional_monthly_cos_df_valid.groupby(["region", "product"]).size())


#%% Monthly Climatology

region_monthly_clim_cos = compute_monthly_climatology_from_regional_series(
    regional_monthly_cos_df_filtered
)

region_monthly_wide_dict = regional_monthly_tidy_to_region_dict(
    regional_monthly_cos_df_filtered,
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
    regional_monthly_cos_df_filtered,
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

#%% YEAR BY YEAR SEASONAL TIMESERIES
# Convert monthly regional values to conventional seasonal means
seasonal_cos_df = monthly_regional_df_to_conventional_seasonal(
    regional_monthly_cos_df_filtered,
    time_col="time",
    region_col="region",
    product_col="product",
    value_col="precipitation",
    require_complete_season=True,
)

# 3. Plot seasonal time series for all regions
fig, axes = plot_seasonal_timeseries_regions(
    seasonal_cos_df,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(
        r"$P_{\mathrm{MB}}$",
        "ERA5",
        "GPCP V3.3",        
        "GPM PMW V07",
    ),
    product_styles=product_styles_corr,
    title_suffix="",
)


#%% Interannual Variability
region_annual_cos = compute_annual_totals_from_regional_series(
    regional_monthly_cos_df_filtered
)

# Drop incomplete 2013 annual totals because corrected PMB starts in March 2013.
region_annual_cos = region_annual_cos[
    (region_annual_cos["year"] >= ANNUAL_YEAR_START) &
    (region_annual_cos["year"] <= ANNUAL_YEAR_END)
].copy()

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
    regional_monthly_cos_df_filtered,
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

#%% ANNUAL TOTALS AND 2014–2020 MEAN ANNUAL FIELDS
# =============================================================================
# Purpose:
# Build annual-total fields and multi-year mean annual fields for basin-scale
# scatter/spread analysis.
#
# Important:
# The corrected PMB product starts in March 2013 because ΔS requires a valid
# previous monthly storage anomaly. Therefore, 2013 is incomplete for PMB.
#
# For annual-total diagnostics, use complete years only:
#     ANNUAL_YEAR_START = 2014
#     ANNUAL_YEAR_END   = 2020
# =============================================================================

annual_period_tag = f"{ANNUAL_YEAR_START}_{ANNUAL_YEAR_END}"

# -----------------------------------------------------------------------------
# 1. Build annual fields [mm/year]
# -----------------------------------------------------------------------------

pmb_annual_01 = monthly_to_annual_totals_field(
    pmb_mon_01_filtered
)

era5_annual_01 = monthly_to_annual_totals_field(
    era5_mnth_01_filtered
)

gpcp_annual_01 = monthly_to_annual_totals_field(
    gpcp_mon_01_filtered
)

gpm_pmw_v07_annual_01 = monthly_to_annual_totals_field(
    gpm_pmw_v07_mon_01_filtered
)

# -----------------------------------------------------------------------------
# 2. Restrict annual fields to complete annual-analysis period
# -----------------------------------------------------------------------------

pmb_annual_01 = pmb_annual_01.sel(
    year=slice(ANNUAL_YEAR_START, ANNUAL_YEAR_END)
)

era5_annual_01 = era5_annual_01.sel(
    year=slice(ANNUAL_YEAR_START, ANNUAL_YEAR_END)
)

gpcp_annual_01 = gpcp_annual_01.sel(
    year=slice(ANNUAL_YEAR_START, ANNUAL_YEAR_END)
)

gpm_pmw_v07_annual_01 = gpm_pmw_v07_annual_01.sel(
    year=slice(ANNUAL_YEAR_START, ANNUAL_YEAR_END)
)

# -----------------------------------------------------------------------------
# 3. Build multi-year mean annual fields [mm/year]
# -----------------------------------------------------------------------------

pmb_annual_mean_01 = annual_to_multiyear_mean_field(
    pmb_annual_01,
    ANNUAL_YEAR_START,
    ANNUAL_YEAR_END,
)

era5_annual_mean_01 = annual_to_multiyear_mean_field(
    era5_annual_01,
    ANNUAL_YEAR_START,
    ANNUAL_YEAR_END,
)

gpcp_annual_mean_01 = annual_to_multiyear_mean_field(
    gpcp_annual_01,
    ANNUAL_YEAR_START,
    ANNUAL_YEAR_END,
)

gpm_pmw_v07_annual_mean_01 = annual_to_multiyear_mean_field(
    gpm_pmw_v07_annual_01,
    ANNUAL_YEAR_START,
    ANNUAL_YEAR_END,
)

print(
    f"Annual mean fields built for {ANNUAL_YEAR_START}–{ANNUAL_YEAR_END}"
)
print("PMB :", pmb_annual_mean_01.shape)
print("ERA5:", era5_annual_mean_01.shape)
print("GPCP:", gpcp_annual_mean_01.shape)
print("GPM :", gpm_pmw_v07_annual_mean_01.shape)


# =============================================================================
# SECTION 10.3. BUILD BASIN MULTI-YEAR MEAN ANNUAL DATAFRAME
# =============================================================================

BASIN_IDS = sorted(AIS_BASINS)

basin_mask_01deg_clean = basin_mask_01deg.where(
    basin_mask_01deg.isin(BASIN_IDS)
)

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
    da_2d=gpm_pmw_v07_annual_mean_01,
    basin_mask_2d=basin_mask_01deg_clean,
    basin_ids=BASIN_IDS,
    value_name="GPM PMW V07",
)

# -----------------------------------------------------------------------------
# Merge basin mean annual dataframe
# -----------------------------------------------------------------------------

df_basin_mean_annual = (
    df_pmb_basin
    .merge(df_era5_basin, on="basin", how="outer")
    .merge(df_gpcp_basin, on="basin", how="outer")
    .merge(df_gpm_pmw_v07, on="basin", how="outer")
    .sort_values("basin")
    .reset_index(drop=True)
)

print("\nBasin mean annual dataframe:")
print(df_basin_mean_annual)

# Optional: save basin mean annual dataframe
df_basin_mean_annual.to_csv(
    os.path.join(
        path_to_dfs,
        f"basin_mean_annual_precip_{annual_period_tag}_{mask_tag}_{cde_run_dte}.csv"
    ),
    index=False,
)


# =============================================================================
# SECTION 10.4. BASIN MEAN ANNUAL SCATTERPLOTS
# =============================================================================

svnme = os.path.join(
    path_to_plots,
    f"annual_precip_basin_mean_scatter_{annual_period_tag}_{mask_tag}_{cde_run_dte}.png"
)

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

fig_sc_basin.savefig(svnme, dpi=300, bbox_inches="tight")
plt.show()

print("\nBasin scatter statistics:")
print(stats_sc_basin)

# Optional: save scatter stats
if isinstance(stats_sc_basin, pd.DataFrame):
    stats_sc_basin.to_csv(
        os.path.join(
            path_to_dfs,
            f"annual_basin_scatter_stats_{annual_period_tag}_{mask_tag}_{cde_run_dte}.csv"
        ),
        index=False,
    )

gc.collect()


# =============================================================================
# SECTION 10.5. BASIN SPREAD POINTS
# =============================================================================

svnme = os.path.join(
    path_to_plots,
    f"basin_spread_points_precip_over_imbie_basins_{annual_period_tag}_{mask_tag}_{cde_run_dte}.png"
)

fig_spread, ax_spread, spread_non_gpm, spread_gpm = plot_basin_spread_points_dual(
    df=df_basin_mean_annual,
    basin_col="basin",
    ref_col=r"$P_{\mathrm{MB}}$",
    prod_cols=["ERA5", "GPCP V3.3", "GPM PMW V07"],
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

fig_spread.savefig(svnme, dpi=300, bbox_inches="tight")
plt.show()

# Optional: print/save spread diagnostics if they are dataframes
print("\nSpread diagnostics excluding GPM:")
print(spread_non_gpm)

print("\nSpread diagnostics including/for GPM:")
print(spread_gpm)

if isinstance(spread_non_gpm, pd.DataFrame):
    spread_non_gpm.to_csv(
        os.path.join(
            path_to_dfs,
            f"basin_spread_non_gpm_{annual_period_tag}_{mask_tag}_{cde_run_dte}.csv"
        ),
        index=False,
    )

if isinstance(spread_gpm, pd.DataFrame):
    spread_gpm.to_csv(
        os.path.join(
            path_to_dfs,
            f"basin_spread_gpm_{annual_period_tag}_{mask_tag}_{cde_run_dte}.csv"
        ),
        index=False,
    )

gc.collect()
#%% ANNUAL MEAN MAPS AND REGIONAL MEAN BARS
# =============================================================================
# Purpose:
# Build basin-mean annual precipitation maps and regional mean annual bars using
# the same complete-year period used in the annual scatter/spread analysis.
#
# Important:
# Corrected PMB starts in March 2013, so annual diagnostics should use complete
# years only:
#     ANNUAL_YEAR_START = 2014
#     ANNUAL_YEAR_END   = 2020
# =============================================================================

BASIN_IDS = sorted(AIS_BASINS)
annual_period_tag = f"{ANNUAL_YEAR_START}_{ANNUAL_YEAR_END}"

basin_mask_01deg_clean = basin_mask_01deg.where(basin_mask_01deg < 1e10)
basin_mask_01deg_clean = basin_mask_01deg_clean.where(
    basin_mask_01deg_clean.isin(BASIN_IDS)
)

# =============================================================================
# 1. Build annual-total fields and multi-year mean fields
# =============================================================================
# These should already exist from the previous annual section, but rebuilding
# here makes this section self-contained and avoids accidentally using monthly
# fields in the annual map panels.

pmb_annual_01 = monthly_to_annual_totals_field(
    pmb_mon_01_filtered
).sel(
    year=slice(ANNUAL_YEAR_START, ANNUAL_YEAR_END)
)

era5_annual_01 = monthly_to_annual_totals_field(
    era5_mnth_01_filtered
).sel(
    year=slice(ANNUAL_YEAR_START, ANNUAL_YEAR_END)
)

gpcp_annual_01 = monthly_to_annual_totals_field(
    gpcp_mon_01_filtered
).sel(
    year=slice(ANNUAL_YEAR_START, ANNUAL_YEAR_END)
)

gpm_pmw_annual_01 = monthly_to_annual_totals_field(
    gpm_pmw_v07_mon_01_filtered
).sel(
    year=slice(ANNUAL_YEAR_START, ANNUAL_YEAR_END)
)

# Individual GPM-family products
atms_annual_01 = monthly_to_annual_totals_field(
    gpm_family_monthly_dict_filtered["ATMS"]
).sel(
    year=slice(ANNUAL_YEAR_START, ANNUAL_YEAR_END)
)

mhs_annual_01 = monthly_to_annual_totals_field(
    gpm_family_monthly_dict_filtered["MHS"]
).sel(
    year=slice(ANNUAL_YEAR_START, ANNUAL_YEAR_END)
)

dmsp_annual_01 = monthly_to_annual_totals_field(
    gpm_family_monthly_dict_filtered["DMSP SSMIS"]
).sel(
    year=slice(ANNUAL_YEAR_START, ANNUAL_YEAR_END)
)

amsr2_annual_01 = monthly_to_annual_totals_field(
    gpm_family_monthly_dict_filtered["AMSR2"]
).sel(
    year=slice(ANNUAL_YEAR_START, ANNUAL_YEAR_END)
)

# Multi-year mean annual fields [mm/year]
pmb_annual_mean_01 = annual_to_multiyear_mean_field(
    pmb_annual_01,
    ANNUAL_YEAR_START,
    ANNUAL_YEAR_END,
)

era5_annual_mean_01 = annual_to_multiyear_mean_field(
    era5_annual_01,
    ANNUAL_YEAR_START,
    ANNUAL_YEAR_END,
)

gpcp_annual_mean_01 = annual_to_multiyear_mean_field(
    gpcp_annual_01,
    ANNUAL_YEAR_START,
    ANNUAL_YEAR_END,
)

gpm_pmw_annual_mean_01 = annual_to_multiyear_mean_field(
    gpm_pmw_annual_01,
    ANNUAL_YEAR_START,
    ANNUAL_YEAR_END,
)

atms_annual_mean_01 = annual_to_multiyear_mean_field(
    atms_annual_01,
    ANNUAL_YEAR_START,
    ANNUAL_YEAR_END,
)

mhs_annual_mean_01 = annual_to_multiyear_mean_field(
    mhs_annual_01,
    ANNUAL_YEAR_START,
    ANNUAL_YEAR_END,
)

dmsp_annual_mean_01 = annual_to_multiyear_mean_field(
    dmsp_annual_01,
    ANNUAL_YEAR_START,
    ANNUAL_YEAR_END,
)

amsr2_annual_mean_01 = annual_to_multiyear_mean_field(
    amsr2_annual_01,
    ANNUAL_YEAR_START,
    ANNUAL_YEAR_END,
)

print(f"\nAnnual map fields prepared for {ANNUAL_YEAR_START}–{ANNUAL_YEAR_END}")
print("PMB annual mean:", pmb_annual_mean_01.shape)
print("ERA5 annual mean:", era5_annual_mean_01.shape)
print("GPCP annual mean:", gpcp_annual_mean_01.shape)
print("GPM PMW annual mean:", gpm_pmw_annual_mean_01.shape)


# =============================================================================
# 2. Build basin-mean plotting packs
# =============================================================================
# IMPORTANT:
# build_basin_mean_plot_product() expects MONTHLY fields, not already-annual
# mean fields. It internally does:
#     monthly -> annual totals -> multi-year annual mean -> basin plot pack
#
# Therefore, pass the monthly fields and control the period using
# year_start/year_end.

pmb_pack = build_basin_mean_plot_product(
    pmb_mon_01_filtered,
    basin_mask_01deg_clean,
    BASIN_IDS,
    r"P$_{MB}$",
    year_start=ANNUAL_YEAR_START,
    year_end=ANNUAL_YEAR_END,
)

era5_pack = build_basin_mean_plot_product(
    era5_mnth_01_filtered,
    basin_mask_01deg_clean,
    BASIN_IDS,
    "ERA5",
    year_start=ANNUAL_YEAR_START,
    year_end=ANNUAL_YEAR_END,
)

gpcp_pack = build_basin_mean_plot_product(
    gpcp_mon_01_filtered,
    basin_mask_01deg_clean,
    BASIN_IDS,
    "GPCP V3.3",
    year_start=ANNUAL_YEAR_START,
    year_end=ANNUAL_YEAR_END,
)

atms_pack = build_basin_mean_plot_product(
    gpm_family_monthly_dict_filtered["ATMS"],
    basin_mask_01deg_clean,
    BASIN_IDS,
    "ATMS",
    year_start=ANNUAL_YEAR_START,
    year_end=ANNUAL_YEAR_END,
)

mhs_pack = build_basin_mean_plot_product(
    gpm_family_monthly_dict_filtered["MHS"],
    basin_mask_01deg_clean,
    BASIN_IDS,
    "MHS",
    year_start=ANNUAL_YEAR_START,
    year_end=ANNUAL_YEAR_END,
)

dmsp_pack = build_basin_mean_plot_product(
    gpm_family_monthly_dict_filtered["DMSP SSMIS"],
    basin_mask_01deg_clean,
    BASIN_IDS,
    "DMSP-SSMIS",
    year_start=ANNUAL_YEAR_START,
    year_end=ANNUAL_YEAR_END,
)

amsr2_pack = build_basin_mean_plot_product(
    gpm_family_monthly_dict_filtered["AMSR2"],
    basin_mask_01deg_clean,
    BASIN_IDS,
    "AMSR2",
    year_start=ANNUAL_YEAR_START,
    year_end=ANNUAL_YEAR_END,
)

gpm_pmw_pack = build_basin_mean_plot_product(
    gpm_pmw_v07_mon_01_filtered,
    basin_mask_01deg_clean,
    BASIN_IDS,
    "GPM PMW V07",
    year_start=ANNUAL_YEAR_START,
    year_end=ANNUAL_YEAR_END,
)


# =============================================================================
# 3. Assemble final basin-map plot list
# =============================================================================

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


svnme = os.path.join(
    path_to_plots,
    f"basin_mean_annual_precip_over_imbie_basins_{annual_period_tag}_{mask_tag}_{cde_run_dte}.png"
)

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

fig.savefig(svnme, dpi=300, bbox_inches="tight")
plt.show()
gc.collect()


# =============================================================================
# 4. Regional mean annual precipitation bars
# =============================================================================
# Use the already-filtered annual dataframe from the annual section.
# region_annual_cos was already restricted to 2014–2020 earlier.
# If not, restrict it again here for safety.

region_annual_cos_bar = region_annual_cos[
    (region_annual_cos["year"] >= ANNUAL_YEAR_START) &
    (region_annual_cos["year"] <= ANNUAL_YEAR_END)
].copy()

df_regional_mean_annual = compute_regional_mean_annual_precip(
    region_annual_cos_bar,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP V3.3", "GPM PMW V07"),
)

print("\nRegional mean annual precipitation:")
print(df_regional_mean_annual)

df_regional_mean_annual.to_csv(
    os.path.join(
        path_to_dfs,
        f"regional_mean_annual_precip_{annual_period_tag}_{mask_tag}_{cde_run_dte}.csv"
    ),
    index=False,
)

svnme = os.path.join(
    path_to_plots,
    f"regional_mean_annual_precip_over_imbie_basins_{annual_period_tag}_{mask_tag}_{cde_run_dte}.png"
)

fig, ax = plot_regional_mean_annual_bars(
    df_regional_mean_annual,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP V3.3", "GPM PMW V07"),
    product_colors=product_styles_corr,
    figsize=(9, 6),
    ylabel="[mm/year]",
    title=f"{ANNUAL_YEAR_START}–{ANNUAL_YEAR_END} mean annual precipitation",
    annotate=True,
)

fig.savefig(svnme, dpi=300, bbox_inches="tight")
plt.show()
gc.collect()
#%% THE CLOUDSAT COMPARATIVE ANALYSIS
cs_ant_file_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/CS_Antartica_analysis_kkk/CS-Antarctica_maps'

all_gpcp_v3pt3_mnthly_files_2007_2010 = [
    f for f in all_gpcp_v3pt3_mnthly_files
    if 2007 <= int(os.path.basename(f).split('_')[2][:4]) <= 2010
]

#----------------------------------------------------------------------------
# 1. LOAD Data
#---------------------------------------------------------------------------
gpcp_ds_v3pt320072010 = xr.open_mfdataset(
    all_gpcp_v3pt3_mnthly_files_2007_2010,
    combine="nested",
    concat_dim="time",
    coords="minimal",
    compat="override",
    parallel=True,
    engine="netcdf4",
    chunks={"time": 120, "lat": 180, "lon": 360},
    cache=False,
)

gpcp_ds_v3pt320072010 = ds_swaplon(gpcp_ds_v3pt320072010)

# Keep the monthly precipitation variable
# Your file shows the variable name is sat_gauge_precip
gpcp_mnth20072013 = gpcp_ds_v3pt320072010["sat_gauge_precip"].copy()

# Normalize monthly timestamps to month-start
gpcp_mnth20072013 = gpcp_mnth20072013.assign_coords(
    time=pd.to_datetime(gpcp_mnth20072013["time"].values).to_period("M").to_timestamp()
)

# Convert from mm/day to mm/month
days_in_month = xr.DataArray(
    pd.to_datetime(gpcp_mnth20072013["time"].values).days_in_month,
    dims=["time"],
    coords={"time": gpcp_mnth20072013["time"]}
)

gpcp_mnth20072013 = gpcp_mnth20072013 * days_in_month
gpcp_mnth20072013.name = "gpcp_mm_month"

# Replace fill/missing with NaN if needed
fillv = gpcp_mnth20072013.attrs.get("_FillValue", None)
if fillv is not None:
    gpcp_mnth20072013 = gpcp_mnth20072013.where(gpcp_mnth20072013 != fillv)
gpcp_mnth20072013 = gpcp_mnth20072013.where(np.isfinite(gpcp_mnth20072013))

# Subset Antarctica
gpcp_mnth20072013 = gpcp_mnth20072013.sel(lat=slice(-60, -90))

# Reproject to target grid
gpcp_mnth20072013 = gpcp_mnth20072013.rio.set_spatial_dims(
    x_dim="lon", y_dim="lat", inplace=False
    )
gpcp_mnth20072013 = gpcp_mnth20072013.rio.write_crs("EPSG:4326", inplace=False)

gpcp_mnth20072013 = gpcp_mnth20072013.rio.reproject_match(
    target_template_01deg,
    resampling=Resampling.nearest
)

rename_map = {}
if "x" in gpcp_mnth20072013.dims:
    rename_map["x"] = "lon"
if "y" in gpcp_mnth20072013.dims:
    rename_map["y"] = "lat"
if rename_map:
    gpcp_mnth20072013 = gpcp_mnth20072013.rename(rename_map)

gpcp_mnth20072013 = gpcp_mnth20072013.sortby("lon")
gpcp_mnth20072013 = gpcp_mnth20072013.sortby("lat", ascending=False)
gpcp_mnth20072013 = gpcp_mnth20072013.where(valid_basin_mask)
gpcp_mnth20072013 = gpcp_mnth20072013.where(gpcp_mnth20072013["lat"] < -60)

#----------------------------------------------------------------------------
# 2. LOAD ERA5
#----------------------------------------------------------------------

era5_mnth20072010 = era5_mnth_ds["tp_mm_month"].copy()
# Subset study period
era5_mnth20072010 = era5_mnth20072010.sel(time=slice("2007-01-01", "2010-12-31"))
era5_mnth20072010 = era5_mnth20072010.sel(latitude=slice(-60, -90))
era5_mnth20072010 = era5_mnth20072010.rename({"latitude": "lat", "longitude": "lon"})

era5_mnth20072010 = era5_mnth20072010.rio.set_spatial_dims(
    x_dim="lon", y_dim="lat", inplace=False)
era5_mnth20072010 = era5_mnth20072010.rio.write_crs(
        "EPSG:4326", inplace=False
                                    )

era5_mnth20072010 = era5_mnth20072010.rio.reproject_match(
    target_template_01deg,
    resampling=Resampling.nearest
)
era5_mnth20072010 = replace_fill_with_nan(era5_mnth20072010)

rename_map = {}
if "x" in era5_mnth20072010.dims:
    rename_map["x"] = "lon"
if "y" in era5_mnth20072010.dims:
    rename_map["y"] = "lat"
if rename_map:
    era5_mnth20072010 = era5_mnth20072010.rename(rename_map)

era5_mnth20072010 = era5_mnth20072010.sortby("lon")
era5_mnth20072010 = era5_mnth20072010.sortby("lat", ascending=False)
era5_mnth20072010 = era5_mnth20072010.where(valid_basin_mask)
era5_mnth20072010 = era5_mnth20072010.where(era5_mnth20072010["lat"] < -60)
#----------------------------------------------------------------------
# 3. LOAD CloudSat
cs_mnthly_clim = xr.open_dataset(
    os.path.join(cs_ant_file_path, 'CS-Antarctica_monthly_climatology_2007-2010.nc')
)

cs_seasonal_clim = xr.open_dataset(
    os.path.join(cs_ant_file_path, 'CS_seasonal_climatology-2007-2010.nc')
)

cs_annual_clim = xr.open_dataset(
    os.path.join(cs_ant_file_path, 'CS_annual_climatology-2007-2010.nc')
)
# first data variable
cs_mnth_da = cs_mnthly_clim[list(cs_mnthly_clim.data_vars)[0]].copy()
cs_season_da = cs_seasonal_clim[list(cs_seasonal_clim.data_vars)[0]].copy()
cs_annual_da = cs_annual_clim[list(cs_annual_clim.data_vars)[0]].copy()

# monthly: rename time->time if needed and build pseudo dates for 12 climatological months
if "time" not in cs_mnth_da.dims:
    if "month" in cs_mnth_da.dims:
        cs_mnth_da = cs_mnth_da.rename({"month": "time"})
    else:
        raise ValueError("CloudSat monthly climatology must have either 'time' or 'month' dimension.")

cs_mnth_da = cs_mnth_da.assign_coords(
    time=pd.date_range("2001-01-01", periods=12, freq="MS")
)

# seasonal: keep season coordinate as-is
if "season" not in cs_season_da.dims:
    raise ValueError("CloudSat seasonal climatology must contain a 'season' dimension.")

# annual: must be 2D
print("CloudSat monthly dims :", cs_mnth_da.dims)
print("CloudSat seasonal dims:", cs_season_da.dims)
print("CloudSat annual dims  :", cs_annual_da.dims)

cs_mnth_01   = regrid_clim_to_target(
                        cs_mnth_da, 
                        target_template_01deg, 
                        valid_basin_mask,
                        method=Resampling.nearest
                        )
cs_season_01 = regrid_clim_to_target(
                        cs_season_da, 
                        target_template_01deg, 
                        valid_basin_mask,
                        method=Resampling.nearest
                        )
cs_annual_01 = regrid_clim_to_target(
                        cs_annual_da, 
                        target_template_01deg, 
                        valid_basin_mask,
                        method=Resampling.nearest
                        )

#%%
product_order_cs = [
    "CloudSat",
    "ERA5",
    "GPCP V3.3",
]

product_styles_cs = {
    "CloudSat":    {"color": "gray",          "marker": "o", "lw": 2.5},
    "ERA5":        {"color": "blue",       "marker": "s", "lw": 2.5},
    "GPCP V3.3":   {"color": "orange", "marker": "D", "lw": 2.5},
}

# =============================================================================
# SECTION 8. MONTHLY REGIONAL TIME SERIES (ERA5, GPCP) + CLOUDSAT MONTHLY CLIM
# =============================================================================
days_in_month_cs = xr.DataArray(
    pd.to_datetime(cs_mnth_01.time.values).days_in_month,
    dims=["time"],
    coords={"time": cs_mnth_01.time}
)

# gpcp_mnth_day = gpcp_mnth20072013 / xr.DataArray(
#     pd.to_datetime(gpcp_mnth20072013.time.values).days_in_month,
#     dims=["time"],
#     coords={"time": gpcp_mnth20072013.time}
# )

# era5_mnth_day = era5_mnth20072010 / xr.DataArray(
#     pd.to_datetime(era5_mnth20072010.time.values).days_in_month,
#     dims=["time"],
#     coords={"time": era5_mnth20072010.time}
# )
cs_mnth_mm_mnth_01 = cs_mnth_01 * days_in_month_cs
cs_mnth_mm_mnth_01.attrs["units"] = "mm/month"

product_monthly_dict_cs = {
    "CloudSat":  cs_mnth_mm_mnth_01,
    "ERA5":      era5_mnth20072010,
    "GPCP V3.3": gpcp_mnth20072013,
}

regional_monthly_cos_df_cs = build_all_region_monthly_series_cosine(
    product_dict=product_monthly_dict_cs,
    region_masks=region_masks_01deg,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

# This works because CloudSat monthly climatology is carried as 12 pseudo-months.
region_monthly_clim_cs = compute_monthly_climatology_from_regional_series(
    regional_monthly_cos_df_cs
)

# =============================================================================
# SECTION 9. MONTHLY CLIMATOLOGY PLOT
# =============================================================================
svnme = os.path.join(
    path_to_plots,
    f"cloudsat_era5_gpcp_monthly_climatology_2007_2010_{cde_run_dte}.png"
)

fig, axes = plot_monthly_climatology(
    region_monthly_clim_cs,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=("CloudSat", "ERA5", "GPCP V3.3"),
    product_styles=product_styles_cs,
    figsize=(10, 9),
    ylabel="mm/month",
    y_nbins=4,
    legend_ncol=3,
)

fig.savefig(svnme, dpi=300, bbox_inches="tight")
plt.show()
gc.collect()
#%% Seasonal and annula
# =============================================================================
# CloudSat seasonal climatology -> mm/season
# =============================================================================

# -----------------------------------------------------------
# 2. Build seasonal climatology directly from monthly climatology
# -----------------------------------------------------------
region_seasonal_clim_cs = compute_seasonal_climatology_from_monthly_climatology(
    region_monthly_clim_cs
)

# print(region_seasonal_clim_cs)

# -----------------------------------------------------------
# 3. Plot
# -----------------------------------------------------------
svnme = os.path.join(
    path_to_plots,
    f"cloudsat_era5_gpcp_seasonal_climatology_2007_2010_{cde_run_dte}.png"
)

fig, axes = plot_seasonal_climatology(
    region_seasonal_clim_cs,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=("CloudSat", "ERA5", "GPCP V3.3"),
    product_styles=product_styles_cs,
    figsize=(10, 8),
    ylabel="mm/season",
    y_nbins=4,
    legend_ncol=3,
)

fig.savefig(svnme, dpi=300, bbox_inches="tight")
plt.show()
gc.collect()
#%%
#%% =========================================================
# REGIONAL MEAN ANNUAL PRECIPITATION BAR PLOT
# CloudSat annual climatology -> mm/year
# ===========================================================

# regional annual mean from monthly climatology
df_regional_mean_annual_cs = (
    region_monthly_clim_cs
    .groupby(["region", "product"], as_index=False)["precipitation"]
    .sum()
)

df_regional_mean_annual_cs["region"] = pd.Categorical(
    df_regional_mean_annual_cs["region"],
    categories=("Antarctica", "West Antarctica", "East Antarctica"),
    ordered=True
)

df_regional_mean_annual_cs["product"] = pd.Categorical(
    df_regional_mean_annual_cs["product"],
    categories=("CloudSat", "ERA5", "GPCP V3.3"),
    ordered=True
)

df_regional_mean_annual_cs = df_regional_mean_annual_cs.sort_values(
    ["region", "product"]
).reset_index(drop=True)

# print(df_regional_mean_annual_cs)
# 5. Plot
svnme = os.path.join(
    path_to_plots,
    f"cloudsat_era5_gpcp_regional_mean_annual_2007_2010_{cde_run_dte}.png"
)

fig, ax = plot_regional_mean_annual_bars(
    df_regional_mean_annual_cs,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=("CloudSat", "ERA5", "GPCP V3.3"),
    product_colors=product_styles_cs,
    figsize=(9, 6),
    ylabel="[mm/year]",
    title="2007–2010 mean annual precipitation",
    annotate=True,
)

fig.savefig(svnme, dpi=300, bbox_inches="tight")
plt.show()
gc.collect()


#%% SENSITIVITY TEST: EXCLUDE SELECTED BASIN/MONTH PMB OUTLIERS
BASIN_ID_TO_NAME = {v: k for k, v in BASIN_NAME_TO_ID.items()}

pmb_basin_month_df = basin_month_pmb_from_grid(
    pmb_da=pmb_mon_01,
    basin_mask=basin_mask_01deg,
    basin_ids=AIS_BASINS,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

pmb_basin_month_df["basin_name"] = pmb_basin_month_df["basin"].map(BASIN_ID_TO_NAME)

#------------------------------------------------------------------------
wais_2014_son_pmb_diag = get_problem_season_basin_months(
    pmb_basin_month_df=pmb_basin_month_df,
    region_name="WAIS",
    basin_ids=WAIS_BASINS,
    year=2014,
    season_months=[9, 10, 11],
)

# If DJF 2017 means Jan-Feb 2017 plus Dec 2016
eais_2017_djf_pmb_diag_a = get_problem_season_basin_months(
    pmb_basin_month_df=pmb_basin_month_df,
    region_name="EAIS",
    basin_ids=EAIS_BASINS,
    year=2017,
    season_months=[1, 2],
)

eais_2016_dec_pmb_diag = get_problem_season_basin_months(
    pmb_basin_month_df=pmb_basin_month_df,
    region_name="EAIS",
    basin_ids=EAIS_BASINS,
    year=2016,
    season_months=[12],
)

# If your DJF label instead uses Dec-Jan-Feb within 2017 label convention
eais_2017_djf_pmb_diag_b = get_problem_season_basin_months(
    pmb_basin_month_df=pmb_basin_month_df,
    region_name="EAIS",
    basin_ids=EAIS_BASINS,
    year=2017,
    season_months=[1, 2, 12],
)
# =============================================================================
# Purpose:
# Test whether negative seasonal PMB means in WAIS-2014 and EAIS-2017 are
# controlled by a few basin/month ΔS-driven PMB outliers.
#
# This section excludes selected basin/months from PMB only, then rebuilds the
# regional cosine-weighted monthly and seasonal series.
#
# Important:
# The selected basin/months are based on the ΔS diagnostic plots, where the
# monthly ΔS time series identifies the actual timing of the strongest drops.
# =============================================================================
# -----------------------------------------------------------------------------
# 2. Candidate outlier basin/months from the ΔS diagnostics
# -----------------------------------------------------------------------------
# Start conservative. These are the most visually dominant ΔS drops in the plots.


OUTLIER_BASIN_MONTHS_BY_NAME_CORE = [

    # WAIS-2014 SON: dominant negative PMB basin/months occur in November

    {"year": 2014, "month": 11, "basins": ["G-H", "Ep-F", "I-Ipp"]},

    # EAIS-2017 DJF: dominant negative PMB basin/months occur in Jan-Feb

    {"year": 2017, "month": 1, "basins": ["Dp-E", "E-Ep"]},

    {"year": 2017, "month": 2, "basins": ["Dp-E", "E-Ep"]},

]

# Broader sensitivity option: includes secondary strong negative ΔS drops
OUTLIER_BASIN_MONTHS_BY_NAME_BROAD = [
    # WAIS-2014 SON: broader set of negative November basin/months
    {"year": 2014, "month": 11, "basins": ["G-H", "Ep-F", "I-Ipp", "H-Hp", "Hp-I", "Ipp-J"]},

    # EAIS-2017 DJF: broader set of negative Jan-Feb basin/months
    {"year": 2017, "month": 1, "basins": ["Dp-E", "E-Ep", "B-C", "Cp-D", "C-Cp", "A-Ap", "Ap-B", "D-Dp", "Jpp-K"]},
    {"year": 2017, "month": 2, "basins": ["Dp-E", "E-Ep", "B-C"]},
]


OUTLIER_BASIN_MONTHS_CORE = convert_outlier_basin_names_to_ids(
    OUTLIER_BASIN_MONTHS_BY_NAME_CORE,
    BASIN_NAME_TO_ID,
)

OUTLIER_BASIN_MONTHS_BROAD = convert_outlier_basin_names_to_ids(
    OUTLIER_BASIN_MONTHS_BY_NAME_BROAD,
    BASIN_NAME_TO_ID,
)

# Choose sensitivity level
OUTLIER_BASIN_MONTHS = OUTLIER_BASIN_MONTHS_CORE
# OUTLIER_BASIN_MONTHS = OUTLIER_BASIN_MONTHS_BROAD

# -----------------------------------------------------------------------------
# 3. Create sensitivity PMB field
# -----------------------------------------------------------------------------

pmb_mon_01_sens = mask_specific_basin_months_in_field(
    da=pmb_mon_01,
    basin_mask=basin_mask_01deg,
    outlier_basin_months=OUTLIER_BASIN_MONTHS,
    time_name="time",
)

# -----------------------------------------------------------------------------
# 4. Rebuild regional cosine monthly series
# -----------------------------------------------------------------------------

product_monthly_dict_sens = {
    r"$P_{\mathrm{MB}}$": pmb_mon_01_sens,
    "ERA5": era5_mnth_01,
    "GPCP V3.3": gpcp_mon_01,
    "GPM PMW V07": gpm_pmw_v07_mon_01,
}

regional_monthly_cos_df_sens = build_all_region_monthly_series_cosine(
    product_dict=product_monthly_dict_sens,
    region_masks=region_masks_01deg,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

# -----------------------------------------------------------------------------
# 5. Rebuild seasonal series
# -----------------------------------------------------------------------------

seasonal_cos_df_sens = monthly_regional_df_to_conventional_seasonal(
    regional_monthly_cos_df_sens,
    time_col="time",
    region_col="region",
    product_col="product",
    value_col="precipitation",
    require_complete_season=True,
)

pmb_sensitivity_compare = compare_original_vs_sensitivity_values(
    seasonal_orig_df=seasonal_cos_df,
    seasonal_sens_df=seasonal_cos_df_sens,
    regions=("West Antarctica", "East Antarctica"),
    product=r"$P_{\mathrm{MB}}$",
    years=(2014, 2017),
)

print(pmb_sensitivity_compare)


#-----------------------------------------------------------------------
fig, axes = plot_seasonal_timeseries_regions(
    seasonal_cos_df_sens,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(
        r"$P_{\mathrm{MB}}$",
        "ERA5",
        "GPCP V3.3",        
        "GPM PMW V07",
    ),
    product_styles=product_styles_corr,
    title_suffix="",
)
#-----------------------------------------------------------------------
fig, ax = plot_monthly_pmb_by_basin(
    pmb_basin_month_df=pmb_basin_month_df,
    year=2014,
    basin_ids=WAIS_BASINS,
    region_name="WAIS",
    highlight_months=[11],
    title="2014 WAIS monthly PMB by basin",
    figsize=(14, 4.8),
    legend_ncol=4,
)

plt.show()

fig, ax = plot_monthly_pmb_by_basin(
    pmb_basin_month_df=pmb_basin_month_df,
    year=2017,
    basin_ids=EAIS_BASINS,
    region_name="EAIS",
    highlight_months=[1, 2],
    title="2017 EAIS monthly PMB by basin",
    figsize=(14, 4.8),
    legend_ncol=5,
)

plt.show()

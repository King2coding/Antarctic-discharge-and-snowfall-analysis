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
out_dfs = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/out_dfs'
# --- Precipitation products ---
gpcp_v3pt3_mnthly_ds_path = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_3_monthly_1983_2024'
era5_mnhtly_file = r'/ra1/pubdat/GPCP/GPCP_Reproduce_GJ/era5_tp_198001202412_monthly.nc'

# --- UA-HIPA DATA ---
# Open UA-HIPA monthly product

ua_file = (

    "/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/"

    "data/uahipa_monthly/uahipa_precip_monthly_2013_2020_20260508.nc"

)

# --- Basin / PMB data ---
basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'
Pmb_mm_fle  = os.path.join(
    basins_path, 
    "Monthly_mass_budget_precip_RignotBasin_in_mm_forward_deltaS_uncorrected_positive_sublimation_loss_20260507.nc")
#    'Monthly_mass_budget_precip_RignotBasin_in_mm_20260226.nc')
Pmb_unc_mm_fle = os.path.join(
    basins_path,
    "Monthly_mass_budget_precip_RignotBasin_uncertainty_in_mm_forward_deltaS_uncorrected_positive_sublimation_loss_20260519.nc"

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
plt.close()

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

print("Loading PMB monthly uncertainty dataset ...")
P_unc_mm_mnth = xr.open_dataarray(Pmb_unc_mm_fle)

print("Loading UA-HIPA monthly dataset ...")
uahipa_ds = xr.open_dataset(ua_file)

uahipa_mnth = uahipa_ds["uahipa_precip_mm_month"]

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

print("Reprojecting PMB monthly uncertainty to common 0.1° grid ...")
pmb_unc_mon_01 = prepare_pmb_monthly_on_target(P_unc_mm_mnth, target_template_01deg)
pmb_unc_mon_01 = subset_common_period(pmb_unc_mon_01)

# Ensure uncertainty is positive
pmb_unc_mon_01 = abs(pmb_unc_mon_01)
#----------------------------------------------------------------------------

print("Reprojecting UA-HIPA monthly to common 0.1° grid ...")
# Rename coords to match comparison workflow

uahipa_mnth = uahipa_mnth.rename({"y": "lat", "x": "lon"})

# Sort like other products

uahipa_mnth = uahipa_mnth.sortby("lon")

uahipa_mnth = uahipa_mnth.sortby("lat", ascending=False)

# Attach CRS

uahipa_mnth = uahipa_mnth.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)

uahipa_mnth = uahipa_mnth.rio.write_crs("EPSG:4326", inplace=False)

# Reproject/remap to your common 0.1° comparison grid

uahipa_mon_01 = uahipa_mnth.rio.reproject_match(

    target_template_01deg,

    resampling=Resampling.nearest

)

# Rename x/y back if needed

rename_map = {}

if "x" in uahipa_mon_01.dims:
    rename_map["x"] = "lon"
if "y" in uahipa_mon_01.dims:
    rename_map["y"] = "lat"

if rename_map:
    uahipa_mon_01 = uahipa_mon_01.rename(rename_map)

uahipa_mon_01 = uahipa_mon_01.sortby("lon")
uahipa_mon_01 = uahipa_mon_01.sortby("lat", ascending=False)
# Apply Antarctic basin mask
uahipa_mon_01 = uahipa_mon_01.where(basin_mask_01deg.notnull())
uahipa_mon_01 = uahipa_mon_01.where(uahipa_mon_01["lat"] < -60)


print("UA-HIPA masked time range:")
print(str(uahipa_mon_01.time.min().values), "->", str(uahipa_mon_01.time.max().values))

print("UA-HIPA shape:", uahipa_mon_01.shape)
print("UA-HIPA Antarctic basin-masked stats [mm/month]:")
print("min :", float(uahipa_mon_01.min(skipna=True)))
print("mean:", float(uahipa_mon_01.mean(skipna=True)))
print("max :", float(uahipa_mon_01.max(skipna=True)))

# First masked month plot
uahipa_mon_01.isel(time=0).plot(vmin=0, vmax=100)
plt.title("UA-HIPA basin-masked precipitation, first month [mm/month]")
plt.show()
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
pmb_mon_01 = pmb_mon_01.where(valid_basin_mask)
pmb_unc_mon_01 = pmb_unc_mon_01.where(valid_basin_mask)
uahipa_mon_01 = uahipa_mon_01.where(valid_basin_mask)

print("✅ Common masked monthly fields ready")
print("GPCP  :", gpcp_mon_01.shape)
print("ERA5  :", era5_mon_01.shape)
print("PMB   :", pmb_mon_01.shape)
print("PMB unc:", pmb_unc_mon_01.shape)
print("UA-HIPA:", uahipa_mon_01.shape)


# =============================================================================
# SECTION 9. QUICK SANITY CHECKS
# =============================================================================

print("\n--- Sanity checks ---")
print("Target grid CRS:", target_template_01deg.rio.crs)
print("Basin mask CRS :", basin_mask_01deg.rio.crs)

print("GPCP time range:", str(gpcp_mon_01.time.min().values), "->", str(gpcp_mon_01.time.max().values))
print("ERA5 time range:", str(era5_mon_01.time.min().values), "->", str(era5_mon_01.time.max().values))
print("PMB time range :", str(pmb_mon_01.time.min().values),  "->", str(pmb_mon_01.time.max().values))
print("PMB uncertainty time range :", str(pmb_unc_mon_01.time.min().values), "->", str(pmb_unc_mon_01.time.max().values))
print("UA-HIPA time range:", str(uahipa_mon_01.time.min().values), "->", str(uahipa_mon_01.time.max().values))

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
# ENFORCE COMMON MONTHLY TIME AXIS FOR MAIN PRODUCT COMPARISON
# =============================================================================
common_time = np.intersect1d(
    pmb_mon_01["time"].values,
    era5_mnth_01["time"].values,
)

common_time = np.intersect1d(
    common_time,
    gpcp_mon_01["time"].values,
)

common_time = np.intersect1d(
    common_time,
    gpm_pmw_v07_mon_01["time"].values,
)

common_time = np.intersect1d(
    common_time,
    pmb_unc_mon_01["time"].values,
)

# common_time = np.intersect1d(
#     common_time,
#     uahipa_mon_01["time"].values,
# )

common_time = np.sort(common_time)

pmb_mon_01 = pmb_mon_01.sel(time=common_time)
pmb_unc_mon_01 = pmb_unc_mon_01.sel(time=common_time)
era5_mnth_01 = era5_mnth_01.sel(time=common_time)
gpcp_mon_01 = gpcp_mon_01.sel(time=common_time)
gpm_pmw_v07_mon_01 = gpm_pmw_v07_mon_01.sel(time=common_time)

# uahipa_mon_01 = uahipa_mon_01.sel(time=common_time)

gpm_family_monthly_dict = {
    name: da.sel(time=common_time)
    for name, da in gpm_family_monthly_dict.items()
}

print("\n✅ Common monthly time axis enforced for main product comparison")
print("PMB :", str(pmb_mon_01.time.min().values), "->", str(pmb_mon_01.time.max().values), pmb_mon_01.sizes["time"])
print("PMB uncertainty:", str(pmb_unc_mon_01.time.min().values), "->", str(pmb_unc_mon_01.time.max().values), pmb_unc_mon_01.sizes["time"])
print("ERA5:", str(era5_mnth_01.time.min().values), "->", str(era5_mnth_01.time.max().values), era5_mnth_01.sizes["time"])
print("GPCP:", str(gpcp_mon_01.time.min().values), "->", str(gpcp_mon_01.time.max().values), gpcp_mon_01.sizes["time"])
print("GPM :", str(gpm_pmw_v07_mon_01.time.min().values), "->", str(gpm_pmw_v07_mon_01.time.max().values), gpm_pmw_v07_mon_01.sizes["time"])
print("UA-HIPA:", str(uahipa_mon_01.time.min().values), "->", str(uahipa_mon_01.time.max().values), uahipa_mon_01.sizes["time"])
#%%
# =============================================================================
# SECTION 2B. BUILD MONTHLY REGIONAL SERIES FOR PMB, ERA5, GPCP
# =============================================================================

product_monthly_dict = {
    r"$P_{\mathrm{MB}}$": pmb_mon_01,
    "ERA5": era5_mnth_01,
    "GPCP V3.3": gpcp_mon_01,
    "GPM PMW V07": gpm_pmw_v07_mon_01,
    # "UA-HIPA": uahipa_mon_01
}

regional_monthly_cos_df = build_all_region_monthly_series_cosine(
    product_dict=product_monthly_dict,
    region_masks=region_masks_01deg,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

# save df
regional_monthly_cos_df.to_csv(os.path.join(out_dfs, 
                                            f"monthly_precip_over_imbie_basins_{cde_run_dte}.csv"), index=False)

# print(regional_monthly_cos_df.head())
# print(regional_monthly_cos_df.tail())
# print(regional_monthly_cos_df.groupby(["region", "product"]).size())

#----------------------------------------------------------------------------
regional_pmb_unc_monthly_df = build_region_monthly_uncertainty_from_basin_painted_field(

    unc_da=pmb_unc_mon_01,
    basin_mask=basin_mask_01deg,
    region_defs=REGION_BASINS,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
    value_name="pmb_uncertainty",
    )

regional_pmb_unc_monthly_df.to_csv(
    os.path.join(out_dfs, 
                 f"monthly_PMB_uncertainty_over_imbie_basins_{cde_run_dte}.csv"),
    index=False,
)

print("\nPMB regional monthly uncertainty:")
print(regional_pmb_unc_monthly_df.head())
print(regional_pmb_unc_monthly_df.tail())

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
) # "UA-HIPA", 

fig.savefig(svnme, dpi=300)
plt.show()
gc.collect()

#----------------------------------------------------------------------------
region_monthly_clim_pmb_unc = compute_monthly_climatology_uncertainty_from_monthly_unc_df(
    regional_pmb_unc_monthly_df
)

region_monthly_clim_pmb_unc.to_csv(
    os.path.join(out_dfs, 
                 f"monthly_climatology_PMB_uncertainty_over_imbie_basins_{cde_run_dte}.csv"),
    index=False,
)

print("\nPMB monthly climatology uncertainty:")
print(region_monthly_clim_pmb_unc.head())

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
        # "UA-HIPA",
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

#----------------------------------------------------------------------------
fig, axes = plot_seasonal_climatology_1x3(
    region_seasonal_clim_corr,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(
        r"$P_{\mathrm{MB}}$",
        "ERA5",
        "GPCP V3.3",
        # "UA-HIPA",
        "GPM PMW V07",
        "GPM PMW V07 (corr.)",
    ),
    product_styles=product_styles_corr,
    figsize=(15, 4.8),
    ylabel="mm/season",
    y_nbins=4,
    legend_ncol=5,
)

svnme = os.path.join(
    path_to_plots,
    f"seasonal_climatology_precip_over_imbie_basins_1x3_{cde_run_dte}.png"
)

fig.savefig(svnme, dpi=500, bbox_inches="tight")
plt.show()
gc.collect()


#--------------- study the seasonal bias structure
bias_ratio_df = compute_seasonal_bias_and_ratio(
    region_seasonal_clim_corr,
    ref_product=r"$P_{\mathrm{MB}}$",
    value_col="precipitation",
)

fig, axes = plot_seasonal_bias_or_ratio(
    bias_ratio_df,
    y_col="pct_diff_vs_ref",
    product_order=("ERA5", "GPCP V3.3"),  # "UA-HIPA" removed
    product_styles=product_styles_corr,
    ylabel=r"% difference relative to $P_{\mathrm{MB}}$",
    ylim=(-35, 35),
    legend_ncol=3,
)


fig, axes = plot_seasonal_bias_or_ratio(
    bias_ratio_df,
    y_col="ref_to_product_ratio",
    product_order=("ERA5", "GPCP V3.3", "UA-HIPA"),
    product_styles=product_styles_corr,
    ylabel=r"$P_{\mathrm{MB}}$ / Product",
    ylim=(0.55, 1.55),
    legend_ncol=3,
)

#----------------------------------------------------------------------------
seasonal_pmb_unc_df = monthly_uncertainty_df_to_conventional_seasonal_uncertainty(
    regional_pmb_unc_monthly_df,
    require_complete_season=True,
)

region_seasonal_clim_pmb_unc = compute_seasonal_climatology_uncertainty_from_seasonal_unc_df(
    seasonal_pmb_unc_df
)

seasonal_pmb_unc_df.to_csv(
    os.path.join(out_dfs, 
                 f"seasonal_timeseries_PMB_uncertainty_over_imbie_basins_{cde_run_dte}.csv"),
    index=False,
    )

region_seasonal_clim_pmb_unc.to_csv(
    os.path.join(out_dfs, 
                 f"seasonal_climatology_PMB_uncertainty_over_imbie_basins_{cde_run_dte}.csv"),
    index=False,
)

print("\nPMB seasonal climatology uncertainty:")

print(region_seasonal_clim_pmb_unc)


#----------------------------------------------------------------------------
fig, axes = plot_seasonal_climatology_with_pmb_uncertainty(
    region_seasonal_clim_corr,
    pmb_unc_df=seasonal_pmb_unc_df,
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
    unc_label=r"$P_{\mathrm{MB}}$ uncertainty ($\pm 1\sigma$)",
)

svnme = os.path.join(
    path_to_plots,
    f"seasonal_climatology_precip_with_PMB_uncertainty_over_imbie_basins_{cde_run_dte}.png"
)

fig.savefig(svnme, dpi=500, bbox_inches="tight")
plt.show()
gc.collect()
#%% YEAR BY YEAR SEASONAL TIMESERIES
# Convert monthly regional values to conventional seasonal means
seasonal_cos_df = monthly_regional_df_to_conventional_seasonal(
    regional_monthly_cos_df,
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
        # "UA-HIPA",      
        "GPM PMW V07",
    ),
    product_styles=product_styles_corr,
    title_suffix="",
)


#%% Interannual Variability
# -----------------------------------------------------------------------------
# 1. Annual product totals
# -----------------------------------------------------------------------------

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
    product_order=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP V3.3", 
                   # "UA-HIPA", 
                   "GPM PMW V07", "GPM PMW V07 (corr.)"),
    product_styles=product_styles_corr,
    figsize=(10, 9),
    ylabel="mm/year",
    y_nbins=4,
    legend_ncol=3,
)

fig.savefig(svnme, dpi=300)
plt.show()
gc.collect()

# -----------------------------------------------------------------------------

# 2. Annual PMB uncertainty

# -----------------------------------------------------------------------------

region_annual_pmb_unc = compute_annual_uncertainty_from_monthly_unc_df(
    regional_pmb_unc_monthly_df,
    min_months_per_year=11,
)

region_annual_pmb_unc.to_csv(
    os.path.join(
        out_dfs,
        f"annual_PMB_uncertainty_over_imbie_basins_{cde_run_dte}.csv"
    ),
    index=False,
)

print("\nPMB annual uncertainty:")
print(region_annual_pmb_unc.head())
print(region_annual_pmb_unc.tail())

svnme = os.path.join(
    path_to_plots,
    f'interannual_variability_precip_over_imbie_basins_with_PMB_uncertainty_{cde_run_dte}.png'
)

fig, axes = plot_interannual_variability(
    region_annual_corr,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(
        r"$P_{\mathrm{MB}}$",
        "ERA5",
        "GPCP V3.3",
        "GPM PMW V07",
        "GPM PMW V07 (corr.)",
    ),
    product_styles=product_styles_corr,
    figsize=(10, 9),
    ylabel="mm/year",
    y_nbins=4,
    legend_ncol=3,
)
# -----------------------------------------------------------------------------
# 3. Add PMB ± 1σ uncertainty band
# -----------------------------------------------------------------------------
region_order = ("Antarctica", "West Antarctica", "East Antarctica")
pmb_label = r"$P_{\mathrm{MB}}$"

for ax, region in zip(axes, region_order):

    pmb_sub = region_annual_corr[
        (region_annual_corr["region"] == region) &
        (region_annual_corr["product"] == pmb_label)
    ].copy()

    unc_sub = region_annual_pmb_unc[
        region_annual_pmb_unc["region"] == region
    ].copy()

    pmb_sub = pmb_sub.sort_values("year")
    unc_sub = unc_sub.sort_values("year")

    band_df = pmb_sub.merge(
        unc_sub[["region", "year", "pmb_uncertainty"]],
        on=["region", "year"],
        how="inner",
    )

    x = band_df["year"].values
    y = band_df["precipitation"].values
    sig = band_df["pmb_uncertainty"].values

    ax.fill_between(
        x,
        y - sig,
        y + sig,
        color="0.75",
        alpha=0.45,
        linewidth=0,
        label=r"$P_{\mathrm{MB}}$ uncertainty ($\pm 1\sigma$)",
        zorder=1,
    )

    # Redraw PMB line above the band
    ax.plot(
        x,
        y,
        color="black",
        marker="o",
        linewidth=2.5,
        label=pmb_label,
        zorder=5,
    )


# -----------------------------------------------------------------------------
# 4. Clean and rebuild legend only once
# -----------------------------------------------------------------------------

for ax in axes:
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

for leg in fig.legends:
    leg.remove()

handles, labels = axes[0].get_legend_handles_labels()

clean_handles = []
clean_labels = []
seen = set()

for h, lab in zip(handles, labels):
    if lab not in seen and not lab.startswith("_"):
        clean_handles.append(h)
        clean_labels.append(lab)
        seen.add(lab)

fig.legend(
    clean_handles,
    clean_labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.015),
    fontsize=13,
)

fig.subplots_adjust(bottom=0.16)

fig.savefig(svnme, dpi=300, bbox_inches="tight")
plt.show()
gc.collect()

#%%
# =============================================================================
# RUN FULL WORKFLOW
# =============================================================================

# IMPORTANT:
# Use the basin mask already reprojected to the common lat/lon grid.
# This should be your basin_mask_01deg / basin_mask_latlon, not raw basins.

basin_ids = range(2, 20)

monthly_df_data_mmmonth = build_basin_monthly_dict_from_gridded_products(
    product_monthly_dict=product_monthly_dict,
    basin_mask_2d=basin_mask_01deg,   # or basin_mask_latlon
    basin_ids=basin_ids,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
)

basin_annual_df, basin_spread_df = build_basin_annual_spread_from_monthly_dict(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    min_months_per_year=12,
    spread_q=(25, 75),
)


fig, axes = plot_region_interannual_original_mean_with_basin_spread(
    region_annual_df=region_annual_cos,
    basin_spread_df=basin_spread_df,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=product_order,
    product_styles=product_styles_corr,
    spread_type="iqr",
    figsize=(10, 10),
    ylabel="Precipitation [mm/year]",
    legend_ncol=4,
    include_band_legend=True,
)

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
ua_haipa_annual_01 = monthly_to_annual_totals_field(uahipa_mon_01)
# Build 2013–2020 mean annual fields [mm/year]
# Build complete-year mean annual fields [mm/year]
pmb_annual_mean_01  = annual_to_multiyear_mean_field(
    pmb_annual_01, ANNUAL_YEAR_START, ANNUAL_YEAR_END
)

era5_annual_mean_01 = annual_to_multiyear_mean_field(
    era5_annual_01, ANNUAL_YEAR_START, ANNUAL_YEAR_END
)

gpcp_annual_mean_01 = annual_to_multiyear_mean_field(
    gpcp_annual_01, ANNUAL_YEAR_START, ANNUAL_YEAR_END
)

gpm_pmw_v07_annual_mean_01 = annual_to_multiyear_mean_field(
    gpm_pmw_v07_01, ANNUAL_YEAR_START, ANNUAL_YEAR_END
)
ua_haipa_annual_mean_01 = annual_to_multiyear_mean_field(
    ua_haipa_annual_01, ANNUAL_YEAR_START, ANNUAL_YEAR_END
)
print(pmb_annual_mean_01.shape, 
      era5_annual_mean_01.shape, 
      gpcp_annual_mean_01.shape,
      gpm_pmw_v07_annual_mean_01.shape,
      ua_haipa_annual_mean_01.shape,)


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

df_ua_haipa_basin = compute_basin_cosine_weighted_means_from_field(
    da_2d=ua_haipa_annual_mean_01,
    basin_mask_2d=basin_mask_01deg_clean,
    basin_ids=BASIN_IDS,
    value_name="UA-HIPA",
)

df_gpm_pmw_v07 = compute_basin_cosine_weighted_means_from_field(
    da_2d=gpm_pmw_v07_annual_mean_01,
    basin_mask_2d=basin_mask_01deg_clean,
    basin_ids=BASIN_IDS,
    value_name="GPM PMW V07",
)

# Merge
df_basin_mean_annual = (
    df_pmb_basin
    .merge(df_era5_basin, on="basin", how="outer")
    .merge(df_gpcp_basin, on="basin", how="outer")
    # .merge(df_ua_haipa_basin, on="basin", how="outer")
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
    prod_cols=["ERA5", "GPCP V3.3", "UA-HIPA", "GPM PMW V07"],   # GPM placeholder okay if absent
    prod_labels=None,
    product_styles=product_styles_corr,
    non_gpm_group=[r"$P_{\mathrm{MB}}$", "ERA5", "GPCP V3.3", "UA-HIPA"],
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
    pmb_mon_01,
    basin_mask_01deg_clean,
    BASIN_IDS,
    r"P$_{MB}$",
    year_start=ANNUAL_YEAR_START,
    year_end=ANNUAL_YEAR_END,
)
era5_pack = build_basin_mean_plot_product(
    era5_mnth_01, 
    basin_mask_01deg_clean, 
    BASIN_IDS, 
    "ERA5",
    year_start=ANNUAL_YEAR_START,
    year_end=ANNUAL_YEAR_END,
)
gpcp_pack = build_basin_mean_plot_product(
    gpcp_mon_01, 
    basin_mask_01deg_clean, 
    BASIN_IDS, 
    "GPCP V3.3",
    year_start=ANNUAL_YEAR_START,
    year_end=ANNUAL_YEAR_END,
)

# ua_hipa_pack = build_basin_mean_plot_product(
#     uahipa_mon_01,
#     basin_mask_01deg_clean,
#     BASIN_IDS,
#     "UA-HIPA",
#     year_start=ANNUAL_YEAR_START,
#     year_end=2019,  
# )
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
    # (ua_hipa_pack["product"],  ua_hipa_pack["plot_grid"],  ua_hipa_pack["panel_mean"]),
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
    group1_idx=[0, 1, 2], # [0, 1, 2,3]
    group2_idx=[3, 4, 5, 6, 7], # [4, 5, 6, 7, 8]
    ncols=4,
    gamma1=0.6,
    vmin1=0,
    vmax1=400,
    cbar_tcks1=[0, 25, 50, 100, 200, 300, 400],
    cbar_label1="axes (a, b, c, d)",
    gamma2=0.6,
    vmin2=0,
    vmax2=80,
    cbar_tcks2=[0, 5, 10, 20, 40, 60, 80],
    cbar_label2="axes (e, f, g, h, i)",
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
) # "UA-HIPA", 
svnme = os.path.join(path_to_plots, f'regional_mean_annual_precip_over_imbie_basins_{cde_run_dte}.png')
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

fig.savefig(svnme, dpi=300)

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


fig, ax = plot_monthly_pmb_by_basin(
    pmb_basin_month_df=pmb_basin_month_df,
    year=2019,
    basin_ids=EAIS_BASINS,
    region_name="EAIS",
    highlight_months=[1, 2],
    title="2019 EAIS monthly PMB by basin",
    figsize=(14, 4.8),
    legend_ncol=5,
)

plt.show()


fig, ax = plot_monthly_pmb_by_basin(
    pmb_basin_month_df=pmb_basin_month_df,
    year=2020,
    basin_ids=EAIS_BASINS,
    region_name="EAIS",
    highlight_months=[1, 2],
    title="2020 EAIS monthly PMB by basin",
    figsize=(14, 4.8),
    legend_ncol=5,
)

plt.show()
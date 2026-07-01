"""
Monthly Mass-Budget Precipitation (2013–2020, Rignot/IMBIE basins)

Inputs
------
1) David (GRACE/altimetry): basin-scale monthly storage/mass anomaly time series (Gt)
   - Original source: DataCombo_RignotBasins.xlsx
   - Working input: LI-filled monthly storage anomaly pickle
     DataCombo_RignotBasins_LI_tier1_20260325.pkl
   - Important: This is storage anomaly S, not ΔS.
   - Monthly ΔS is computed after gap filling using the active forward-difference convention:
     ΔS_m = S_{m+1} - S_m and assigned to the starting month m.

2) Chad (discharge & basal melt): Excel with annual values per basin (Gt/yr)
   - Discharge sheet: "Discharge (Gt yr^-1)"
   - Basal melt source: "Summary"

3) RACMO2 sublimation: NetCDF monthly ANT11 domain (RACMO2.4p1)
   - Variable: "subltot"
   - Integrated over each basin and converted to Gt/month.

Output
------
- Monthly P_MB per basin/map in Gt/month and mm/month.
- Annual, seasonal, and monthly climatological P_MB summaries.

Main PMB equation
-----------------
P_MB = discharge + basal melt + ΔS + sublimation

Author: K. K. Kumah
Updated for GRACE storage-anomaly correction workflow.
"""
#%%
# import libraries
from program_utils import *
from Extra_util_functions import *
from program_utile_13Apr2026 import *
#%%
# define paths
basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'
racmo_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/RACMO2pt4p1'
path_to_plots = r'/home/kkumah/Projects/Antarctic_discharge_work/plots'

#%%
# -----------------------------------------------------------------------------
# Analysis settings
# -----------------------------------------------------------------------------
YEAR_START = 2013
YEAR_END = 2020
YEARS = np.arange(YEAR_START, YEAR_END + 1)

# Main GRACE-derived ΔS correction switch
APPLY_GRACE_DELTAS_CLIM_CORRECTION = False

# Monthly ΔS climatology statistic: "mean" or "median"
GRACE_CLIM_STAT = "mean"

# Exclude flagged ΔS values when computing the monthly ΔS climatology
EXCLUDE_FLAGGED_FROM_GRACE_CLIM = True

# =============================================================================
# PMB UNCERTAINTY SETTINGS
# =============================================================================
# Current approach:
#   PMB = D + BM + ΔS + SUB
#
# Available/usable uncertainty:
#   1) ΔS uncertainty from David's 1-sigma_Error(Gt) sheet
#   2) Optional fractional uncertainty for discharge
#   3) Optional fractional uncertainty for basal melt
#   4) Optional fractional uncertainty for RACMO sublimation
#
# For now, ΔS uncertainty is the main formal uncertainty term.
# Other terms can be set to 0.0 until we decide defensible values.

COMPUTE_PMB_UNCERTAINTY = True

# Fractional uncertainty assumptions for budget terms.
# Keep these at 0.0 if we want PMB uncertainty to reflect only GRACE/altimetry ΔS error.
DISCHARGE_FRAC_UNC = 0.0
BASAL_MELT_FRAC_UNC = 0.0
SUBLIMATION_FRAC_UNC = 0.0


#%%

# basin_fle = os.path.join(basin_path, 'bedmap3_basins.nc')
# bedmap3_basins = xr.open_dataset(basin_fle)
# basins_imbie = bedmap3_basins['imbie'].copy()
# # Specs from the basin grid
# basin_transform = basins_imbie.rio.transform()
# height, width = basins_imbie.shape
# xmin, ymax = basin_transform.c, basin_transform.f
# xres, yres = basin_transform.a, -basin_transform.e
# xmax = xmin + width * xres
# ymin = ymax - height * yres
# print(f"Basin grid: width={width}, height={height}, xres={xres}, yres={yres}")
# basin_bounds = (xmin, xmax, ymin, ymax)  # (minx, maxx, miny, maxy)
# print(f"Basin bounds: {basin_bounds}")
# # Print the bounds for verification
# print(f"Basin bounds: x_min={xmin}, x_max={xmax}, y_min={ymin}, y_max={ymax}")

# # remap basin grid to a 0.1 deg (10km) grid

# # If the mask is a subdataset, point to it explicitly, e.g.:
# src = f'NETCDF:"{basin_fle}":imbie'
# # src = basin_nc  # if gdalwarp sees the first variable as the raster; else use the NETCDF:"...":imbie form

dst = os.path.join(basin_path, 'bedmap3_basins_0.1deg.tif')

# Antarctica 0.1° lat/lon template: lon [-180,180], lat [-90,-60]
# cmd = [
#     "gdalwarp",
#     "-s_srs", crs_stereo, #'+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +datum=WGS84',
#     "-t_srs", crs_stereo, #"EPSG:4326",
#     "-te",  str(xmin), str(ymin), str(xmax), str(ymax),      # extent: lon min, lat min, lon max, lat max
#     "-tr",  "10000", "10000",                    # 0.1° grid (# 10 km pixels (meters))
#     "-r", "near",                            # preserve integer IDs
#     "-ot", "Int16",
#     "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2", "-co", "TILED=YES",
#     "-dstnodata", "0",
#     src, dst
# ]
# subprocess.run(cmd, check=True)

# Read the remapped file
basins_imbie = xr.open_dataarray(dst)
if not basins_imbie.rio.crs:
    basins_imbie = basins_imbie.rio.write_crs(CRS.from_proj4(crs_stereo))
basin_transform = basins_imbie.rio.transform()
height, width = basins_imbie.shape[1:]
xmin, ymax = basin_transform.c, basin_transform.f
xres, yres = basin_transform.a, -basin_transform.e
xmax = xmin + width * xres
ymin = ymax - height * yres
print(f"Basin grid: width={width}, height={height}, xres={xres}, yres={yres}")
basin_bounds = (xmin, xmax, ymin, ymax)  # (minx, maxx, miny, maxy)
print(f"Basin bounds: {basin_bounds}")
basins_imbie = basins_imbie.where((basins_imbie > 1) & (basins_imbie.notnull()))

# --- Helper: carry mapping in DataArray coords (nice for groupby) ---

# Create a labelled copy where 'basin_name' is a coordinate matching each pixel
name_lookup = xr.DataArray(
    np.vectorize(lambda v: id2name.get(int(v), "NA") if np.isfinite(v) else "NA")(basins_imbie.values),
    coords=basins_imbie.coords, dims=basins_imbie.dims
)
name_lookup.name = "basin_name"
basin_imbie_with_name_map = xr.Dataset(dict(basin_id=basins_imbie, basin_name=name_lookup))

# 0) Squeeze band, mask invalid, and (optionally) drop Islands (ID==1)
basin_id = basin_imbie_with_name_map['basin_id']
basin_name = basin_imbie_with_name_map['basin_name']

# - - - - - - - - - - - - - - - - - - - - - - - -- - - -- - - - - - - -- - - - - 

# rignot_deltaS_err = pd.read_excel(
#     os.path.join(basin_path, 
#     'DataCombo_RignotBasins.xlsx'), 
#     sheet_name='1-sigma_Error(Gt)')

# print("\n[Rignot uncertainty] Loaded 1-sigma error table:")
# print(rignot_deltaS_err.head())
# print("Columns:")
# print(list(rignot_deltaS_err.columns))

# tmp = rignot_deltaS_err.copy()
# tmp.columns = [str(c).strip() for c in tmp.columns]

# tmp["date"] = tmp["Time"].apply(
#     lambda x: decimal_year_to_month_start(x, mode="nearest")
# )

# dup_dates = tmp["date"][tmp["date"].duplicated()].unique()

# print("Number of duplicate converted dates:", len(dup_dates))
# print("First few duplicate converted dates:")
# print(dup_dates[:10])

rignot_sigmaS_filled_pkl = os.path.join(
    basin_path,
    "DataCombo_RignotBasins_1sigma_Error_LI_tier1_GRACE_updated_20260609.pkl"    
) # "DataCombo_RignotBasins_1sigma_Error_LI_tier1_20260519.pkl"

# - - - - - - - - - - - - - - - - - - - - - - - -- - - -- - - - - - - -- - - - - 

#%% - - - - - - - - - - Plot basins - - - - - - - - - - - - - - - - - - - - - - - 

# Set up colormap and norm for 27 discrete basins
colors = plt.cm.gist_ncar(np.linspace(0, 1, 19))

# Give Basin 19 a unique neutral color not used elsewhere
colors[-1] = np.array([0.60, 0.60, 0.60, 1.0])   # medium gray

cmap = mcolors.ListedColormap(colors)
cmap.set_bad(color='white')  # Set background (masked or NaN) to white

# Use min and max values for levels
vmin, vmax = 1,19 #1, 27
levels = np.linspace(vmin, vmax, vmax - vmin + 2)  # 27 basins + 1 for boundaries
norm = mcolors.BoundaryNorm(levels, cmap.N)

da = basins_imbie.isel(band=0)


# Plot
proj = ccrs.SouthPolarStereo()

# --- Pretty plot to sanity-check labels on the map ---
fig, ax = plt.subplots(figsize=(9, 10), subplot_kw={'projection': proj}, dpi=200)
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

im = da.plot(
    ax=ax,
    transform=proj,   # data are in SouthPolarStereo
    cmap=cmap,
    norm=norm,
    vmin=vmin,
    vmax=vmax,
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

# ids = sorted(id2name.keys())
# for bid in ids:
#     mask = (da == bid)
#     if mask.any():
#         yy, xx = np.where(mask.values)
#         if len(xx) == 0:
#             continue
#         cx = float(da['x'].values[xx].mean())
#         cy = float(da['y'].values[yy].mean())
#         ax.text(
#             cx, cy, id2name[bid],
#             fontsize=12, ha='center', va='center',
#             transform=proj, color='k',
#             bbox=dict(boxstyle="round,pad=0.2",
#                       fc="white", ec="none", alpha=0.6),
#         )

# --- choose which basins are "small" and need external labels ---
# adjust ids and offsets after a quick test render
small_basin_offsets = {
    11: (-4.0e5, -1.0e5),  # F-G  (example offsets, tweak!)
    # 12: (-2.0e5,  0.8e5),  # G-H
    13: (-4.8e5,  1.3e5),  # H-Hp
    14: (-8.4e5,  1.9e5),  # Hp-I
    15: (-4.8e5,  2.5e5),  # I-Ipp
    16: (-0.8e5,  3.7e5),  # Ipp-J
    17: (-0.5e5,  4.2e5),  # J-Jpp
    # add / tweak as needed
}

for bid in sorted(id2name.keys()):
    mask = (da == bid)
    if not mask.any():
        continue

    yy, xx = np.where(mask.values)
    if len(xx) == 0:
        continue

    cx = float(da['x'].values[xx].mean())
    cy = float(da['y'].values[yy].mean())
    label = id2name[bid]

    if bid in small_basin_offsets:
        # move label outside & draw leader line
        dx, dy = small_basin_offsets[bid]
        lx, ly = cx + dx, cy + dy

        ax.annotate(
            label,
            xy=(cx, cy),      # basin centroid (tail of line)
            xytext=(lx, ly),  # label position
            textcoords='data',
            xycoords='data',
            ha='center',
            va='center',
            fontsize=12,
            transform=proj,
            arrowprops=dict(
                arrowstyle="-",   # simple line, no arrow head
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
        # regular in-basin label
        ax.text(
            cx, cy, label,
            fontsize=12,
            ha='center',
            va='center',
            transform=proj,
            color='k',
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.6
            ),
        )
# Remove axis
ax.axis('off')

# Final cleanup
# ax.set_title("IMBIE Basins with IDs ", fontsize=18)
plt.tight_layout()
# plt.close()

# Save the imbie basin plot
# output_path = os.path.join(path_to_plots, 'imbie_basins_with_ids.png')
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
gc.collect()

#%% 1) Read David's LI-filled GRACE/altimetry basin storage anomaly data
#     and compute monthly ΔS [Gt/month] for each basin.
#
# Important:
# David's working GRACE/altimetry file provides monthly basin storage anomaly S [Gt].
# The LI gap filling was already applied before this pickle was saved.
#
# TEST CONVENTION USED HERE:
#     ΔS_m = S_{m+1} - S_m
#
# This is the previous/old workflow convention.
# The ΔS value is assigned to the starting month m.
#
# Example:
#     ΔS_Feb2013 = S_Mar2013 - S_Feb2013
#
# This differs from pandas .diff(), which gives:
#     ΔS_Mar2013 = S_Mar2013 - S_Feb2013
#
# The numerical value is the same, but the month label shifts by one month.
# This test checks whether the recent PMB differences are mainly due to this
# ΔS month-label convention.

rignot_storage = pd.read_pickle(
    os.path.join(basin_path, 
                 "DataCombo_RignotBasins_LI_tier1_GRACE_updated_20260609.pkl")
                 ) 
# "DataCombo_RignotBasins_LI_tier1_20260325.pkl"

# Ensure monthly datetime index
rignot_storage = rignot_storage.copy()
rignot_storage.index = (
    pd.to_datetime(rignot_storage.index)
    .to_period("M")
    .to_timestamp()
)

# Add year/month for grouping
rignot_storage["Year"] = rignot_storage.index.year
rignot_storage["Month"] = rignot_storage.index.month

# Identify basin columns only
basin_cols = [
    c for c in rignot_storage.columns
    if c not in ("Time", "Date", "Year", "Month")
]

# If multiple entries exist within a month, average them.
# For the LI-filled monthly pickle, this should usually be one value per basin/month.
dfm = (
    rignot_storage
    .groupby(["Year", "Month"], as_index=False)[basin_cols]
    .mean()
)

# Build monthly date index
dfm["date"] = pd.to_datetime(
    dict(year=dfm["Year"], month=dfm["Month"], day=1)
)

dfm = dfm.set_index("date").sort_index()

# Restrict to analysis window
dfm = dfm.loc[
    (dfm.index.year >= YEAR_START) &
    (dfm.index.year <= YEAR_END)
].copy()

# This is the LI-filled monthly storage anomaly series S [Gt]
S_li = dfm[basin_cols].copy()

print(
    f"[David] LI-filled storage anomaly loaded for months "
    f"{S_li.index.min().date()} to {S_li.index.max().date()}"
)
print(f"[David] Number of basins in storage file: {len(S_li.columns)}")


# =============================================================================
# Compute original monthly ΔS using the previous forward-difference convention
# =============================================================================
# ΔS_m = S_{m+1} - S_m
# Assigned to starting month m.
#
# Because Jan 2013 is all NaN in the current LI-filled file, the first usable
# ΔS month should become Feb 2013:
#     ΔS_Feb2013 = S_Mar2013 - S_Feb2013
# =============================================================================

S_next = S_li.shift(-1)

dS_original = S_next - S_li
dS_original = dS_original.dropna(how="all")

print(
    f"[David] Original forward ΔS computed for months "
    f"{dS_original.index.min().date()} to {dS_original.index.max().date()}"
)
print(f"[David] Number of original forward ΔS months: {len(dS_original.index)}")

# =============================================================================
# Use original LI-derived forward ΔS without climatology correction
# =============================================================================
# Old-method replication:
#     ΔS_m = S_{m+1} - S_m
# assigned to the starting month m.
#
# No flagged PMB months are used here.
# No ΔS monthly climatology replacement is applied.
# =============================================================================

dS_used = dS_original.copy()
deltaS_correction_log = pd.DataFrame()

GRACE_DELTAS_CORRECTION_VARIANT = "none"

print("\n[GRACE forward-ΔS] No climatology correction applied.")
print("[GRACE forward-ΔS] Using original LI-derived forward ΔS.")

# =============================================================================
# Prepare monthly ΔS uncertainty [Gt/month]
# =============================================================================
# This uncertainty is kept at basin-month level first. It will later be aligned
# with D, basal melt, ΔS, and sublimation in the PMB uncertainty propagation step.

if COMPUTE_PMB_UNCERTAINTY:

    dS_unc_used, dS_unc_long = prepare_rignot_deltaS_uncertainty(
    basin_cols=basin_cols,
    target_dates=dS_used.index,
    deltaS_convention="forward",
    filled_error_pickle=rignot_sigmaS_filled_pkl,
    )

    print("\n[PMB uncertainty] ΔS uncertainty prepared.")
    print("ΔS uncertainty date range:")
    print(dS_unc_used.index.min(), "to", dS_unc_used.index.max())
    print("ΔS uncertainty shape:", dS_unc_used.shape)
    print("Mean ΔS uncertainty [Gt/month]:", np.nanmean(dS_unc_used.values))

else:
    dS_unc_used = None
    dS_unc_long = pd.DataFrame()

# =============================================================================
# Convert active ΔS to long dataframe for basin-raster painting
# =============================================================================

dS_long = (
    dS_used
    .reset_index(names="date")
    .melt(id_vars="date", var_name="basin", value_name="dS_Gt")
    .dropna(subset=["dS_Gt"])
    .reset_index(drop=True)
)

print(
    f"[David] Active forward ΔS computed for months "
    f"{dS_long['date'].min().date()} to {dS_long['date'].max().date()}"
)

print(f"[David] Number of active forward ΔS rows: {len(dS_long)}")

#%%
# =============================================================================
# CREATE ΔS RASTER FROM ACTIVE FORWARD-DIFFERENCE ΔS DATAFRAME
# =============================================================================
# dS_long comes from the active ΔS series:
#     ΔS_m = S_{m+1} - S_m
#
# The ΔS value is assigned to the starting month m.
# Example:
#     ΔS_Feb2013 = S_Mar2013 - S_Feb2013
# =============================================================================

# -----------------------------------------------------------------------------
# Recover basin-id and basin-name DataArrays safely
# -----------------------------------------------------------------------------
# Important:
# Do not use the variable name "basin_id" for loop counters elsewhere.
# It can overwrite the basin DataArray and cause:
#     AttributeError: 'int' object has no attribute 'where'
# -----------------------------------------------------------------------------

basin_id_da = basin_imbie_with_name_map["basin_id"].copy()
basin_name_da = basin_imbie_with_name_map["basin_name"].copy()

# Mask out non-basin pixels
basin_id_da = basin_id_da.where(np.isfinite(basin_id_da))
basin_name_da = basin_name_da.where(basin_name_da != "NA")

# Exclude islands if ID == 1
basin_id_da = basin_id_da.where(basin_id_da != 1)
basin_name_da = basin_name_da.where(basin_id_da.notnull())

# Carry the basin_id and basin_name into the dataframe for mapping
dS_long_df = generate_basin_id_mapping(
    basin_id_da,
    basin_name_da,
    dS_long
)

# =============================================================================
# Build time-by-space ΔS raster
# =============================================================================

attributes = {
    "description": (
        "Monthly basin-total ΔS painted to all pixels of each basin "
        "(not areal density)."
    ),
    "long_name": "Monthly basin mass anomaly change",
    "units": "Gt/month",
    "source": (
        "Computed from David's GRACE/altimetry Rignot-basin storage anomaly "
        "time series after LI gap filling."
    ),
    "deltaS_convention": "forward_difference_S_next_minus_S_current",
    "note": (
        "ΔS_m = S_{m+1} - S_m, assigned to the starting month m. "
        "For example, ΔS_Feb2013 = S_Mar2013 - S_Feb2013. "
        "Islands (ID=1) excluded; basin names matched via modal name per ID "
        "from the basin grid."
    ),
}

attributes["correction"] = "No GRACE-derived ΔS climatology correction applied."
attributes["correction_variant"] = "none"

dS_raster = create_basin_xrr(
    basin_id_da[0],
    basin_name_da[0],
    dS_long_df,
    "dS_Gt",
    attributes,
    "deltaS_Gt_per_month",
)

# Quick plot/check of first timestamp
tms_plt = dS_raster["date"][0]

dS_raster.isel(date=0).plot(
    cmap="jet",
    vmin=-10,
    vmax=10,
    cbar_kwargs={"label": "ΔS [Gt/month]"},
)

plt.title(f"ΔS for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")

print("First ΔS raster month:", pd.to_datetime(dS_raster["date"].values[0]).date())
print("Last ΔS raster month :", pd.to_datetime(dS_raster["date"].values[-1]).date())
print("Number of ΔS raster months:", dS_raster.sizes["date"])

vals = dS_raster["basin_name"].values.ravel()
non_nan = vals[pd.notna(vals)]
print("Unique basin names in ΔS raster:")
print(np.unique(non_nan))

# =============================================================================
# Save to disk
# =============================================================================

correction_tag = "forward_deltaS_uncorrected"

out_flnme = os.path.join(
    basin_path,
    (
        f"rignot_deltaS_monthly_{YEAR_START}_{YEAR_END}_"
        f"LI_gap_filled_GRACE_tier1_{correction_tag}_GRACE_updated_{cde_run_dte}.nc"
    )
)

dS_raster.to_netcdf(out_flnme)

print(f"[David] ΔS raster saved to: {out_flnme}")

# =============================================================================
# CREATE ΔS UNCERTAINTY RASTER FROM RIGNOT 1-SIGMA ERROR TABLE
# =============================================================================

if COMPUTE_PMB_UNCERTAINTY:

    dS_unc_long_df = generate_basin_id_mapping(
        basin_id_da,
        basin_name_da,
        dS_unc_long,
    )

    unc_attributes = {
        "description": (
            "Monthly basin-level 1-sigma ΔS uncertainty painted to all pixels "
            "of each basin. Values are basin-total uncertainty, not areal density."
        ),
        "long_name": "Monthly basin mass-change uncertainty",
        "units": "Gt/month",
        "source": (
            "David/Rignot basin 1-sigma error table from "
            "DataCombo_RignotBasins.xlsx, sheet '1-sigma_Error(Gt)'."
        ),
        "deltaS_convention": "forward_difference_S_next_minus_S_current",
        "note": (
            "This uncertainty is used as sigma_deltaS in PMB uncertainty "
            "propagation. Islands (ID=1) excluded."
        ),
    }

    dS_unc_raster = create_basin_xrr(
        basin_id_da[0],
        basin_name_da[0],
        dS_unc_long_df,
        "dS_unc_Gt",
        unc_attributes,
        "deltaS_uncertainty_Gt_per_month",
    )

    dS_unc_outfile = os.path.join(
        basin_path,
        (
            f"rignot_deltaS_uncertainty_monthly_{YEAR_START}_{YEAR_END}_"
            f"LI_gap_filled_GRACE_tier1_{correction_tag}_GRACE_updated_{cde_run_dte}.nc"
        )
    )

    dS_unc_raster.to_netcdf(dS_unc_outfile)

    print("\n[PMB uncertainty] ΔS uncertainty raster saved:")
    print(dS_unc_outfile)

    print("First ΔS uncertainty raster month:", pd.to_datetime(dS_unc_raster["date"].values[0]).date())
    print("Last ΔS uncertainty raster month :", pd.to_datetime(dS_unc_raster["date"].values[-1]).date())
    print("Number of ΔS uncertainty months:", dS_unc_raster.sizes["date"])

else:
    dS_unc_raster = None

# =============================================================================
# Recover basin-id and basin-name DataArrays safely
# =============================================================================
# This protects against accidental overwriting of "basin_id" as an integer
# from plotting loops such as: for basin_id in range(...)

basin_id_da = basin_imbie_with_name_map["basin_id"].copy()
basin_name_da = basin_imbie_with_name_map["basin_name"].copy()

# Mask invalid pixels
basin_id_da = basin_id_da.where(np.isfinite(basin_id_da))
basin_name_da = basin_name_da.where(basin_name_da != "NA")

# Exclude Islands if ID == 1
basin_id_da = basin_id_da.where(basin_id_da != 1)
basin_name_da = basin_name_da.where(basin_id_da.notnull())

print("basin_id_da type:", type(basin_id_da))
print("basin_name_da type:", type(basin_name_da))
print("basin_id_da dims:", basin_id_da.dims)
#%% 2) Read Chad's discharge & basal melt (annual Gt/yr) and spread to months

discharge_data = pd.read_excel(
    os.path.join(basin_path, "antarctic_discharge_2013-2022_imbie.xlsx"),
    sheet_name=[
        "Discharge (Gt yr^-1)",
        "Discharge Error (Gt yr^-1)",
        "Summary",
    ]
)

basin_discharge = discharge_data["Discharge (Gt yr^-1)"].copy()
basin_discharge_err = discharge_data["Discharge Error (Gt yr^-1)"].copy()

# Chad's file appears to use Ep-F, while David's GRACE file may use Ep-f.
# For Chad's discharge/basal melt table, use Chad-style basin names.
basin_cols_chad = basin_cols.copy()
basin_cols_chad = [
    b.replace("Ep-f", "Ep-F") if b == "Ep-f" else b
    for b in basin_cols_chad
]

basin_discharge = (
    basin_discharge
    .loc[
        basin_discharge["IMBIE basin"].isin(basin_cols_chad),
        ["IMBIE basin"] + [str(yr) for yr in YEARS]
    ]
    .rename(columns={"IMBIE basin": "basin"})
    .copy()
)

# =============================================================================
# Prepare Chad discharge uncertainty table
# =============================================================================
# Chad's discharge-error sheet contains annual 1-sigma discharge uncertainty
# in Gt/yr for each IMBIE basin.
#
# For monthly PMB uncertainty, annual uncertainty is distributed to monthly
# uncertainty as:
#
#     sigma_D_month = sigma_D_annual / 12
#
# This assumes the annual discharge uncertainty is spread uniformly across months,
# consistent with how annual discharge itself is converted to monthly discharge.
# =============================================================================

basin_discharge_err = (
    basin_discharge_err
    .loc[
        basin_discharge_err["IMBIE basin"].isin(basin_cols_chad),
        ["IMBIE basin"] + [str(yr) for yr in YEARS]
    ]
    .rename(columns={"IMBIE basin": "basin"})
    .copy()
)

# Convert annual discharge uncertainty [Gt/yr] to monthly uncertainty [Gt/month].
# Note: annual_to_monthly_long() already divides annual values by 12.
D_unc_month = annual_to_monthly_long(
    basin_discharge_err,
    YEARS,
    "discharge_unc_Gt"
)
# Map basin names to basin IDs
D_unc_month_df = generate_basin_id_mapping(
    basin_id_da,
    basin_name_da,
    D_unc_month
)

print("[Chad] Discharge uncertainty monthly rows:", len(D_unc_month))
print("[Chad] Mean monthly discharge uncertainty [Gt/month]:", D_unc_month["discharge_unc_Gt"].mean())

# --- Basal melt ---
basal_melt = discharge_data["Summary"].copy()

basal_melt = (
    basal_melt
    .loc[
        basal_melt["IMBIE basin"].isin(basin_cols_chad),
        ["IMBIE basin", "Basal melt total Gt/yr"]
    ]
    .rename(columns={"IMBIE basin": "basin"})
    .copy()
)

# Convert Chad-style Ep-F back to basin-mask / basin-name style if needed.
# Your basin map/name dictionary appears to use Ep-F, so this may not change anything.
# Keep this only if generate_basin_id_mapping expects map names such as Ep-F.
# If your basin_name_da uses Ep-f instead, switch this mapping accordingly.
# basin_discharge["basin"] = basin_discharge["basin"].replace({"Ep-F": "Ep-f"})
# basal_melt["basin"] = basal_melt["basin"].replace({"Ep-F": "Ep-f"})

# Convert annual discharge (Gt/yr) → monthly discharge (Gt/month)
D_month = annual_to_monthly_long(
    basin_discharge,
    YEARS,
    "discharge_Gt"
)

# Map basin names to basin IDs using protected DataArrays
D_month_df = generate_basin_id_mapping(
    basin_id_da,
    basin_name_da,
    D_month
)

# Convert basal melt annual total to monthly
basal_melt["basal_Gt_per_month"] = (
    basal_melt["Basal melt total Gt/yr"] / 12.0
)

# Expand basal melt to monthly values
rows = []

for _, row in basal_melt.iterrows():
    for y in YEARS:
        for m in range(1, 13):
            rows.append({
                "date": pd.Timestamp(year=int(y), month=int(m), day=1),
                "basin": row["basin"],
                "basal_Gt": row["basal_Gt_per_month"],
            })

B_month = pd.DataFrame(rows)

B_month_df = generate_basin_id_mapping(
    basin_id_da,
    basin_name_da,
    B_month
)

Q_month = D_month.merge(
    B_month,
    on=["date", "basin"],
    how="outer"
)

print(f"[Chad] Qnet monthly rows: {len(Q_month)}")
print("[Chad] Discharge monthly rows:", len(D_month))
print("[Chad] Basal melt monthly rows:", len(B_month))


# =============================================================================
# Create discharge raster
# =============================================================================

attributes = {
    "description": (
        "Monthly basin-total discharge painted to all pixels of each basin "
        "(not areal density)."
    ),
    "long_name": "Monthly basin discharge",
    "units": "Gt/month",
    "source": "Computed from Chad's discharge Excel (antarctic_discharge_2013-2022_imbie.xlsx)",
    "note": (
        "Annual discharge values divided by 12. Islands (ID=1) excluded; "
        "basin names matched via modal name per ID from the basin grid."
    ),
}

discharge_raster = create_basin_xrr(
    basin_id_da[0],
    basin_name_da[0],
    D_month_df,
    "discharge_Gt",
    attributes,
    "discharge_Gt_per_month",
)

# =============================================================================
# Create discharge uncertainty raster
# =============================================================================

if COMPUTE_PMB_UNCERTAINTY:

    attributes = {
        "description": (
            "Monthly basin-total 1-sigma discharge uncertainty painted to all "
            "pixels of each basin. Values are basin-total uncertainty, not areal density."
        ),
        "long_name": "Monthly basin discharge uncertainty",
        "units": "Gt/month",
        "source": (
            "Computed from Chad's discharge Excel "
            "(antarctic_discharge_2013-2022_imbie.xlsx), sheet "
            "'Discharge Error (Gt yr^-1)'."
        ),
        "note": (
            "Annual discharge uncertainty values divided by 12 to match the "
            "monthly discharge time step. Islands (ID=1) excluded; basin names "
            "matched via modal name per ID from the basin grid."
        ),
    }

    discharge_unc_raster = create_basin_xrr(
        basin_id_da[0],
        basin_name_da[0],
        D_unc_month_df,
        "discharge_unc_Gt",
        attributes,
        "discharge_uncertainty_Gt_per_month",
    )

    discharge_unc_outfile = os.path.join(
        basin_path,
        (
            f"discharge_uncertainty_monthly_{YEAR_START}_{YEAR_END}_"
            f"Gt_per_month_{cde_run_dte}.nc"
        )
    )

    discharge_unc_raster.to_netcdf(discharge_unc_outfile)

    print("\n[PMB uncertainty] Discharge uncertainty raster saved:")
    print(discharge_unc_outfile)

else:
    discharge_unc_raster = None


# =============================================================================
# Create basal melt raster
# =============================================================================

attributes = {
    "description": (
        "Monthly basin-total basal melt painted to all pixels of each basin "
        "(not areal density)."
    ),
    "long_name": "Monthly basin basal melt",
    "units": "Gt/month",
    "source": "Computed from Chad's discharge Excel (antarctic_discharge_2013-2022_imbie.xlsx)",
    "note": (
        "Annual basal melt values divided by 12. Islands (ID=1) excluded; "
        "basin names matched via modal name per ID from the basin grid."
    ),
}

basal_melt_raster = create_basin_xrr(
    basin_id_da[0],
    basin_name_da[0],
    B_month_df,
    "basal_Gt",
    attributes,
    "basal_Gt_per_month",
)


# =============================================================================
# Quick plots
# =============================================================================

tms_plt = discharge_raster["date"][0]

discharge_raster.isel(date=0).plot(
    cmap="jet",
    vmin=0,
    vmax=30,
    cbar_kwargs={"label": "Discharge [Gt/month]"},
)
plt.title(f"Discharge for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")
plt.show()

if discharge_unc_raster is not None:
    discharge_unc_raster.isel(date=0).plot(
        cmap="jet",
        vmin=0,
        vmax=0.5,
        cbar_kwargs={"label": "Discharge uncertainty [Gt/month]"},
    )
    plt.title(f"Discharge uncertainty for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")
    plt.show()

basal_melt_raster.isel(date=0).plot(
    cmap="jet",
    vmin=0,
    vmax=0.5,
    cbar_kwargs={"label": "Basal melt [Gt/month]"},
)
plt.title(f"Basal Melt for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")
plt.show()


#%% 3) RACMO: regrid subltot to IMBIE grid and save as positive sublimation loss
# =============================================================================
# Purpose:
# Regrid RACMO2.4p1 monthly snowdrift sublimation field to the IMBIE basin grid.
#
# Important sign convention:
# Behrangi et al. use sublimation as a positive loss term in:
#
#     P_MB = D + basal melt + ΔS + sublimation
#
# RACMO subltot may be stored using the model convention where sublimation
# mass loss is negative. Therefore, after regridding, we convert the field to
# positive sublimation loss ONCE and save it that way.
#
# Downstream PMB code should then use the saved field directly, with no extra
# sign flip.
# =============================================================================

racmo_sublim_file = os.path.join(
    racmo_path,
    "subltot_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc"
)

# -----------------------------------------------------------------------------
# 1. Open and subset RACMO to analysis window
# -----------------------------------------------------------------------------

da_src = xr.open_dataset(racmo_sublim_file)["subltot"].sel(
    time=slice("2013-01-01", "2022-12-31")
)

# RACMO file has height dimension of size 1. Remove it safely.
if "height" in da_src.dims:
    da_src = da_src.squeeze("height", drop=True)

print("Raw RACMO subltot:")
print(da_src)
print("Raw units:", da_src.attrs.get("units", "unknown"))

# Pull 2D lon/lat from RACMO curvilinear grid
lon2d = da_src["lon"].values
lat2d = da_src["lat"].values

# Safety: mask impossible lon/lat
bad = ~np.isfinite(lon2d) | ~np.isfinite(lat2d)

if bad.any():
    lon2d = lon2d.copy()
    lat2d = lat2d.copy()
    lon2d[bad] = np.nan
    lat2d[bad] = np.nan


# -----------------------------------------------------------------------------
# 2. Build target IMBIE x/y coordinates from basin transform/shape
# -----------------------------------------------------------------------------

# These variables should already exist from the basin-grid setup:
#     width, height, xmin, ymax, xres, yres, basin_transform, crs_stereo

jj = np.arange(width)
ii = np.arange(height)

x_target = xmin + (jj + 0.5) * xres
y_target = ymax - (ii + 0.5) * abs(yres)

X, Y = np.meshgrid(x_target, y_target)  # shape: (height, width)


# -----------------------------------------------------------------------------
# 3. Convert target polar stereographic x/y to lon/lat for interpolation
# -----------------------------------------------------------------------------

crs_out = CRS.from_proj4(crs_stereo)
transform_to_geo = Transformer.from_crs(
    crs_out,
    CRS.from_epsg(4326),
    always_xy=True,
)

lon_tgt, lat_tgt = transform_to_geo.transform(X, Y)


# -----------------------------------------------------------------------------
# 4. Prepare source and target interpolation points
# -----------------------------------------------------------------------------

pts_src = np.column_stack([lon2d.ravel(), lat2d.ravel()])
mask_src = np.isfinite(pts_src).all(axis=1)
pts_src = pts_src[mask_src]

tgt_points = np.column_stack([lon_tgt.ravel(), lat_tgt.ravel()])


# -----------------------------------------------------------------------------
# 5. Domain-overlap sanity check
# -----------------------------------------------------------------------------

src_lon_min, src_lon_max = np.nanmin(lon2d), np.nanmax(lon2d)
src_lat_min, src_lat_max = np.nanmin(lat2d), np.nanmax(lat2d)

tgt_lon_min, tgt_lon_max = np.nanmin(lon_tgt), np.nanmax(lon_tgt)
tgt_lat_min, tgt_lat_max = np.nanmin(lat_tgt), np.nanmax(lat_tgt)

overlap_lon = (tgt_lon_min <= src_lon_max) and (tgt_lon_max >= src_lon_min)
overlap_lat = (tgt_lat_min <= src_lat_max) and (tgt_lat_max >= src_lat_min)

if not (overlap_lon and overlap_lat):
    print(
        "WARNING: target grid is outside the RACMO domain in lon/lat. "
        "Interpolation may return all NaN."
    )


# -----------------------------------------------------------------------------
# 6. Interpolate each time slice to IMBIE grid
# -----------------------------------------------------------------------------

out = np.full(
    (da_src.sizes["time"], height, width),
    np.nan,
    dtype=np.float32,
)

for tt in range(da_src.sizes["time"]):

    v = da_src.isel(time=tt).values.astype(np.float64)
    v_flat = v.ravel()[mask_src]

    # Linear interpolation
    interp_lin = griddata(
        pts_src,
        v_flat,
        tgt_points,
        method="linear",
    )

    # Nearest-neighbor fill for edge holes
    nan_mask = ~np.isfinite(interp_lin)

    if nan_mask.any():
        interp_nn = griddata(
            pts_src,
            v_flat,
            tgt_points[nan_mask],
            method="nearest",
        )
        interp_lin[nan_mask] = interp_nn

    out[tt, :, :] = interp_lin.reshape(height, width).astype(np.float32)


# -----------------------------------------------------------------------------
# 7. Wrap into xarray DataArray on IMBIE grid
# -----------------------------------------------------------------------------

racmo_on_imbie_raw = xr.DataArray(
    out,
    name="subltot_raw",
    dims=("time", "y", "x"),
    coords={
        "time": da_src["time"].values,
        "x": x_target,
        "y": y_target,
    },
    attrs=da_src.attrs,
)

racmo_on_imbie_raw = racmo_on_imbie_raw.rio.write_crs(
    CRS.from_proj4(crs_stereo).to_wkt(),
    inplace=False,
)

racmo_on_imbie_raw = racmo_on_imbie_raw.rio.write_transform(
    basin_transform,
    inplace=False,
)


# -----------------------------------------------------------------------------
# 8. Normalize time stamps to month-start
# -----------------------------------------------------------------------------
# RACMO monthly timestamps are usually mid-month, e.g., 2013-01-16.
# Convert to clean month-start dates so they align with D, BM, and ΔS.

racmo_on_imbie_raw = racmo_on_imbie_raw.assign_coords(
    time=pd.to_datetime(racmo_on_imbie_raw["time"].values)
    .to_period("M")
    .to_timestamp(how="start")
)


# -----------------------------------------------------------------------------
# 9. Convert RACMO subltot to positive sublimation loss
# -----------------------------------------------------------------------------
# If RACMO subltot is negative on average, interpret negative as mass loss and
# flip sign. If it is already positive on average, keep it unchanged.

raw_mean = float(racmo_on_imbie_raw.mean(skipna=True).values)
raw_min = float(racmo_on_imbie_raw.min(skipna=True).values)
raw_max = float(racmo_on_imbie_raw.max(skipna=True).values)

print("\nRACMO subltot raw regridded statistics:")
print("mean:", raw_mean)
print("min :", raw_min)
print("max :", raw_max)

if raw_mean < 0:
    print(
        "\nRACMO subltot is negative on average. "
        "Flipping sign so positive values represent sublimation loss."
    )
    racmo_on_imbie = -racmo_on_imbie_raw
else:
    print(
        "\nRACMO subltot is positive on average. "
        "Keeping sign as positive sublimation loss."
    )
    racmo_on_imbie = racmo_on_imbie_raw.copy()

racmo_on_imbie.name = "sublimation_loss"

racmo_on_imbie.attrs.update({
    "long_name": "Snowdrift sublimation loss",
    "units": "kg m-2",
    "source_variable": "RACMO2.4p1 subltot",
    "sign_convention": (
        "Positive values represent sublimation mass loss. "
        "This field is intended to be added as a positive loss term in the "
        "PMB equation: P_MB = D + basal melt + ΔS + sublimation."
    ),
    "time_note": (
        "Original RACMO monthly timestamps were normalized to month-start dates."
    ),
})


# -----------------------------------------------------------------------------
# 10. Clean metadata before NetCDF writing
# -----------------------------------------------------------------------------

da = racmo_on_imbie.copy()

# Remove problematic CF-encoding attributes from data variable
for key in ("_FillValue", "grid_mapping", "scale_factor", "add_offset"):
    if key in da.attrs:
        da.attrs.pop(key)

# Remove problematic encoding attrs from coordinates
for c in list(da.coords):
    for key in ("_FillValue", "grid_mapping", "scale_factor", "add_offset"):
        if key in da[c].attrs:
            da[c].attrs.pop(key)

# Drop unused scalar coord if present
if "rotated_pole" in da.coords and da["rotated_pole"].ndim == 0:
    da = da.drop_vars("rotated_pole")

da = da.astype("float32")

var_name = da.name or "sublimation_loss"

encoding = {
    var_name: {
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(np.nan),
        "dtype": "float32",
    }
}


# -----------------------------------------------------------------------------
# 11. Save positive sublimation-loss field
# -----------------------------------------------------------------------------
# fnme = f"sublimation_loss_positive_monthlyS_ANT11_RACMO2.4p1_ERA5_2013_2022_{cde_run_dte}.nc"
fnme = "sublimation_loss_positive_monthlyS_ANT11_RACMO2.4p1_ERA5_2013_2022_20260507.nc"

out_nc = os.path.join(
    racmo_path,
    fnme
)

da.to_netcdf(out_nc, encoding=encoding)

print("\nSaved positive sublimation-loss field:")
print(out_nc)


# -----------------------------------------------------------------------------
# 12. Reload saved field for downstream PMB calculation
# -----------------------------------------------------------------------------

racmo_on_imbie = xr.open_dataarray(out_nc)

print("\nReloaded RACMO sublimation-loss field:")
print(racmo_on_imbie)
print("Reloaded mean:", float(racmo_on_imbie.mean(skipna=True).values))
print("Reloaded min :", float(racmo_on_imbie.min(skipna=True).values))
print("Reloaded max :", float(racmo_on_imbie.max(skipna=True).values))
#%% COMPUTE PMB USING POSITIVE SUBLIMATION LOSS
# =============================================================================
# Compute basin series for each mass-budget component:
#
#     P_MB = D + BM + ΔS + SUB
#
# where:
#     D   = discharge, Gt/month
#     BM  = basal melt, Gt/month
#     ΔS  = storage change, Gt/month
#     SUB = positive sublimation loss, Gt/month
#
# IMPORTANT:
# SUB is already positive-loss because the RACMO preprocessing section converted
# the sign once and saved "sublimation_loss_positive...nc".
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Time align
# -----------------------------------------------------------------------------

subl = racmo_on_imbie.rename(time="date")

subl = subl.assign_coords(
    date=pd.DatetimeIndex(subl.date.values)
    .to_period("M")
    .to_timestamp(how="start")
)

common_dates = np.intersect1d(
    discharge_raster.date,
    np.intersect1d(basal_melt_raster.date, subl.date)
)

common_dates = np.intersect1d(common_dates, dS_raster.date)

D  = discharge_raster.sel(date=common_dates)
BM = basal_melt_raster.sel(date=common_dates)
DS = dS_raster.sel(date=common_dates)
SUB = subl.sel(date=common_dates)

print("\nCommon budget dates:")
print(pd.to_datetime(common_dates).min(), "to", pd.to_datetime(common_dates).max())
print("Number of common dates:", len(common_dates))


# -----------------------------------------------------------------------------
# 2. Basin labels / mask
# -----------------------------------------------------------------------------

basin_labels = D["basin_id"]

SUBm = mask_to_basins(SUB, basin_labels)

tms_plt = SUBm["date"][0]

SUBm.isel(date=0).plot(
    cmap="jet",
    vmin=0,
    vmax=5,
    cbar_kwargs={"label": "Positive sublimation loss [kg m$^{-2}$]"},
)
plt.title(f"Positive sublimation loss for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")
plt.show()

print("finite basin pixels:", int(np.isfinite(basin_labels).sum()))
print("finite SUB pixels after mask, first month:", int(np.isfinite(SUBm.isel(date=0)).sum()))

print("\nMasked SUB statistics [kg m-2]:")
print("mean:", float(SUBm.mean(skipna=True).values))
print("min :", float(SUBm.min(skipna=True).values))
print("max :", float(SUBm.max(skipna=True).values))


# -----------------------------------------------------------------------------
# 3. Basin area
# -----------------------------------------------------------------------------

cell_area = xr.full_like(basin_labels, 10000.0**2)  # m²
cell_area = cell_area.where(np.isfinite(basin_labels))

basin_area_m2 = cell_area.groupby(
    basin_labels.rename("basin_id")
).sum()

print("\nBasin area dims:", basin_area_m2.dims)
print("Basin IDs in area array:", basin_area_m2["basin_id"].values)


# -----------------------------------------------------------------------------
# 4. Ensure SUB carries basin_id coordinate
# -----------------------------------------------------------------------------

if "basin_id" not in SUBm.coords:
    SUBm = SUBm.assign_coords(basin_id=basin_labels)


# -----------------------------------------------------------------------------
# 5. Basin series for each component
# -----------------------------------------------------------------------------

D_basin  = painted_to_basin_series(D)      # Gt/month
BM_basin = painted_to_basin_series(BM)     # Gt/month
dS_basin = painted_to_basin_series(DS)     # Gt/month

SUB_basin = areal_to_basin_series(
    SUBm,
    pixel_area_m2=10000.0 * 10000.0,
    units="kg m-2",
)
SUB_basin.name = "subl_Gt_per_month"


# -----------------------------------------------------------------------------
# 6. Align all basin components
# -----------------------------------------------------------------------------
# Use inner join so PMB is computed only where all terms exist.

D_basin, BM_basin, dS_basin, SUB_basin = xr.align(
    D_basin,
    BM_basin,
    dS_basin,
    SUB_basin,
    join="inner",
)

print("\nAligned component shapes:")
print("D   :", D_basin.shape)
print("BM  :", BM_basin.shape)
print("dS  :", dS_basin.shape)
print("SUB :", SUB_basin.shape)

print("\nAligned PMB dates:")
print(
    pd.to_datetime(D_basin["date"].values).min(),
    "to",
    pd.to_datetime(D_basin["date"].values).max(),
)
print("Number of aligned dates:", D_basin.sizes["date"])


# -----------------------------------------------------------------------------
# 7. Convert sublimation to mm/month for diagnostics
# -----------------------------------------------------------------------------

SUB_basin_mm = SUB_basin * 1e12 / basin_area_m2
SUB_basin_mm.name = "subl_mm_per_month"

SUB_basin_mm_annual = SUB_basin_mm.groupby("date.year").sum("date")

print("\nSublimation-loss annual diagnostic [mm/yr]:")
print("mean:", float(SUB_basin_mm_annual.mean(skipna=True).values))
print("min basin mean:", float(SUB_basin_mm_annual.mean("year", skipna=True).min(skipna=True).values))
print("max basin mean:", float(SUB_basin_mm_annual.mean("year", skipna=True).max(skipna=True).values))


# -----------------------------------------------------------------------------
# 8. Compute PMB
# -----------------------------------------------------------------------------

Precip_basin = D_basin + BM_basin + dS_basin + SUB_basin
Precip_basin.name = "precip_Gt_per_month"

Precip_basin_mm = Precip_basin * 1e12 / basin_area_m2
Precip_basin_mm.name = "precip_mm_per_month"

print("\nPMB basin monthly diagnostic [mm/month]:")
print("mean:", float(Precip_basin_mm.mean(skipna=True).values))
print("min :", float(Precip_basin_mm.min(skipna=True).values))
print("max :", float(Precip_basin_mm.max(skipna=True).values))

# -----------------------------------------------------------------------------
# 8b. Sublimation contribution to PMB in mm/year
# -----------------------------------------------------------------------------
# Goal:
#   Report regional mean annual sublimation and PMB in mm/yr, plus the
#   sublimation fraction relative to PMB.
#
# Important:
#   We compute area-weighted regional means, not simple basin averages.
#   This keeps the mm/yr ratios consistent with the Gt-based mass ratios.

REGION_BASINS = {
    "Antarctica": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "West Antarctica": [10, 11, 12, 13, 14, 15, 16, 17],
    "East Antarctica": [2, 3, 4, 5, 6, 7, 8, 9, 18, 19],
}

# Convert sublimation from Gt/month to mm/month
SUB_basin_mm = SUB_basin * 1e12 / basin_area_m2
SUB_basin_mm.name = "sublimation_mm_per_month"

sublimation_fraction_rows = []

for region_name, basin_ids in REGION_BASINS.items():

    # Select region
    sub_mm_reg = SUB_basin_mm.sel(basin_id=basin_ids)
    pmb_mm_reg = Precip_basin_mm.sel(basin_id=basin_ids)
    area_reg = basin_area_m2.sel(basin_id=basin_ids)

    # Monthly area-weighted regional mean in mm/month
    sub_reg_mm_month = sub_mm_reg.weighted(area_reg).mean("basin_id", skipna=True)
    pmb_reg_mm_month = pmb_mm_reg.weighted(area_reg).mean("basin_id", skipna=True)

    # Annual regional totals in mm/year
    sub_reg_mm_year = sub_reg_mm_month.groupby("date.year").sum("date", skipna=True)
    pmb_reg_mm_year = pmb_reg_mm_month.groupby("date.year").sum("date", skipna=True)

    # Mean annual mm/year over 2013–2020
    mean_sub_mm_yr = sub_reg_mm_year.mean("year", skipna=True)
    mean_pmb_mm_yr = pmb_reg_mm_year.mean("year", skipna=True)

    # Fraction based on mean annual mm/year
    ratio_mean_mm = 100.0 * mean_sub_mm_yr / mean_pmb_mm_yr

    # Annual fraction range
    ratio_ann = 100.0 * sub_reg_mm_year / pmb_reg_mm_year

    sublimation_fraction_rows.append({
        "region": region_name,
        "mean_annual_sublimation_mm_yr": float(mean_sub_mm_yr.values),
        "mean_annual_PMB_mm_yr": float(mean_pmb_mm_yr.values),
        "sublimation_fraction_mean_percent": float(ratio_mean_mm.values),
        "sublimation_fraction_annual_min_percent": float(ratio_ann.min("year", skipna=True).values),
        "sublimation_fraction_annual_max_percent": float(ratio_ann.max("year", skipna=True).values),
    })

df_sublimation_fraction_mm = pd.DataFrame(sublimation_fraction_rows)

print("\nSublimation contribution to PMB using area-weighted mm/year:")
print(df_sublimation_fraction_mm)

out_csv = os.path.join(
    path_to_plots,
    f"sublimation_fraction_of_PMB_by_region_mm_yr_{YEAR_START}_{YEAR_END}_{cde_run_dte}.csv"
)
df_sublimation_fraction_mm.to_csv(out_csv, index=False)

print("\nSaved sublimation fraction diagnostic in mm/year:")
print(out_csv)

# -----------------------------------------------------------------------------
# 8b. Compute PMB uncertainty
# -----------------------------------------------------------------------------
# PMB uncertainty is propagated at basin-month level:
#
#     sigma_PMB = sqrt(
#         sigma_D^2 +
#         sigma_BM^2 +
#         sigma_dS^2 +
#         sigma_SUB^2
#     )
#
# Current implemented uncertainty inputs:
#     sigma_D  : Chad discharge uncertainty from "Discharge Error (Gt yr^-1)"
#     sigma_dS : David/Rignot 1-sigma error table
#
# Optional assumption-based terms:
#     sigma_BM  = abs(BM)  * BASAL_MELT_FRAC_UNC
#     sigma_SUB = abs(SUB) * SUBLIMATION_FRAC_UNC
# -----------------------------------------------------------------------------

if COMPUTE_PMB_UNCERTAINTY:

    if dS_unc_raster is None:
        raise RuntimeError(
            "COMPUTE_PMB_UNCERTAINTY=True, but dS_unc_raster was not created."
        )

    if discharge_unc_raster is None:
        raise RuntimeError(
            "COMPUTE_PMB_UNCERTAINTY=True, but discharge_unc_raster was not created."
        )

    # Align uncertainty rasters to the same dates used in PMB calculation
    DS_unc = dS_unc_raster.sel(date=common_dates)
    D_unc = discharge_unc_raster.sel(date=common_dates)

    # Convert painted uncertainty rasters to basin series [Gt/month]
    dS_unc_basin = painted_to_basin_series(DS_unc)
    dS_unc_basin.name = "deltaS_uncertainty_Gt_per_month"

    D_unc_basin = painted_to_basin_series(D_unc)
    D_unc_basin.name = "discharge_uncertainty_Gt_per_month"

    # Align all component and uncertainty series
    D_unc_basin, dS_unc_basin, D_basin, BM_basin, dS_basin, SUB_basin = xr.align(
        D_unc_basin,
        dS_unc_basin,
        D_basin,
        BM_basin,
        dS_basin,
        SUB_basin,
        join="inner",
    )

    # Optional uncertainty assumptions for terms without direct error files
    BM_unc_basin = np.abs(BM_basin) * BASAL_MELT_FRAC_UNC
    SUB_unc_basin = np.abs(SUB_basin) * SUBLIMATION_FRAC_UNC

    BM_unc_basin.name = "basal_melt_uncertainty_Gt_per_month"
    SUB_unc_basin.name = "sublimation_uncertainty_Gt_per_month"

    # Propagate uncertainty in quadrature
    Pmb_unc_basin_Gt = np.sqrt(
        D_unc_basin**2 +
        dS_unc_basin**2 +
        BM_unc_basin**2 +
        SUB_unc_basin**2
    )

    Pmb_unc_basin_Gt.name = "P_MB_uncertainty_Gt_per_month"

    # Convert PMB uncertainty from Gt/month to mm/month
    Pmb_unc_basin_mm = Pmb_unc_basin_Gt * 1e12 / basin_area_m2
    Pmb_unc_basin_mm.name = "P_MB_uncertainty_mm_per_month"

    print("\nPMB uncertainty diagnostic [Gt/month]:")
    print("mean:", float(Pmb_unc_basin_Gt.mean(skipna=True).values))
    print("min :", float(Pmb_unc_basin_Gt.min(skipna=True).values))
    print("max :", float(Pmb_unc_basin_Gt.max(skipna=True).values))

    print("\nPMB uncertainty diagnostic [mm/month]:")
    print("mean:", float(Pmb_unc_basin_mm.mean(skipna=True).values))
    print("min :", float(Pmb_unc_basin_mm.min(skipna=True).values))
    print("max :", float(Pmb_unc_basin_mm.max(skipna=True).values))

else:
    Pmb_unc_basin_Gt = None
    Pmb_unc_basin_mm = None

# -----------------------------------------------------------------------------
# 8b. Supplementary Table S3: PMB components and annual uncertainty terms
# -----------------------------------------------------------------------------
# Convert PMB components from Gt/month to mm/month

D_basin_mm   = D_basin   * 1e12 / basin_area_m2
BM_basin_mm  = BM_basin  * 1e12 / basin_area_m2
dS_basin_mm  = dS_basin  * 1e12 / basin_area_m2
SUB_basin_mm = SUB_basin * 1e12 / basin_area_m2

# Mean annual components in mm/yr

df_D   = mean_annual_mm(D_basin_mm, "Ice discharge")
df_BM  = mean_annual_mm(BM_basin_mm, "Basal melt")
df_dS  = mean_annual_mm(dS_basin_mm, "GRACE dS")
df_SUB = mean_annual_mm(SUB_basin_mm, "RACMO sublimation term")
df_PMB = mean_annual_mm(Precip_basin_mm, "P_MB")

# -----------------------------------------------------------------------------
# Annual uncertainty terms
# -----------------------------------------------------------------------------

# GRACE dS uncertainty:
# monthly dS uncertainty is propagated to annual totals by root-sum-square.
# -----------------------------------------------------------------------------
# GRACE dS annual uncertainty from storage-anomaly endpoints
# -----------------------------------------------------------------------------
# The annual storage-change term is S_end - S_start. Therefore, annual
# uncertainty is computed from endpoint storage-anomaly uncertainties, not
# by RSS accumulation of monthly deltaS uncertainties.

sigmaS_df = load_sigmaS_storage_dataframe(
    rignot_sigmaS_filled_pkl,
    basin_cols,
)

sigmaS_xr = storage_sigma_to_xarray(
    sigmaS_df,
    basin_id_da,
    basin_name_da,
)

dS_unc_annual_Gt = annual_deltaS_uncertainty_from_storage_endpoints(
    sigmaS_xr,
    pmb_dates=D_basin["date"].values,
)

dS_unc_annual_mm = dS_unc_annual_Gt * 1e12 / basin_area_m2

df_dS_unc = mean_over_years_df(
    dS_unc_annual_mm,
    "GRACE dS uncertainty 1sigma"
)
# Discharge uncertainty:
# D_unc_basin is monthly because annual_to_monthly_long() distributes
# annual discharge uncertainty uniformly across the 12 months.
# For annual uncertainty, restore the annual value by multiplying the
# monthly value by 12.
D_unc_annual_Gt = D_unc_basin.groupby("date.year").mean("date") * 12.0
D_unc_annual_mm = D_unc_annual_Gt * 1e12 / basin_area_m2
df_D_unc = mean_over_years_df(
    D_unc_annual_mm,
    "Ice discharge uncertainty 1sigma"

)

# Correct annual PMB uncertainty:
# combine annual GRACE dS uncertainty and annual discharge uncertainty.
# Basal-melt and RACMO sublimation uncertainties are not included.

D_unc_annual_mm, dS_unc_annual_mm = xr.align(
    D_unc_annual_mm,
    dS_unc_annual_mm,
    join="inner",
)

P_MB_unc_annual_mm = np.sqrt(
    D_unc_annual_mm ** 2 + dS_unc_annual_mm ** 2
)

df_PMB_unc = mean_over_years_df(
    P_MB_unc_annual_mm,
    "P_MB uncertainty 1sigma"
)

# -----------------------------------------------------------------------------
# Merge Table S3
# -----------------------------------------------------------------------------

df_table_s3 = (
    df_D.merge(df_D_unc, on="basin_id", how="outer")
        .merge(df_BM, on="basin_id", how="outer")
        .merge(df_dS, on="basin_id", how="outer")
        .merge(df_dS_unc, on="basin_id", how="outer")
        .merge(df_SUB, on="basin_id", how="outer")
        .merge(df_PMB, on="basin_id", how="outer")
        .merge(df_PMB_unc, on="basin_id", how="outer")
)

df_table_s3 = df_table_s3.rename(columns={"basin_id": "basin"})
df_table_s3["basin_label"] = df_table_s3["basin"].map(id2name)
df_table_s3["region"] = df_table_s3["basin"].apply(basin_region)

df_table_s3 = df_table_s3[
    [
        "basin",
        "basin_label",
        "region",
        "Ice discharge",
        "Ice discharge uncertainty 1sigma",
        "Basal melt",
        "GRACE dS",
        "GRACE dS uncertainty 1sigma",
        "RACMO sublimation term",
        "P_MB",
        "P_MB uncertainty 1sigma",
    ]

].round(1)

out_s3 = os.path.join(
    path_to_plots,
    f"Table_S3_basin_mean_annual_PMB_components_uncertainty_2013_2020_{cde_run_dte}.csv"
)

df_table_s3.to_csv(out_s3, index=False)
print("Saved Table S3:", out_s3)

# -----------------------------------------------------------------------------
# 8c Annual regional PMB uncertainty for Figure 5
# -----------------------------------------------------------------------------
# Regional uncertainty is computed from basin-level annual uncertainties using
# normalized basin-area weights:
#
# sigma_R = sqrt(sum_b (w_b * sigma_b)^2)
# -----------------------------------------------------------------------------

REGION_BASINS = {
    "Antarctica": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "West Antarctica": [10, 11, 12, 13, 14, 15, 16, 17],
    "East Antarctica": [2, 3, 4, 5, 6, 7, 8, 9, 18, 19],
}

regional_unc_rows = []

for region_name, basin_ids in REGION_BASINS.items():

    sigma_b = P_MB_unc_annual_mm.sel(basin_id=basin_ids)
    area_b = basin_area_m2.sel(basin_id=basin_ids)

    # normalized basin-area weights
    w_b = area_b / area_b.sum("basin_id")

    sigma_region = np.sqrt(((w_b * sigma_b) ** 2).sum("basin_id"))

    df_reg = (
        sigma_region
        .to_dataframe(name="pmb_uncertainty")
        .reset_index()
    )

    df_reg["region"] = region_name
    regional_unc_rows.append(df_reg[["region", "year", "pmb_uncertainty"]])

df_region_annual_pmb_unc_corrected = pd.concat(
    regional_unc_rows,
    ignore_index=True
)

out_unc_region = os.path.join(
    path_to_plots,
    f"annual_PMB_uncertainty_AIS_WAIS_EAIS_corrected_2013_2020_{cde_run_dte}.csv"
)

df_region_annual_pmb_unc_corrected.to_csv(out_unc_region, index=False)

print("Saved corrected annual regional PMB uncertainty:", out_unc_region)
print(df_region_annual_pmb_unc_corrected.head())
print(df_region_annual_pmb_unc_corrected.tail())

# -----------------------------------------------------------------------------
# 9. Paint basin series back to maps
# -----------------------------------------------------------------------------

if "basin_id_da" not in globals() or "basin_name_da" not in globals():

    basin_id_da = basin_imbie_with_name_map["basin_id"].copy()
    basin_name_da = basin_imbie_with_name_map["basin_name"].copy()

    basin_id_da = basin_id_da.where(np.isfinite(basin_id_da))
    basin_name_da = basin_name_da.where(basin_name_da != "NA")

    basin_id_da = basin_id_da.where(basin_id_da != 1)
    basin_name_da = basin_name_da.where(basin_id_da.notnull())

template = xr.Dataset(
    dict(
        basin_id=basin_id_da[0],
        basin_name=basin_name_da[0],
    )
)

Precip_map_Gt = paint_basin_series_to_grid(
    Precip_basin,
    template,
)

Precip_map_mm = paint_basin_series_to_grid(
    Precip_basin_mm,
    template,
)

# Paint PMB uncertainty back to maps
if COMPUTE_PMB_UNCERTAINTY:

    Pmb_unc_map_Gt = paint_basin_series_to_grid(
        Pmb_unc_basin_Gt,
        template,
    )

    Pmb_unc_map_mm = paint_basin_series_to_grid(
        Pmb_unc_basin_mm,
        template,
    )

else:
    Pmb_unc_map_Gt = None
    Pmb_unc_map_mm = None

# -----------------------------------------------------------------------------
# 10. Metadata
# -----------------------------------------------------------------------------

deltaS_convention_note = (
    "ΔS convention follows the active dS_raster input. "
    "If using forward difference, ΔS_m = S_{m+1} - S_m and is assigned to month m. "
    "If using backward difference, ΔS_m = S_m - S_{m-1} and is assigned to month m."
)

Precip_map_Gt.attrs.update({
    "units": "Gt/month",
    "description": "Monthly accumulated mass-budget precipitation per basin, painted to pixels.",
    "equation": "P_MB = discharge + basal melt + deltaS + positive sublimation loss",
    "note": (
        "D, basal melt, and ΔS are basin totals. Sublimation is converted from "
        "positive areal sublimation loss [kg m-2] to basin-total Gt/month before "
        "summation. "
        + deltaS_convention_note
    ),
})

Precip_map_mm.attrs.update({
    "units": "mm/month",
    "description": "Monthly accumulated mass-budget precipitation per basin in water-equivalent height, painted to pixels.",
    "equation": "P_MB = discharge + basal melt + deltaS + positive sublimation loss",
    "note": (
        "Same as Gt/month result but divided by basin area. "
        "Conversion uses 1 Gt = 1e12 kg and 1 mm water over 1 m2 = 1 kg. "
        + deltaS_convention_note
    ),
})

if COMPUTE_PMB_UNCERTAINTY:

    Pmb_unc_map_Gt.attrs.update({
        "units": "Gt/month",
        "description": "Monthly propagated 1-sigma uncertainty of PMB precipitation per basin, painted to pixels.",
        "equation": (
            "sigma_PMB = sqrt(sigma_D^2 + sigma_basal_melt^2 + "
            "sigma_deltaS^2 + sigma_sublimation^2)"
        ),
        "uncertainty_source": (
            "sigma_deltaS from DataCombo_RignotBasins.xlsx sheet "
            "'1-sigma_Error(Gt)'; sigma_discharge from "
            "antarctic_discharge_2013-2022_imbie.xlsx sheet "
            "'Discharge Error (Gt yr^-1)'. Basal melt and sublimation use optional "
            "fractional uncertainty settings if nonzero."
        ),
        "discharge_fractional_uncertainty": DISCHARGE_FRAC_UNC,
        "basal_melt_fractional_uncertainty": BASAL_MELT_FRAC_UNC,
        "sublimation_fractional_uncertainty": SUBLIMATION_FRAC_UNC,
        "note": (
            "This uncertainty applies only to PMB. It should not be interpreted "
            "as uncertainty for ERA5, GPCP, GPM, or UA-HIPA."
        ),
    })

    Pmb_unc_map_mm.attrs.update({
        "units": "mm/month",
        "description": "Monthly propagated 1-sigma uncertainty of PMB precipitation in water-equivalent height.",
        "equation": (
            "sigma_PMB = sqrt(sigma_D^2 + sigma_basal_melt^2 + "
            "sigma_deltaS^2 + sigma_sublimation^2)"
        ),
        "conversion": "Gt/month converted to mm/month using basin area.",
        "uncertainty_source": (
                "sigma_deltaS from DataCombo_RignotBasins.xlsx sheet "
                "'1-sigma_Error(Gt)'; sigma_discharge from "
                "antarctic_discharge_2013-2022_imbie.xlsx sheet "
                "'Discharge Error (Gt yr^-1)'. Basal melt and sublimation use optional "
                "fractional uncertainty settings if nonzero."
            ),
        "discharge_fractional_uncertainty": DISCHARGE_FRAC_UNC,
        "basal_melt_fractional_uncertainty": BASAL_MELT_FRAC_UNC,
        "sublimation_fractional_uncertainty": SUBLIMATION_FRAC_UNC,
        "note": (
            "This uncertainty applies only to PMB. It should be carried as a "
            "PMB-only companion field in the comparison workflow."
        ),
    })


# -----------------------------------------------------------------------------
# 11. Save outputs
# -----------------------------------------------------------------------------

deltaS_output_tag = (
    correction_tag if "correction_tag" in globals()
    else "unknown_deltaS_version"
)

subl_tag = "positive_sublimation_loss"

out_gt = os.path.join(
    basin_path,
    f"Monthly_mass_budget_precip_RignotBasin_in_GT_{deltaS_output_tag}_{subl_tag}_GRACE_updated_{cde_run_dte}.nc"
)

out_mm = os.path.join(
    basin_path,
    f"Monthly_mass_budget_precip_RignotBasin_in_mm_{deltaS_output_tag}_{subl_tag}_GRACE_updated_{cde_run_dte}.nc"
)

Precip_map_Gt.to_netcdf(out_gt)
Precip_map_mm.to_netcdf(out_mm)

print("\nSaved PMB Gt/month:", out_gt)
print("Saved PMB mm/month:", out_mm)

if COMPUTE_PMB_UNCERTAINTY:

    out_unc_gt = os.path.join(
        basin_path,
        f"Monthly_mass_budget_precip_RignotBasin_uncertainty_in_GT_{deltaS_output_tag}_{subl_tag}_GRACE_updated_{cde_run_dte}.nc"
    )

    out_unc_mm = os.path.join(
        basin_path,
        f"Monthly_mass_budget_precip_RignotBasin_uncertainty_in_mm_{deltaS_output_tag}_{subl_tag}_GRACE_updated_{cde_run_dte}.nc"
    )

    Pmb_unc_map_Gt.to_netcdf(out_unc_gt)
    Pmb_unc_map_mm.to_netcdf(out_unc_mm)

    print("Saved PMB uncertainty Gt/month:", out_unc_gt)
    print("Saved PMB uncertainty mm/month:", out_unc_mm)

print("\nPMB map time range:")
print(
    pd.to_datetime(Precip_map_mm["date"].values).min(),
    "to",
    pd.to_datetime(Precip_map_mm["date"].values).max(),
)
print("Number of PMB months:", Precip_map_mm.sizes["date"])
#%%
# =============================================================================
# POST-PROCESS PMB MAPS: BASIN SERIES, ANNUAL MAPS, SEASONAL MEANS, CLIMATOLOGY
# =============================================================================
# Input:
#   Precip_map_mm : monthly PMB map [mm/month], dims = (date, y, x)
#
# Important:
#   Do NOT use basin_id[0] or basin_name[0] here, because basin_id may have been
#   overwritten by a loop variable. Use protected variables:
#       basin_id_da
#       basin_name_da
#
# Notes:
#   The main monthly PMB file used in the comparative-analysis script is saved
#   earlier as:
#
#       Monthly_mass_budget_precip_RignotBasin_in_mm_{deltaS_output_tag}_{cde_run_dte}.nc
#
#   This post-processing block saves annual, seasonal, and monthly-climatology
#   diagnostic products derived from that monthly PMB map.
# =============================================================================


# =============================================================================
# 0. Short alias, protected tags, and valid basin mask
# =============================================================================

Pmm = Precip_map_mm.copy()

if COMPUTE_PMB_UNCERTAINTY:
    Punc_mm = Pmb_unc_map_mm.copy()
    Punc_mm = Punc_mm.where(np.isfinite(Punc_mm["basin_id"]))
else:
    Punc_mm = None

# Keep only valid basin pixels
Pmm = Pmm.where(np.isfinite(Pmm["basin_id"]))

# Use the same output tag used when saving the main monthly PMB file.
# This keeps all derived files traceable to the exact ΔS convention/correction.
deltaS_output_tag = (
    correction_tag if "correction_tag" in globals()
    else "unknown_deltaS_version"
)

print("\nPMB monthly map time range:")
print(
    pd.to_datetime(Pmm["date"].values).min(),
    "to",
    pd.to_datetime(Pmm["date"].values).max()
)
print("Number of PMB monthly maps:", Pmm.sizes["date"])

print("\nActive PMB/ΔS output tag:")
print(deltaS_output_tag)

print("\nMain monthly PMB file expected from earlier save step:")
print(
    os.path.join(
        basin_path,
        f"Monthly_mass_budget_precip_RignotBasin_in_mm_{deltaS_output_tag}_GRACE_updated_{cde_run_dte}.nc"
    )
)


# =============================================================================
# 1. Basin-time series [mm/month]
# =============================================================================
# Since PMB is constant within each basin after painting, the spatial mean simply
# recovers the basin monthly value. This is still useful for annual aggregation.
# =============================================================================

Pmm_basin = (
    stack_space(Pmm)
    .groupby(basin_labels_from(Pmm))
    .mean("space", skipna=True)
    .transpose("date", "basin_id")
)

Pmm_basin.name = "P_MB_mm_month"

print("\nPMB basin monthly series:")
print(Pmm_basin)


# =============================================================================
# 2. Annual accumulation [mm/year]
# =============================================================================
# Safer handling:
#   - Build monthly counts per year.
#   - Keep only complete 12-month years for formal annual totals.
#   - Save incomplete-year information for diagnostics.
#
# With backward difference:
#     ΔS_m = S_m - S_{m-1}
#     first valid ΔS may begin after the first storage month.
#
# With forward difference:
#     ΔS_m = S_{m+1} - S_m
#     final valid ΔS may end before the final storage month.
#
# Either way, incomplete years should not be treated as full annual totals unless
# clearly marked as diagnostic.
# =============================================================================

month_count_by_year = (
    Pmm_basin
    .notnull()
    .any("basin_id")
    .groupby("date.year")
    .sum("date")
)

print("\nNumber of available PMB months per year:")
print(month_count_by_year)

complete_years = (
    month_count_by_year
    .where(month_count_by_year == 12, drop=True)
    ["year"]
    .values
)

complete_years = np.array([int(y) for y in complete_years])

print("\nComplete PMB years available for formal annual totals:")
print(complete_years)

# Restrict to requested YEARS if defined
requested_years = np.array([int(y) for y in YEARS])

complete_requested_years = np.array(
    [y for y in requested_years if y in complete_years]
)

print("\nComplete requested years used for formal annual maps:")
print(complete_requested_years)

# Annual totals for all years, including incomplete years, for diagnostics only
Pmm_basin_ann_all = (
    Pmm_basin
    .groupby("date.year")
    .sum("date", skipna=True)
    .rename(year="year")
)

Pmm_basin_ann_all.name = "P_MB_mm_year_diagnostic"

# Formal annual totals: complete years only
if len(complete_requested_years) > 0:
    Pmm_basin_ann = Pmm_basin_ann_all.sel(year=complete_requested_years)
else:
    Pmm_basin_ann = Pmm_basin_ann_all.isel(year=slice(0, 0))

Pmm_basin_ann.name = "P_MB_mm_year"


# =============================================================================
# 3. Build protected basin template
# =============================================================================

if "basin_id_da" not in globals() or "basin_name_da" not in globals():

    basin_id_da = basin_imbie_with_name_map["basin_id"].copy()
    basin_name_da = basin_imbie_with_name_map["basin_name"].copy()

    basin_id_da = basin_id_da.where(np.isfinite(basin_id_da))
    basin_name_da = basin_name_da.where(basin_name_da != "NA")

    basin_id_da = basin_id_da.where(basin_id_da != 1)
    basin_name_da = basin_name_da.where(basin_id_da.notnull())

template = xr.Dataset(
    dict(
        basin_id=basin_id_da[0],
        basin_name=basin_name_da[0],
    )
)


# =============================================================================
# 4. Paint annual basin values back to maps
# =============================================================================

if len(complete_requested_years) > 0:

    Pmm_ann_maps = paint_basin_series_to_grid(Pmm_basin_ann, template)

    Pmm_ann_maps.attrs.update({
        "units": "mm/year",
        "description": "Annual mass-budget precipitation from complete monthly PMB years only.",
        "note": (
            "Annual totals are computed only for years with 12 available monthly PMB values. "
            "Incomplete edge years are excluded from formal annual maps."
        ),
        "deltaS_output_tag": deltaS_output_tag,
    })

    annual_year_tag = (
        f"{int(complete_requested_years.min())}_{int(complete_requested_years.max())}"
    )

    annual_outfile = os.path.join(
        basin_path,
        f"Pmb_annual_complete_years_{annual_year_tag}_mm_{deltaS_output_tag}_GRACE_updated_{cde_run_dte}.nc"
    )

    Pmm_ann_maps.to_netcdf(annual_outfile)

    print("\nSaved complete-year annual PMB maps:")
    print(annual_outfile)

else:
    Pmm_ann_maps = None
    annual_year_tag = "no_complete_years"

    print("\nNo complete requested years available. Formal annual PMB map not saved.")


# =============================================================================
# 5. Optional diagnostic annual maps including incomplete years
# =============================================================================
# Keep this only for checking, not final product comparison.
# =============================================================================

Pmm_ann_maps_all = paint_basin_series_to_grid(Pmm_basin_ann_all, template)

Pmm_ann_maps_all.attrs.update({
    "units": "mm/year",
    "description": "Diagnostic annual PMB totals including incomplete edge years.",
    "note": (
        "This file includes incomplete years and should not be used for formal "
        "annual climatology or product comparison."
    ),
    "deltaS_output_tag": deltaS_output_tag,
})

annual_diag_outfile = os.path.join(
    basin_path,
    f"Pmb_annual_DIAGNOSTIC_including_incomplete_years_{YEARS[0]}_{YEARS[-1]}_mm_{deltaS_output_tag}_GRACE_updated_{cde_run_dte}.nc"
)

Pmm_ann_maps_all.to_netcdf(annual_diag_outfile)

print("\nSaved diagnostic annual PMB maps including incomplete years:")
print(annual_diag_outfile)


# =============================================================================
# 6. Quick annual maps
# =============================================================================

if Pmm_ann_maps is not None:

    Pmm_basin_ann_arrs = [
        (f"P_MB annual accumulation — {yr}", Pmm_ann_maps.sel(year=yr))
        for yr in complete_requested_years
    ]

    compare_mean_precp_plot(
        Pmm_basin_ann_arrs,
        vmin=0,
        vmax=300,
        cbar_tcks=[0, 50, 100, 150, 200, 250, 300],
    )

else:
    print("No complete requested years available for annual PMB plotting.")


# =============================================================================
# 7. Seasonal mean maps
# =============================================================================
# This is a climatological seasonal mean over available months.
#
# Important:
#   This output is mm/month, not mm/season.
#
# For formal seasonal time-series analysis, use the comparative-analysis script,
# because that script can enforce common product time support and complete-season
# handling consistently.
# =============================================================================

Pmm_season = Pmm.groupby("date.season").mean(dim="date", skipna=True)

# Force conventional season order if all seasons exist
season_order = ["DJF", "MAM", "JJA", "SON"]
available_seasons = [s for s in season_order if s in Pmm_season["season"].values]
Pmm_season = Pmm_season.sel(season=available_seasons)

Pmm_season.attrs.update({
    "units": "mm/month",
    "description": "Seasonal mean monthly PMB precipitation.",
    "note": (
        "Computed as mean of available monthly PMB maps by season. "
        "This is mm/month, not mm/season. "
        "DJF and edge-year seasons may have unequal sampling depending on ΔS convention."
    ),
    "deltaS_output_tag": deltaS_output_tag,
})

season_outfile = os.path.join(
    basin_path,
    f"Pmb_seasonal_mean_monthly_mm_{YEARS[0]}_{YEARS[-1]}_{deltaS_output_tag}_GRACE_updated_{cde_run_dte}.nc"
)

Pmm_season.to_netcdf(season_outfile)

print("\nSaved seasonal mean monthly PMB maps:")
print(season_outfile)

Pmm_season_arrs = [
    (f"P_MB seasonal mean monthly — {s}", Pmm_season.sel(season=s))
    for s in available_seasons
]

compare_mean_precp_plot(
    Pmm_season_arrs,
    vmin=0,
    vmax=30,
    cbar_tcks=[0, 5, 10, 15, 20, 25, 30],
)

gc.collect()


# =============================================================================
# 8. Monthly climatology maps
# =============================================================================
# This gives calendar-month mean PMB over available years.
#
# Jan/Feb or Nov/Dec may have fewer contributing years depending on whether the
# active ΔS convention is backward or forward difference.
# =============================================================================

Pmm_clim = Pmm.groupby("date.month").mean(dim="date", skipna=True)

# Ensure months are ordered 1–12 where available
available_months = np.array(Pmm_clim["month"].values).astype(int)
Pmm_clim = Pmm_clim.sel(month=np.sort(available_months))

Pmm_clim.attrs.update({
    "units": "mm/month",
    "description": "Monthly climatological PMB precipitation.",
    "note": (
        "Computed from available monthly PMB maps. Calendar months near the time-series "
        "edges may have fewer contributing years depending on ΔS convention."
    ),
    "deltaS_output_tag": deltaS_output_tag,
})

clim_outfile = os.path.join(
    basin_path,
    f"Pmb_monthly_climatology_mm_{YEARS[0]}_{YEARS[-1]}_{deltaS_output_tag}_GRACE_updated_{cde_run_dte}.nc"
)

Pmm_clim.to_netcdf(clim_outfile)

print("\nSaved monthly climatology PMB maps:")
print(clim_outfile)

month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

Pmm_clim_arrs = [
    (f"P_MB climatological mean — {month_names[m-1]}", Pmm_clim.sel(month=m))
    for m in available_months
]

compare_mean_precp_plot(
    Pmm_clim_arrs,
    vmin=0,
    vmax=30,
    cbar_tcks=[0, 2.5, 5, 7.5, 10, 15, 20, 25, 30],
)

gc.collect()

# =============================================================================
# PMB UNCERTAINTY MONTHLY CLIMATOLOGY MAPS
# =============================================================================
# This gives uncertainty of the calendar-month climatological mean:
#
#     sigma_clim = sqrt(sum(sigma_i^2)) / n
#
# where i represents all available years for that calendar month.
# =============================================================================

if COMPUTE_PMB_UNCERTAINTY and Punc_mm is not None:

    def monthly_clim_uncertainty_xarray(da_unc):
        out_list = []

        for month in sorted(np.unique(da_unc["date.month"].values)):
            sub = da_unc.where(da_unc["date.month"] == month, drop=True)
            n = sub.notnull().sum("date")
            sig = np.sqrt((sub ** 2).sum("date", skipna=True)) / n
            sig = sig.where(n > 0)
            sig = sig.expand_dims(month=[int(month)])
            out_list.append(sig)

        return xr.concat(out_list, dim="month")

    Punc_clim = monthly_clim_uncertainty_xarray(Punc_mm)

    Punc_clim.name = "P_MB_uncertainty_monthly_climatology_mm"

    Punc_clim.attrs.update({
        "units": "mm/month",
        "description": "Uncertainty of monthly climatological PMB mean.",
        "equation": "sigma_clim = sqrt(sum(sigma_monthly^2)) / n",
        "deltaS_output_tag": deltaS_output_tag,
    })

    unc_clim_outfile = os.path.join(
        basin_path,
        f"Pmb_monthly_climatology_uncertainty_mm_{YEARS[0]}_{YEARS[-1]}_{deltaS_output_tag}_GRACE_updated_{cde_run_dte}.nc"
    )

    Punc_clim.to_netcdf(unc_clim_outfile)

    print("\nSaved PMB monthly climatology uncertainty maps:")
    print(unc_clim_outfile)


# =============================================================================
# 9. Month-count diagnostics
# =============================================================================

month_counts = (
    Pmm_basin
    .notnull()
    .any("basin_id")
    .groupby("date.month")
    .sum("date")
)

print("\nNumber of years/months contributing to each calendar-month climatology:")
print(month_counts)

season_counts = (
    Pmm_basin
    .notnull()
    .any("basin_id")
    .groupby("date.season")
    .sum("date")
)

print("\nNumber of monthly maps contributing to each seasonal climatology:")
print(season_counts)

print("\nPost-processing complete.")

"""
Monthly Mass-Budget Precipitation (2013–2020, Rignot/IMBIE basins)

Inputs
------
1) David (GRACE/altimetry): basin-scale monthly storage/mass anomaly time series (Gt)
   - Original source: DataCombo_RignotBasins.xlsx
   - Working input: LI-filled monthly storage anomaly pickle
     DataCombo_RignotBasins_LI_tier1_20260325.pkl
   - Important: This is storage anomaly S, not ΔS.
   - Monthly ΔS is computed after gap filling as:
         ΔS_m = S_m - S_{m-1}
     and assigned to the current month m.

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
APPLY_GRACE_DELTAS_CLIM_CORRECTION = True

# Monthly ΔS climatology statistic: "mean" or "median"
GRACE_CLIM_STAT = "mean"

# Exclude flagged ΔS values when computing the monthly ΔS climatology
EXCLUDE_FLAGGED_FROM_GRACE_CLIM = True
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

rignot_deltaS_err = pd.read_excel(os.path.join(basin_path, 'DataCombo_RignotBasins.xlsx'), sheet_name='1-sigma_Error(Gt)')

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
    os.path.join(basin_path, "DataCombo_RignotBasins_LI_tier1_20260325.pkl")
)

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
# PMB-negative-month trigger list
# =============================================================================
# These basin-months come from the diagnostic slide section:
# "Diagnosing PMB negative months"
#
# They identify where the final PMB estimate became physically problematic.
# They do NOT by themselves prove that the same GRACE storage anomaly month is bad.
#
# Important terminology:
#
# flagged_pmb_table:
#     Basin-months where final P_MB is negative/unphysical.
#     This tells us where the problem appears.
#
# flagged_deltaS_table:
#     Actual GRACE-derived ΔS basin-months selected for replacement.
#     This tells us what enters the correction.
#
# For this controlled correction experiment, we use the PMB-negative basin-months
# as the ΔS correction targets.
# =============================================================================

FLAGGED_PMB_MONTHS_BY_NAME = [
    # WAIS 2014 SON problem season: dominant issue in November
    {"year": 2014, "month": 11, "basins": ["G-H", "Ep-f", "I-Ipp"]},

    # EAIS 2017 DJF problem season: dominant issue in Jan-Feb
    {"year": 2017, "month": 1, "basins": ["Dp-E", "E-Ep"]},
    {"year": 2017, "month": 2, "basins": ["Dp-E", "E-Ep"]},
]

flagged_pmb_rows = []

for item in FLAGGED_PMB_MONTHS_BY_NAME:
    yy = int(item["year"])
    mm = int(item["month"])

    for b in item["basins"]:
        flagged_pmb_rows.append({
            "time": pd.Timestamp(year=yy, month=mm, day=1),
            "basin": normalize_basin_name_for_grace(b),
            "basin_original": b,
            "source": "PMB_negative_or_unphysical",
        })

flagged_pmb_table = pd.DataFrame(flagged_pmb_rows)

# Variant A: all PMB-triggered ΔS months
flagged_deltaS_table_all = flagged_pmb_table[["time", "basin"]].copy()
flagged_deltaS_table_all["time"] = (
    pd.to_datetime(flagged_deltaS_table_all["time"])
    .dt.to_period("M")
    .dt.to_timestamp()
)

# Variant B: only PMB-triggered months where original forward ΔS is negative
dS_original_long_for_filter = (
    dS_original
    .reset_index(names="time")
    .melt(id_vars="time", var_name="basin", value_name="dS_original_Gt")
)

flagged_deltaS_table_neg_only = (
    flagged_deltaS_table_all
    .merge(
        dS_original_long_for_filter,
        on=["time", "basin"],
        how="left"
    )
)

flagged_deltaS_table_neg_only = (
    flagged_deltaS_table_neg_only[
        flagged_deltaS_table_neg_only["dS_original_Gt"] < 0
    ][["time", "basin"]]
    .reset_index(drop=True)
)

# Choose correction variant
# Variant A = correct all PMB-triggered ΔS months
# Variant B = correct only PMB-triggered months where forward ΔS is negative
GRACE_DELTAS_CORRECTION_VARIANT = "Variant_B_negative_deltaS_only"
# GRACE_DELTAS_CORRECTION_VARIANT = "Variant_A_all_PMB_triggered"

if GRACE_DELTAS_CORRECTION_VARIANT == "Variant_A_all_PMB_triggered":
    flagged_deltaS_table_active = flagged_deltaS_table_all.copy()
elif GRACE_DELTAS_CORRECTION_VARIANT == "Variant_B_negative_deltaS_only":
    flagged_deltaS_table_active = flagged_deltaS_table_neg_only.copy()
else:
    raise ValueError(
        "Unknown GRACE_DELTAS_CORRECTION_VARIANT: "
        f"{GRACE_DELTAS_CORRECTION_VARIANT}"
    )

print("\n[GRACE forward-ΔS correction] PMB-triggered basin-months:")
print(flagged_pmb_table)

print("\n[GRACE forward-ΔS correction] Variant A candidate targets: all PMB-triggered months")
print(flagged_deltaS_table_all)

print("\n[GRACE forward-ΔS correction] Variant B candidate targets: negative-forward-ΔS-only")
print(flagged_deltaS_table_neg_only)

print(f"\n[GRACE forward-ΔS correction] Active correction variant: {GRACE_DELTAS_CORRECTION_VARIANT}")
print(flagged_deltaS_table_active)


# =============================================================================
# Safety checks
# =============================================================================

missing_flagged_basins = sorted(
    set(flagged_deltaS_table_all["basin"]) - set(dS_original.columns)
)

if missing_flagged_basins:
    raise ValueError(
        "Some flagged basins are not present in the GRACE forward-ΔS dataframe: "
        f"{missing_flagged_basins}"
    )

flagged_times_all = (
    pd.to_datetime(flagged_deltaS_table_all["time"])
    .dt.to_period("M")
    .dt.to_timestamp()
)

missing_flagged_times = sorted(set(flagged_times_all) - set(dS_original.index))

if missing_flagged_times:
    raise ValueError(
        "Some flagged correction months are not present in the GRACE forward-ΔS index: "
        f"{missing_flagged_times}"
    )


# =============================================================================
# Apply optional GRACE-derived ΔS monthly-climatology correction
# =============================================================================

if APPLY_GRACE_DELTAS_CLIM_CORRECTION:

    dS_used, deltaS_correction_log = replace_flagged_deltaS_with_monthly_climatology(
        dS_df=dS_original,
        flagged_deltaS_table=flagged_deltaS_table_active,
        clim_stat=GRACE_CLIM_STAT,
        exclude_flagged_from_clim=EXCLUDE_FLAGGED_FROM_GRACE_CLIM,
    )

    deltaS_correction_log["correction_variant"] = GRACE_DELTAS_CORRECTION_VARIANT
    deltaS_correction_log["deltaS_convention"] = "forward_difference_S_next_minus_S_current"
    deltaS_correction_log["clim_stat"] = GRACE_CLIM_STAT
    deltaS_correction_log["exclude_flagged_from_clim"] = EXCLUDE_FLAGGED_FROM_GRACE_CLIM

    print("\n[GRACE forward-ΔS correction] ΔS values replaced:")
    print(deltaS_correction_log)

    deltaS_log_file = os.path.join(
        basin_path,
        (
            "grace_forward_deltaS_monthly_climatology_correction_log_"
            f"{GRACE_DELTAS_CORRECTION_VARIANT}_{cde_run_dte}.csv"
        )
    )

    deltaS_correction_log.to_csv(deltaS_log_file, index=False)
    print(f"[GRACE forward-ΔS correction] Log saved to: {deltaS_log_file}")

else:
    dS_used = dS_original.copy()
    deltaS_correction_log = pd.DataFrame()

    print("\n[GRACE forward-ΔS correction] Correction is OFF. Using original LI-derived forward ΔS.")


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


# =============================================================================
# Diagnostic: compare original LI-derived forward ΔS and active ΔS
# =============================================================================

dS_original_long = (
    dS_original
    .reset_index(names="date")
    .melt(id_vars="date", var_name="basin", value_name="dS_original_Gt")
    .dropna(subset=["dS_original_Gt"])
    .reset_index(drop=True)
)

dS_compare_all_triggers = (
    dS_long
    .rename(columns={"dS_Gt": "dS_used_Gt"})
    .merge(
        dS_original_long,
        on=["date", "basin"],
        how="left"
    )
)

flagged_keys_all = flagged_deltaS_table_all.rename(columns={"time": "date"}).copy()
flagged_keys_all["date"] = pd.to_datetime(flagged_keys_all["date"])

flagged_keys_active = flagged_deltaS_table_active.rename(columns={"time": "date"}).copy()
flagged_keys_active["date"] = pd.to_datetime(flagged_keys_active["date"])

dS_compare_all_triggers = dS_compare_all_triggers.merge(
    flagged_keys_all.assign(pmb_triggered=True),
    on=["date", "basin"],
    how="inner"
)

dS_compare_all_triggers = dS_compare_all_triggers.merge(
    flagged_keys_active.assign(was_actively_corrected=True),
    on=["date", "basin"],
    how="left"
)

dS_compare_all_triggers["was_actively_corrected"] = (
    dS_compare_all_triggers["was_actively_corrected"].fillna(False)
)

dS_compare_all_triggers["dS_change_Gt"] = (
    dS_compare_all_triggers["dS_used_Gt"] -
    dS_compare_all_triggers["dS_original_Gt"]
)

dS_compare_all_triggers["correction_variant"] = GRACE_DELTAS_CORRECTION_VARIANT
dS_compare_all_triggers["deltaS_convention"] = "forward_difference_S_next_minus_S_current"

dS_compare_active = dS_compare_all_triggers[
    dS_compare_all_triggers["was_actively_corrected"]
].copy()

print("\n[GRACE forward-ΔS correction] ΔS before/after comparison for ACTIVE correction targets:")
print(dS_compare_active)

print("\n[GRACE forward-ΔS correction] ΔS before/after comparison for ALL PMB-triggered months:")
print(dS_compare_all_triggers)

if APPLY_GRACE_DELTAS_CLIM_CORRECTION:

    dS_compare_active_file = os.path.join(
        basin_path,
        (
            "grace_forward_deltaS_before_after_ACTIVE_targets_"
            f"{GRACE_DELTAS_CORRECTION_VARIANT}_{cde_run_dte}.csv"
        )
    )

    dS_compare_all_file = os.path.join(
        basin_path,
        (
            "grace_forward_deltaS_before_after_ALL_PMB_triggers_"
            f"{GRACE_DELTAS_CORRECTION_VARIANT}_{cde_run_dte}.csv"
        )
    )

    dS_compare_active.to_csv(dS_compare_active_file, index=False)
    dS_compare_all_triggers.to_csv(dS_compare_all_file, index=False)

    print(
        "[GRACE forward-ΔS correction] Active-target ΔS before/after comparison saved to: "
        f"{dS_compare_active_file}"
    )
    print(
        "[GRACE forward-ΔS correction] All-trigger ΔS before/after comparison saved to: "
        f"{dS_compare_all_file}"
    )
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

if APPLY_GRACE_DELTAS_CLIM_CORRECTION:
    attributes["correction"] = (
        "Selected flagged GRACE-derived ΔS basin-months were replaced "
        "with basin-specific monthly climatological ΔS values before computing PMB."
    )
    attributes["correction_variant"] = GRACE_DELTAS_CORRECTION_VARIANT
    attributes["clim_stat"] = GRACE_CLIM_STAT
    attributes["exclude_flagged_from_clim"] = str(EXCLUDE_FLAGGED_FROM_GRACE_CLIM)
else:
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

correction_tag = (
    f"forward_deltaS_{GRACE_DELTAS_CORRECTION_VARIANT}"
    if APPLY_GRACE_DELTAS_CLIM_CORRECTION
    else "forward_deltaS_uncorrected"
)

out_flnme = os.path.join(
    basin_path,
    (
        f"rignot_deltaS_monthly_{YEAR_START}_{YEAR_END}_"
        f"LI_gap_filled_GRACE_tier1_{correction_tag}_{cde_run_dte}.nc"
    )
)

dS_raster.to_netcdf(out_flnme)

print(f"[David] ΔS raster saved to: {out_flnme}")

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
    sheet_name=["Discharge (Gt yr^-1)", "Summary"]
)

basin_discharge = discharge_data["Discharge (Gt yr^-1)"].copy()

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

basal_melt_raster.isel(date=0).plot(
    cmap="jet",
    vmin=0,
    vmax=0.5,
    cbar_kwargs={"label": "Basal melt [Gt/month]"},
)
plt.title(f"Basal Melt for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")
plt.show()


#%% 3) RACMO: integrate subltot (mm/month) to basin Gt/month
# racmo_sublim_file = os.path.join(racmo_path, 'subltot_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc')
# # --- open & subset RACMO to 2013–2022 ---
# # --- 1) open & subset RACMO on its native curvilinear grid ---
# # 1) Open and subset RACMO to 2013–2022
# da_src = xr.open_dataset(racmo_sublim_file)['subltot'].sel(time=slice('2013-01-01', '2022-12-31'))

# # Pull 2-D lon/lat (RACMO supplies these on the curvilinear grid)
# lon2d = da_src['lon'].values
# lat2d = da_src['lat'].values

# # Safety: mask impossible lon/lat
# bad = ~np.isfinite(lon2d) | ~np.isfinite(lat2d)
# if bad.any():
#     lon2d = lon2d.copy()
#     lat2d = lat2d.copy()
#     lon2d[bad] = np.nan
#     lat2d[bad] = np.nan

# # 2) Build target x/y coordinates (pixel centers) from your transform/shape
# #    (x = xmin + (j + 0.5)*xres, y = ymax - (i + 0.5)*|yres|)
# # height, width = basins_imbie.shape

# jj = np.arange(width)
# ii = np.arange(height)
# x_target = xmin + (jj + 0.5) * xres
# y_target = ymax - (ii + 0.5) * abs(yres)

# X, Y = np.meshgrid(x_target, y_target)  # shape (height, width)

# # 3) Convert target X/Y (stereo) -> lon/lat for interpolation
# crs_out = CRS.from_proj4(crs_stereo)
# transform_to_geo = Transformer.from_crs(crs_out, CRS.from_epsg(4326), always_xy=True)
# lon_tgt, lat_tgt = transform_to_geo.transform(X, Y)  # each (height, width)

# # 4) Prepare source points and target points for griddata
# #    Flatten source and target; drop any NaN lon/lat in the source
# pts_src = np.column_stack([lon2d.ravel(), lat2d.ravel()])
# mask_src = np.isfinite(pts_src).all(axis=1)
# pts_src = pts_src[mask_src]

# # To save RAM, we’ll pre-allocate the output and loop over time
# out = np.full((da_src.sizes['time'], height, width), np.nan, dtype=np.float32)

# # 5) Fast pre-check to avoid "all-NaN" surprises: ensure target overlaps source bbox
# src_lon_min, src_lon_max = np.nanmin(lon2d), np.nanmax(lon2d)
# src_lat_min, src_lat_max = np.nanmin(lat2d), np.nanmax(lat2d)
# tgt_lon_min, tgt_lon_max = np.nanmin(lon_tgt), np.nanmax(lon_tgt)
# tgt_lat_min, tgt_lat_max = np.nanmin(lat_tgt), np.nanmax(lat_tgt)

# overlap_lon = (tgt_lon_min <= src_lon_max) and (tgt_lon_max >= src_lon_min)
# overlap_lat = (tgt_lat_min <= src_lat_max) and (tgt_lat_max >= src_lat_min)
# if not (overlap_lon and overlap_lat):
#     print("WARNING: target grid is outside the RACMO domain in lon/lat — interpolation would be all NaN.")

# # 6) Interpolate each time slice with bilinear (griddata 'linear'); fall back to nearest for edge holes
# tgt_points = np.column_stack([lon_tgt.ravel(), lat_tgt.ravel()])  # (height*width, 2)

# for tt in range(da_src.sizes['time']):
#     v = da_src.isel(time=tt).values.astype(np.float64)  # (rlat, rlon)
#     v_flat = v.ravel()[mask_src]

#     # linear interpolation
#     interp_lin = griddata(pts_src, v_flat, tgt_points, method='linear')

#     # nearest-neighbor fill for anything linear missed (edges/outside convex hull)
#     nan_mask = ~np.isfinite(interp_lin)
#     if nan_mask.any():
#         interp_nn = griddata(pts_src, v_flat, tgt_points[nan_mask], method='nearest')
#         interp_lin[nan_mask] = interp_nn

#     out[tt, :, :] = interp_lin.reshape(height, width).astype(np.float32)

# # 7) Wrap into an xarray.DataArray on the IMBIE grid, stamp georeferencing
# racmo_on_imbie = xr.DataArray(
#     out,
#     name='subltot',
#     dims=('time', 'y', 'x'),
#     coords={
#         'time': da_src['time'].values,
#         'x': x_target,  # meters, polar stereo
#         'y': y_target,  # meters, polar stereo
#     },
#     attrs=da_src.attrs,  # keep units/long_name
# )

# # Attach CRS/transform so it plays nicely with rioxarray
# racmo_on_imbie = racmo_on_imbie.rio.write_crs(CRS.from_proj4(crs_stereo).to_wkt(), inplace=False)
# racmo_on_imbie = racmo_on_imbie.rio.write_transform(basin_transform, inplace=False)

# # print(racmo_on_imbie)
# # sve to disk
# # start from the reprojected DataArray you showed
# # start from your DataArray
# da = racmo_on_imbie

# # (1) Make a copy so we can mutate safely
# da = da.copy()

# # (2) Strip problematic CF-encoding attrs on the data variable
# for key in ("_FillValue", "grid_mapping", "scale_factor", "add_offset"):
#     if key in da.attrs:
#         da.attrs.pop(key)

# # (3) Also remove those from coords if any lib snuck them in
# for c in list(da.coords):
#     for key in ("_FillValue", "grid_mapping", "scale_factor", "add_offset"):
#         if key in da[c].attrs:
#             da[c].attrs.pop(key)

# # (4) Drop the unused scalar coord that can confuse CF writing
# if "rotated_pole" in da.coords and da["rotated_pole"].ndim == 0:
#     da = da.drop_vars("rotated_pole")

# # (5) Ensure a consistent dtype and NaN fill
# da = da.astype("float32")

# # (6) Build encoding ONLY on the data variable (not coords)
# var_name = da.name or "subltot"
# encoding = {
#     var_name: {
#         "zlib": True,
#         "complevel": 4,
#         "_FillValue": np.float32(np.nan),
#         # optional but often nice:
#         "dtype": "float32",
#         "chunksizes": None,  # let engine choose; omit if you want specific chunking
#     }
# }

# # (7) Write to netCDF
# out_nc = os.path.join(racmo_path, f"subltot_monthlyS_ANT11_RACMO2.4p1_ERA5_2013_2022_{cde_run_dte}.nc")

# da.to_netcdf(out_nc, encoding=encoding)
# print("wrote:", out_nc)

racmo_on_imbie = xr.open_dataarray(os.path.join(racmo_path, 'subltot_monthlyS_ANT11_RACMO2.4p1_ERA5_2013_2022_20260325.nc'))

# plot one of the time stamps
tms_plt = racmo_on_imbie['time'][0]
racmo_on_imbie.isel(time=0).plot(cmap='jet', vmin=-10, vmax=10, cbar_kwargs={'label': 'Sublimation [mm/month]'})
# plt.title(f"Sublimation for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")
#%%
# --- compute basin series for each component -------------------------------

# 1) Time align
subl = racmo_on_imbie.rename(time="date")
subl = subl.assign_coords(
    date=pd.DatetimeIndex(subl.date.values).to_period("M").to_timestamp(how="start")
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

print("Common budget dates:")
print(pd.to_datetime(common_dates).min(), "to", pd.to_datetime(common_dates).max())
print("Number of common dates:", len(common_dates))


# =============================================================================
# Basin labels / mask
# =============================================================================
# Use basin labels already carried by the discharge raster.
# This avoids relying on overwritten variables such as basin_id or basin_name.

basin_labels = D["basin_id"]  # 2D, dims usually (y, x)

# Mask SUB to basin interiors
SUBm = mask_to_basins(SUB, basin_labels)

tms_plt = SUBm["date"][0]
SUBm.isel(date=0).plot(
    cmap="jet",
    vmin=-10,
    vmax=5,
    cbar_kwargs={"label": "Sublimation [mm/month]"},
)
plt.title(f"Sublimation for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")
plt.show()

# Optional convention check:
# If RACMO sublimation is negative for mass loss and you want positive loss in:
# P = D + BM + ΔS + SUB
# then uncomment below.
# if float(SUBm.mean(skipna=True)) < 0:
#     SUBm = -SUBm


# Quick sanity prints
print("finite basin pixels (mask):", int(np.isfinite(basin_labels).sum()))
print("finite SUB pixels after mask (first month):", int(np.isfinite(SUBm.isel(date=0)).sum()))


# =============================================================================
# Basin area
# =============================================================================
# Your working grid is 10 km × 10 km in polar stereographic coordinates.
# Therefore each pixel is 10000 m × 10000 m.
# This is consistent with your existing workflow.

cell_area = xr.full_like(basin_labels, 10000.0**2)  # m²
cell_area = cell_area.where(np.isfinite(basin_labels))

basin_area_m2 = cell_area.groupby(
    basin_labels.rename("basin_id")
).sum()

print("Basin area dims:", basin_area_m2.dims)
print("Basin IDs in area array:", basin_area_m2["basin_id"].values)


# =============================================================================
# Ensure sublimation carries basin_id coordinate
# =============================================================================

if "basin_id" not in SUBm.coords:
    SUBm = SUBm.assign_coords(basin_id=basin_labels)


# =============================================================================
# Basin series for each component
# =============================================================================

D_basin   = painted_to_basin_series(D)     # Gt/month
BM_basin  = painted_to_basin_series(BM)    # Gt/month
dS_basin  = painted_to_basin_series(DS)    # Gt/month

SUB_basin = areal_to_basin_series(
    SUBm,
    pixel_area_m2=10000.0 * 10000.0,
    units="kg m-2"
)
SUB_basin.name = "subl_Gt_per_month"


# Optional plot of masked sublimation
SUBm.isel(date=0).plot(
    vmin=0,
    vmax=1.5,
    cbar_kwargs={"label": "Snowdrift sublimation [kg m⁻²]"},
)
plt.title(f"Masked sublimation for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")
plt.show()


# =============================================================================
# Align all basin components
# =============================================================================
# Use inner join here so PMB is computed only where all terms are available.
# This is safer than outer + fillna(0), because fillna(0) can silently create
# PMB values when one component is actually missing.

D_basin, BM_basin, dS_basin, SUB_basin = xr.align(
    D_basin,
    BM_basin,
    dS_basin,
    SUB_basin,
    join="inner"
)

print("Aligned component shapes:")
print("D   :", D_basin.shape)
print("BM  :", BM_basin.shape)
print("dS  :", dS_basin.shape)
print("SUB :", SUB_basin.shape)

print("Aligned PMB dates:")
print(
    pd.to_datetime(D_basin["date"].values).min(),
    "to",
    pd.to_datetime(D_basin["date"].values).max(),
)
print("Number of aligned dates:", D_basin.sizes["date"])


# =============================================================================
# Compute PMB
# =============================================================================
# Mass-budget equation used here:
# P_MB = D + basal melt + ΔS + sublimation
#
# where all terms are basin totals in Gt/month after converting sublimation.

Precip_basin = D_basin + BM_basin + dS_basin + SUB_basin
Precip_basin.name = "precip_Gt_per_month"


# Convert Gt/month to mm/month:
# 1 Gt = 1e12 kg
# 1 mm water over 1 m² = 1 kg
# therefore mm = Gt * 1e12 / area_m2

Precip_basin_mm = Precip_basin * 1e12 / basin_area_m2
Precip_basin_mm.name = "precip_mm_per_month"


# =============================================================================
# Paint basin series back to maps
# =============================================================================
# IMPORTANT:
# Do NOT use basin_id[0] or basin_name[0], because basin_id may have been
# overwritten by a plotting loop.
#
# Use protected DataArray variables instead:
#     basin_id_da
#     basin_name_da
#
# If they do not exist, recover them from basin_imbie_with_name_map.

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
    template
)

Precip_map_mm = paint_basin_series_to_grid(
    Precip_basin_mm,
    template
)


# =============================================================================
# Metadata
# =============================================================================

deltaS_convention_note = (
    "ΔS convention follows the active dS_raster input. "
    "If using forward difference, ΔS_m = S_{m+1} - S_m and is assigned to month m. "
    "If using backward difference, ΔS_m = S_m - S_{m-1} and is assigned to month m."
)

Precip_map_Gt.attrs.update({
    "units": "Gt/month",
    "description": "Monthly accumulated mass-budget precipitation per basin, painted to pixels.",
    "note": (
        "Computed as Discharge + Basal melt + ΔS + Sublimation. "
        "D, basal melt, and ΔS are basin totals. Sublimation is converted from areal units "
        "to basin-total Gt/month before summation. "
        + deltaS_convention_note
    ),
})

Precip_map_mm.attrs.update({
    "units": "mm/month",
    "description": "Monthly accumulated mass-budget precipitation per basin in water-equivalent height, painted to pixels.",
    "note": (
        "Same as Gt/month result but divided by basin area. "
        "Conversion uses 1 Gt = 1e12 kg and 1 mm water over 1 m² = 1 kg. "
        + deltaS_convention_note
    ),
})


# =============================================================================
# Save outputs
# =============================================================================

deltaS_output_tag = (
    correction_tag if "correction_tag" in globals()
    else "unknown_deltaS_version"
)

out_gt = os.path.join(
    basin_path,
    f"Monthly_mass_budget_precip_RignotBasin_in_GT_{deltaS_output_tag}_{cde_run_dte}.nc"
)

out_mm = os.path.join(
    basin_path,
    f"Monthly_mass_budget_precip_RignotBasin_in_mm_{deltaS_output_tag}_{cde_run_dte}.nc"
)

Precip_map_Gt.to_netcdf(out_gt)
Precip_map_mm.to_netcdf(out_mm)

print("Saved PMB Gt/month:", out_gt)
print("Saved PMB mm/month:", out_mm)

print("PMB map time range:")
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
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Short alias and valid basin mask
# -----------------------------------------------------------------------------

Pmm = Precip_map_mm.copy()

# Keep only valid basin pixels
Pmm = Pmm.where(np.isfinite(Pmm["basin_id"]))

print("\nPMB monthly map time range:")
print(
    pd.to_datetime(Pmm["date"].values).min(),
    "to",
    pd.to_datetime(Pmm["date"].values).max()
)
print("Number of PMB monthly maps:", Pmm.sizes["date"])


# -----------------------------------------------------------------------------
# 1. Basin-time series [mm/month]
# -----------------------------------------------------------------------------
# Since PMB is constant within each basin after painting, spatial mean simply
# recovers the basin monthly value. This is still useful for annual aggregation.

Pmm_basin = (
    stack_space(Pmm)
    .groupby(basin_labels_from(Pmm))
    .mean("space", skipna=True)
    .transpose("date", "basin_id")
)

Pmm_basin.name = "P_MB_mm_month"

print("\nPMB basin monthly series:")
print(Pmm_basin)


# -----------------------------------------------------------------------------
# 2. Annual accumulation [mm/year]
# -----------------------------------------------------------------------------
# Safer handling:
#   - Build monthly counts per year.
#   - Keep only complete 12-month years for formal annual totals.
#   - Save incomplete-year information for diagnostics.
#
# With backward difference ΔS = S_m - S_{m-1}, 2013 starts in March if Jan is NaN.
# With forward difference ΔS = S_{m+1} - S_m, 2013 may start in Feb and 2020 may
# end in Nov. Either way, incomplete years should not be treated as full annual
# totals unless clearly marked as diagnostic.
# -----------------------------------------------------------------------------

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

print("\nComplete PMB years available for annual totals:")
print(complete_years)

# Restrict to your requested YEARS if defined
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

# Formal annual totals: complete years only
Pmm_basin_ann = Pmm_basin_ann_all.sel(year=complete_requested_years)

Pmm_basin_ann.name = "P_MB_mm_year"


# -----------------------------------------------------------------------------
# 3. Paint annual basin values back to maps
# -----------------------------------------------------------------------------
# Recover protected basin template if needed.

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

Pmm_ann_maps = paint_basin_series_to_grid(Pmm_basin_ann, template)

Pmm_ann_maps.attrs.update({
    "units": "mm/year",
    "description": "Annual mass-budget precipitation from complete monthly PMB years only.",
    "note": (
        "Annual totals are computed only for years with 12 available monthly PMB values. "
        "Incomplete edge years are excluded from formal annual maps."
    ),
})

annual_year_tag = (
    f"{int(complete_requested_years.min())}_{int(complete_requested_years.max())}"
    if len(complete_requested_years) > 0
    else "no_complete_years"
)

deltaS_output_tag = (
    correction_tag if "correction_tag" in globals()
    else "unknown_deltaS_version"
)

annual_outfile = os.path.join(
    basin_path,
    f"Pmb_annual_complete_years_{annual_year_tag}_mm_{deltaS_output_tag}_{cde_run_dte}.nc"
)

Pmm_ann_maps.to_netcdf(annual_outfile)
print("\nSaved complete-year annual PMB maps:")
print(annual_outfile)


# Optional diagnostic annual maps including incomplete years
# Keep this only for checking, not final analysis.
Pmm_ann_maps_all = paint_basin_series_to_grid(Pmm_basin_ann_all, template)

Pmm_ann_maps_all.attrs.update({
    "units": "mm/year",
    "description": "Diagnostic annual PMB totals including incomplete edge years.",
    "note": (
        "This file includes incomplete years and should not be used for formal "
        "annual climatology or product comparison."
    ),
})

annual_diag_outfile = os.path.join(
    basin_path,
    f"Pmb_annual_DIAGNOSTIC_including_incomplete_years_{YEARS[0]}_{YEARS[-1]}_mm_{deltaS_output_tag}_{cde_run_dte}.nc"
)

Pmm_ann_maps_all.to_netcdf(annual_diag_outfile)
print("\nSaved diagnostic annual PMB maps including incomplete years:")
print(annual_diag_outfile)


# -----------------------------------------------------------------------------
# 4. Quick annual maps
# -----------------------------------------------------------------------------

Pmm_basin_ann_arrs = [
    (f"P_MB annual accumulation — {yr}", Pmm_ann_maps.sel(year=yr))
    for yr in complete_requested_years
]

if len(Pmm_basin_ann_arrs) > 0:
    compare_mean_precp_plot(
        Pmm_basin_ann_arrs,
        vmin=0,
        vmax=300,
        cbar_tcks=[0, 50, 100, 150, 200, 250, 300],
    )
else:
    print("No complete requested years available for annual PMB plotting.")


# -----------------------------------------------------------------------------
# 5. Seasonal mean maps
# -----------------------------------------------------------------------------
# This is a climatological seasonal mean over available months.
# It is okay as a diagnostic, but note that DJF may have fewer Jan/Feb/Dec samples
# depending on the ΔS convention and edge-year coverage.
# -----------------------------------------------------------------------------

Pmm_season = Pmm.groupby("date.season").mean(dim="date", skipna=True)

# Force conventional season order
Pmm_season = Pmm_season.sel(season=["DJF", "MAM", "JJA", "SON"])

Pmm_season.attrs.update({
    "units": "mm/month",
    "description": "Seasonal mean monthly PMB precipitation.",
    "note": (
        "Computed as mean of available monthly PMB maps by season. "
        "This is mm/month, not mm/season."
    ),
})

season_outfile = os.path.join(
    basin_path,
    f"Pmb_seasonal_mean_monthly_mm_{YEARS[0]}_{YEARS[-1]}_{deltaS_output_tag}_{cde_run_dte}.nc"
)

Pmm_season.to_netcdf(season_outfile)
print("\nSaved seasonal mean monthly PMB maps:")
print(season_outfile)

Pmm_season_arrs = [
    (f"P_MB seasonal mean monthly — {s}", Pmm_season.sel(season=s))
    for s in ["DJF", "MAM", "JJA", "SON"]
]

compare_mean_precp_plot(
    Pmm_season_arrs,
    vmin=0,
    vmax=30,
    cbar_tcks=[0, 5, 10, 15, 20, 25, 30],
)

gc.collect()


# -----------------------------------------------------------------------------
# 6. Monthly climatology maps
# -----------------------------------------------------------------------------
# This gives calendar-month mean PMB over available years.
# Jan and Feb may have one fewer contributing year than other months if PMB starts
# after Jan/Feb 2013.
# -----------------------------------------------------------------------------

Pmm_clim = Pmm.groupby("date.month").mean(dim="date", skipna=True)
Pmm_clim = Pmm_clim.assign_coords(month=np.arange(1, 13))

Pmm_clim.attrs.update({
    "units": "mm/month",
    "description": "Monthly climatological PMB precipitation.",
    "note": (
        "Computed from available monthly PMB maps. Calendar months near the time-series "
        "edges may have fewer contributing years."
    ),
})

clim_outfile = os.path.join(
    basin_path,
    f"Pmb_monthly_climatology_mm_{YEARS[0]}_{YEARS[-1]}_{deltaS_output_tag}_{cde_run_dte}.nc"
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
    for m in np.arange(1, 13)
]

compare_mean_precp_plot(
    Pmm_clim_arrs,
    vmin=0,
    vmax=30,
    cbar_tcks=[0, 2.5, 5, 7.5, 10, 15, 20, 25, 30],
)

gc.collect()


# -----------------------------------------------------------------------------
# 7. Month-count diagnostic for climatology
# -----------------------------------------------------------------------------

month_counts = (
    Pmm_basin
    .notnull()
    .any("basin_id")
    .groupby("date.month")
    .sum("date")
)

print("\nNumber of years/months contributing to each calendar-month climatology:")
print(month_counts)


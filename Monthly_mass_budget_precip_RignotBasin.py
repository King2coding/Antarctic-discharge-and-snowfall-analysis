"""
Monthly Mass-Budget Precipitation (2019–2020, Rignot/IMBIE basins)

Inputs
------
1) David (GRACE/altimetry): Excel with sub-annual basin mass anomalies (Gt)
   - Sheet: "Basin_Timeseries (Gt)" (typical)
   - Columns: Time (decimal year), then one column per basin (e.g., "A-Ap", "Ap-B", ...)

2) Chad (discharge & basal melt): Excel with annual values per basin (Gt/yr)
   - Provide sheet/column names in CONFIG below.

3) RACMO2 sublimation: NetCDF monthly ANT11 domain (RACMO2.4p1)
   - Variable: "subltot" (mm water equivalent per month) [common name]
   - Requires cell areas to convert to Gt. Tries variable "cell_area" (m^2) or "AREA" (km^2).
   - Basin mask on the same grid: tries to find "basin" integer IDs in the NetCDF; if not present,
     set BASIN_MASK_PATH to a co-registered mask (same shape/projection as RACMO grid).

Output
------
- CSV with monthly P_MB per basin (Gt) for 2019–2020, and (if basin areas known) mm/month.
- Columns: date, basin, dS_Gt, discharge_Gt, basal_Gt, subl_Gt, Pmb_Gt, (optional) basin_area_km2, Pmb_mm

Author: (K. K. Kumah)
Date: 2025-09-10
"""
#%%
# import libraries
from program_utils import *

#%%
# define paths
basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'
racmo_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/RACMO2pt4p1'
path_to_plots = r'/home/kkumah/Projects/Antarctic_discharge_work/plots'

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
# Year window
YEARS = [2019, 2020]
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
output_path = os.path.join(path_to_plots, 'imbie_basins_with_ids.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')


#%% 1) Read David's Excel and compute ΔS (Gt/month) for 2019–2020
rignot_deltaS = pd.read_excel(os.path.join(basin_path, 'DataCombo_RignotBasins.xlsx'), sheet_name='Basin_Timeseries (Gt)')

rignot_deltaS["Date"] = rignot_deltaS["Time"].apply(decimal_year_to_date).dt.strftime('%Y-%m-%d')
rignot_deltaS['Date'] = pd.to_datetime(rignot_deltaS['Date'])
rignot_deltaS["Year"] = pd.to_datetime(rignot_deltaS["Date"]).dt.year
rignot_deltaS["Month"] = pd.to_datetime(rignot_deltaS["Date"]).dt.month

# Keep only 2019–2020 rows
rignot_deltaS = rignot_deltaS[rignot_deltaS["Year"].isin(YEARS)].copy()

# Identify basin columns
basin_cols = [c for c in rignot_deltaS.columns if c not in ("Time","Date","Year","Month")]

# ΔS(m) = S(t_{m+1}) - S(t_m) (per basin)
# First, ensure monthly step by grouping per (Year, Month); if multiple epochs per month, average
dfm = rignot_deltaS.groupby(["Year","Month"], as_index=False)[basin_cols].mean()
# Create a "month index" for easy shift
dfm["date"] = pd.to_datetime(dict(year=dfm["Year"], month=dfm["Month"], day=1))
dfm = dfm.sort_values("date")

# Compute difference forward in time per basin
# ΔS for a given month is next_month_state - current_month_state
dfm_shift = dfm.copy()
dfm_shift[basin_cols] = dfm[basin_cols].shift(-1)
dS = dfm_shift[basin_cols] - dfm[basin_cols]
dS["date"] = dfm["date"]  # assign ΔS to the current month index

# Keep only months that have a valid "next month" within the range
dS = dS.dropna(subset=basin_cols, how="all")

# Melt to long format
dS_long = dS.melt(id_vars="date", var_name="basin", value_name="dS_Gt")
dS_long = dS_long.dropna(subset=["dS_Gt"]).reset_index(drop=True)

print(f"[David] ΔS computed for months {dS_long['date'].min().date()} to {dS_long['date'].max().date()}")

# create an xarray data based on the df above mapping basin mae to the GT values
# ds_long_xr = dS_long.set_index(["date", "basin"]).to_xarray()


# Mask out non-basin pixels
basin_id = basin_id.where(np.isfinite(basin_id))
basin_name = basin_name.where(basin_name != 'NA')

# If you want to exclude islands up front:
basin_id = basin_id.where(basin_id != 1)
basin_name = basin_name.where(basin_id.notnull())
# Carry the basin_id and basin_name into the dataframe for mapping
dS_long_df = generate_basin_id_mapping(basin_id, basin_name, dS_long)

# Build the time-by-space raster

attributes = {
    'description': 'Monthly basin-total ΔS painted to all pixels of each basin (not areal density). ',
    'long_name': 'Monthly basin mass anomaly change',
    'units': 'Gt/month',
    'source': 'Computed from David Rignot basin Excel (DataCombo_RignotBasins.xlsx)',
    'note': 'Islands (ID=1) excluded; names matched via modal name per ID from the basin grid.'
}

dS_raster = create_basin_xrr(basin_id[0], basin_name[0], dS_long_df, 
                             'dS_Gt', attributes, "deltaS_Gt_per_month")

# plot one of the time stamps
tms_plt = dS_raster['date'][0]
dS_raster.isel(date=0).plot(cmap='jet',vmin=-10,vmax=10, cbar_kwargs={'label': 'ΔS [Gt/month]'})
plt.title(f"ΔS for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")

vals = dS_raster['basin_name'].values.ravel()
non_nan = vals[pd.notna(vals)]
np.unique(non_nan)
# save to disk
# out_flnme = os.path.join(basin_path, 'rignot_deltaS_monthly_2019_2020.nc')
# dS_raster.to_netcdf(out_flnme)
# print(f"[David] ΔS raster saved to {out_flnme}")
#%% 2) Read Chad's discharge & basal melt (annual Gt/yr) and spread to months
discharge_data = pd.read_excel(os.path.join(basin_path, 'antarctic_discharge_2013-2022_imbie.xlsx'), sheet_name=['Discharge (Gt yr^-1)', 'Summary'])
basin_discharge = discharge_data['Discharge (Gt yr^-1)']
basin_cols_ = basin_cols.copy()
basin_cols_ = [b.replace('-f', '-F') if b == 'Ep-f' else b for b in basin_cols_]

basin_discharge = basin_discharge.loc[basin_discharge['IMBIE basin'].isin(basin_cols_), ['IMBIE basin', '2019', '2020']].rename(columns={'IMBIE basin': 'basin'})

# --- Basal melt ---
basal_melt = discharge_data['Summary']
basal_melt = basal_melt.loc[basal_melt['IMBIE basin'].isin(basin_cols_), ['IMBIE basin', 'Basal melt total Gt/yr']].rename(columns={'IMBIE basin': 'basin'})


# Convert annual (Gt/yr) → monthly (Gt/mo), then expand to months
D_month = annual_to_monthly_long(basin_discharge,YEARS, "discharge_Gt")
D_month_df = generate_basin_id_mapping(basin_id, basin_name, D_month)

# Convert annual to monthly
basal_melt["basal_Gt_per_month"] = basal_melt["Basal melt total Gt/yr"] / 12.0

# Expand to months for requested years
rows = []
for _, row in basal_melt.iterrows():
    for y in YEARS:
        for m in range(1, 13):
            rows.append({
                "date": pd.to_datetime(f"{y}-{m:02d}-01"),
                "basin": row["basin"],
                "basal_Gt": row["basal_Gt_per_month"]
            })
B_month = pd.DataFrame(rows)
B_month_df = generate_basin_id_mapping(basin_id, basin_name, B_month)

Q_month = D_month.merge(B_month, on=["date","basin"], how="outer")
print(f"[Chad] Qnet monthly rows: {len(Q_month)}")

# create x array for discharge and basal melt separately
attributes = {
    'description': 'Monthly basin-total discharge + basal melt painted to all pixels of each basin (not areal density). ',
    'long_name': 'Monthly basin discharge ',
    'units': 'Gt/month ',
    'source': 'Computed from Chad’s discharge Excel (antarctic_discharge_2013-2022_imbie.xlsx)',
    'note': 'Islands (ID=1) excluded; names matched via modal name per ID from the basin grid.'
}
discharge_raster = create_basin_xrr(basin_id[0], basin_name[0], 
                                    D_month_df, 'discharge_Gt', 
                                    attributes, "discharge_Gt_per_month")



attributes = {
    'description': 'Monthly basin-total basal melt painted to all pixels of each basin (not areal density). ',
    'long_name': 'Monthly basin basal melt ',
    'units': 'Gt/month ',
    'source': 'Computed from Chad’s discharge Excel (antarctic_discharge_2013-2022_imbie.xlsx)',
    'note': 'Islands (ID=1) excluded; names matched via modal name per ID from the basin grid.'
}
# plot one of the time stamps
discharge_raster.isel(date=0).plot(cmap='jet',vmin=0,vmax=30, cbar_kwargs={'label': 'Discharge [Gt/month]'})
plt.title(f"Discharge for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")

basal_melt_raster = create_basin_xrr(basin_id[0], basin_name[0], B_month_df, 
                                     'basal_Gt', attributes, "basal_Gt_per_month")
# plot one of the time stamps
basal_melt_raster.isel(date=0).plot(cmap='jet', vmin=0, vmax=0.5, cbar_kwargs={'label': 'Basal melt [Gt/month]'})
plt.title(f"Basal Melt for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")

# --- Provisional ID -> Name mapping by sorted unique IDs (excluding Islands) ---
# def make_id_name_map_from_excel_order(basin_da, names):
#     uniq_ids = np.sort(np.unique(basin_da.values[~np.isnan(basin_da.values)]).astype(int))
#     uniq_ids = [i for i in uniq_ids if i != 1]  # drop Islands if present
#     if len(uniq_ids) != len(names):
#         raise ValueError(f"Found {len(uniq_ids)} non-island IDs but have {len(names)} names. "
#                          "Please check inputs or mask.")
#     return dict(zip(uniq_ids, names))

# id2name = make_id_name_map_from_excel_order(basins_imbie, basin_cols_)



#%% 3) RACMO: integrate subltot (mm/month) to basin Gt/month
# racmo_sublim_file = os.path.join(racmo_path, 'subltot_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc')
# # --- open & subset RACMO to 2019–2020 ---
# # --- 1) open & subset RACMO on its native curvilinear grid ---
# # 1) Open and subset RACMO to 2019–2020
# da_src = xr.open_dataset(racmo_sublim_file)['subltot'].sel(time=slice('2019-01-01', '2020-12-31'))

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

# (7) Write to netCDF
out_nc = os.path.join(racmo_path, "subltot_monthlyS_ANT11_RACMO2.4p1_ERA5_2019_2020.nc")
# da.to_netcdf(out_nc, encoding=encoding)
# print("wrote:", out_nc)

racmo_on_imbie = xr.open_dataarray(out_nc)

# plot one of the time stamps
tms_plt = racmo_on_imbie['time'][0]
racmo_on_imbie.isel(time=0).plot(cmap='jet', vmin=-10, vmax=10, cbar_kwargs={'label': 'Sublimation [mm/month]'})
plt.title(f"Sublimation for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")
#%%
# --- compute basin series for each component -------------------------------

# 1) Time align
subl = racmo_on_imbie.rename(time="date")
subl = subl.assign_coords(date=pd.DatetimeIndex(subl.date.values).to_period("M").to_timestamp(how="start"))

common_dates = np.intersect1d(discharge_raster.date, np.intersect1d(basal_melt_raster.date, subl.date))
D  = discharge_raster.sel(date=common_dates)
BM = basal_melt_raster.sel(date=common_dates)
SUB = subl.sel(date=common_dates)

# start from the same (y,x) grid as your discharge raster (or any template with basin_id)
basin_labels = D["basin_id"]                     # (y,x) float32 with NaNs outside basins

# 1) Mask SUB to basin interiors (for plotting and for the budget)
SUBm = mask_to_basins(SUB, basin_labels)
tms_plt = SUBm['time'][0]
SUBm.isel(date=0).plot(cmap='jet', vmin=-10, vmax=5, cbar_kwargs={'label': 'Sublimation [mm/month]'})
plt.title(f"Sublimation for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")

# (optional) If RACMO’s convention makes sublimation negative (loss),
# flip to a positive loss so your P=D+BM+ΔS+SUB uses magnitudes.
# if float(SUBm.mean()) < 0:
#     SUBm = -SUBm


# quick sanity prints
print("finite basin pixels (mask):", int(np.isfinite(basin_labels).sum()))
print("finite SUB pixels after mask (first month):", int(np.isfinite(SUBm.isel(date=0)).sum()))

cell_area    = xr.full_like(basin_labels, 10000.0**2)  # m² per pixel (10 km × 10 km)
cell_area    = cell_area.where(np.isfinite(basin_labels))

# area per basin_id (m²); result dims: ('basin_id',)
basin_area_m2 = cell_area.groupby(basin_labels.rename("basin_id")).sum()

# 2) Ensure sublimation has basin_id
if "basin_id" not in SUB.coords:
    SUB = SUB.assign_coords(basin_id=D["basin_id"])

# 3) Basin series
D_basin   = painted_to_basin_series(D)            # Gt/month
BM_basin  = painted_to_basin_series(BM)           # Gt/month
dS_basin  = painted_to_basin_series(dS_raster)    # Gt/month  (← use your raster)
SUB_basin = areal_to_basin_series(
    SUBm, pixel_area_m2=10000.0 * 10000.0, units="kg m-2"
)
SUB_basin.name = "subl_Gt_per_month"
SUB_basin.isel(date=0).plot(cmap='jet', vmin=-10, vmax=10, cbar_kwargs={'label': 'Sublimation [mm/month]'})
plt.title(f"Sublimation for {pd.to_datetime(tms_plt.values).strftime('%Y-%m-%d')}")
# (optional) Plot a masked frame to confirm
SUBm.isel(date=0).plot(
    cmap="RdBu_r",
    vmin=0, vmax=15,  # adjust to your range
    cbar_kwargs={"label": "Snowdrift Sublimation [kg m⁻2]"},
)

# 4) Sum to precipitation in Gt/month
D_basin, BM_basin, dS_basin, SUB_basin = xr.align(D_basin, BM_basin, dS_basin, SUB_basin, join="outer")
Precip_basin = (D_basin.fillna(0) + BM_basin.fillna(0) + dS_basin.fillna(0) + SUB_basin.fillna(0))
Precip_basin.name = "precip_Gt_per_month"

# 5) Convert to mm/month per basin using basin areas
#    mm = (Gt * 1e12 kg/Gt) / (rho_w * area[m²]) * 1000 mm/m, with rho_w=1000 → mm = Gt * 1e12 / area_m2
Precip_basin_mm = Precip_basin * 1e12 / basin_area_m2
Precip_basin_mm.name = "precip_mm_per_month"

# 6) Paint back to maps
template = xr.Dataset(dict(basin_id=basin_id[0], basin_name=basin_name[0]))

Precip_map_Gt = paint_basin_series_to_grid(Precip_basin,    template)
Precip_map_mm = paint_basin_series_to_grid(Precip_basin_mm, template)

Precip_map_Gt.attrs.update({
    "units": "Gt/month",
    "description": "Monthly accumulated precipitation per basin, painted to pixels.",
    "note": "Computed as Discharge + Basal melt + ΔS + Sublimation; D/BM/ΔS treated as basin totals, Sublimation as areal."
})
Precip_map_mm.attrs.update({
    "units": "mm/month",
    "description": "Monthly accumulated precipitation per basin in water-equivalent height, painted to pixels.",
    "note": "Same as Gt/month result but divided by basin area (ρ=1000 kg/m³)."
})

# save to disk with good memory usage
svnme = os.path.join(basin_path, 'Monthly_mass_budget_precip_RignotBasin_in_GT.nc')
Precip_map_Gt.to_netcdf(svnme)
svnme = os.path.join(basin_path, 'Monthly_mass_budget_precip_RignotBasin_in_mm.nc')
Precip_map_mm.to_netcdf(svnme)


#%%
# 0) Short alias
Pmm = Precip_map_mm  # (date, y, x), mm/month, constant within each basin_id

# 1) Basin-time series (mm/month) by taking spatial mean inside basins

Pmm = Precip_map_mm.where(np.isfinite(Precip_map_mm["basin_id"]))
Pmm_basin = (
    stack_space(Pmm)                                   # (date, space)
    .groupby(basin_labels_from(Pmm))                   # group by basin_id labels
    .mean("space", skipna=True)                         # (date, basin_id)
    .transpose("date", "basin_id")
)
# 2) Annual accumulation (sum over months) → mm/year
Pmm_basin_ann = (
    Pmm_basin
    .groupby("date.year")
    .sum("date")                         # sum of 12 months
    .rename(year="year")
    .sel(year=[2019, 2020])              # pick the two years you want
)

# 3) Paint the annual basin values back to (y, x) maps
template = xr.Dataset(dict(basin_id=basin_id[0], basin_name=basin_name[0]))
Pmm_ann_maps = paint_basin_series_to_grid(Pmm_basin_ann, template)  # (year, y, x)
Pmm_ann_maps.attrs.update(dict(units="mm/year", description="Annual mass-budget precipitation"))

# 4) (optional) Save & quick plots
# Pmm_ann_maps.to_netcdf(os.path.join(basin_path, "Pmb_annual_2019_2020_mm.nc"))

# for yr in [2019, 2020]:
#     Pmm_ann_maps.sel(year=yr).plot(
#         vmin=0, vmax=300,
#         cmap="jet",  # or any diverging you like
#         robust=True,
#         cbar_kwargs={"label": "Mass-budget precipitation (mm/year)"}
#     )
#     plt.title(f"P_MB annual accumulation — {yr}")
#     plt.show()

Pmm_basin_ann_arrs = [(f'P_MB annual accumulation — {yr}',Pmm_ann_maps.sel(year=yr)) for yr in [2019, 2020]]
compare_mean_precp_plot(Pmm_basin_ann_arrs, 
                        vmin=0, vmax=300, 
                        cbar_tcks=[0,  50, 100, 150, 200, 250, 300])

# - -- - - - - - - -  include seasonal and climatological plots - - - - 
# Add a 'season' coordinate
# seaons
Pmm_season = Pmm.groupby('date.season').mean(dim="date")
Pmm_season = Pmm_season.assign_coords(season=["DJF", "MAM", "JJA", "SON"])

# save to disk
# Pmm_season.to_netcdf(os.path.join(basin_path, "Pmb_seasonal_mm_2019_2020.nc"))

# make array for plot
Pmm_season_arrs = [(f'P_MB seasonal mean — {s}', Pmm_season.sel(season=s)) for s in ["DJF", "MAM", "JJA", "SON"]]
compare_mean_precp_plot(Pmm_season_arrs, 
                        vmin=0, vmax=30, 
                        cbar_tcks=[0,  5, 10, 15, 20, 25, 30])
gc.collect()

# climatological
Pmm_clim = Pmm.groupby('date.month').mean(dim="date")
Pmm_clim = Pmm_clim.assign_coords(month=np.arange(1,13))
# make array for plot
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Pmm_clim_arrs = [(f'P_MB climatological mean — {month_names[m-1]}', Pmm_clim.sel(month=m)) for m in np.arange(1,13)]
compare_mean_precp_plot(Pmm_clim_arrs, 
                        vmin=0, vmax=30, 
                        cbar_tcks=[0, 2.5, 5, 7.5,10, 15, 20, 25, 30])
gc.collect()

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
# floating variables
id2name = {
    2: "A-Ap",
    3: "Ap-B",
    4: "B-C",
    5: "C-Cp",
    6: "Cp-D",
    7: "D-Dp",
    8: "Dp-E",
    9: "E-Ep",
    10: "Ep-F",
    11: "F-G",
    12: "G-H",
    13: "H-Hp",
    14: "Hp-I",
    15: "I-Ipp",
    16: "Ipp-J",
    17: "J-Jpp",
    18: "Jpp-K",
    19: "K-A"
}

basin_fle = os.path.join(basin_path, 'bedmap3_basins.nc')
bedmap3_basins = xr.open_dataset(basin_fle)
basins_imbie = bedmap3_basins['imbie'].copy()
# Specs from the basin grid
basin_transform = basins_imbie.rio.transform()
height, width = basins_imbie.shape
xmin, ymax = basin_transform.c, basin_transform.f
xres, yres = basin_transform.a, -basin_transform.e
xmax = xmin + width * xres
ymin = ymax - height * yres
print(f"Basin grid: width={width}, height={height}, xres={xres}, yres={yres}")
basin_bounds = (xmin, xmax, ymin, ymax)  # (minx, maxx, miny, maxy)
print(f"Basin bounds: {basin_bounds}")
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
basin_id = basin_imbie_with_name_map['basin_id'].squeeze('band', drop=True)
basin_name = basin_imbie_with_name_map['basin_name'].squeeze('band', drop=True)

# - - - - - - - - - - - - - - - - - - - - - - - -- - - -- - - - - - - -- - - - - 

rignot_deltaS_err = pd.read_excel(os.path.join(basin_path, 'DataCombo_RignotBasins.xlsx'), sheet_name='1-sigma_Error(Gt)')

# - - - - - - - - - - - - - - - - - - - - - - - -- - - -- - - - - - - -- - - - - 
# Year window
YEARS = [2019, 2020]
#%% - - - - - - - - - - Plot basins - - - - - - - - - - - - - - - - - - - - - - - 

# Set up colormap and norm for 27 discrete basins
colors = plt.cm.gist_ncar(np.linspace(0, 1, 19))
cmap = mcolors.ListedColormap(colors)
cmap.set_bad(color='white')  # Set background (masked or NaN) to white

# Use min and max values for levels
vmin, vmax = 1,19 #1, 27
levels = np.linspace(vmin, vmax, vmax - vmin + 2)  # 27 basins + 1 for boundaries
norm = mcolors.BoundaryNorm(levels, cmap.N)

# Plot
proj = ccrs.SouthPolarStereo()

# --- Pretty plot to sanity-check labels on the map ---
fig, ax = plt.subplots(figsize=(9,10), subplot_kw={'projection': proj}, dpi=200)
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
im = basins_imbie.plot(ax=ax, transform=proj, cmap='gist_ncar', add_colorbar=False)
# Add white background
ax.set_facecolor('white')
ax.coastlines(resolution = '110m', color="k", linewidth=0.6)

# annotate using pixel centroids
ids = sorted(id2name.keys())
for bid in ids:
    mask = (basins_imbie == bid)
    if mask.any():
        yy, xx = np.where(mask[0].values)
        if len(xx) == 0: 
            continue
        cx = float(basins_imbie['x'].values[xx].mean())
        cy = float(basins_imbie['y'].values[yy].mean())
        ax.text(cx, cy, id2name[bid], fontsize=12, ha='center', va='center',
                transform=proj, color='k', bbox=dict(boxstyle="round,pad=0.2",
                fc="white", ec="none", alpha=0.6))

# Remove axis
ax.axis('off')

# Final cleanup
ax.set_title("IMBIE Basins with IDs ", fontsize=18)
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

# Build a robust ID↔name mapping from the grid itself (use most frequent name per ID)


def generate_basin_id_mapping(basin_id, basin_name, input_df):
    ids = np.sort(np.unique(basin_id.values[~np.isnan(basin_id.values)])).astype(int)
    id_to_name = {}
    for bid in ids:
        mask = (basin_id == bid)
        names_here = basin_name.where(mask).values
        names_here = [n for n in names_here.ravel() if isinstance(n, str) and n != 'NA']
        if len(names_here):
        # modal name in the basin
            modal = Counter(names_here).most_common(1)[0][0]
            id_to_name[bid] = modal

# Also keep a normalized dictionary for matching to David’s column labels
    id_to_name_norm = {bid: norm_name(nm) for bid, nm in id_to_name.items() if nm is not None}
    name_to_id_norm = {nm: bid for bid, nm in id_to_name_norm.items()}

# Normalize basin names in David’s table and attach basin_id
    df = input_df.copy()
    df['basin_norm'] = df['basin'].map(norm_name)
    df['basin_id'] = df['basin_norm'].map(name_to_id_norm)

# Drop rows we can’t map, and Islands (ID==1) just in case
    df = df.dropna(subset=['basin_id']).copy()
    df['basin_id'] = df['basin_id'].astype(int)
    df = df[df['basin_id'] != 1]
    return df

dS_long_df = generate_basin_id_mapping(basin_id, basin_name, dS_long)

# Build the time-by-space raster
def create_basin_xrr(basin_id, basin_name, df, colnme, attrs):
    dates = np.sort(df['date'].unique())
    frames = []
    for t in dates:
        slic = df.loc[df['date'] == t, ['basin_id', colnme]]
        values = dict(zip(slic['basin_id'].astype(int), slic[colnme].astype(float)))
        raster_t = paint_by_id(basin_id, values)
        raster_t = raster_t.assign_coords(date=np.datetime64(t)).expand_dims('date')
        frames.append(raster_t)

    dS_raster = xr.concat(frames, dim='date')
    dS_raster.name = 'deltaS_Gt_per_month'
    dS_raster.attrs.update(attrs)

# (Optional) carry through your basin metadata alongside
    dS_raster = dS_raster.assign_coords(
    basin_id=(('y','x'), basin_id.values),
    basin_name=(('y','x'), basin_name.values))
    return dS_raster
attributes = {
    'description': 'Monthly basin-total ΔS painted to all pixels of each basin (not areal density). ',
    'long_name': 'Monthly basin mass anomaly change',
    'units': 'Gt/month',
    'source': 'Computed from David Rignot basin Excel (DataCombo_RignotBasins.xlsx)',
    'note': 'Islands (ID=1) excluded; names matched via modal name per ID from the basin grid.'
}

dS_raster = create_basin_xrr(basin_id, basin_name, dS_long_df, 'dS_Gt', attributes)
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
discharge_raster = create_basin_xrr(basin_id, basin_name, D_month_df, 'discharge_Gt', attributes)

attributes = {
    'description': 'Monthly basin-total basal melt painted to all pixels of each basin (not areal density). ',
    'long_name': 'Monthly basin basal melt ',
    'units': 'Gt/month ',
    'source': 'Computed from Chad’s discharge Excel (antarctic_discharge_2013-2022_imbie.xlsx)',
    'note': 'Islands (ID=1) excluded; names matched via modal name per ID from the basin grid.'
}
basal_melt_raster = create_basin_xrr(basin_id, basin_name, B_month_df, 'basal_Gt', attributes)

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
racmo_sublim_file = os.path.join(racmo_path, 'subltot_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc')

# 2) Build gdalwarp call
src = f'NETCDF:"{racmo_sublim_file}":subltot'  # select the variable
dst = os.path.join(misc_out, "racmo_subl_on_basins.tif")

stereo = '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +datum=WGS84'
src = f'NETCDF:"{racmo_sublim_file}":subltot'
dst = os.path.join(misc_out, "racmo_subl_on_basins.tif")

cmd = [
    "gdalwarp",
    "-s_srs", "EPSG:4326",
    "-t_srs", stereo,
    "-te", str(xmin), str(ymin), str(xmax), str(ymax),
    "-ts", str(width), str(height),
    "-r", "bilinear",
    "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
    "-srcnodata", "nan", "-dstnodata", "nan",
    "-of", "GTiff",
    src, dst
]

subprocess.run(cmd, check=True) 

racmo_sublim_data = xr.open_dataset(racmo_sublim_file)
racmo_sublim_data = racmo_sublim_data['subltot'].sel(time=slice(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31"))
racmo_ant_msk = xr.open_dataset(os.path.join(racmo_path, 'ANT11_masks.nc'))

# plt.imshow(racmo_sublim_data['lat'])
# plt.colorbar(label='Latitude (degrees)')

# plt.imshow(racmo_sublim_data['lon'])
# plt.colorbar(label='Longitude (degrees)')

# plt.imshow(np.diff(racmo_sublim_data['lat'].values), cmap='jet')
# plt.colorbar(label='Latitude difference (degrees)')


transfm = racmo_sublim_data.rio.transform()
minx = racmo_sublim_data['lon'].min().item()
maxy = racmo_sublim_data['lat'].max().item()
px_sz = round(racmo_sublim_data['lon'].diff('lon').mean().item(), 2)

# Reproject RACMO sublimation data to the basin grid
racmo_sublim_data = racmo_sublim_data.rio.write_crs(crs, inplace=True)
yshp, xshp = racmo_sublim_data.shape[2:4]

dest_flnme = os.path.join(misc_out, 'subltot_monthlyS_ANT11_RACMO2.4p1_ERA5_2019_2020.tif')

gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, 
                              crs, crs_format, racmo_sublim_data.data)

output_file_stereo = os.path.join(misc_out, 'subltot_monthlyS_ANT11_RACMO2.4p1_ERA5_2019_2020_stere.tif')

gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'

subprocess.run(gdalwarp_command, shell=True)

# Read the stereographic projection file
racmo_stereo = xr.open_dataset(output_file_stereo)['band_data']

os.remove(dest_flnme)
os.remove(output_file_stereo)

# Clip the data to the bounds of the basin dataset using the queried basin bounds
racmo_stereo_clip = racmo_stereo.sel(x=slice(basin_bounds[0], basin_bounds[1]), 
                                     y=slice(basin_bounds[3], basin_bounds[2]))
# Explicitly set the CRS before reprojecting
racmo_stereo_clip.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

racmo_stereo_clip_res = racmo_stereo_clip.rio.reproject(
    racmo_stereo_clip.rio.crs,
    shape=basins_imbie.shape,  # set the shape as the basin data shape
    resampling=Resampling.nearest,
    transform=basins_imbie.rio.transform()
)
#%%
# import packages

from program_utils import *

#%%
# define paths
basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'

path_to_avhrr_precp = r'/ra1/pubdat/AVHRR_CloudSat_proj/preci_maps/2010_IMERG_based_0.5_res_mean_daily_ERA5_added_0_train_thresh_variable_mask_7_upto_70_smoothed_win_2_Ebtehaj_masking_params_1/'
path_to_era5 = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_0.25deg/ERA5_to_netcdf_files'
path_to_airs_ir = r'/ra1/pubdat/AVHRR_CloudSat_proj/TOVSAIRS/AIRS_from_Eric_to_netcdf_files'
path_to_imerg = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/IMERGV7/DataV7_2007-2020'
path_to_ssmis_17 = r'/ra1/pubdat/AVHRR_CloudSat_proj/SSMI/data/daily/SSMIS-F17-ncfiles_pnt5'
gpcpv3pt3_path = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_3_2010_2020'
racmo_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/RACMO2pt4p1'

# paths to put satellite precip over basins data
imerg_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/imerg_precip'
avhrr_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/avhrr_precip'
ssmis_17_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/ssmis_17_precip'
airs_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/airs_precip'
era5_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/era5_precip'
gpcpv3pt3_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/gpcpv3pt3'
#%%
# floating variables

# load the different precipitation files
imerg_files = sorted([os.path.join(path_to_imerg,x) for x in os.listdir(path_to_imerg) if x.endswith('.nc4')])
# Filter file paths to include only files from the year 2019 and 2020
imerg_files_2019_2020 = sorted([
    file for file in imerg_files 
    if os.path.basename(file).split('.')[4][:4] in ['2019', '2020']
])

gpcpv3pt3_files = sorted([os.path.join(gpcpv3pt3_path,x) for x in os.listdir(gpcpv3pt3_path) if x.endswith('.nc4')])
# Filter file paths to include only files from the year 2019 and 2020
gpcpv3pt3_files_2019_2020 = sorted([file for file in gpcpv3pt3_files 
                             if os.path.basename(file).split('_')[2][:4] in ['2019', '2020']]
)
# avhrr_precp_files = sorted([os.path.join(path_to_avhrr_precp,x) for x in os.listdir(path_to_avhrr_precp) if x.endswith('.tif')])

era5_file_2019 = os.path.join(path_to_era5,'ERA5_daily_precipitation_2019_0.25res.nc') #Total_precip_2019_0.5.nc
era5_file_2020 = os.path.join(path_to_era5,'ERA5_daily_precipitation_2020_0.25res.nc') #Total_precip_2020_0.5.nc

# airs_file_2019 = os.path.join(path_to_airs_ir,'3A_AIRSV6_IR_HDD_daily_precipitation_2019.nc')
# # airs_file_2020 = os.path.join(path_to_airs_ir,'3A_AIRSV6_IR_HDD_daily_precipitation_2020.nc')

# ssmis_17_files_2010 = sorted([os.path.join(path_to_ssmis_17,x) for x in os.listdir(path_to_ssmis_17) if x.endswith('.nc')])
#-----------------------------------------------------------------------------------------

# read ancilary data
basins  = xr.open_dataarray(os.path.join(basins_path, 'bedmap3_basins_0.1deg.tif'))
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
basins = basins.where((basins > 1) & (basins.notnull()))

#%%
# read and process satellite precipitation data

# process imerg files
for idx, im in enumerate(imerg_files_2019_2020, start=1):
    fle_svnme = os.path.join(imerg_basin_path,os.path.basename(im).replace('.nc4', '_imbie_basin_precip.nc'))
    img_bsn = process_imerg_file_to_basin(im, misc_out, basins)

    encoding = {img_bsn.name:{"zlib": True, "complevel": 9}}
    img_bsn.to_netcdf(os.path.join(imerg_basin_path, fle_svnme), 
                      mode='w', 
                      format='NETCDF4', 
                      encoding=encoding)

    # Print progress every 100 files
    if idx % 100 == 0:
        print(f"Processed {idx} files: {os.path.basename(fle_svnme)}")

#-----------------------------------------------------------------------------------------

# process avhrr files
# for idx, avhrr in enumerate(avhrr_precp_files, start=1):
#     fle_svnme = os.path.join(avhrr_basin_path,os.path.basename(avhrr).replace('.tif', '_imbie_basin_precip.nc'))
#     avh_bsn = process_avhrr_file_to_basin(avhrr, '2010', basins['imbie'])

#     encoding = {avh_bsn.name:{"zlib": True, "complevel": 9}}
#     avh_bsn.to_netcdf(os.path.join(avhrr_basin_path, fle_svnme), 
#                       mode='w', 
#                       format='NETCDF4', 
#                       encoding=encoding)

#     # Print progress every 25 files
#     if idx % 25 == 0:
#         print(f"Processed {idx} files: {os.path.basename(fle_svnme)}")

#-----------------------------------------------------------------------------------------

# # process ssmis_17 files
# for idx, ss in enumerate(ssmis_17_files_2010, start=1):
#     fle_svnme = os.path.join(ssmis_17_basin_path,os.path.basename(ss).replace('.nc', '_imbie_basin_precip.nc'))
#     ssmis_bsn = process_ssmis_file_to_basin(ss, basins['imbie'])

#     encoding = {ssmis_bsn.name:{"zlib": True, "complevel": 9}}
#     ssmis_bsn.to_netcdf(os.path.join(ssmis_17_basin_path, fle_svnme), 
#                         mode='w', 
#                         format='NETCDF4', 
#                         encoding=encoding)

#     # Print progress every 25 files
#     if idx % 25 == 0:
#         print(f"Processed {idx} files: {os.path.basename(fle_svnme)}")

#-----------------------------------------------------------------------------------------

# process airs files
# airs_data = xr.open_dataarray(airs_file_2019)
# if 'x' in airs_data.coords and 'y' in airs_data.coords:
#     airs_data = airs_data.rename({'x': 'lon', 'y': 'lat'})

# for idx, ai_tme in enumerate(airs_data.time.values, start=1):
#     airs_time = pd.to_datetime(ai_tme).strftime('%Y%m%d')
#     fle_svnme = os.path.join(airs_basin_path, f'AIRS_3A_AIRSV6_IR_HDD_daily_precipitation_imbie_basin_{airs_time}.nc')

#     airs_bsn = process_airs_file_to_basin(airs_data, ai_tme, basins['imbie'], fle_svnme)

#     encoding = {airs_bsn.name: {"zlib": True, "complevel": 9}}
#     airs_bsn.to_netcdf(os.path.join(airs_basin_path, fle_svnme), 
#                         mode='w', 
#                         format='NETCDF4', 
#                         encoding=encoding)

#     # Print progress every 25 files
#     if idx % 25 == 0:
#         print(f"Processed {idx} files: {os.path.basename(fle_svnme)}")


#-----------------------------------------------------------------------------------------
# process era5 files
for er5 in [era5_file_2019, era5_file_2020]:
    if not os.path.exists(er5):
        raise FileNotFoundError(f"ERA5 file not found: {er5}")
    era5_data = xr.open_dataarray(er5)
    if 'longitude' in era5_data.coords and 'latitude' in era5_data.coords:
        era5_data = era5_data.rename({'longitude': 'lon', 'latitude': 'lat'})    

    for idx, er_tme in enumerate(era5_data.time.values, start=1):
        era5_time = pd.to_datetime(er_tme).strftime('%Y%m%d')
        fle_svnme = os.path.join(era5_basin_path, f'ERA5_daily_precipitation_imbie_basin_{era5_time}.nc')

        er_time = pd.to_datetime(er_tme).strftime('%Y%m%d')
        
        era5_bsn = process_era5_file_to_basin(era5_data, er_tme, basins, fle_svnme)

        encoding = {era5_bsn.name: {"zlib": True, "complevel": 9}}
        era5_bsn.to_netcdf(os.path.join(era5_basin_path, fle_svnme), 
                            mode='w', 
                            format='NETCDF4', 
                            encoding=encoding)

        # Print progress every 100 files
        if idx % 100 == 0:
            print(f"Processed {idx} files: {os.path.basename(fle_svnme)}")

#-----------------------------------------------------------------------------------------
# process gpcpv3pt3 files
# Iterate through GPCP files and process
for idx, gp in enumerate(gpcpv3pt3_files_2019_2020, start=1):
    # Open the GPCP file
    gpcp_fle_time = os.path.basename(gp).split('_')[2]

    fle_svnme = os.path.join(gpcpv3pt3_basin_path, f'GPCP_v3_pnt_3_imbie_basin_{gpcp_fle_time}.nc')
    gpcp_bsn = process_gpcp_file_to_basin(gp, basins)

    encoding = {gpcp_bsn.name:{"zlib": True, "complevel": 9}}
    gpcp_bsn.to_netcdf(os.path.join(gpcpv3pt3_basin_path, fle_svnme), 
                      mode='w', 
                      format='NETCDF4', 
                      encoding=encoding)

    # Print progress every 100 files
    if idx % 100 == 0:
        print(f"Processed {idx} files: {os.path.basename(fle_svnme)}")

#-----------------------------------------------------------------------------------------
# process racmo files
racmo_sublim_file = os.path.join(racmo_path, 'pr_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc')
# --- open & subset RACMO to 2019–2020 ---
# --- 1) open & subset RACMO on its native curvilinear grid ---
# 1) Open and subset RACMO to 2019–2020
da_src = xr.open_dataset(racmo_sublim_file)['pr'].sel(time=slice('2019-01-01', '2020-12-31'))
da_src[0].plot(cmap='jet',vmax=100)
# Pull 2-D lon/lat (RACMO supplies these on the curvilinear grid)
lon2d = da_src['lon'].values
lat2d = da_src['lat'].values

# Safety: mask impossible lon/lat
bad = ~np.isfinite(lon2d) | ~np.isfinite(lat2d)
if bad.any():
    lon2d = lon2d.copy()
    lat2d = lat2d.copy()
    lon2d[bad] = np.nan
    lat2d[bad] = np.nan

# # 2) Build target x/y coordinates (pixel centers) from your transform/shape
# #    (x = xmin + (j + 0.5)*xres, y = ymax - (i + 0.5)*|yres|)
# # height, width = basins_imbie.shape

jj = np.arange(width)
ii = np.arange(height)
x_target = xmin + (jj + 0.5) * xres
y_target = ymax - (ii + 0.5) * abs(yres)

X, Y = np.meshgrid(x_target, y_target)  # shape (height, width)

# 3) Convert target X/Y (stereo) -> lon/lat for interpolation
crs_out = CRS.from_proj4(crs_stereo)
transform_to_geo = Transformer.from_crs(crs_out, CRS.from_epsg(4326), always_xy=True)
lon_tgt, lat_tgt = transform_to_geo.transform(X, Y)  # each (height, width)

# # 4) Prepare source points and target points for griddata
#    Flatten source and target; drop any NaN lon/lat in the source
pts_src = np.column_stack([lon2d.ravel(), lat2d.ravel()])
mask_src = np.isfinite(pts_src).all(axis=1)
pts_src = pts_src[mask_src]

# # To save RAM, we’ll pre-allocate the output and loop over time
out = np.full((da_src.sizes['time'], height, width), np.nan, dtype=np.float32)

# 5) Fast pre-check to avoid "all-NaN" surprises: ensure target overlaps source bbox
src_lon_min, src_lon_max = np.nanmin(lon2d), np.nanmax(lon2d)
src_lat_min, src_lat_max = np.nanmin(lat2d), np.nanmax(lat2d)
tgt_lon_min, tgt_lon_max = np.nanmin(lon_tgt), np.nanmax(lon_tgt)
tgt_lat_min, tgt_lat_max = np.nanmin(lat_tgt), np.nanmax(lat_tgt)

overlap_lon = (tgt_lon_min <= src_lon_max) and (tgt_lon_max >= src_lon_min)
overlap_lat = (tgt_lat_min <= src_lat_max) and (tgt_lat_max >= src_lat_min)
if not (overlap_lon and overlap_lat):
    print("WARNING: target grid is outside the RACMO domain in lon/lat — interpolation would be all NaN.")

# 6) Interpolate each time slice with bilinear (griddata 'linear'); fall back to nearest for edge holes
tgt_points = np.column_stack([lon_tgt.ravel(), lat_tgt.ravel()])  # (height*width, 2)

for tt in range(da_src.sizes['time']):
    v = da_src.isel(time=tt).values.astype(np.float64)  # (rlat, rlon)
    v_flat = v.ravel()[mask_src]

    # linear interpolation
    interp_lin = griddata(pts_src, v_flat, tgt_points, method='linear')

    # nearest-neighbor fill for anything linear missed (edges/outside convex hull)
    nan_mask = ~np.isfinite(interp_lin)
    if nan_mask.any():
        interp_nn = griddata(pts_src, v_flat, tgt_points[nan_mask], method='nearest')
        interp_lin[nan_mask] = interp_nn

    out[tt, :, :] = interp_lin.reshape(height, width).astype(np.float32)

# # 7) Wrap into an xarray.DataArray on the IMBIE grid, stamp georeferencing
racmo_pr_on_imbie = xr.DataArray(
    out,
    name='subltot',
    dims=('time', 'y', 'x'),
    coords={
        'time': da_src['time'].values,
        'x': x_target,  # meters, polar stereo
        'y': y_target,  # meters, polar stereo
    },
    attrs=da_src.attrs,  # keep units/long_name
)

# # Attach CRS/transform so it plays nicely with rioxarray
racmo_pr_on_imbie = racmo_pr_on_imbie.rio.write_crs(CRS.from_proj4(crs_stereo).to_wkt(), inplace=False)
racmo_pr_on_imbie = racmo_pr_on_imbie.rio.write_transform(basin_transform, inplace=False)


da = racmo_pr_on_imbie

# # (1) Make a copy so we can mutate safely
da = da.copy()

# (2) Strip problematic CF-encoding attrs on the data variable
for key in ("_FillValue", "grid_mapping", "scale_factor", "add_offset"):
    if key in da.attrs:
        da.attrs.pop(key)

# (3) Also remove those from coords if any lib snuck them in
for c in list(da.coords):
    for key in ("_FillValue", "grid_mapping", "scale_factor", "add_offset"):
        if key in da[c].attrs:
            da[c].attrs.pop(key)

# (4) Drop the unused scalar coord that can confuse CF writing
if "rotated_pole" in da.coords and da["rotated_pole"].ndim == 0:
    da = da.drop_vars("rotated_pole")

# (5) Ensure a consistent dtype and NaN fill
da = da.astype("float32")

# (6) Build encoding ONLY on the data variable (not coords)
var_name = da.name or "subltot"
encoding = {
    var_name: {
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.float32(np.nan),
        # optional but often nice:
        "dtype": "float32",
        "chunksizes": None,  # let engine choose; omit if you want specific chunking
    }
}

# 1) Mask SUB to basin interiors (for plotting and for the budget)
da = mask_to_basins(da, basins)

# (7) Write to netCDF
out_nc = os.path.join(racmo_path, "pr_monthlyS_ANT11_RACMO2.4p1_ERA5_2019_2022.nc")

da.to_netcdf(out_nc, encoding=encoding)
print("wrote:", out_nc)

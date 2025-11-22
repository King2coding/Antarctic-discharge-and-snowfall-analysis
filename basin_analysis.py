#%%
# packages
import gc
import os
import pandas as pd
import numpy as np

import xarray as xr

from program_utils import *
from affine import Affine
from datetime import date

# from concurrent.futures import ProcessPoolExecutor
# from functools import partial
# from multiprocessing import Pool

#%%
# file paths
basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'
# grace_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/grace/GRCTellus.JPL.200204_202501.GLO.RL06.3M.MSCNv04.nc'

# paths to put satellite precip over basins data
imerg_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/imerg_precip'
# avhrr_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/avhrr_precip'
# ssmis_17_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/ssmis_17_precip'
# airs_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/airs_precip'
era5_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/era5_precip'
gpcpv3pt3_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/gpcpv3pt3'
# paths to put satellite precip over basins data

annual_precip_in_basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/precip_in_basins/annual'
seasonal_precip_in_basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/precip_in_basins/seasonal'

# path to put outs e.g. plots, dfs
path_to_plots = r'/home/kkumah/Projects/Antarctic_discharge_work/plots'
path_to_dfs = r'/home/kkumah/Projects/Antarctic_discharge_work/dfs'
#%%
# floating variables

crs = "+proj=longlat +datum=WGS84 +no_defs"  
crs_format = 'proj4' 

batch_size = 10

cde_run_dte = str(date.today().strftime('%Y%m%d'))

#----------------------------------------------------------------------------------

basins  = xr.open_dataset(os.path.join(basins_path,'bedmap3_basins_0.1deg.tif'))
# basins_zwally = basins['zwally']

# basins_imbie = basins['imbie']

# Set up colormap and norm for 27 discrete basins
colors = plt.cm.gist_ncar(np.linspace(0, 1, 19))
cmap = mcolors.ListedColormap(colors)
cmap.set_bad(color='white')  # Set background (masked or NaN) to white

# Use min and max values for levels
vmin, vmax = 1,19 #1, 27
levels = np.linspace(vmin, vmax, vmax - vmin + 2)  # 27 basins + 1 for boundaries
norm = mcolors.BoundaryNorm(levels, cmap.N)

# Mask out invalid values (0 or NaN)
# zwally_data = basins_zwally.where((basins_zwally > 0) & (basins_zwally.notnull()))
basins = basins.where((basins > 0) & (basins.notnull()))
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

# Plot
proj = ccrs.SouthPolarStereo()
fig, ax = plt.subplots(figsize=(12, 8), dpi=300, subplot_kw={'projection': proj})

# Set extent for Antarctica
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

# Plot the data
p = basins.plot(
    ax=ax,
    transform=proj,
    cmap=cmap,
    norm=norm,
    add_colorbar=False
)

# Add white background
ax.set_facecolor('white')

# Annotate each basin with its ID
for basin_id in range(1, 20):
    # Create a mask for the current basin
    basin_mask = basins == basin_id

    # Get the centroid of the basin
    y, x = np.where(basin_mask)
    if len(x) > 0 and len(y) > 0:
        centroid_x = basins['x'].values[x].mean()
        centroid_y = basins['y'].values[y].mean()
        ax.text(
            centroid_x, centroid_y, str(basin_id),
            color='black', fontsize=15, ha='center', va='center', zorder=5,
            transform=ccrs.SouthPolarStereo()
        )

# Add coastlines
ax.coastlines(resolution='110m', color='black', linewidth=0.5)

# Remove axis
ax.axis('off')

# Final cleanup
ax.set_title("IMBIE Basins with IDs ", fontsize=18)
plt.tight_layout()
# plt.show()
# Save the imbie basin plot
output_path = os.path.join(path_to_plots, 'imbie_basins_with_ids.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# we may resample the precip data to course resolution for plotting
# new_resolution = 5000  # meters
# new_transform = Affine(
#     new_resolution, 0.0, -3333500.0,
#     0.0, -new_resolution, 3333500.0
# )

# new_shape = (
#     int((ymax - ymin) / new_resolution),
#     int((xmax - xmin) / new_resolution)
# )

# Load the GRACE dataset
# grace = xr.open_dataset(grace_path)

img_fle_lst = [os.path.join(imerg_basin_path, x) for x in os.listdir(imerg_basin_path) if 'imbie_basin' in x]
# avhrr_fle_lst = [os.path.join(avhrr_basin_path, x) for x in os.listdir(avhrr_basin_path) if 'imbie_basin' in x]
# ssmis_17_fle_lst = [os.path.join(ssmis_17_basin_path, x) for x in os.listdir(ssmis_17_basin_path) if 'imbie_basin' in x]
# airs_fle_lst = [os.path.join(airs_basin_path, x) for x in os.listdir(airs_basin_path) if 'imbie_basin' in x]
era5_fle_lst = [os.path.join(era5_basin_path, x) for x in os.listdir(era5_basin_path) if 'imbie_basin' in x]
gpcpv3pt3_fle_lst = [os.path.join(gpcpv3pt3_basin_path, x) for x in os.listdir(gpcpv3pt3_basin_path) if 'imbie_basin' in x]
#%%
# read and process satellite precipitation data
# Split the file list into batches of 10
print('Processing IMERG data')
batches = [img_fle_lst[i:i + batch_size] for i in range(0, len(img_fle_lst), batch_size)]

img_batch_results = run_batched_processing(batches, basins)

# Calculate the final mean across all batches
img_final_result = xr.concat(img_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_img_xrr_basin_mm_per_year = img_final_result * 365

fle_svnme = os.path.join(annual_precip_in_basins_path, 'IMERG_2010_imbie_basin_annual_precip.nc')
encoding = {stereo_img_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
stereo_img_xrr_basin_mm_per_year.to_netcdf(os.path.join(imerg_basin_path, fle_svnme), 
                                           mode='w', format='NETCDF4', encoding=encoding)

# resample the data to the new resolution
stereo_img_xrr_basin_mm_per_year = xr.open_dataset(os.path.join(annual_precip_in_basins_path,
                                                              'IMERG_2010_imbie_basin_annual_precip.nc')) 
# Explicitly set the CRS before reprojecting
# stereo_img_xrr_basin_mm_per_year.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

# stereo_img_xrr_basin_mm_per_year_5km = stereo_img_xrr_basin_mm_per_year.rio.reproject(
#                                     dst_crs=stereo_img_xrr_basin_mm_per_year.rio.crs,
#                                     shape=(1333, 1333),
#                                     transform=new_transform,
#                                     resampling=Resampling.nearest
# )

gc.collect()

#----------------------------------------------------------------------------------

# read and process era5 data
print('Processing ERA5 data')

batches = [era5_fle_lst[i:i + batch_size] for i in range(0, len(era5_fle_lst), batch_size)]

era5_batch_results = run_batched_processing(batches, basins)

# Calculate the final mean across all batches
era5_final_result = xr.concat(era5_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_era5_xrr_basin_mm_per_year = era5_final_result * 365

fle_svnme = os.path.join(annual_precip_in_basins_path, 'ERA5_2010_imbie_basin_annual_precip.nc')
encoding = {stereo_era5_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
stereo_era5_xrr_basin_mm_per_year.to_netcdf(os.path.join(era5_basin_path, fle_svnme), 
                                            mode='w', format='NETCDF4', encoding=encoding)

# resample the data to the new resolution
stereo_era5_xrr_basin_mm_per_year = xr.open_dataset(os.path.join(annual_precip_in_basins_path,
                                                              'ERA5_2010_imbie_basin_annual_precip.nc')) 
# Explicitly set the CRS before reprojecting
# stereo_era5_xrr_basin_mm_per_year.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

# stereo_era5_xrr_basin_mm_per_year_5km = stereo_era5_xrr_basin_mm_per_year.rio.reproject(
#                                     dst_crs=stereo_era5_xrr_basin_mm_per_year.rio.crs,
#                                     shape=(1333, 1333),
#                                     transform=new_transform,
#                                     resampling=Resampling.nearest
# )

gc.collect()

#----------------------------------------------------------------------------------

# read and process avhrr data
# print('Processing AVHRR data')

# batches = [avhrr_fle_lst[i:i + batch_size] for i in range(0, len(avhrr_fle_lst), batch_size)]

# avhrr_batch_results = run_batched_processing(batches, basins)

# # Calculate the final mean across all batches
# avhrr_final_result = xr.concat(avhrr_batch_results, dim='batch').mean(dim='batch', skipna=True)

# stereo_avhrr_xrr_basin_mm_per_year = avhrr_final_result * 365

# fle_svnme = os.path.join(annual_precip_in_basins_path, 'AVHRR_2010_imbie_basin_annual_precip.nc')
# encoding = {stereo_avhrr_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
# stereo_avhrr_xrr_basin_mm_per_year.to_netcdf(os.path.join(avhrr_basin_path, fle_svnme), 
#                                              mode='w', format='NETCDF4', encoding=encoding)

# # resample the data to the new resolution
# stereo_avhrr_xrr_basin_mm_per_year = xr.open_dataset(os.path.join(annual_precip_in_basins_path,
#                                                               'AVHRR_2010_imbie_basin_annual_precip.nc')) 

# # Explicitly set the CRS before reprojecting

# stereo_avhrr_xrr_basin_mm_per_year.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

# stereo_avhrr_xrr_basin_mm_per_year_5km = stereo_avhrr_xrr_basin_mm_per_year.rio.reproject(
#                                     dst_crs=stereo_avhrr_xrr_basin_mm_per_year.rio.crs,
#                                     shape=(1333, 1333),
#                                     transform=new_transform,
#                                     resampling=Resampling.nearest
# )

gc.collect()

#----------------------------------------------------------------------------------
# read and process ssmi_17 data
print('Processing SSMI_17 data')

# batches = [ssmis_17_fle_lst[i:i + batch_size] for i in range(0, len(ssmis_17_fle_lst), batch_size)]

# ssmis_batch_results = run_batched_processing(batches, basins)

# # Calculate the final mean across all batches
# ssmis_final_result = xr.concat(ssmis_batch_results, dim='batch').mean(dim='batch', skipna=True)

# stereo_ssmi_17_xrr_basin_mm_per_year = ssmis_final_result * 365

# fle_svnme = os.path.join(annual_precip_in_basins_path, 'SSMIS_17_2010_imbie_basin_annual_precip.nc')
# encoding = {stereo_ssmi_17_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
# stereo_ssmi_17_xrr_basin_mm_per_year.to_netcdf(os.path.join(ssmis_17_basin_path, fle_svnme), 
#                                                mode='w', format='NETCDF4', encoding=encoding)

# # resample the data to the new resolution
# stereo_ssmi_17_xrr_basin_mm_per_year = xr.open_dataset(os.path.join(annual_precip_in_basins_path,
#                                                               'SSMIS_17_2010_imbie_basin_annual_precip.nc')) 

# # Explicitly set the CRS before reprojecting

# stereo_ssmi_17_xrr_basin_mm_per_year.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

# stereo_ssmi_17_xrr_basin_mm_per_year_5km = stereo_ssmi_17_xrr_basin_mm_per_year.rio.reproject(
#                                     dst_crs=stereo_ssmi_17_xrr_basin_mm_per_year.rio.crs,
#                                     shape=(1333, 1333),
#                                     transform=new_transform,
#                                     resampling=Resampling.nearest
# )

gc.collect()

#----------------------------------------------------------------------------------
# read and process airs data
print('Processing AIRS data')

# batches = [airs_fle_lst[i:i + batch_size] for i in range(0, len(airs_fle_lst), batch_size)]

# airs_batch_results = run_batched_processing(batches, basins)

# # Calculate the final mean across all batches
# airs_final_result = xr.concat(airs_batch_results, dim='batch').mean(dim='batch', skipna=True)

# stereo_airs_xrr_basin_mm_per_year = airs_final_result * 365

# fle_svnme = os.path.join(annual_precip_in_basins_path, 'AIRS_2010_imbie_basin_annual_precip.nc')
# encoding = {stereo_airs_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
# stereo_airs_xrr_basin_mm_per_year.to_netcdf(os.path.join(airs_basin_path, fle_svnme), 
#                                             mode='w', format='NETCDF4', encoding=encoding)

# # resample the data to the new resolution
# stereo_airs_xrr_basin_mm_per_year = xr.open_dataset(os.path.join(annual_precip_in_basins_path,
#                                                               'AIRS_2010_imbie_basin_annual_precip.nc')) 

# # Explicitly set the CRS before reprojecting

# stereo_airs_xrr_basin_mm_per_year.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

# stereo_airs_xrr_basin_mm_per_year_5km = stereo_airs_xrr_basin_mm_per_year.rio.reproject(
#                                     dst_crs=stereo_airs_xrr_basin_mm_per_year.rio.crs,
#                                     shape=(1333, 1333),
#                                     transform=new_transform,
#                                     resampling=Resampling.nearest
# )

gc.collect()

#----------------------------------------------------------------------------------
# bring CloudSat into the discusiion
# read and process CloudSat data
# ant_data_path = r"/ra1/pubdat/AVHRR_CloudSat_proj/CS_Antartica_analysis_kkk/miscellaneous"  # Replace with your actual path
# cs_ant_filename = os.path.join(ant_data_path,"CS_seasonal_climatology-2007-2010.nc")
# cs_ant = xr.open_dataarray(cs_ant_filename)

# cs_ant_annual_clim = cs_ant.mean(dim='season',skipna=True)#.where(new_mask_ == 1)

# we will fill nan areas in cs ant with era5 data so create that data
era5_data = xr.open_dataarray(r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_0.25deg/ERA5_to_netcdf_files/ERA5_daily_precipitation_2010.nc')
if 'lon' in era5_data.coords and 'lat' in era5_data.coords:
    era5_data = era5_data.rename({'lon': 'x', 'lat': 'y'})
era5_data_ant = era5_data.sel(y=slice(-55, -90), x=slice(-180, 180))
era5_data_ant = era5_data_ant.mean(dim='time',skipna=True)

era5_data_ant.rio.write_crs(CRS.from_proj4(crs).to_string(), inplace=True)
# era5_trns = Affine(round(np.unique(np.diff(era5_data_ant['x'].values))[0],2),
#                    0.0,
#                    era5_data_ant['x'].min().item(),
#                    0.0,
#                    round(np.unique(np.diff(era5_data_ant['y'].values))[0],2),
#                    era5_data_ant['y'].max().item())

era5_data_ant = era5_data_ant.rio.reproject(
                                    dst_crs=era5_data_ant.rio.crs,
                                    shape=(70, 720),                                    
                                    resampling=Resampling.nearest
)

# Set the fill value to NaN
# era5_data_ant = era5_data_ant.where(era5_data_ant != era5_data_ant.attrs['_FillValue'], np.nan)

cs_annual_path = r'/ra1/pubdat/Reza_archive/CS_2022_Maps/annual'
cs_annual_filename = os.path.join(cs_annual_path,"2007_2010.npy")
raw = np.load(cs_annual_filename, allow_pickle=True)
cs_annual = np.flip(raw[:,:].transpose(), axis=0)
cs_annual = xr.DataArray(cs_annual,
                         dims=['lat', 'lon'],
                         coords={'lat': np.arange(90,-89.75, -0.5),
                          'lon': np.arange(-179.75, 180, 0.5)},
                          name = 'precipitation')
cs_annual_ant = cs_annual.sel(lat=slice(-55, -90), lon=slice(-180, 180))
cs_annual_ant = cs_annual_ant*24
# set areas in cs with na to era5 values
cs_annual_ant = cs_annual_ant.fillna(era5_data_ant.values)

yshp, xshp = cs_annual_ant.shape

minx = cs_annual_ant['lon'].min().item()
maxy = cs_annual_ant['lat'].max().item()
px_sz = round(cs_annual_ant['lon'].diff('lon').mean().item(), 2)

dest_flnme = os.path.join(misc_out, os.path.basename(cs_annual_filename).replace('.npy', '.tif'))

gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, 
                              crs, crs_format, cs_annual_ant.data)

output_file_stereo = os.path.join(misc_out, os.path.basename(cs_annual_filename).replace('.npy', '_stere.tif'))

gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'

subprocess.run(gdalwarp_command, shell=True)

# Read the stereographic projection file
cs_ant_xrr_sh_stereo = xr.open_dataset(output_file_stereo)['band_data']

os.remove(dest_flnme)
os.remove(output_file_stereo)

# Clip the data to the bounds of the basin dataset
cs_ant_xrr_clip = cs_ant_xrr_sh_stereo.sel(
    x=slice(-3333250, 3333250),
    y=slice(3333250, -3333250)
).squeeze()


# Explicitly set the CRS before reprojecting
cs_ant_xrr_clip.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

cs_ant_xrr_clip_res = cs_ant_xrr_clip.rio.reproject(
    cs_ant_xrr_clip.rio.crs,
    shape=basins.shape,  # set the shape as the basin data shape
    resampling=Resampling.nearest,
    transform=basins.rio.transform()
)

cs_ant_xrr_clip_res_arr = cs_ant_xrr_clip_res.values
cs_ant_xrr_clip_res_arr = np.where(basins.values > 0, cs_ant_xrr_clip_res_arr, np.nan)
cs_ant_xrr_clip_res = xr.DataArray(
    cs_ant_xrr_clip_res_arr,  # Use the 2D numpy array directly
    dims=['y', 'x'],  # Define dimensions
    coords={'y': cs_ant_xrr_clip_res.coords['y'], 
            'x': cs_ant_xrr_clip_res.coords['x']},
    name='precipitation'  # Rename the DataArray to 'precipitation'
)

# Create an empty DataArray to store mean precipitation mapped to basins
cs_ant_precip_xrr_basin_mapped = xr.full_like(basins, np.nan, dtype=float)

# Loop through each basin ID (Zwally basins are numbered from 1 to 27)
for basin_id in range(1, 20):
    # print(f"Processing basin {basin_id}")
    # Create a mask for the current basin
    basin_mask = basins == basin_id

    # Mask the precipitation data for the current basin
    basin_precip = cs_ant_xrr_clip_res.where(basin_mask.data)

    # Calculate the mean precipitation for the current basin across time and spatial dimensions
    basin_mean_precip = basin_precip.mean(dim=['x', 'y'], skipna=True)

    # Map the calculated mean precipitation back to the basin region
    cs_ant_precip_xrr_basin_mapped = cs_ant_precip_xrr_basin_mapped.where(~basin_mask, basin_mean_precip)

# Clean up variables to free memory
del(basin_precip, basin_mean_precip, basin_mask, basin_id)

cs_ant_precip_xrr_basin_mapped_mm_per_year = cs_ant_precip_xrr_basin_mapped * 365


# resample the data to the new resolution
# cs_ant_precip_xrr_basin_mapped.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

# cs_ant_precip_xrr_basin_mapped_res_5km = cs_ant_precip_xrr_basin_mapped.rio.reproject(
#                                     dst_crs=cs_ant_precip_xrr_basin_mapped.rio.crs,
#                                     shape=(1333, 1333),
#                                     transform=new_transform,
#                                     resampling=Resampling.nearest
# )

gc.collect()
#%%
# work on the reference data

imbie_basin_discharge_params = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins/antarctic_discharge_2013-2022_imbie.xlsx'
# xls = pd.ExcelFile(imbie_basin_discharge_params)
# print("Sheets:", xls.sheet_names)


# # 2) Preview each sheet
# for sheet in xls.sheet_names:
#     df = xls.parse(sheet)
#     print(f"\n--- {sheet} (first 5 rows) ---")
#     print(df.head())

df_sum = pd.read_excel(
    imbie_basin_discharge_params,
    sheet_name="Summary",
    engine="openpyxl"
)

# 2) Build lookup but only for the first 19 basins
lookup_gt = {
    i+1: gt
    for i, gt in enumerate(df_sum["SMB total 2013-2022 Gt/yr"].values[:19])
}
# 3) Convert Gt/yr → mm/yr on the uniform 500 m grid
cell_area = 500.0 * 500.0   # m² per grid cell (500 m × 500 m)
P_MB_mm = xr.full_like(basins_imbie, np.nan, dtype=float)

# 4) Loop over each basin code
for code, smb_gt in lookup_gt.items():
    mask = (basins_imbie == code)               # DataArray of True/False
    n_cells = int(mask.sum(dim=("y","x")).values)  # extract integer count
    area = n_cells * cell_area
    # Convert: Gt/yr → m³/yr → m/yr depth → mm/yr
    depth_mm = (smb_gt * 1e9) / area * 1000.0
    P_MB_mm = P_MB_mm.where(~mask, other=depth_mm)

# 2) For a proper map projection (optional)
ax = plt.axes(projection=ccrs.SouthPolarStereo())
P_MB_mm.plot.imshow(
    x="x", y="y",
    transform=ccrs.SouthPolarStereo(),
    cmap="jet",
    origin="lower",
    yincrease=False,
    ax=ax,
    vmin=0,
    vmax=300
)
ax.coastlines(color="k", linewidth=0.5)
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
# plt.title("SMB Total Mapped to IMBIE Basins (Polar Stereo)")
plt.show()

# save imbie basins 
encoding = {P_MB_mm.name:{"zlib": True, "complevel": 9}}
svename = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins/P_MB_2013-2022_imbie_basin_annual_precip.nc'
P_MB_mm.to_netcdf(svename, mode='w', format='NETCDF4', encoding=encoding)

gc.collect()
#%%
print('Plotting')

plot_arras = [('P$_{MB}$', P_MB_mm),
              ('AVHRR', stereo_avhrr_xrr_basin_mm_per_year['imbie']), 
              ('ERA5', stereo_era5_xrr_basin_mm_per_year['imbie']),
              ('AIRS', stereo_airs_xrr_basin_mm_per_year['imbie']),
              ('CS', cs_ant_precip_xrr_basin_mapped_mm_per_year) 
             ]
            #   ('IMERG', stereo_img_xrr_basin_mm_per_year_5km),
            #   
            #   ] #

svnme = os.path.join(path_to_plots, 'annual_snowfall_accumulation_over_imbie_basins.png')
compare_mean_precp_plot(plot_arras, 
                        vmin=0, vmax=300, 
                        cbar_tcks=[0, 50, 100, 150, 200, 250, 300])
plt.savefig(svnme,  dpi=1000, bbox_inches='tight')
gc.collect()


plot_arras = [ ('SSMIS-F17', stereo_ssmi_17_xrr_basin_mm_per_year['imbie']),
               ('IMERG', stereo_img_xrr_basin_mm_per_year['imbie']), ] #

svnme = os.path.join(path_to_plots, 'IMERG-SSMIS-F17-annual_snowfall_accumulation_over_imbie_basins.png')
compare_mean_precp_plot(plot_arras, 
                        vmin=0, vmax=100, 
                        cbar_tcks=[0, 10 ,25, 50, 75, 85 ,100])

plt.savefig(svnme,  dpi=1000, bbox_inches='tight')
gc.collect()

# Example usage
# single_precp_plot(stereo_img_xrr_basin_mm_per_year['imbie'], 'IMERG', vmin=0, vmax=100)
# single_precp_plot(stereo_avhrr_xrr_basin_mm_per_year['imbie'], 'AVHRR', vmin=0, vmax=350)
# single_precp_plot(stereo_era5_xrr_basin_mm_per_year['imbie'], 'ERA5', vmin=0, vmax=350)
# single_precp_plot(stereo_ssmi_17_xrr_basin_mm_per_year['imbie'], 'SSMIS-F17', vmin=0, vmax=100)
# single_precp_plot(stereo_airs_xrr_basin_mm_per_year['imbie'], 'AIRS', vmin=0, vmax=350)
# single_precp_plot(cs_ant_precip_xrr_basin_mapped_mm_per_year, 'CloudSat', vmin=0, vmax=350)


#%%
print('Making a table of annual mean precipitation for each basin')
arras = [('P$_{MB}$', P_MB_mm),
        ('IMERG', stereo_img_xrr_basin_mm_per_year), 
        ('AVHRR', stereo_avhrr_xrr_basin_mm_per_year), 
        ('ERA5', stereo_era5_xrr_basin_mm_per_year),
        ('SSMIS-F17', stereo_ssmi_17_xrr_basin_mm_per_year), 
        ('AIRS', stereo_airs_xrr_basin_mm_per_year),
        ('CS', cs_ant_precip_xrr_basin_mapped_mm_per_year)] #
# make a table of the mean precipitation for each basin
annual_mean_df = pd.DataFrame(columns=list(range(1, 20)), index=[x[0] for x in arras])
for product_name, data in arras:
    print(product_name)

    for basin_id in range(1, 20):
        # Create a mask for the current basin
        basin_mask = basins_zwally == basin_id

        # Mask the precipitation data for the current basin
        # Ensure 'data' is a DataArray by selecting the first variable if it's a Dataset
        if isinstance(data, xr.Dataset):
            data = list(data.data_vars.values())[0]

        basin_precip = np.unique(data.where(basin_mask.data).values[~np.isnan(data.where(basin_mask.data).values)])[0]

        annual_mean_df.loc[product_name, basin_id] = basin_precip

annual_mean_df = annual_mean_df.applymap(lambda x: round(x, 2) if pd.notnull(x) else x)

# Save the DataFrame to a CSV file
annual_mean_df.to_csv(os.path.join(path_to_dfs, f'annual_mean_precip_over_imbie_basins_{cde_run_dte}.csv'))

# make a scatter plot of all products agianst P_MB
svename = os.path.join(path_to_plots, f'scatter_compare_annual_mean_precip_over_imbie_basins_{cde_run_dte}.png')

x = annual_mean_df.loc['P_MB'].astype(float).values
products = [p for p in annual_mean_df.index if p != 'P_MB']

rows, cols = 2, 3
fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True, sharey=True)
axes_flat = axes.flatten()

for ax, prod in zip(axes_flat, products):
    y = annual_mean_df.loc[prod].astype(float).values
    valid = (~np.isnan(x)) & (~np.isnan(y))
    cc = np.corrcoef(x[valid], y[valid])[0,1]
    bias = np.nanmean(y) / np.nanmean(x)
    ax.scatter(x, y, alpha=0.7)
    lims = [min(np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y))]
    ax.plot(lims, lims, 'k--', linewidth=1)
    ax.text(0.05, 0.95, f"CC={cc:.2f}\nBias={bias:.2f}", transform=ax.transAxes, fontsize=20, 
            verticalalignment='top', horizontalalignment='left')
    ax.set_xlabel("P$_{MB}$ (mm/yr)", fontsize=20)
    ax.set_ylabel(f"{prod} (mm/yr)", fontsize=20)

for ax in axes_flat[len(products):]:
    ax.axis('off')

plt.tight_layout()
plt.savefig(svename,  dpi=1000, bbox_inches='tight')
# plt.show()

#------------------------------------------------------------------------------
df = annual_mean_df.copy().astype(float)

# 2) Products list (excluding P_MB)
products = [p for p in df.index if p != 'P_MB']

# 3) References
refs = ['CS', 'ERA5']

# 4) Prepare results table
cols = [f"CC_vs_{ref}" for ref in refs] + [f"Bias_vs_{ref}" for ref in refs]
results = pd.DataFrame(index=products, columns=cols)

# 5) Compute CC and Bias for each scenario
for ref in refs:
    x_ref = df.loc[ref].values
    for prod in products:
        if prod == ref:
            results.loc[prod, f"CC_vs_{ref}"] = 1.0
            results.loc[prod, f"Bias_vs_{ref}"] = 1.0
        else:
            y = df.loc[prod].values
            valid = (~np.isnan(x_ref)) & (~np.isnan(y))
            cc = np.corrcoef(x_ref[valid], y[valid])[0,1]
            bias = np.nanmean(y) / np.nanmean(x_ref)
            results.loc[prod, f"CC_vs_{ref}"] = round(cc, 2)
            results.loc[prod, f"Bias_vs_{ref}"] = round(bias, 2)

svnme = os.path.join(path_to_dfs, f'metrics_compare_annual_mean_precip_over_imbie_basins_{cde_run_dte}.csv')

results.to_csv(svnme)

#%%
from scipy.io import loadmat

mask_path = "/ra1/pubdat/mask_land_ocean/mask50km.mat"
mask = loadmat(mask_path)["mask50"][-60:, :].swapaxes(0, 1)
mask = np.flip(mask, axis=1)
mask_binary = mask < 75
new_mask = mask_binary.copy()
new_mask[:360, :] = mask_binary[360:, :].copy()
new_mask[360:, :] = mask_binary[:360, :].copy()
new_mask = np.flip(new_mask.T, axis=0)
new_mask_ = new_mask.astype(int)


cs_annual_path = r'/home/kkumah/CS_2022_Maps/annual'
cs_annual_filename = os.path.join(cs_annual_path,"2007_2010.npy")
raw = np.load(cs_annual_filename, allow_pickle=True)
cs_annual = np.flip(raw[:,:].transpose(), axis=0)
cs_annual = xr.DataArray(raw)
cs_antarctica_data = cs_annual.where(new_mask_ == 1)


#%%

# ncolors = 15

# # 2) Get discrete ‘jet’ colormap
# cmap = plt.get_cmap('jet', ncolors)

# # 3) Define bin edges
# levels = np.linspace(0,
#                      500,
#                      ncolors + 1)
# norm = BoundaryNorm(levels, ncolors)

# # 4) Plot
# plt.figure(figsize=(12, 4))
# im = plt.imshow(annual_mean_df, aspect='auto', cmap=cmap, norm=norm)
# cbar = plt.colorbar(im, ticks=levels, spacing='proportional')
# cbar.set_label('Annual Precipitation (mm)')
# plt.xticks(ticks=np.arange(annual_mean_df.shape[1]),
#            labels=annual_mean_df.columns, rotation=90)
# plt.yticks(ticks=np.arange(annual_mean_df.shape[0]),
#            labels=annual_mean_df.index)
# plt.xlabel('Basin ID')
# plt.ylabel('Satellite Product')
# plt.title('Annual Precipitation by Basin (Discrete Jet Colormap)')
# plt.tight_layout()
# plt.show()
#%%
# Ensure the DataArray is sorted by its coordinates before plotting
# Ensure the DataArray is sorted by both 'lat' and 'lons' in increasing order
# grace['lwe_thickness'].sortby('lat').sortby('lons')[0].plot()

# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np

# # Custom colormap and normalization for 27 discrete basin values
# colors = plt.cm.gist_ncar(np.linspace(0, 1, 27))  # Use other palettes like 'tab20b' or 'Set3' for variety
# cmap = mcolors.ListedColormap(colors)
# norm = mcolors.BoundaryNorm(np.arange(-0.5, 27.5), cmap.N)

# fig, ax = plt.subplots(figsize=(8, 8))

# # Use masked plotting to handle NaNs gracefully
# basin_data = basins['zwally']

# # Ensure data is a DataArray and mask invalid values (e.g., where basin == 0 or NaN)
# masked = basin_data.where(basin_data.notnull() & (basin_data >= 1) & (basin_data <= 26))

# # Plot
# p = masked.plot.imshow(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)

# # Add colorbar with integer ticks only
# cbar = plt.colorbar(p, ax=ax, ticks=np.arange(0, 27))
# cbar.set_label("Basin index")

# # Aesthetic clean-up
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_xlabel('')
# ax.set_ylabel('')
# ax.set_title('Zwally Basin Map')

# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import numpy as np

# # Set up colormap and norm for 27 discrete basins
# colors = plt.cm.gist_ncar(np.linspace(0, 1, 27))
# cmap = mcolors.ListedColormap(colors)
# cmap.set_bad(color='white')  # Set background (masked or NaN) to white

# norm = mcolors.BoundaryNorm(np.arange(-0.5, 27.5), cmap.N)

# # Define Antarctic stereographic projection
# proj = ccrs.SouthPolarStereo()

# # Mask out invalid values (0 or NaN)
# zwally_data = basins['zwally'].where((basins['zwally'] > 0) & (basins['zwally'].notnull()))

# # Plot
# fig, ax = plt.su<truncated__content/>
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
basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins/bedmap3_basins.nc'
grace_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/grace/GRCTellus.JPL.200204_202501.GLO.RL06.3M.MSCNv04.nc'

# paths to put satellite precip over basins data
imerg_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/imerg_precip'
avhrr_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/avhrr_precip'
ssmis_17_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/ssmis_17_precip'
airs_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/airs_precip'
era5_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/era5_precip'

# paths to put satellite precip over basins data
annual_precip_in_basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/precip_in_basins/annual'
seasonal_precip_in_basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/precip_in_basins/seasonal'

# path to put outs e.g. plots, dfs
path_to_plots = r'/home/kkumah/Projects/Antarctic_discharge_work/plots'
path_to_dfs = r'/home/kkumah/Projects/Antarctic_discharge_work/dfs'
#%%
# floating variables
misc_out = r'/ra1/pubdat/AVHRR_CloudSat_proj/miscelaneous_outs'

crs = "+proj=longlat +datum=WGS84 +no_defs"  
crs_format = 'proj4' 

batch_size = 10

cde_run_dte = str(date.today().strftime('%Y%m%d'))

#----------------------------------------------------------------------------------

basins  = xr.open_dataset(basins_path)
basins_zwally = basins['zwally']

basins_imbie = basins['imbie']

# Set up colormap and norm for 27 discrete basins
colors = plt.cm.gist_ncar(np.linspace(0, 1, 19))
cmap = mcolors.ListedColormap(colors)
cmap.set_bad(color='white')  # Set background (masked or NaN) to white

# Use min and max values for levels
vmin, vmax = 1,19 #1, 27
levels = np.linspace(vmin, vmax, vmax - vmin + 2)  # 27 basins + 1 for boundaries
norm = mcolors.BoundaryNorm(levels, cmap.N)

# Mask out invalid values (0 or NaN)
zwally_data = basins_zwally.where((basins_zwally > 0) & (basins_zwally.notnull()))
imbie_data = basins_imbie.where((basins_imbie > 0) & (basins_imbie.notnull()))
# Plot
proj = ccrs.SouthPolarStereo()
fig, ax = plt.subplots(figsize=(12, 8), dpi=300, subplot_kw={'projection': proj})

# Set extent for Antarctica
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

# Plot the data
p = imbie_data.plot(
    ax=ax,
    transform=ccrs.SouthPolarStereo(),
    cmap=cmap,
    norm=norm,
    add_colorbar=False
)

# Add white background
ax.set_facecolor('white')

# Annotate each basin with its ID
for basin_id in range(1, 20):
    # Create a mask for the current basin
    basin_mask = basins_imbie == basin_id

    # Get the centroid of the basin
    y, x = np.where(basin_mask)
    if len(x) > 0 and len(y) > 0:
        centroid_x = basins_imbie['x'].values[x].mean()
        centroid_y = basins_imbie['y'].values[y].mean()
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
ax.set_title("Zwally Basins with IDs (Polar Stereographic)", fontsize=18)
plt.tight_layout()
plt.show()

# Extract the bounds of the Zwally basins data
x_min, x_max = basins_imbie['x'].values.min(), basins_imbie['x'].values.max()
y_min, y_max = basins_imbie['y'].values.min(), basins_imbie['y'].values.max()
basin_bounds = (x_min, x_max, y_min, y_max)

# Print the bounds for verification
print(f"Basin bounds: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

# we may resample the precip data to course resolution for plotting
new_resolution = 5000  # meters
new_transform = Affine(
    new_resolution, 0.0, -3333500.0,
    0.0, -new_resolution, 3333500.0
)

new_shpe = (
    int((y_max - y_min) / new_resolution),
    int((x_max - x_min) / new_resolution)
)

# Load the GRACE dataset
grace = xr.open_dataset(grace_path)

img_fle_lst = [os.path.join(imerg_basin_path, x) for x in os.listdir(imerg_basin_path) if 'imbie_basin' in x]
avhrr_fle_lst = [os.path.join(avhrr_basin_path, x) for x in os.listdir(avhrr_basin_path) if 'imbie_basin' in x]
ssmis_17_fle_lst = [os.path.join(ssmis_17_basin_path, x) for x in os.listdir(ssmis_17_basin_path) if 'imbie_basin' in x]
airs_fle_lst = [os.path.join(airs_basin_path, x) for x in os.listdir(airs_basin_path) if 'imbie_basin' in x]
era5_fle_lst = [os.path.join(era5_basin_path, x) for x in os.listdir(era5_basin_path) if 'imbie_basin' in x]

gc.collect()
#%%
# read and process satellite precipitation data
# Split the file list into batches of 10
print('Processing IMERG data')
batches = [img_fle_lst[i:i + batch_size] for i in range(0, len(img_fle_lst), batch_size)]

img_batch_results = run_batched_processing(batches, basins_imbie)

# Calculate the final mean across all batches
img_final_result = xr.concat(img_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_img_xrr_basin_mm_per_year = img_final_result * 365

fle_svnme = os.path.join(annual_precip_in_basins_path, 'IMERG_2010_basin_annual_precip.nc')
encoding = {stereo_img_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
stereo_img_xrr_basin_mm_per_year.to_netcdf(os.path.join(imerg_basin_path, fle_svnme), 
                                           mode='w', format='NETCDF4', encoding=encoding)

# resample the data to the new resolution
# stereo_img_xrr_basin_mm_per_year = xr.open_dataset(os.path.join(annual_precip_in_basins_path,
#                                                               'IMERG_2010_basin_annual_precip.nc')) 
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

era5_batch_results = run_batched_processing(batches, basins_imbie)

# Calculate the final mean across all batches
era5_final_result = xr.concat(era5_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_era5_xrr_basin_mm_per_year = era5_final_result * 365

fle_svnme = os.path.join(annual_precip_in_basins_path, 'ERA5_2010_basin_annual_precip.nc')
encoding = {stereo_era5_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
stereo_era5_xrr_basin_mm_per_year.to_netcdf(os.path.join(era5_basin_path, fle_svnme), 
                                            mode='w', format='NETCDF4', encoding=encoding)

# resample the data to the new resolution
# stereo_era5_xrr_basin_mm_per_year = xr.open_dataset(os.path.join(annual_precip_in_basins_path,
#                                                               'ERA5_2010_basin_annual_precip.nc')) 
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
print('Processing AVHRR data')

batches = [avhrr_fle_lst[i:i + batch_size] for i in range(0, len(avhrr_fle_lst), batch_size)]

avhrr_batch_results = run_batched_processing(batches, basins_imbie)

# Calculate the final mean across all batches
avhrr_final_result = xr.concat(avhrr_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_avhrr_xrr_basin_mm_per_year = avhrr_final_result * 365

fle_svnme = os.path.join(annual_precip_in_basins_path, 'AVHRR_2010_basin_annual_precip.nc')
encoding = {stereo_avhrr_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
stereo_avhrr_xrr_basin_mm_per_year.to_netcdf(os.path.join(avhrr_basin_path, fle_svnme), 
                                             mode='w', format='NETCDF4', encoding=encoding)

# resample the data to the new resolution
# stereo_avhrr_xrr_basin_mm_per_year = xr.open_dataset(os.path.join(annual_precip_in_basins_path,
#                                                               'AVHRR_2010_basin_annual_precip.nc')) 

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

batches = [ssmis_17_fle_lst[i:i + batch_size] for i in range(0, len(ssmis_17_fle_lst), batch_size)]

ssmis_batch_results = run_batched_processing(batches, basins_imbie)

# Calculate the final mean across all batches
ssmis_final_result = xr.concat(ssmis_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_ssmi_17_xrr_basin_mm_per_year = ssmis_final_result * 365

fle_svnme = os.path.join(annual_precip_in_basins_path, 'SSMIS_17_2010_basin_annual_precip.nc')
encoding = {stereo_ssmi_17_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
stereo_ssmi_17_xrr_basin_mm_per_year.to_netcdf(os.path.join(ssmis_17_basin_path, fle_svnme), 
                                               mode='w', format='NETCDF4', encoding=encoding)

# resample the data to the new resolution
# stereo_ssmi_17_xrr_basin_mm_per_year = xr.open_dataset(os.path.join(annual_precip_in_basins_path,
#                                                               'SSMIS_17_2010_basin_annual_precip.nc')) 

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

batches = [airs_fle_lst[i:i + batch_size] for i in range(0, len(airs_fle_lst), batch_size)]

airs_batch_results = run_batched_processing(batches, basins_imbie)

# Calculate the final mean across all batches
airs_final_result = xr.concat(airs_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_airs_xrr_basin_mm_per_year = airs_final_result * 365

fle_svnme = os.path.join(annual_precip_in_basins_path, 'AIRS_2010_basin_annual_precip.nc')
encoding = {stereo_airs_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
stereo_airs_xrr_basin_mm_per_year.to_netcdf(os.path.join(airs_basin_path, fle_svnme), 
                                            mode='w', format='NETCDF4', encoding=encoding)

# resample the data to the new resolution
# stereo_airs_xrr_basin_mm_per_year = xr.open_dataset(os.path.join(annual_precip_in_basins_path,
#                                                               'AIRS_2010_basin_annual_precip.nc')) 

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
ant_data_path = r"/ra1/pubdat/AVHRR_CloudSat_proj/CS_Antartica_analysis_kkk/miscellaneous"  # Replace with your actual path
cs_ant_filename = os.path.join(ant_data_path,"CS_seasonal_climatology-2007-2010.nc")
cs_ant = xr.open_dataarray(cs_ant_filename)

cs_ant_annual_clim = cs_ant.mean(dim='season',skipna=True)#.where(new_mask_ == 1)

yshp, xshp = cs_ant_annual_clim.shape

minx = cs_ant_annual_clim['lon'].min().item()
maxy = cs_ant_annual_clim['lat'].max().item()
px_sz = round(cs_ant_annual_clim['lon'].diff('lon').mean().item(), 2)

dest_flnme = os.path.join(misc_out, os.path.basename(cs_ant_filename))

gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, 
                              crs, crs_format, cs_ant_annual_clim.data)

output_file_stereo = os.path.join(misc_out, os.path.basename(cs_ant_filename).replace('.nc', '_stere.nc'))

gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'

subprocess.run(gdalwarp_command, shell=True)

# Read the stereographic projection file
cs_ant_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

os.remove(dest_flnme)
os.remove(output_file_stereo)

# Clip the data to the bounds of the basin dataset
cs_ant_xrr_clip = cs_ant_xrr_sh_stereo.sel(
    x=slice(-3333250, 3333250),
    y=slice(-3333250, 3333250)
).squeeze()


# Explicitly set the CRS before reprojecting
cs_ant_xrr_clip.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

cs_ant_xrr_clip_res = cs_ant_xrr_clip.rio.reproject(
    cs_ant_xrr_clip.rio.crs,
    shape=basins_imbie.shape,  # set the shape as the basin data shape
    resampling=Resampling.nearest,
    transform=basins['zwally'].rio.transform()
)

cs_ant_xrr_clip_res_arr = cs_ant_xrr_clip_res['Band1'].values
cs_ant_xrr_clip_res_arr = np.where(basins_imbie.values > 0, cs_ant_xrr_clip_res_arr, np.nan)
cs_ant_xrr_clip_res = xr.DataArray(
    cs_ant_xrr_clip_res_arr,  # Use the 2D numpy array directly
    dims=['y', 'x'],  # Define dimensions
    coords={'y': cs_ant_xrr_clip_res.coords['y'], 
            'x': cs_ant_xrr_clip_res.coords['x']},
    name='precipitation'  # Rename the DataArray to 'precipitation'
)

# Create an empty DataArray to store mean precipitation mapped to basins
cs_ant_precip_xrr_basin_mapped = xr.full_like(basins_imbie, np.nan, dtype=float)

# Loop through each basin ID (Zwally basins are numbered from 1 to 27)
for basin_id in range(1, 20):
    # print(f"Processing basin {basin_id}")
    # Create a mask for the current basin
    basin_mask = basins_imbie == basin_id

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
print('Plotting')


plot_arras = [ 
              ('AVHRR', stereo_avhrr_xrr_basin_mm_per_year), 
              ('ERA5', stereo_era5_xrr_basin_mm_per_year),
              ('AIRS', stereo_airs_xrr_basin_mm_per_year),
              ('CS', cs_ant_precip_xrr_basin_mapped_mm_per_year), ]
            #   ('IMERG', stereo_img_xrr_basin_mm_per_year_5km),
            #   
            #   ] #

svnme = os.path.join(path_to_plots, 'annual_snowfall_accumulation_over_imbie_basins.png')
compare_mean_precp_plot(plot_arras, vmin=0, vmax=300, cbar_tcks=[0, 50, 100, 150, 200, 250, 300])
plt.savefig(svnme,  dpi=1000, bbox_inches='tight')


plot_arras = [ ('SSMIS-F17', stereo_ssmi_17_xrr_basin_mm_per_year),
              ('IMERG', stereo_img_xrr_basin_mm_per_year), ] #

svnme = os.path.join(path_to_plots, 'annual_snpwfall_accumulation_over_basins.png')
compare_mean_precp_plot(plot_arras, vmin=0, vmax=100, cbar_tcks=[0, 10 ,25, 50, 75, 85 ,100])
plt.savefig(svnme,  dpi=1000, bbox_inches='tight')

# Example usage
single_precp_plot(stereo_img_xrr_basin_mm_per_year['imbie'], 'IMERG', vmin=0, vmax=100)
single_precp_plot(stereo_avhrr_xrr_basin_mm_per_year['imbie'], 'AVHRR', vmin=0, vmax=350)
single_precp_plot(stereo_era5_xrr_basin_mm_per_year['imbie'], 'ERA5', vmin=0, vmax=350)
single_precp_plot(stereo_ssmi_17_xrr_basin_mm_per_year['imbie'], 'SSMIS-F17', vmin=0, vmax=100)
single_precp_plot(stereo_airs_xrr_basin_mm_per_year['imbie'], 'AIRS', vmin=0, vmax=350)
single_precp_plot(cs_ant_precip_xrr_basin_mapped_res*365, 'CloudSat', vmin=0, vmax=350)


#%%
arras = [('IMERG', stereo_img_xrr_basin_mm_per_year), 
        ('AVHRR', stereo_avhrr_xrr_basin_mm_per_year), 
        ('ERA5', stereo_era5_xrr_basin_mm_per_year),
        ('SSMIS-F17', stereo_ssmi_17_xrr_basin_mm_per_year), 
        ('AIRS', stereo_airs_xrr_basin_mm_per_year),
        ('CS', cs_ant_precip_xrr_basin_mapped * 365)] #
# make a table of the mean precipitation for each basin
annual_mean_df = pd.DataFrame(columns=list(range(1, 28)), index=[x[0] for x in arras])
for product_name, data in arras:
    print(product_name)

    for basin_id in range(1, 28):
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
annual_mean_df.to_csv(os.path.join(path_to_dfs, f'annual_mean_precip_over_basins_{cde_run_dte}.csv'))


ncolors = 15

# 2) Get discrete ‘jet’ colormap
cmap = plt.get_cmap('jet', ncolors)

# 3) Define bin edges
levels = np.linspace(0,
                     500,
                     ncolors + 1)
norm = BoundaryNorm(levels, ncolors)

# 4) Plot
plt.figure(figsize=(12, 4))
im = plt.imshow(annual_mean_df, aspect='auto', cmap=cmap, norm=norm)
cbar = plt.colorbar(im, ticks=levels, spacing='proportional')
cbar.set_label('Annual Precipitation (mm)')
plt.xticks(ticks=np.arange(annual_mean_df.shape[1]),
           labels=annual_mean_df.columns, rotation=90)
plt.yticks(ticks=np.arange(annual_mean_df.shape[0]),
           labels=annual_mean_df.index)
plt.xlabel('Basin ID')
plt.ylabel('Satellite Product')
plt.title('Annual Precipitation by Basin (Discrete Jet Colormap)')
plt.tight_layout()
plt.show()
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
# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': proj})
# p = zwally_data.plot(
#     ax=ax,
#     transform=ccrs.SouthPolarStereo(),  # If data is already projected in EPSG:3031
#     cmap=cmap,
#     norm=norm,
#     add_colorbar=False
# )

# # Add white background
# ax.set_facecolor('white')

# # Add colorbar
# cbar = plt.colorbar(p, ax=ax, orientation='vertical', shrink=0.5, pad=0.05, ticks=np.arange(27))
# cbar.set_label("Basin Index")

# # Optional: Add coastlines or other features
# ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
# ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.4)

# # Set limits for Antarctica view
# ax.set_extent([-2800000, 2800000, -2800000, 2800000], crs=ccrs.SouthPolarStereo())

# # Final cleanup
# ax.set_title("Zwally Basins")
# plt.tight_layout()
# plt.show()

# # Ensure img_xrr_clip is a DataArray by selecting a specific variable if it's a Dataset
# if isinstance(img_xrr_clip, xr.Dataset):
#     variable_name = list(img_xrr_clip.data_vars.keys())[0]  # Select the first variable
#     img_xrr_clip = img_xrr_clip[variable_name]

# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.SouthPolarStereo()})
# img_xrr_clip.plot(
#     ax=ax,
#     transform=ccrs.SouthPolarStereo(),  # Data is already in polar stereographic projection
#     cmap='jet',
#     add_colorbar=True
# )
# ax.set_title("Projected Data")
# plt.show()


new_res = 5000#50000  # meters

new_shpe_ = (
    int((y_max - y_min) / new_res),
    int((x_max - x_min) / new_res)
)


new_trans = Affine(
    50000, 0.0, -3333500.0,
    0.0, -50000, 3333500.0
)

avhrr_precip_xrr_basin_mapped_res_50km = stereo_avhrr_xrr_basin_mm_per_year_5km.rio.reproject(
                                    dst_crs=stereo_avhrr_xrr_basin_mm_per_year_5km.rio.crs,
                                    shape=new_shpe_,
                                    transform=new_trans,
                                    resampling=Resampling.nearest
)

fig = plt.figure(figsize=(12, 10))
proj = ccrs.SouthPolarStereo()

# Use the 'jet' colormap
cmap = plt.cm.jet
levels = np.linspace(0, 300, 28)  # 27 basins + 1 for boundaries
norm = BoundaryNorm(levels, cmap.N)

ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())
ax.coastlines(lw=0.25, resolution="110m", zorder=2)

# Plot the data
(cs_ant_precip_xrr_basin_mapped_res_5km*360).plot(
    ax=ax,
    transform=ccrs.SouthPolarStereo(),
    cmap=cmap,
    norm=norm,
    add_colorbar=False
)

ax.add_feature(cfeature.OCEAN, zorder=1, edgecolor=None, lw=0, color="silver", alpha=0.5)
ax.set_title("CS", fontsize=20)

# Add gridlines
gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                  linestyle='--', color='k', linewidth=0.75)
gl.xlocator = MaxNLocator(nbins=5)
gl.ylocator = MaxNLocator(nbins=5)
gl.xlabel_style = {'size': 20, 'color': 'k'}
gl.ylabel_style = {'size': 20, 'color': 'k'}

# Only show specific labels
gl.top_labels = False
gl.bottom_labels = True
gl.left_labels = True
gl.right_labels = False

# Create a colorbar at the bottom
cb = fig.colorbar(
    ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    orientation="horizontal",
    fraction=0.04,  # Fraction of the original axes height
    pad=0.15,  # Distance from the bottom of the subplots
    extend="max"
)
cb.set_ticks([0,50,100,150,200,250,300])
cb.ax.tick_params(labelsize=20)
cb.set_label("Precipitation [mm/year]", fontsize=20)

# Show the plot
plt.tight_layout()
plt.show()

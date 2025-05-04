#%%
# packages
import gc
import os
import pandas as pd
import numpy as np

import xarray as xr

from program_utils import *

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


#%%
# floating variables
misc_out = r'/ra1/pubdat/AVHRR_CloudSat_proj/miscelaneous_outs'

crs = "+proj=longlat +datum=WGS84 +no_defs"  
crs_format = 'proj4' 

batch_size = 10

#----------------------------------------------------------------------------------

basins  = xr.open_dataset(basins_path)
basins_zwally = basins['zwally']
# Extract the bounds of the Zwally basins data
x_min, x_max = basins_zwally['x'].values.min(), basins_zwally['x'].values.max()
y_min, y_max = basins_zwally['y'].values.min(), basins_zwally['y'].values.max()
basin_bounds = (x_min, x_max, y_min, y_max)

# Print the bounds for verification
print(f"Basin bounds: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

# Load the GRACE dataset
grace = xr.open_dataset(grace_path)

img_fle_lst = [os.path.join(imerg_basin_path, x) for x in os.listdir(imerg_basin_path)]
avhrr_fle_lst = [os.path.join(avhrr_basin_path, x) for x in os.listdir(avhrr_basin_path)]
ssmis_17_fle_lst = [os.path.join(ssmis_17_basin_path, x) for x in os.listdir(ssmis_17_basin_path)]
airs_fle_lst = [os.path.join(airs_basin_path, x) for x in os.listdir(airs_basin_path)]
era5_fle_lst = [os.path.join(era5_basin_path, x) for x in os.listdir(era5_basin_path)]

#%%
# read and process satellite precipitation data
# Split the file list into batches of 10
batches = [img_fle_lst[i:i + batch_size] for i in range(0, len(img_fle_lst[:21]), batch_size)]

img_batch_results = run_batched_processing(batches, basins_zwally)

# Calculate the final mean across all batches
img_final_result = xr.concat(img_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_img_xrr_basin_mm_per_year = img_final_result * 365

# Initialize a list to store the results from each batch
# batch_results = []

# for batch in batches:
#     # Process each batch
#     img_precip_bsn_xrr = xr.concat([xr.open_dataarray(x) for x in batch], dim='time')

#     # Calculate mean precipitation for each basin
#     stereo_img_xrr_basin_mapped = xr.full_like(basins_zwally, np.nan, dtype=float)

#     for basin_id in range(1, 28):  # Zwally basins are numbered from 1 to 27
#         basin_mask = basins_zwally == basin_id  # Create a mask for the current basin
#         basin_precip = img_precip_bsn_xrr.where(basin_mask.data)  # Mask the precipitation data for the basin
#         basin_mean_precip = basin_precip.mean(dim=['time', 'x', 'y'], skipna=True)  # Calculate a single mean precipitation

#         stereo_img_xrr_basin_mapped = stereo_img_xrr_basin_mapped.where(~basin_mask, basin_mean_precip)

#     del(img_precip_bsn_xrr, basin_precip, basin_mean_precip, basin_mask)

#     # Append the result for this batch
#     batch_results.append(stereo_img_xrr_basin_mapped)


# unique_values = np.unique(stereo_img_xrr_basin_mm_per_year.data[~np.isnan(stereo_img_xrr_basin_mm_per_year.data)])

# # Convert the unique values into a human-readable format
# readable_values = [f"{value:.2f}" for value in unique_values]
# print("Unique non-NaN values (rounded):", readable_values)
#----------------------------------------------------------------------------------

# read and process era5 data

batches = [era5_fle_lst[i:i + batch_size] for i in range(0, len(era5_fle_lst[:21]), batch_size)]

era5_batch_results = run_batched_processing(batches, basins_zwally)

# Calculate the final mean across all batches
era5_final_result = xr.concat(era5_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_era5_xrr_basin_mm_per_year = era5_final_result * 365

# era5_precip_bsn_xrr = xr.concat([xr.open_dataarray(x) for x in era5_fle_lst], dim='time')

# # Calculate mean precipitation for each basin
# stereo_era5_xrr_basin_mapped = xr.full_like(basins_zwally, np.nan, dtype=float)

# for basin_id in range(1, 28):  # Zwally basins are numbered from 1 to 27
#     basin_mask = basins_zwally == basin_id  # Create a mask for the current basin
#     basin_precip = era5_precip_bsn_xrr.where(basin_mask.data)  # Mask the precipitation data for the basin
#     basin_mean_precip = basin_precip.mean(dim=['time','x', 'y'], skipna=True)  # Calculate a single mean precipitation

#     stereo_era5_xrr_basin_mapped = stereo_era5_xrr_basin_mapped.where(~basin_mask, basin_mean_precip)
# del(basin_precip, basin_mean_precip, basin_mask)

#----------------------------------------------------------------------------------

# read and process avhrr data

batches = [avhrr_fle_lst[i:i + batch_size] for i in range(0, len(avhrr_fle_lst[:21]), batch_size)]

avhrr_batch_results = run_batched_processing(batches, basins_zwally)

# Calculate the final mean across all batches
avhrr_final_result = xr.concat(avhrr_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_avhrr_xrr_basin_mm_per_year = avhrr_final_result * 365

# avhrr_precip_bsn_xrr = xr.concat([xr.open_dataarray(x) for x in avhrr_fle_lst], dim='time')

# # Calculate mean precipitation for each basin
# stereo_avhrr_xrr_basin_mapped = xr.full_like(basins_zwally, np.nan, dtype=float)

# for basin_id in range(1, 28):  # Zwally basins are numbered from 1 to 27
#     basin_mask = basins_zwally == basin_id  # Create a mask for the current basin
#     basin_precip = avhrr_precip_bsn_xrr.where(basin_mask.data)  # Mask the precipitation data for the basin
#     basin_mean_precip = basin_precip.mean(dim=['time','x', 'y'], skipna=True)  # Calculate a single mean precipitation

#     stereo_avhrr_xrr_basin_mapped = stereo_avhrr_xrr_basin_mapped.where(~basin_mask, basin_mean_precip)

# del(basin_precip, basin_mean_precip, basin_mask)


#----------------------------------------------------------------------------------
# read and process ssmi_17 data

batches = [ssmis_17_fle_lst[i:i + batch_size] for i in range(0, len(ssmis_17_fle_lst[:21]), batch_size)]

ssmis_batch_results = run_batched_processing(batches, basins_zwally)

# Calculate the final mean across all batches
ssmis_final_result = xr.concat(ssmis_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_ssmi_17_xrr_basin_mm_per_year = ssmis_final_result * 365

# ssmis_17_precip_bsn_xrr = xr.concat([xr.open_dataarray(x) for x in ssmis_17_fle_lst], dim='time')

# # Calculate mean precipitation for each basin
# stereo_ssmi_17_xrr_basin_mapped = xr.full_like(basins_zwally, np.nan, dtype=float)

# for basin_id in range(1, 28):  # Zwally basins are numbered from 1 to 27
#     basin_mask = basins_zwally == basin_id  # Create a mask for the current basin
#     basin_precip = ssmis_17_precip_bsn_xrr.where(basin_mask.data)  # Mask the precipitation data for the basin
#     basin_mean_precip = basin_precip.mean(dim=['time','x', 'y'], skipna=True)  # Calculate a single mean precipitation

#     stereo_ssmi_17_xrr_basin_mapped = stereo_ssmi_17_xrr_basin_mapped.where(~basin_mask, basin_mean_precip)
# del(basin_precip, basin_mean_precip, basin_mask)

#----------------------------------------------------------------------------------
# read and process airs data
batches = [airs_fle_lst[i:i + batch_size] for i in range(0, len(airs_fle_lst[:21]), batch_size)]

airs_batch_results = run_batched_processing(batches, basins_zwally)

# Calculate the final mean across all batches
airs_final_result = xr.concat(airs_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_airs_xrr_basin_mm_per_year = airs_final_result * 365

# airs_precip_bsn_xrr = xr.concat([xr.open_dataarray(x) for x in airs_fle_lst], dim='time')

# # Calculate mean precipitation for each basin
# stereo_airs_xrr_basin_mapped = xr.full_like(basins_zwally, np.nan, dtype=float)

# for basin_id in range(1, 28):  # Zwally basins are numbered from 1 to 27
#     basin_mask = basins_zwally == basin_id  # Create a mask for the current basin
#     basin_precip = airs_precip_bsn_xrr.where(basin_mask.data)  # Mask the precipitation data for the basin
#     basin_mean_precip = basin_precip.mean(dim=['time','x', 'y'], skipna=True)  # Calculate a single mean precipitation

#     stereo_airs_xrr_basin_mapped = stereo_airs_xrr_basin_mapped.where(~basin_mask, basin_mean_precip)
# del(basin_precip, basin_mean_precip, basin_mask)

#----------------------------------------------------------------------------------

#%%

# Ensure the DataArray is sorted by its coordinates before plotting
# Ensure the DataArray is sorted by both 'lat' and 'lons' in increasing order
grace['lwe_thickness'].sortby('lat').sortby('lons')[0].plot()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Custom colormap and normalization for 27 discrete basin values
colors = plt.cm.gist_ncar(np.linspace(0, 1, 27))  # Use other palettes like 'tab20b' or 'Set3' for variety
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(np.arange(-0.5, 27.5), cmap.N)

fig, ax = plt.subplots(figsize=(8, 8))

# Use masked plotting to handle NaNs gracefully
basin_data = basins['zwally']

# Ensure data is a DataArray and mask invalid values (e.g., where basin == 0 or NaN)
masked = basin_data.where(basin_data.notnull() & (basin_data >= 1) & (basin_data <= 26))

# Plot
p = masked.plot.imshow(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)

# Add colorbar with integer ticks only
cbar = plt.colorbar(p, ax=ax, ticks=np.arange(0, 27))
cbar.set_label("Basin index")

# Aesthetic clean-up
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('Zwally Basin Map')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# Set up colormap and norm for 27 discrete basins
colors = plt.cm.gist_ncar(np.linspace(0, 1, 27))
cmap = mcolors.ListedColormap(colors)
cmap.set_bad(color='white')  # Set background (masked or NaN) to white

norm = mcolors.BoundaryNorm(np.arange(-0.5, 27.5), cmap.N)

# Define Antarctic stereographic projection
proj = ccrs.SouthPolarStereo()

# Mask out invalid values (0 or NaN)
zwally_data = basins['zwally'].where((basins['zwally'] > 0) & (basins['zwally'].notnull()))

# Plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': proj})
p = zwally_data.plot(
    ax=ax,
    transform=ccrs.SouthPolarStereo(),  # If data is already projected in EPSG:3031
    cmap=cmap,
    norm=norm,
    add_colorbar=False
)

# Add white background
ax.set_facecolor('white')

# Add colorbar
cbar = plt.colorbar(p, ax=ax, orientation='vertical', shrink=0.5, pad=0.05, ticks=np.arange(27))
cbar.set_label("Basin Index")

# Optional: Add coastlines or other features
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.4)

# Set limits for Antarctica view
ax.set_extent([-2800000, 2800000, -2800000, 2800000], crs=ccrs.SouthPolarStereo())

# Final cleanup
ax.set_title("Zwally Basins")
plt.tight_layout()
plt.show()

# Ensure img_xrr_clip is a DataArray by selecting a specific variable if it's a Dataset
if isinstance(img_xrr_clip, xr.Dataset):
    variable_name = list(img_xrr_clip.data_vars.keys())[0]  # Select the first variable
    img_xrr_clip = img_xrr_clip[variable_name]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.SouthPolarStereo()})
img_xrr_clip.plot(
    ax=ax,
    transform=ccrs.SouthPolarStereo(),  # Data is already in polar stereographic projection
    cmap='jet',
    add_colorbar=True
)
ax.set_title("Projected Data")
plt.show()

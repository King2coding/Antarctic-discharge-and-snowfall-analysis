#%%
# packages
import gc
import os
import pandas as pd
import numpy as np

import xarray as xr

from program_utils import *

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Pool

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

#%%
# read and process satellite precipitation data

img_precip_bsn_xrr = xr.concat(img_precip_bsn, dim='time')

# Calculate mean precipitation for each basin
img_mean_precip_bsn = {}
for basin_id in range(1, 28):  # Zwally basins are numbered from 1 to 27
    basin_mask = basins['zwally'] == basin_id  # Create a mask for the current basin
    basin_precip = img_precip_bsn_xrr.where(basin_mask.data)  # Mask the precipitation data for the basin
    basin_mean_precip = basin_precip.mean(dim=['time','x', 'y'], skipna=True)  # Calculate a single mean precipitation
    img_mean_precip_bsn[basin_id] = basin_mean_precip

# Combine the mean precipitation values into a single DataArray
stereo_img_xrr_basin = xr.concat(stereo_img_xrr_basin, dim='basin')
stereo_img_xrr_basin.coords['basin'] = range(1, 28)  # Add basin IDs as a coordinate

# Multiply by 365 to get precipitation in mm/year
stereo_img_xrr_basin_mapped = xr.full_like(basins['zwally'], np.nan, dtype=float)

for basin_id in range(1, 28):  # Zwally basins are numbered from 1 to 27
    basin_mask = basins['zwally'] == basin_id  # Create a mask for the current basin
    basin_precip_value = stereo_img_xrr_basin[basin_id].values#.sel(basin=basin_id).values[0]  # Extract the scalar value
    stereo_img_xrr_basin_mapped = stereo_img_xrr_basin_mapped.where(~basin_mask, basin_precip_value)

stereo_img_xrr_basin_mm_per_year = stereo_img_xrr_basin_mapped * 365

era5_data = xr.open_dataarray(era5_file_2010)
# Map the mean precipitation back to the basin IDs

if 'x' in era5_data.coords and 'y' in era5_data.coords:
    era5_data = era5_data.rename({'x': 'lon', 'y': 'lat'})
era5_data = mean_res(era5_data)

airs_data = xr.open_dataarray(airs_file_2010)
if 'x' in airs_data.coords and 'y' in airs_data.coords:
    airs_data = airs_data.rename({'x': 'lon', 'y': 'lat'})
airs_data = mean_res(airs_data)

imerg_fn = process_imerg(imerg_files_2010,'imerg_fn')
imerg_fn = mean_res(imerg_fn)

avhrr_precip = process_sim(avhrr_precp_files,'2010')

ssmis_17_precip = xr.concat([xr.open_dataarray(x) for x in ssmis_17_files_2010], dim='time')
ssmis_17_precip = ssmis_17_precip * 24  # Multiply the "precipitation" data by 24
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

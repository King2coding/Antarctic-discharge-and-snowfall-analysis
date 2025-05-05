#%%
# packages
import gc
import os
import pandas as pd
import numpy as np

import xarray as xr

from program_utils import *
from affine import Affine

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

img_fle_lst = [os.path.join(imerg_basin_path, x) for x in os.listdir(imerg_basin_path)]
avhrr_fle_lst = [os.path.join(avhrr_basin_path, x) for x in os.listdir(avhrr_basin_path)]
ssmis_17_fle_lst = [os.path.join(ssmis_17_basin_path, x) for x in os.listdir(ssmis_17_basin_path)]
airs_fle_lst = [os.path.join(airs_basin_path, x) for x in os.listdir(airs_basin_path)]
era5_fle_lst = [os.path.join(era5_basin_path, x) for x in os.listdir(era5_basin_path)]

gc.collect()
#%%
# read and process satellite precipitation data
# Split the file list into batches of 10
batches = [img_fle_lst[i:i + batch_size] for i in range(0, len(img_fle_lst), batch_size)]

img_batch_results = run_batched_processing(batches, basins_zwally)

# Calculate the final mean across all batches
img_final_result = xr.concat(img_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_img_xrr_basin_mm_per_year = img_final_result * 365

fle_svnme = os.path.join(annual_precip_in_basins_path, 'IMERG_2010_basin_annual_precip.nc')
encoding = {stereo_img_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
stereo_img_xrr_basin_mm_per_year.to_netcdf(os.path.join(imerg_basin_path, fle_svnme), 
                                           mode='w', format='NETCDF4', encoding=encoding)

# resample the data to the new resolution

# Explicitly set the CRS before reprojecting
stereo_img_xrr_basin_mm_per_year.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

stereo_img_xrr_basin_mm_per_year_5km = stereo_img_xrr_basin_mm_per_year.rio.reproject(
                                    dst_crs=stereo_img_xrr_basin_mm_per_year.rio.crs,
                                    shape=(1333, 1333),
                                    transform=new_transform,
                                    resampling=Resampling.nearest
)

gc.collect()

#----------------------------------------------------------------------------------

# read and process era5 data

batches = [era5_fle_lst[i:i + batch_size] for i in range(0, len(era5_fle_lst), batch_size)]

era5_batch_results = run_batched_processing(batches, basins_zwally)

# Calculate the final mean across all batches
era5_final_result = xr.concat(era5_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_era5_xrr_basin_mm_per_year = era5_final_result * 365

fle_svnme = os.path.join(annual_precip_in_basins_path, 'ERA5_2010_basin_annual_precip.nc')
encoding = {stereo_era5_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
stereo_era5_xrr_basin_mm_per_year.to_netcdf(os.path.join(era5_basin_path, fle_svnme), 
                                            mode='w', format='NETCDF4', encoding=encoding)

# resample the data to the new resolution
stereo_era5_xrr_basin_mm_per_year.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

stereo_era5_xrr_basin_mm_per_year_5km = stereo_era5_xrr_basin_mm_per_year.rio.reproject(
                                    dst_crs=stereo_era5_xrr_basin_mm_per_year.rio.crs,
                                    shape=(1333, 1333),
                                    transform=new_transform,
                                    resampling=Resampling.nearest
)

gc.collect()

#----------------------------------------------------------------------------------

# read and process avhrr data

batches = [avhrr_fle_lst[i:i + batch_size] for i in range(0, len(avhrr_fle_lst), batch_size)]

avhrr_batch_results = run_batched_processing(batches, basins_zwally)

# Calculate the final mean across all batches
avhrr_final_result = xr.concat(avhrr_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_avhrr_xrr_basin_mm_per_year = avhrr_final_result * 365

fle_svnme = os.path.join(annual_precip_in_basins_path, 'AVHRR_2010_basin_annual_precip.nc')
encoding = {stereo_avhrr_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
stereo_avhrr_xrr_basin_mm_per_year.to_netcdf(os.path.join(avhrr_basin_path, fle_svnme), 
                                             mode='w', format='NETCDF4', encoding=encoding)

# resample the data to the new resolution
stereo_avhrr_xrr_basin_mm_per_year.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

stereo_avhrr_xrr_basin_mm_per_year_5km = stereo_avhrr_xrr_basin_mm_per_year.rio.reproject(
                                    dst_crs=stereo_avhrr_xrr_basin_mm_per_year.rio.crs,
                                    shape=(1333, 1333),
                                    transform=new_transform,
                                    resampling=Resampling.nearest
)

gc.collect()

#----------------------------------------------------------------------------------
# read and process ssmi_17 data

batches = [ssmis_17_fle_lst[i:i + batch_size] for i in range(0, len(ssmis_17_fle_lst), batch_size)]

ssmis_batch_results = run_batched_processing(batches, basins_zwally)

# Calculate the final mean across all batches
ssmis_final_result = xr.concat(ssmis_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_ssmi_17_xrr_basin_mm_per_year = ssmis_final_result * 365

fle_svnme = os.path.join(annual_precip_in_basins_path, 'SSMIS_17_2010_basin_annual_precip.nc')
encoding = {stereo_ssmi_17_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
stereo_ssmi_17_xrr_basin_mm_per_year.to_netcdf(os.path.join(ssmis_17_basin_path, fle_svnme), 
                                               mode='w', format='NETCDF4', encoding=encoding)

# resample the data to the new resolution
stereo_ssmi_17_xrr_basin_mm_per_year.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

stereo_ssmi_17_xrr_basin_mm_per_year_5km = stereo_ssmi_17_xrr_basin_mm_per_year.rio.reproject(
                                    dst_crs=stereo_ssmi_17_xrr_basin_mm_per_year.rio.crs,
                                    shape=(1333, 1333),
                                    transform=new_transform,
                                    resampling=Resampling.nearest
)

gc.collect()

#----------------------------------------------------------------------------------
# read and process airs data
batches = [airs_fle_lst[i:i + batch_size] for i in range(0, len(airs_fle_lst), batch_size)]

airs_batch_results = run_batched_processing(batches, basins_zwally)

# Calculate the final mean across all batches
airs_final_result = xr.concat(airs_batch_results, dim='batch').mean(dim='batch', skipna=True)

stereo_airs_xrr_basin_mm_per_year = airs_final_result * 365

fle_svnme = os.path.join(annual_precip_in_basins_path, 'AIRS_2010_basin_annual_precip.nc')
encoding = {stereo_airs_xrr_basin_mm_per_year.name:{"zlib": True, "complevel": 9}}
stereo_airs_xrr_basin_mm_per_year.to_netcdf(os.path.join(airs_basin_path, fle_svnme), 
                                            mode='w', format='NETCDF4', encoding=encoding)

# resample the data to the new resolution
stereo_airs_xrr_basin_mm_per_year.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

stereo_airs_xrr_basin_mm_per_year_5km = stereo_airs_xrr_basin_mm_per_year.rio.reproject(
                                    dst_crs=stereo_airs_xrr_basin_mm_per_year.rio.crs,
                                    shape=(1333, 1333),
                                    transform=new_transform,
                                    resampling=Resampling.nearest
)

gc.collect()

#%%
# plot
import cartopy.crs as ccrs
import cartopy.crs as ccrs

from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.cm import ScalarMappable

import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm

def compare_mean_precp_plot(arr_lst_mean, vmin=0, vmax=300):
    """
    Plots a multi-row grid of mean precipitation for different products over 27 basins.

    Parameters:
    arr_lst_mean (list of tuples): List of tuples with product names and their mean precipitation data.
    vmin (float): Minimum value for colorbar.
    vmax (float): Maximum value for colorbar.
    """
    proj = ccrs.SouthPolarStereo()

    # Use the 'jet' colormap
    cmap = plt.cm.jet
    levels = np.linspace(vmin, vmax, 28)  # 27 basins + 1 for boundaries
    norm = BoundaryNorm(levels, cmap.N)

    # Determine the number of rows and columns for the grid
    n_products = len(arr_lst_mean)
    ncols = 3  # Number of columns
    nrows = (n_products + ncols - 1) // ncols  # Calculate rows needed

    # Create a GridSpec with precise control over spacing
    fig = plt.figure(figsize=(28, 10 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.025, hspace=0.15)

    # Loop through each dataset to create the plots
    axes = []
    for i, (product_name, data) in enumerate(arr_lst_mean):
        ax = fig.add_subplot(gs[i], projection=proj)
        ax.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())
        ax.coastlines(lw=0.25, resolution="110m", zorder=2)

        # Plot the data
        data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False
        )

        ax.add_feature(cfeature.OCEAN, zorder=1, edgecolor=None, lw=0, color="silver", alpha=0.5)
        ax.set_title(product_name, fontsize=20)

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                          linestyle='--', color='k', linewidth=0.75)
        gl.xlocator = MaxNLocator(nbins=5)
        gl.ylocator = MaxNLocator(nbins=5)
        gl.xlabel_style = {'size': 20, 'color': 'k'}
        gl.ylabel_style = {'size': 20, 'color': 'k'}

        # Only show specific labels
        gl.top_labels = False
        gl.bottom_labels = i >= n_products - ncols  # Bottom labels only for the last row
        gl.left_labels = i % ncols == 0  # Left labels for the first column
        gl.right_labels = i % ncols == ncols - 1  # Right labels for the last column

        axes.append(ax)

    # Create a colorbar at the bottom spanning all subplots
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,  # Attach the colorbar to all axes for better placement
        orientation="horizontal",
        fraction=0.04,  # Fraction of the original axes height
        pad=0.1,  # Distance from the bottom of the subplots
        extend="max"
    )
    cb.ax.tick_params(labelsize=20)
    cb.set_label("Snowfall Rate (mm/year)", fontsize=20)

    # Show the plot
    plt.tight_layout()


plot_arras = [('IMERG', stereo_img_xrr_basin_mm_per_year_5km), 
              ('AVHRR', stereo_avhrr_xrr_basin_mm_per_year_5km), 
              ('SSMIS-F17', stereo_ssmi_17_xrr_basin_mm_per_year_5km), 
              ('AIRS', stereo_airs_xrr_basin_mm_per_year_5km),
              ('ERA5', stereo_era5_xrr_basin_mm_per_year_5km)] #

svnme = os.path.join(misc_out, 'mean_precp_over_basins.png')
compare_mean_precp_plot(plot_arras, vmin=0, vmax=300)

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

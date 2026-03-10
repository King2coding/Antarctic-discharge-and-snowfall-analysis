#%%
# packages
import warnings
warnings.filterwarnings("ignore")

import gc
import os
import glob
import pandas as pd
import numpy as np
import datetime
import calendar
from functools import reduce

from datetime import datetime, timedelta, date
from scipy.stats import linregress
import math
from collections import Counter
import HydroErr as he

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FixedLocator
from matplotlib.colors import LogNorm, Normalize

from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.cm import get_cmap
import matplotlib.gridspec as gridspec

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib import cm
import matplotlib as mpl

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import subprocess
from scipy.interpolate import griddata
import xarray as xr
# import xesmf as xe
from rasterio.warp import Resampling
from pyproj import CRS, Transformer

from osgeo import gdal, osr

import h5py

#%%
# floating variables
cde_run_dte = date.today().strftime('%Y%m%d')
misc_out = r'/ra1/pubdat/AVHRR_CloudSat_proj/miscelaneous_outs'

crs = "+proj=longlat +datum=WGS84 +no_defs"  
crs_format = 'proj4'

crs_stereo = "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84"

SEAS = ["DJF", "MAM", "JJA", "SON"]

SEAS_ORDER = ("DJF", "MAM", "JJA", "SON")

color_cycle = ["k", "tab:blue", "tab:orange", "tab:green", "tab:red"]
marker_cycle = ["o", "s", "D", "^", "v"]
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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

#----------------------------------------------------------------------------

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------------------- Stable basin palette (IDs 2..19) ----------------------
BASIN_IDS = np.arange(2, 20)  # [2, 3, ..., 19]
PALETTE   = plt.cm.gist_ncar(np.linspace(0, 1, len(BASIN_IDS)))
ID2COLOR  = dict(zip(BASIN_IDS, PALETTE))

#%%
def p_corr(obs,model):
    return he.pearson_r(model,obs)

def rmsqe(obs,model):
    return he.rmse(model,obs)
# fucntions
def relative_bias(obs, model):
    obs = np.asarray(obs, dtype=float)
    model = np.asarray(model, dtype=float)

    m = np.isfinite(obs) & np.isfinite(model)
    if m.sum() == 0:
        return np.nan

    mu_obs = np.mean(obs[m])
    if mu_obs == 0:
        return np.nan  # or np.inf / raise, depending on your preference

    mu_residuals = np.mean(model[m] - obs[m])
    return mu_residuals / mu_obs

# rmse
# Range 0 RMSE < inf, smaller is better.
# Notes: The standard deviation of the residuals. A lower spread indicates that the points are better concentrated
# around the line of best fit (linear). Random errors do not cancel. This metric will highlights larger errors.

def rmsqe(obs,model):
    return he.rmse(model,obs)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def calculate_metrics(df, xcol, ycol):
    df = df.dropna(axis=0, how='any', subset=[xcol, ycol])  # Drop rows with NaN in specified columns
    xdf = df[xcol].values
    ydf = df[ycol].values
    
    
    # if len(xdf) == 0 or len(ydf) == 0:
    #     return np.nan, np.nan, np.nan
    
    # Calculate metrics
    rb = round(relative_bias(xdf, ydf) * 100,1)  # Relative Bias in %
    rmse = round(rmsqe(xdf, ydf), 2)  # Root Mean Square Error
    cc = round(p_corr(xdf, ydf), 2)   # Pearson Correlation Coefficient    
    mae  = round(np.mean(np.abs(ydf - xdf)), 2) # Mean Absolute Error

    return {
        'Bias':rb, 
        'RMSE':rmse, 
        'CC':cc, 
        'MAE':mae
        } # rb, rmse, cc, mae


def ds_swaplon(ds):
    """
    Swap longitude coordinates from [0, 360] to [-180, 180] and sort.
    Handles both 'lon' and 'longitude' coordinate names.
    """
    var = 'lon' if 'lon' in ds.coords else 'longitude' if 'longitude' in ds.coords else None
    if var is None:
        raise ValueError("No longitude coordinate found in dataset (expected 'lon' or 'longitude').")
    new_lon = (((ds[var] + 180) % 360) - 180)
    ds = ds.assign_coords({var: new_lon})
    ds = ds.sortby(var)
    return ds
#----------------------------------------------------------------------------
# function to read and process ERA5 precip file
def process_era5_file(file_info):
    # idx, file_path = file_info
    # if idx % 5 == 0:
    #     print(f"Processing ERA5 file {idx+1}")
    era5_xr = xr.open_dataset(file_info, engine='netcdf4')
    era5_xr = era5_xr.rename({'valid_time': 'time'})
    era5_xr = ds_swaplon(era5_xr)
    # data units are in m per day, convert to mm/day using 1000 factor
    era5_xr = era5_xr['tp'] * 1000  # mm/h
    era5_xr = era5_xr * 24  # mm/day
    # # write crs and resample to gpcp resolution
    # era5_xr.rio.write_crs(cc.to_string(), inplace=True)
    # era5_xr = era5_xr.rio.reproject(
    #     era5_xr.rio.crs,
    #     shape=(360, 720),#gpcp_ds_v3pt2_xr['precip'].shape[1:], # # set the shape as the GPCP data
    #     resampling=Resampling.average,
    # )
    return era5_xr
def process_era5_file_to_basin(era5_xr_data, er_tme, basin, fle_svnme):

    basin_transform = basin.rio.transform()
    height, width = basin.data.shape[1:]
    xmin, ymax = basin_transform.c, basin_transform.f
    xres, yres = basin_transform.a, -basin_transform.e
    xmax = xmin + width * xres
    ymin = ymax - height * yres

    er_time = pd.to_datetime(er_tme).strftime('%Y%m%d')

    er_precip_dat = era5_xr_data.sel(time=er_tme).sel(lat=slice(-55, -90))

    yshp, xshp = er_precip_dat.shape

    minx = er_precip_dat['lon'].min().item()
    maxy = er_precip_dat['lat'].max().item()
    px_sz = round(er_precip_dat['lon'].diff('lon').mean().item(), 2)

    dest_flnme = os.path.join(misc_out, os.path.basename(fle_svnme).replace('.nc', '_sh.nc'))

    gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, er_precip_dat.data)

    output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('_sh.nc', '_sh_stere.nc'))

    gdalwarp_command = [
        'gdalwarp',
        '-t_srs', '+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84',
        '-r', 'near',
        dest_flnme,
        output_file_stereo
    ]

    subprocess.run(gdalwarp_command, 
                   shell=False, check=True,
                   stdout=subprocess.DEVNULL, 
                   stderr=subprocess.DEVNULL)

    # Read the stereographic projection file
    er_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

    os.remove(dest_flnme)
    os.remove(output_file_stereo)

    # Clip the data to the bounds of the basin dataset
    er_xrr_clip = er_xrr_sh_stereo.sel(
        x=slice(xmin, xmax),
        y=slice(ymin, ymax)
    ).squeeze()

    ccrs = CRS.from_proj4('+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84')

    # Explicitly set the CRS before reprojecting
    er_xrr_clip.rio.write_crs(ccrs.to_string(), inplace=True)

    er_xrr_clip_res = er_xrr_clip.rio.reproject(
        er_xrr_clip.rio.crs,
        shape=(height, width),  # set the shape as the basin data shape
        resampling=Resampling.bilinear,
        transform=basin.rio.transform()
    )

    # Add the time coordinate to the reprojected DataArray
    er_xrr_clip_res_arr = er_xrr_clip_res['Band1'].values
    er_xrr_clip_res_arr = np.where(basin.values > 0, er_xrr_clip_res_arr, np.nan)
    er_xrr_clip_res = xr.DataArray(
        np.expand_dims(er_xrr_clip_res_arr[0], axis=0),  # Expand dimensions to match ['time', 'y', 'x']
        dims=['time', 'y', 'x'],
        coords={'time': [np.datetime64(pd.to_datetime(er_time, format='%Y%m%d'), 'D')], 
                'y': er_xrr_clip_res.coords['y'], 
                'x': er_xrr_clip_res.coords['x']},
        name='precipitation'  # Change the variable name to 'precipitation'
    )

    return er_xrr_clip_res
#----------------------------------------------------------------------------
def process_gpm_precip_file(xr_da, tme, basin, flenme2svnme):

        basin_transform = basin.rio.transform()
        height, width = basin.data.shape[1:]
        xmin, ymax = basin_transform.c, basin_transform.f
        xres, yres = basin_transform.a, -basin_transform.e
        xmax = xmin + width * xres
        ymin = ymax - height * yres       

        ant_precip_dat = xr_da.sel(lat=slice(-90, -55)).compute()

        # Make it match GDAL north-up convention: row 0 = max lat
        if ant_precip_dat["lat"][0] < ant_precip_dat["lat"][-1]:
            ant_precip_dat = ant_precip_dat.sortby("lat", ascending=False)
        
        # print("lat[0], lat[-1] =", float(ant_precip_dat.lat[0]), float(ant_precip_dat.lat[-1]))

        yshp, xshp = ant_precip_dat.shape

        # minx = ant_precip_dat['lon'].min().item()
        # maxy = ant_precip_dat['lat'].max().item()
        px_sz = float(ant_precip_dat["lon"].diff("lon").mean())
        origin_x = float(ant_precip_dat["lon"].min()) - px_sz/2
        origin_y = float(ant_precip_dat["lat"].max()) + px_sz/2
        
        dest_flnme = os.path.join(misc_out, os.path.basename(flenme2svnme).replace('.nc', '_sh.tif'))

        gdal_based_save_array_to_disk2(dest_flnme, xshp, yshp, px_sz, origin_x, origin_y, crs, crs_format, ant_precip_dat.values)

        # print(gdal.Info(dest_flnme) + ' of file')

        output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('_sh.tif', '_sh_stere.tif'))

        gdalwarp_command = [
        "gdalwarp",
        "-s_srs", "EPSG:4326",
        "-t_srs", "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84",
        "-of", "GTiff",
        "-r", "bilinear",                 # keep near if you insist, but see note below
        "-tr", str(xres), str(yres),  # 10000 10000 (meters)
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-tap",
        dest_flnme,
        output_file_stereo,
        ]
        gdalwarp_command += ["-srcnodata", "-9999.9", "-dstnodata", "nan"]

        subprocess.run(gdalwarp_command, 
                    shell=False, check=True,
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL)
        
        # print(gdal.Info(output_file_stereo) + ' of stereo file')

        # Read the stereographic projection file
        ant_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

        os.remove(dest_flnme)
        os.remove(output_file_stereo)

        # Clip the data to the bounds of the basin dataset
        # ant_xrr_clip = ant_xrr_sh_stereo.sel(
        #     x=slice(xmin, xmax),
        #     y=slice(ymin, ymax)
        # ).squeeze()

        ycoord = ant_xrr_sh_stereo["y"]
        ysel = slice(ymax, ymin) if ycoord[0] > ycoord[-1] else slice(ymin, ymax)

        ant_xrr_clip = ant_xrr_sh_stereo.sel(x=slice(xmin, xmax), y=ysel).squeeze()

        ccrs = CRS.from_proj4('+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84')

        # Explicitly set the CRS before reprojecting
        ant_xrr_clip.rio.write_crs(ccrs.to_string(), inplace=True)

        ant_xrr_clip_res = ant_xrr_clip.rio.reproject(
            ant_xrr_clip.rio.crs,
            shape=(height, width),  # set the shape as the basin data shape
            resampling=Resampling.nearest,
            transform=basin.rio.transform()
        )

        # Add the time coordinate to the reprojected DataArray
        # after reprojection:
        # ant_xrr_clip_res is a Dataset -> extract the data variable as a DataArray
        if isinstance(ant_xrr_clip_res, xr.Dataset):
            if "band_data" in ant_xrr_clip_res:
                ant_xrr_clip_res_da = ant_xrr_clip_res["band_data"]
            elif "Band1" in ant_xrr_clip_res:
                ant_xrr_clip_res_da = ant_xrr_clip_res["Band1"]
            else:
                # fall back: take the first data variable
                ant_xrr_clip_res_da = next(iter(ant_xrr_clip_res.data_vars.values()))
        else:
            ant_xrr_clip_res_da = ant_xrr_clip_res
        ant_xrr_clip_res_da = ant_xrr_clip_res_da.where(basin.squeeze() > 0)
        ant_xrr_clip_res_da = ant_xrr_clip_res_da.expand_dims(
        time=[np.datetime64(pd.to_datetime(tme, format="%Y%m%d"), "D")]
        ).rename("precipitation")

        return ant_xrr_clip_res_da
#----------------------------------------------------------------------------
# function to read and process AIRS precip file
def process_airs_file_to_basin(airs_xrr_data, ai_tme, basin, fle_svnme):

    basin_transform = basin.rio.transform()
    height, width = basin.data.shape[1:]
    xmin, ymax = basin_transform.c, basin_transform.f
    xres, yres = basin_transform.a, -basin_transform.e
    xmax = xmin + width * xres
    ymin = ymax - height * yres
        
    airs_time = pd.to_datetime(ai_tme).strftime('%Y%m%d')

    airs_precip_dat = airs_xrr_data.sel(time=ai_tme).sel(lat=slice(-55, -90))


    yshp, xshp = airs_precip_dat.shape

    minx = airs_precip_dat['lon'].min().item()
    maxy = airs_precip_dat['lat'].max().item()
    px_sz = round(airs_precip_dat['lon'].diff('lon').mean().item(), 2)

    dest_flnme = os.path.join(misc_out, os.path.basename(fle_svnme).replace('.nc', '_sh.nc'))

    gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, airs_precip_dat.data)

    output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('_sh.nc', '_sh_stere.nc'))

    gdalwarp_command = [
        'gdalwarp',
        '-t_srs', '+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84',
        '-r', 'near',
        dest_flnme,
        output_file_stereo
    ]

    subprocess.run(gdalwarp_command, 
                   shell=False, check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    # Read the stereographic projection file
    ai_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

    os.remove(dest_flnme)
    os.remove(output_file_stereo)

    # Clip the data to the bounds of the basin dataset
    ai_xrr_clip = ai_xrr_sh_stereo.sel(
        x=slice(xmin, xmax),
        y=slice(ymax, ymin)
    ).squeeze()

    ccrs = CRS.from_proj4('+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84')

    # Explicitly set the CRS before reprojecting
    ai_xrr_clip.rio.write_crs(ccrs.to_string(), inplace=True)
    ai_xrr_clip_res = ai_xrr_clip.rio.reproject(
        ai_xrr_clip.rio.crs,
        shape=(height, width),  # set the shape as the basin data shape
        resampling=Resampling.bilinear,
        transform=basin.rio.transform()
    )

    # Add the time coordinate to the reprojected DataArray
    ai_xrr_clip_res_arr = ai_xrr_clip_res['Band1'].values
    ai_xrr_clip_res_arr = np.where(basin.values > 0, ai_xrr_clip_res_arr, np.nan)
    ai_xrr_clip_res = xr.DataArray(
        np.expand_dims(ai_xrr_clip_res_arr[0], axis=0),  # Expand dimensions to match ['time', 'y', 'x']
        dims=['time', 'y', 'x'],
        coords={'time': [np.datetime64(pd.to_datetime(airs_time, format='%Y%m%d'), 'D')], 'y': ai_xrr_clip_res.coords['y'], 'x': ai_xrr_clip_res.coords['x']},
        name='precipitation'  # Change the variable name to 'precipitation'
    )

    return ai_xrr_clip_res
#----------------------------------------------------------------------------
def process_ssmis_file_to_basin(ss, basin):
       
    basin_transform = basin.rio.transform()
    height, width = basin.data.shape[1:]
    xmin, ymax = basin_transform.c, basin_transform.f
    xres, yres = basin_transform.a, -basin_transform.e
    xmax = xmin + width * xres
    ymin = ymax - height * yres

    ss_time = os.path.basename(ss).split('.')[4].split('-')[0]

    ssmi_precip = xr.open_dataarray(ss)
    ssmi_precip = ssmi_precip * 24  # Multiply the "precipitation" data by 24

    ss_xrr_sh = ssmi_precip.sel(lat=slice(-55, -90)).squeeze()
    yshp, xshp = ss_xrr_sh.shape

    minx = ss_xrr_sh['lon'].min().item()
    maxy = ss_xrr_sh['lat'].max().item()
    px_sz = round(ss_xrr_sh['lon'].diff('lon').mean().item(), 2)

    dest_flnme = os.path.join(misc_out, os.path.basename(ss).replace('.nc', '_sh.nc'))

    gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, ss_xrr_sh.data)

    output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('_sh.nc', '_sh_stere.nc'))

    gdalwarp_command = [
        'gdalwarp',
        '-t_srs', '+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84',
        '-r', 'near',
        dest_flnme,
        output_file_stereo
    ]
    subprocess.run(gdalwarp_command, 
                   shell=False, check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    # Read the stereographic projection file
    ss_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

    os.remove(dest_flnme)
    os.remove(output_file_stereo)

    # Clip the data to the bounds of the basin dataset
    ss_xrr_clip = ss_xrr_sh_stereo.sel(
        x=slice(xmin, xmax),
        y=slice(ymax, ymin)
    ).squeeze()

    ccrs = CRS.from_proj4('+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84')

    # Explicitly set the CRS before reprojecting
    ss_xrr_clip.rio.write_crs(ccrs.to_string(), inplace=True)
    ss_xrr_clip_res = ss_xrr_clip.rio.reproject(
        ss_xrr_clip.rio.crs,
        shape=(height, width),  # set the shape as the basin data shape
        resampling=Resampling.bilinear,
        transform=basin.rio.transform()
    )

    # Add the time coordinate to the reprojected DataArray
    ss_xrr_clip_res_arr = ss_xrr_clip_res['Band1'].values
    ss_xrr_clip_res_arr = np.where(basin.values > 0, ss_xrr_clip_res_arr, np.nan)
    ss_xrr_clip_res = xr.DataArray(
        np.expand_dims(ss_xrr_clip_res_arr, axis=0),  # Expand dimensions to match ['time', 'y', 'x']
        dims=['time', 'y', 'x'],
        coords={'time': [ss_time], 'y': ss_xrr_clip_res.coords['y'], 'x': ss_xrr_clip_res.coords['x']},
        name='precipitation'  # Change the variable name to 'precipitation'
    )

    return ss_xrr_clip_res
#----------------------------------------------------------------------------
def process_avhrr_file_to_basin(file, yr, basin):

    # Extract basin shape and transformation details
    basin_transform = basin.rio.transform()
    height, width = basin.data.shape[1:]
    xmin, ymax = basin_transform.c, basin_transform.f
    xres, yres = basin_transform.a, -basin_transform.e
    xmax = xmin + width * xres
    ymin = ymax - height * yres

    # Read AVHRR file and extract coordinates and data
    av_x, av_y = return_sim_cords(file)
    av_arr, av_time = read_tif_sim_file(file, yr)

    av_arr = np.expand_dims(av_arr, axis=0) if av_arr.ndim == 2 else av_arr  # Ensure av_arr has 3 dimensions
    av_xrr = xr.DataArray(av_arr, dims=['time', 'lat', 'lon'], coords={'time': [av_time], 'lat': av_y, 'lon': av_x})
    av_xrr_sh = av_xrr.sel(lat=slice(-55, -90)).squeeze()
    yshp, xshp = av_xrr_sh.shape

    # Define spatial extent and pixel size
    minx = av_xrr_sh['lon'].min().item()
    maxy = av_xrr_sh['lat'].max().item()
    px_sz = round(av_xrr_sh['lon'].diff('lon').mean().item(), 2)

    # Save the data to disk using GDAL
    dest_flnme = os.path.join(misc_out, os.path.basename(file).replace('.tif', '_sh.tif'))
    gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, av_xrr_sh.data)

    # Reproject the data to stereographic projection
    output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('_sh.tif', '_sh_stere.tif'))
    gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'
    subprocess.run(gdalwarp_command, shell=True)

    # Read the stereographic projection file
    av_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

    # Remove intermediate files
    os.remove(dest_flnme)
    os.remove(output_file_stereo)

    # Clip the data to the bounds of the basin dataset
    av_xrr_clip = av_xrr_sh_stereo.sel(
        x=slice(xmin, xmax),
        y=slice(ymax, ymin)
    ).squeeze()

    # Reproject the clipped data to match the basin shape
    av_xrr_clip_res = av_xrr_clip.rio.reproject(
        av_xrr_clip.rio.crs,
        shape=(height, width),  # Set the shape as the basin data shape
        resampling=Resampling.bilinear,
        transform=basin.rio.transform()
    )

    # Add the time coordinate to the reprojected DataArray
    av_xrr_clip_res_arr = av_xrr_clip_res['band_data'].values
    av_xrr_clip_res_arr = np.where(basin.values > 0, av_xrr_clip_res_arr, np.nan)
    av_xrr_clip_res = xr.DataArray(
        np.expand_dims(av_xrr_clip_res_arr, axis=0),  # Expand dimensions to match ['time', 'y', 'x']
        dims=['time', 'y', 'x'],
        coords={'time': [av_time], 'y': av_xrr_clip_res.coords['y'], 'x': av_xrr_clip_res.coords['x']},
        name='precipitation'  # Change the variable name to 'precipitation'
    )

    return av_xrr_clip_res
#----------------------------------------------------------------------------
# function to process IMERG file to basin
def process_imerg_file_to_basin(im, misc_out, basin):
    basin_transform = basin.rio.transform()
    height, width = basin.data.shape[1:]
    xmin, ymax = basin_transform.c, basin_transform.f
    xres, yres = basin_transform.a, -basin_transform.e
    xmax = xmin + width * xres
    ymin = ymax - height * yres

    img_x, img_y = return_imerg_cords(im)
    img_arr, img_time = read_nc_imger_file(im, 'imerg_fn')

    img_arr = np.expand_dims(img_arr, axis=0) if img_arr.ndim == 2 else img_arr  # Ensure img_arr has 3 dimensions
    img_xrr = xr.DataArray(img_arr, dims=['time', 'lat', 'lon'], coords={'time': [img_time], 'lat': img_y, 'lon': img_x})

    img_xrr_sh = img_xrr.sel(lat=slice(-55, -90)).squeeze()
    yshp, xshp = img_xrr_sh.shape

    minx = img_xrr_sh['lon'].min().item()
    maxy = img_xrr_sh['lat'].max().item()

    px_sz = round(img_xrr_sh['lon'].diff('lon').mean().item(), 2)

    dest_flnme = os.path.join(misc_out, os.path.basename(im).replace('.nc4', '.tif'))

    gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, img_xrr_sh.data)

    output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('.tif', '_sh_stere.tif'))
    gdalwarp_command = [
        'gdalwarp',
        '-t_srs', '+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84',
        '-r', 'near',
        dest_flnme,
        output_file_stereo
    ]
    subprocess.run(gdalwarp_command, 
                   shell=False, check=True,
                   stdout=subprocess.DEVNULL, 
                   stderr=subprocess.DEVNULL)

    # Read the stereographic projection file
    img_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

    os.remove(dest_flnme)
    os.remove(output_file_stereo)

    # Clip the data to the bounds of the basin dataset
    img_xrr_clip = img_xrr_sh_stereo.sel(
        x=slice(xmin, xmax),
        y=slice(ymax, ymin)
    ).squeeze()

    # Reproject the clipped data to match the basin shape
    img_xrr_clip_res = img_xrr_clip.rio.reproject(
        img_xrr_clip.rio.crs,
        shape=(height, width),  # Use the basin's height and width
        resampling=Resampling.bilinear,
        transform=basin.rio.transform()
    )
    # Add the time coordinate to the reprojected DataArray
    img_xrr_clip_res_arr = img_xrr_clip_res['band_data'].values
    img_xrr_clip_res_arr = np.where(basin.values > 0, img_xrr_clip_res_arr, np.nan)
    img_xrr_clip_res = xr.DataArray(
        np.expand_dims(img_xrr_clip_res_arr[0], axis=0),  # Expand dimensions to match ['time', 'y', 'x']
        dims=['time', 'y', 'x'],
        coords={'time': [img_time], 'y': img_xrr_clip_res.coords['y'], 'x': img_xrr_clip_res.coords['x']},
        name='precipitation'  # Change the variable name to 'precipitation'
    )       

    return img_xrr_clip_res

#----------------------------------------------------------------------------
def process_gpcp_file_to_basin(gp, basin):
    basin_transform = basin.rio.transform()
    height, width = basin.data.shape[1:]
    xmin, ymax = basin_transform.c, basin_transform.f
    xres, yres = basin_transform.a, -basin_transform.e
    xmax = xmin + width * xres
    ymin = ymax - height * yres

    with xr.open_dataset(gp) as ds:
        ds = ds_swaplon(ds)

        # Extract precipitation data
        gp_arr = ds['precip']
        
        gp_time = ds['time'].values[0]

        gp_x, gp_y = ds['lon'].values, ds['lat'].values

        # gp_arr = np.expand_dims(gp_arr, axis=0) if gp_arr.ndim == 2 else gp_arr  # Ensure gp_arr has 3 dimensions
        # gp_xrr = xr.DataArray(gp_arr, dims=['time', 'lat', 'lon'], coords={'time': [gp_time], 'lat': gp_y, 'lon': gp_x})

        gp_xrr_sh = gp_arr.sel(lat=slice(-55, -90)).squeeze()
        yshp, xshp = gp_xrr_sh.shape

        # plot_data_on_antarctica(gp_xrr_sh, 
        #                         title=f'GPCP v3.3 Precipitation on {pd.to_datetime(gp_time).strftime("%Y-%m-%d")}', 
        #                         cmap='jet', vmax=10)

        minx = gp_xrr_sh['lon'].min().item()
        maxy = gp_xrr_sh['lat'].max().item()

        px_sz = round(gp_xrr_sh['lon'].diff('lon').mean().item(), 2)

        dest_flnme = os.path.join(misc_out, os.path.basename(gp).replace('.nc4', '.tif'))

        gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, gp_xrr_sh.data)

        output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('.tif', '_sh_stere.tif'))
        gdalwarp_command = [
        'gdalwarp',
        '-t_srs', '+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84',
        '-r', 'near',
        dest_flnme,
        output_file_stereo
        ]
        subprocess.run(gdalwarp_command, 
                       shell=False, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Read the stereographic projection file
        gp_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

        os.remove(dest_flnme)
        os.remove(output_file_stereo)

        # Clip the data to the bounds of the basin dataset
        gp_xrr_clip = gp_xrr_sh_stereo.sel(
            x=slice(xmin, xmax),
            y=slice(ymax, ymin)
        ).squeeze()

        # Reproject the clipped data to match the basin shape
        gp_xrr_clip_res = gp_xrr_clip.rio.reproject(
            gp_xrr_clip.rio.crs,
            shape=(height, width),  # Use the basin's height and width
            resampling=Resampling.bilinear,
            transform=basin_transform
        )
        # Add the time coordinate to the reprojected DataArray
        # Extract the 'band_data' variable from the dataset
        gp_xrr_clip_res_arr = gp_xrr_clip_res['band_data'].values

        # Mask the array using the basin values
        gp_xrr_clip_res_arr = np.where(basin.values > 0, gp_xrr_clip_res_arr, np.nan)

        # Create a new DataArray with the correct dimensions and coordinates
        gp_xrr_clip_res = xr.DataArray(
            np.expand_dims(gp_xrr_clip_res_arr[0], axis=0),  # Add a time dimension
            dims=['time', 'y', 'x'],
            coords={
            'time': [gp_time],
            'y': gp_xrr_clip_res.coords['y'],
            'x': gp_xrr_clip_res.coords['x']
            },
            name='precipitation'  # Use the original variable name
        )   

    return gp_xrr_clip_res

#----------------------------------------------------------------------------
# function to save 2d arrays to disk using gdal
def gdal_based_save_array_to_disk(dstflnme, xpx, ypx, px_sz, minx, maxy, crs, crs_format, arrayTosave):
        
        """
        Save an array to disk using GDAL.

        Parameters:
        - dstflnme: Destination filename including the file path.
        - xpx: Number of pixels in x (width).
        - ypx: Number of pixels in y (height).
        - px_sz: Size of the pixel (e.g., 0.5 degrees).
        - minx: Minimum x (longitude) value.
        - maxy: Maximum y (latitude) value (top-left corner).
        - crs: Coordinate reference system in proj4 or WKT format.
        - crs_format: 'proj4' or 'WKT', indicating the format of the CRS.
        - arrayTosave: The array to save to disk.
        """        
        
        # Calculate the bottom latitude and clamp it to -90.0 if necessary
        bottom_latitude = maxy - (ypx * px_sz)
        if bottom_latitude < -90.0:
            bottom_latitude = -90.0
            # Adjust ypx based on the clamped latitude
            ypx = int((maxy - bottom_latitude) / px_sz)
            # Slice the array to match the new dimensions
            arrayTosave = arrayTosave[:ypx, :]

        srs = osr.SpatialReference()
        if crs_format == 'proj4':
            srs.ImportFromProj4(crs)
            srs = srs.ExportToProj4()
        elif crs_format == 'WKT':
            srs.ImportFromWkt(crs)
            srs = srs.ExportToWkt()

        driver = gdal.GetDriverByName('GTiff')

        # Print dimensions to debug
        # print(f"Array Shape After Clamping: {arrayTosave.shape}")
        # print(f"Raster Size: (ypx={ypx}, xpx={xpx})")

        dataset = driver.Create(dstflnme, xpx, ypx, 1, gdal.GDT_Float64)

        dataset.SetGeoTransform((
            minx,    
            px_sz,   
            0.0,     
            maxy,    
            0.0,     
            -px_sz   
        ))  

        dataset.SetProjection(srs)
        dataset.GetRasterBand(1).WriteArray(arrayTosave)
        dataset.FlushCache()
        del(dataset)
#----------------------------------------------------------------------------
def gdal_based_save_array_to_disk2(dstflnme, xpx, ypx, px_sz, minx, maxy, crs, crs_format, arrayTosave):
        
        """
        Save an array to disk using GDAL.

        Parameters:
        - dstflnme: Destination filename including the file path.
        - xpx: Number of pixels in x (width).
        - ypx: Number of pixels in y (height).
        - px_sz: Size of the pixel (e.g., 0.5 degrees).
        - minx: Minimum x (longitude) value.
        - maxy: Maximum y (latitude) value (top-left corner).
        - crs: Coordinate reference system in proj4 or WKT format.
        - crs_format: 'proj4' or 'WKT', indicating the format of the CRS.
        - arrayTosave: The array to save to disk.
        """        
        
        # Calculate the bottom latitude and clamp it to -90.0 if necessary
        bottom_latitude = maxy - (ypx * px_sz)
        if bottom_latitude < -90.0:
            bottom_latitude = -90.0
            # Adjust ypx based on the clamped latitude
            ypx = int((maxy - bottom_latitude) / px_sz)
            # Slice the array to match the new dimensions
            arrayTosave = arrayTosave[:ypx, :]

        srs = osr.SpatialReference()
        if crs_format == "proj4":
            srs.ImportFromProj4(crs)
        elif crs_format == "WKT":
            srs.ImportFromWkt(crs)

        driver = gdal.GetDriverByName('GTiff')

        # Print dimensions to debug
        # print(f"Array Shape After Clamping: {arrayTosave.shape}")
        # print(f"Raster Size: (ypx={ypx}, xpx={xpx})")

        dataset = driver.Create(dstflnme, xpx, ypx, 1, gdal.GDT_Float64)

        dataset.SetGeoTransform((
            minx,    
            px_sz,   
            0.0,     
            maxy,    
            0.0,     
            -px_sz   
        ))  

        dataset.SetProjection(srs.ExportToWkt())
        dataset.GetRasterBand(1).WriteArray(arrayTosave)
        dataset.FlushCache()
        del(dataset)
#----------------------------------------------------------------------------
# function to reproject data using gdal
def gdal_based_stereo_reproj(input_file,hemisphere, y_min,y_max):     

    misc_out = r'/ra1/pubdat/AVHRR_CloudSat_proj/miscelaneous_outs'

    input_xr_dat = xr.open_dataset(input_file)     

    # Apply conditional selection based on the hemisphere and latitude bounds
    if 'x' in input_xr_dat.dims or 'y' in input_xr_dat.dims:
        input_xr_dat = input_xr_dat.rename({'x': 'lon', 'y': 'lat'})

    xr_dat = input_xr_dat.sel(lat=slice(y_max,y_min))
    # xr_dat = input_xr_dat.where((input_xr_dat.y >= y_min) & (input_xr_dat.y <= y_max), drop=True)

    if 'band' in xr_dat.dims and xr_dat.sizes['band'] == 1:
        xr_dat = xr_dat.squeeze('band')

    # xr_dat = xr_dat.transpose('y', 'x')
    array2sve = xr_dat.data

    x_dim = 'x' if 'x' in xr_dat.dims else 'lon'
    y_dim = 'y' if 'y' in xr_dat.dims else 'lat'

    ypx, xpx = xr_dat.shape  
    px_sz = xr_dat[x_dim].diff(x_dim).mean().item()  
    minx = xr_dat[x_dim].min().item()  
    maxy = xr_dat[y_dim].max().item()  
    
    crs = "+proj=longlat +datum=WGS84 +no_defs"  
    crs_format = 'proj4'  
    
    output_file_wgs = os.path.join(misc_out, os.path.basename(input_file).replace('.tif', f'_{hemisphere.lower()}.tif'))
    
    gdal_based_save_array_to_disk(output_file_wgs, xpx, ypx, px_sz, minx, maxy, crs, crs_format, array2sve)

    # Print latitude bounds for debugging
    miny = xr_dat[y_dim].min().item()
    maxy = xr_dat[y_dim].max().item()
    print(f"{hemisphere} Data Latitude Bounds: miny={miny}, maxy={maxy}")

    # if hemisphere == 'NH':
    #     output_file_stereo = os.path.join(misc_out, os.path.basename(input_file).replace('.tif', '_nh_stere.tif'))
    #     gdalwarp_command = f'gdalwarp -t_srs "EPSG:3413" -r near {output_file_wgs} {output_file_stereo}'
    # elif hemisphere == 'SH':
    #     output_file_stereo = os.path.join(misc_out, os.path.basename(input_file).replace('.tif', '_sh_stere.tif'))
    #     gdalwarp_command = f'gdalwarp -t_srs "EPSG:3976" -r near {output_file_wgs} {output_file_stereo}'

    if hemisphere == 'NH':
        output_file_stereo = os.path.join(misc_out, os.path.basename(input_file).replace('.tif', '_nh_stere.tif'))
        gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=0 +datum=WGS84" -r near {output_file_wgs} {output_file_stereo}'
    elif hemisphere == 'SH':
        output_file_stereo = os.path.join(misc_out, os.path.basename(input_file).replace('.tif', '_sh_stere.tif'))
        gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +datum=WGS84" -r near {output_file_wgs} {output_file_stereo}'


    # print(f"Reprojecting {hemisphere} data...")
    subprocess.run(gdalwarp_command, shell=True)

    xr_dt_stereo = xr.open_dataarray(output_file_stereo).squeeze()

    os.remove(output_file_wgs)

    os.remove(output_file_stereo)   

    return xr_dt_stereo

#----------------------------------------------------------------------------
# function to read IMGER file
def read_nc_imger_file(file_path, product):
    imerg_precip_data = xr.open_dataset(file_path)
    if product == 'imerg_fn':
        precip_aray = imerg_precip_data.precipitation.data 
    elif product == 'imerg_mw':
        precip_aray = imerg_precip_data.MWprecipitation.data 
    precip_aray = np.flip(precip_aray[0,:,:].transpose(), axis=0)
    imerg_time = imerg_precip_data.attrs['BeginDate']
    imerg_precip_data.close()
    
    # Convert time to pandas datetime
    imerg_time_index = pd.to_datetime(imerg_time,format='%Y-%m-%d')

    del(imerg_precip_data,imerg_time)
    
    return precip_aray, imerg_time_index

#----------------------------------------------------------------------------
# function to read IMGER file and return coordinates
def return_imerg_cords(file):
    file_dat = xr.open_dataset(file)
    lon = file_dat.coords['lon'].values
    lat = np.flip(file_dat.coords['lat']).values

    del(file_dat)

    return lon, lat

#----------------------------------------------------------------------------
# function to read SSMIS file
def read_tif_sim_file(file,yr):

    sim_precip_array = xr.open_dataarray(file).data[0] 

    sim_time = [x for x in os.path.split(file)[1].split('_') if x.startswith(yr)][0]


    sim_time_index = pd.to_datetime(sim_time,format='%Y%m%d')

    return sim_precip_array, sim_time_index

#----------------------------------------------------------------------------
# function to read SSMIS file and return coordinates
def return_sim_cords(file):
    file_dat = xr.open_dataarray(file)
    lon = file_dat.coords['x'].values
    lat = file_dat.coords['y'].values

    file_dat.close()
    return lon, lat

#----------------------------------------------------------------------------
def preprocess(ds):

    file = ds.encoding["source"]

    file_date = pd.to_datetime(
        os.path.basename(file).split('.')[4].split('-')[0],
        format='%Y%m%d'
    )

    ds = ds.assign_coords(time=("time", [file_date]))

    return ds
#----------------------------------------------------------------------------
# function to run batched processing
def run_batched_processing(batches, basin):
    """
    Function to process precipitation data in batches and calculate mean precipitation for each basin.

    Parameters:
    - batches: List of lists, where each inner list contains file paths to precipitation data for a batch.
    - basins: xarray DataArray representing the basin regions, with unique IDs for each basin.

    Returns:
    - batch_results: List of xarray DataArrays containing mean precipitation for each basin in each batch.
    """
     
    # Initialize a list to store the results from each batch
    batch_results = []

    # Get the total number of batches to process
    total_batches = len(batches)
    print(f"Total number of batches to process: {total_batches}")

    # Loop through each batch
    for idx, batch in enumerate(batches, start=1):
        # Display progress for the current batch
        print(f"Processing batch {idx}/{total_batches} ({(idx / total_batches) * 100:.2f}%)...")

        # Concatenate all precipitation data in the current batch along the 'time' dimension
        precip_bsn_xrr = xr.concat([xr.open_dataarray(x) for x in batch], dim='time')

        # Create an empty DataArray to store mean precipitation mapped to basins
        precip_xrr_basin_mapped = xr.full_like(basin, np.nan, dtype=float)

        # Loop through each basin ID (Zwally basins are numbered from 1 to 27)
        bsn_ids = np.unique(basin.data[~np.isnan(basin.data)])
        for basin_id in bsn_ids:
            # Create a mask for the current basin
            basin_mask = basin == basin_id

            # Mask the precipitation data for the current basin
            basin_precip = precip_bsn_xrr.where(basin_mask.data)

            # Calculate the mean precipitation for the current basin across time and spatial dimensions
            basin_mean_precip = basin_precip.mean(dim=['time', 'x', 'y'], skipna=True)

            # Map the calculated mean precipitation back to the basin region
            precip_xrr_basin_mapped = precip_xrr_basin_mapped.where(~basin_mask, basin_mean_precip)

        # Clean up variables to free memory
        del(precip_bsn_xrr, basin_precip, basin_mean_precip, basin_mask, basin_id)

        # Append the result for this batch to the results list
        batch_results.append(precip_xrr_basin_mapped)

    # Indicate that all batches have been processed
    print("All batches have been processed.")

    return batch_results

#-----------------------------------------------------------------------------
def decimal_year_to_date(decimal_year):
    """
    Converts a decimal year representation to a datetime object.

    A decimal year is a floating-point number where the integer part represents the year,
    and the fractional part represents the fraction of the year that has passed.

    Args:
        decimal_year (float): The year in decimal format (e.g., 2023.5 for mid-2023).

    Returns:
        datetime.datetime: A datetime object representing the corresponding date.

    Notes:
        - The function accounts for leap years when calculating the number of days in a year.
        - A leap year is defined as a year divisible by 4, except for years divisible by 100
          unless they are also divisible by 400.

    Example:
        >>> from datetime import datetime
        >>> decimal_year_to_date(2023.5)
        datetime.datetime(2023, 7, 2, 12, 0)
    """
    year = int(decimal_year)
    rem = decimal_year - year
    base = datetime(year, 1, 1)
    days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    return base + timedelta(days=rem * days_in_year)

#------------------------------------------------------------------------------
def decimal_year_to_month_start(decimal_year, mode="nearest"):
    """
    Convert decimal year -> month-start Timestamp (freq='MS').

    mode:
      - "nearest": nearest month (best if your decimal years are mid-month-ish)
      - "floor"  : always floor to earlier month
    """
    y = float(decimal_year)
    year = int(np.floor(y))
    frac = y - year

    m_float = frac * 12.0  # 0..12
    if mode == "nearest":
        month = int(np.round(m_float)) + 1
    elif mode == "floor":
        month = int(np.floor(m_float)) + 1
    else:
        raise ValueError("mode must be 'nearest' or 'floor'")

    month = int(np.clip(month, 1, 12))
    return pd.Timestamp(year=year, month=month, day=1)
#-----------------------------------------------------------------------------

def annual_slopes_with_se(df_year, cols):
    """
    Calculate annual mass balance rates (slopes) and their uncertainties for each basin.

    Parameters
    ----------
    df_year : DataFrame
        Subset of the basin time series for a single calendar year. 
        Must contain 'Time' (decimal years) and basin columns (mass anomalies in Gt).
    cols : list of str
        Names of the basin columns to process.

    Returns
    -------
    results : dict
        Dictionary keyed by basin name. Each entry is a dict with:
            - slope_Gt_per_yr : float
                Estimated rate of storage change (Gt/yr) for that year.
            - slope_stderr : float
                Standard error of the slope (Gt/yr).
            - n : int
                Number of valid (non-NaN) data points used in the regression.
    """

    # Extract decimal year values for the given year
    x = df_year["Time"].values

    # Center the time axis around its mean for numerical stability in regression
    # x_center = x - x.mean()

    results = {}
    for c in cols:
        # Mass anomaly values for the current basin
        y = df_year[c].values

        # Only proceed if at least 3 valid points exist (needed for a meaningful regression)
        if np.isfinite(y).sum() >= 3:
            # Perform linear regression: slope is ΔS rate (Gt/yr), stderr is its uncertainty
            slope, intercept, r, p, stderr = linregress(x, y)
            results[c] = {
                "slope_Gt_per_yr": slope,    # rate of mass change (Gt/yr)
                "slope_stderr": stderr,      # uncertainty in the slope
                "n": np.isfinite(y).sum()    # number of valid samples
            }
        else:
            # Not enough data to compute a slope
            results[c] = {
                "slope_Gt_per_yr": np.nan,
                "slope_stderr": np.nan,
                "n": int(np.isfinite(y).sum())
            }
    return results



#
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MaxNLocator
import numpy as np

def plot_seasonal_precip_maps(products_for_year,
                              seasons=("DJF", "MAM", "JJA", "SON"),
                              vmin=0, vmax=30,
                              cbar_ticks=None):
    """
    Plot seasonal precipitation maps: seasons in rows, products in columns.

    Parameters
    ----------
    products_for_year : list of (str, xarray.DataArray)
        Each DataArray must have a 'season' dimension that includes all `seasons`.
    seasons : sequence of str
        Order of seasons to plot as rows (e.g., ("DJF", "MAM", "JJA", "SON")).
    vmin, vmax : float
        Range for the color scale.
    cbar_ticks : list of float, optional
        Tick locations for the colorbar.
    """
    proj = ccrs.SouthPolarStereo()
    cmap = plt.cm.jet
    levels = np.linspace(vmin, vmax, 20)
    norm = BoundaryNorm(levels, cmap.N)

    nrows = len(seasons)              # seasons = rows
    ncols = len(products_for_year)    # products = columns

    fig = plt.figure(figsize=(5.5 * ncols, 4.8 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.05, hspace=0.10)

    axes = []
    for r, season in enumerate(seasons):
        for c, (prod_name, da) in enumerate(products_for_year):
            idx = r * ncols + c
            ax = fig.add_subplot(gs[idx], projection=proj)

            ax.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())
            ax.coastlines(resolution="110m", color="k", linewidth=0.6)

            # Select that season
            field = da.sel(season=season)

            field.plot(
                ax=ax,
                transform=ccrs.SouthPolarStereo(),
                cmap=cmap,
                norm=norm,
                add_colorbar=False
            )

            ax.add_feature(cfeature.OCEAN, zorder=1, edgecolor=None,
                           lw=0, color="silver", alpha=0.5)

            # Column titles = product names (top row only)
            if r == 0:
                ax.set_title(prod_name, fontsize=20, pad=8)
            else:
                ax.set_title("")

            # Row labels = season names (first column only)
            if c == 0:
                ax.text(-0.10, 0.5, season,
                        transform=ax.transAxes,
                        rotation=90,
                        va="center", ha="right",
                        fontsize=20, fontweight="bold")

            # Gridlines with cleaner labels
            gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                              linestyle="--", color="k", linewidth=0.6)
            gl.xlocator = MaxNLocator(nbins=4)
            gl.ylocator = MaxNLocator(nbins=4)
            gl.xlabel_style = {"size": 14, "color": "k"}
            gl.ylabel_style = {"size": 14, "color": "k"}

            # Only show bottom labels on last row, left labels on first column
            gl.top_labels = False
            gl.right_labels = False
            gl.bottom_labels = (r == nrows - 1)
            gl.left_labels = (c == 0)

            axes.append(ax)

    # Shared colorbar at bottom
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        orientation="horizontal",
        fraction=0.035,
        pad=0.06,
        extend="max"
    )

    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Precipitation [mm/season]", fontsize=20)

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.97])
    return fig, axes

#-----------------------------------------------------------------------------
# single plot fucntion
def single_precp_plot(data, product_name, vmin=0, vmax=300):
    """
    Plots mean precipitation for a single product over 27 basins.

    Parameters:
    data (xarray.DataArray): The precipitation data to plot.
    product_name (str): Name of the product for the title.
    vmin (float): Minimum value for colorbar.
    vmax (float): Maximum value for colorbar.
    """
    # Use the 'jet' colormap
    cmap = plt.cm.jet
    levels = np.linspace(vmin, vmax, 20)  # 27 basins + 1 for boundaries
    norm = BoundaryNorm(levels, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the data
    im = data.plot(
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=False
    )

    ax.set_title(product_name, fontsize=18)
    ax.set_xlabel("Longitude", fontsize=18)
    ax.set_ylabel("Latitude", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)  # Increase tick font sizes

    # Create a colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("Precipitation [mm/yr]", fontsize=15)
    cbar.set_ticks([round(tick, 0) for tick in cbar.get_ticks()])  # Round tick values to 0 decimal places

    plt.tight_layout()
    plt.show()


#-----------------------------------------------------------------------------
def annual_to_monthly_long(df_years, YEARS, value_name):
    longy = df_years.melt(id_vars="basin", var_name="year", value_name=value_name)
    longy["year"] = longy["year"].astype(int)
    # make 12 rows per basin-year, one per month
    longy = longy.loc[longy["year"].isin(YEARS)].copy()
    longy = longy.merge(pd.DataFrame({"month": np.arange(1,13)}), how="cross")
    longy["date"] = pd.to_datetime(dict(year=longy["year"], month=longy["month"], day=1))
    longy[value_name] = longy[value_name] / 12.0
    return longy[["date","basin", value_name]]

#-----------------------------------------------------------------------------
def paint_by_id(basin_id_da: xr.DataArray, values_by_id: dict) -> xr.DataArray:
    # Build a small lookup table for vectorized indexing
    max_id = int(np.nanmax(basin_id_da.values))
    lut = np.full(max_id + 1, np.nan, dtype='float64')
    for k, v in values_by_id.items():
        if k is not None and k >= 0 and k <= max_id:
            lut[int(k)] = float(v)

    # Vectorized index, preserving NaNs
    def _map_ids(arr):
        out = np.full(arr.shape, np.nan, dtype='float64')
        valid = np.isfinite(arr)
        out[valid] = lut[arr[valid].astype(int)]
        return out

    mapped = xr.apply_ufunc(
        _map_ids,
        basin_id_da,
        input_core_dims=[['y','x']],
        output_core_dims=[['y','x']],
        dask='parallelized',
        output_dtypes=[float],
    )
    mapped.attrs['units'] = 'Gt month-1'
    return mapped
#----------------------------------------------------------------------------
def norm_name(s):
    if s is None:
        return None
    return (str(s)
            .replace('–','-').replace('—','-')  # dash variants
            .replace(' ', '')                   # remove spaces
            .upper())

#----------------------------------------------------------------------------
# Build a robust ID↔name mapping from the grid itself (use most frequent name per ID)

def generate_basin_id_mapping(basin_id, basin_name, input_df):
    """
    Generates a mapping between basin IDs and their corresponding names, and maps basin IDs to an input DataFrame.

    Parameters:
    -----------
    basin_id : xarray.DataArray
        A 2D array where each cell contains the ID of the basin it belongs to.
    basin_name : xarray.DataArray
        A 2D array where each cell contains the name of the basin it belongs to.
    input_df : pandas.DataFrame
        A DataFrame containing basin data with a 'basin' column for basin names.

    Returns:
    --------
    pandas.DataFrame
        A modified DataFrame with an additional 'basin_id' column mapping basin names to their IDs.
        Rows with unmapped basins or islands (ID==1) are dropped.

    Notes:
    ------
    - The function uses the most common name within each basin ID to create the mapping.
    - Basin names are normalized to ensure consistent matching.
    - Rows with unmapped basins or islands (ID==1) are excluded from the output DataFrame.
    """
    # Extract unique basin IDs, ignoring NaN values, and sort them
    ids = np.sort(np.unique(basin_id.values[~np.isnan(basin_id.values)])).astype(int)

    # Create a dictionary to map basin IDs to their most common names
    id_to_name = {}
    for bid in ids:
        # Create a mask for the current basin ID
        mask = (basin_id == bid)

        # Extract names corresponding to the current basin ID
        names_here = basin_name.where(mask).values
        names_here = [n for n in names_here.ravel() if isinstance(n, str) and n != 'NA']

        if len(names_here):
            # Find the most common name in the basin
            modal = Counter(names_here).most_common(1)[0][0]
            id_to_name[bid] = modal

    # Create a normalized dictionary for matching to the input DataFrame's column labels
    id_to_name_norm = {bid: norm_name(nm) for bid, nm in id_to_name.items() if nm is not None}
    name_to_id_norm = {nm: bid for bid, nm in id_to_name_norm.items()}

    # Normalize basin names in the input DataFrame and attach basin IDs
    df = input_df.copy()
    df['basin_norm'] = df['basin'].map(norm_name)  # Normalize basin names
    df['basin_id'] = df['basin_norm'].map(name_to_id_norm)  # Map normalized names to basin IDs

    # Drop rows with unmapped basins and exclude islands (ID==1)
    df = df.dropna(subset=['basin_id']).copy()
    df['basin_id'] = df['basin_id'].astype(int)
    df = df[df['basin_id'] != 1]

    return df
#----------------------------------------------------------------------------

def create_basin_xrr(basin_id, basin_name, df, colnme, attrs, var_name):
    """
    Creates an xarray DataArray representing a time series of raster data for a specific basin.

    Parameters:
    -----------
    basin_id : xarray.DataArray
        A 2D array where each cell contains the ID of the basin it belongs to.
    basin_name : xarray.DataArray
        A 2D array where each cell contains the name of the basin it belongs to.
    df : pandas.DataFrame
        A DataFrame containing the input data. It must include the following columns:
        - 'date': Timestamps for each observation.
        - 'basin_id': IDs of the basins corresponding to the data.
        - `colnme`: The column name specified by the `colnme` parameter, containing the values to be rasterized.
    colnme : str
        The name of the column in `df` that contains the values to be rasterized.
    attrs : dict
        A dictionary of attributes to be added to the resulting xarray DataArray.

    Returns:
    --------
    xarray.DataArray
        A 3D DataArray with dimensions ('date', 'y', 'x') representing the rasterized time series.
        - The 'date' dimension corresponds to the time steps.
        - The 'y' and 'x' dimensions correspond to the spatial grid.
        - The DataArray includes the following additional coordinates:
            - 'basin_id': The basin IDs for each grid cell.
            - 'basin_name': The basin names for each grid cell.
        - The DataArray is named 'deltaS_Gt_per_month' and includes the provided attributes.

    Notes:
    ------
    - The function assumes that the `paint_by_id` function is available and is used to rasterize the data.
    - The `df['date']` column is expected to contain unique timestamps for each observation.
    - The resulting DataArray is concatenated along the 'date' dimension.

    Example:
    --------
    >>> dS_raster = create_basin_xrr(basin_id, basin_name, df, 'value_column', {'unit': 'Gt/month'})
    >>> print(dS_raster)
    """
    dates = np.sort(df['date'].unique())
    frames = []
    for t in dates:
        slic = df.loc[df['date'] == t, ['basin_id', colnme]]
        values = dict(zip(slic['basin_id'].astype(int), slic[colnme].astype(float)))
        raster_t = paint_by_id(basin_id, values)
        raster_t = raster_t.assign_coords(date=np.datetime64(t)).expand_dims('date')
        frames.append(raster_t)

    dS_raster = xr.concat(frames, dim='date')
    dS_raster.name = var_name
    dS_raster.attrs.update(attrs)

# (Optional) carry through your basin metadata alongside
    dS_raster = dS_raster.assign_coords(
    basin_id=(('y','x'), basin_id.values),
    basin_name=(('y','x'), basin_name.values))
    return dS_raster
#----------------------------------------------------------------------------

# --- helpers ---------------------------------------------------------------

def stack_space(da):
    if "stacked_y_x" in da.dims:
        return da.rename({"stacked_y_x": "space"})
    return da.stack(space=("y", "x"))

def basin_labels_from(da):
    lbl = da["basin_id"]
    if "stacked_y_x" in lbl.dims:
        lbl = lbl.rename({"stacked_y_x":"space"})
    else:
        lbl = lbl.stack(space=("y","x"))
    return lbl

def painted_to_basin_series(da):
    """
    da holds basin totals painted to pixels (units already Gt/month).
    Return (date, basin_id) by taking spatial mean within each basin.
    """
    data   = stack_space(da)
    labels = basin_labels_from(da)
    valid  = np.isfinite(labels)
    data   = data.where(valid)
    labels = labels.where(valid)
    out = data.groupby(labels).mean(dim="space", skipna=True)
    return out.transpose("date", "basin_id")

def areal_to_basin_series(da, *, pixel_area_m2, units="mm_we_per_month"):
    """
    da is an areal field over (date, y, x) (e.g., sublimation).
    Convert to Gt/month by summing over pixels within each basin.
    units: "mm_we_per_month" | "m_we_per_month" | "kg_m2_s"
    """
    data   = stack_space(da)
    labels = basin_labels_from(da)
    valid  = np.isfinite(labels)
    data   = data.where(valid)
    labels = labels.where(valid)

    if units == "mm_we_per_month":     # mm water equivalent accumulated per month
        kg_per_m2 = (data / 1000.0) * 1000.0     # mm -> m, then * rho_w
    elif units == "m_we_per_month":    # m w.e. accumulated per month
        kg_per_m2 = data * 1000.0
    elif units == "kg_m2_s":           # flux kg m^-2 s^-1 -> accumulate over month
        secs = xr.DataArray(da["date"].dt.days_in_month * 86400,
                            coords={"date": da["date"]}, dims=("date",))
        kg_per_m2 = data * secs
    elif units in ("kg_m2_per_month", "kg m-2"):
        kg_per_m2 = data
    else:
        raise ValueError("Unsupported units for sublimation")

    kg = kg_per_m2 * pixel_area_m2
    Gt = kg.groupby(labels).sum(dim="space", skipna=True) / 1e12
    return Gt.transpose("date", "basin_id")

def paint_basin_series_to_grid(series, template_with_basin_id):
    """
    series: DataArray (date, basin_id)
    template_with_basin_id: DataArray with coords basin_id(y,x) (and maybe basin_name)
    returns: DataArray (date, y, x)
    """
    labels = template_with_basin_id["basin_id"]
    if "stacked_y_x" in labels.dims:
        labels = labels.rename({"stacked_y_x": "space"})
    else:
        labels = labels.stack(space=("y", "x"))

    series_ids = series["basin_id"].values
    exist = xr.apply_ufunc(np.isin, labels, series_ids)

    dummy_id = series_ids[0]
    safe_labels = labels.where(exist).astype(series["basin_id"].dtype)
    safe_labels = safe_labels.fillna(dummy_id)

    painted = series.sel(basin_id=safe_labels)  # (date, space)
    painted = painted.where(exist)
    out = painted.unstack("space")

    coords_to_add = {"basin_id": template_with_basin_id["basin_id"]}
    if "basin_name" in template_with_basin_id.coords:
        coords_to_add["basin_name"] = template_with_basin_id.coords["basin_name"]
    out = out.assign_coords(**coords_to_add)
    return out


def mask_to_basins(da, basin_labels_like):
    """
    Keep values only where basin_labels_like is finite.
    Also carry basin_id / basin_name coords forward so later
    groupby(label) works without surprises.
    """
    mask = np.isfinite(basin_labels_like)
    out = da.where(mask)
    # attach the labels so they travel with the DataArray
    coords = {"basin_id": basin_labels_like}
    if "basin_name" in basin_labels_like.coords:
        coords["basin_name"] = basin_labels_like.coords["basin_name"]
    return out.assign_coords(**coords)
#-----------------------------------------------------------------------------
# def plot_data_on_antarctica(data, title="Data on Antarctica", cmap='jet', vmax=None):
#     """
#     Plot xarray data on a polar stereographic projection focusing on Antarctica.

#     Parameters:
#     data (xarray.DataArray): The data to be plotted. It should have latitude and longitude coordinates.
#     title (str): Title of the plot.
#     cmap (str): Colormap to use for the plot.
#     vmax (float): Maximum value for the color scale. If None, it will be determined automatically.

#     Returns:
#     None
#     """   

#     # Plot the data on a polar stereographic projection
#     fig, ax = plt.subplots(subplot_kw={'projection': ccrs.SouthPolarStereo()}, 
#                            figsize=(10, 10))
#     data.plot(ax=ax, transform=ccrs.PlateCarree(), 
#               cmap=cmap, vmax=vmax)

#     # Add features to the map
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
#     ax.gridlines(dms=True, x_inline=False, y_inline=False)  # Removed draw_labels=True for performance

#     # Set the extent to focus on Antarctica
#     ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

#     plt.title(title)
#     plt.show(block=False)  # Ensure the plot does not block further execution
#-----------------------------------------------------------------------------
# helper to map back with a dummy '0' basin so NaNs don't crash .sel
def process_precipitation_data(
    source,                      # list/str of files OR xr.DataArray / xr.Dataset
    basins,
    variable_name="precipitation",
    is_accumulated=False,        # True for RACMO monthly totals (kg m-2 per month)
    seasonal_resample="QS-DEC",  # seasons starting Dec (DJF, MAM, JJA, SON)
    compute=False,               # set True if you want concrete arrays at the end
):
    """
    Returns:
      annual_map:   (year, y, x)
      seasonal_map: (season, y, x)
    """

    # ---- 0) Load precip into P: (time, y, x) ----
    if isinstance(source, xr.DataArray):
        P = source
    elif isinstance(source, xr.Dataset):
        P = source[variable_name]
    else:
        # assume file path or list of paths
        ds = xr.open_mfdataset(source, combine="by_coords")
        P = ds[variable_name].compute()
        ds.close()

    # squeeze away any singleton dims (e.g., RACMO had band=1)
    for d in list(P.dims):
        if P.sizes[d] == 1 and d not in ("time", "y", "x"):
            P = P.squeeze(d, drop=True)

    # make sure we have the grid dims
    for need in ("y", "x"):
        if need not in P.dims:
            raise ValueError(f"Expected '{need}' in precip dims, found {P.dims}")
    if "time" in P.dims:
        P = P.transpose("time", "y", "x", ...)

    # ---- 1) Clean basin labels ----
    labels   = basins.squeeze(drop=True).where(lambda a: a.notnull() & (a > 1))
    valid    = labels.notnull()
    labels_i = labels.fillna(0).astype("int32")  # 0 = background
    basin_ids = np.sort(np.unique(labels_i.values[valid.values])).astype("int32")

    # ---- 2) Basin means (time, basin) ----
    P_st   = P.stack(stacked_y_x=("y", "x"))
    lab_st = labels_i.stack(stacked_y_x=("y", "x")).rename("basin")
    P_basin = (P_st.groupby(lab_st)
                  .mean("stacked_y_x", skipna=True)
                  .sel(basin=basin_ids))

    # ---- 3) Annual stats ----
    if is_accumulated:
        P_basin_annual = P_basin.groupby("time.year").sum("time")
        annual_units = "kg m-2 yr-1"
    else:
        P_basin_annual = P_basin.groupby("time.year").mean("time")
        annual_units = P.attrs.get("units", "")

    # ---- 4) Seasonal climatology ----
    if is_accumulated:
        P_season_blocks = P_basin.resample(time=seasonal_resample).mean()
    else:
        P_season_blocks = P_basin.resample(time=seasonal_resample).mean()

    P_season_blocks = P_season_blocks.assign_coords(
        season=("time", P_season_blocks["time.season"].values)
    )
    P_basin_seasonal = P_season_blocks.groupby("season").mean("time")
    seasonal_units = "kg m-2 per season" if is_accumulated else P.attrs.get("units", "")

    # ---- 5) Map back to (y, x) ----
    def map_back(field_basin, indexer_ij, valid_mask):
        fb = field_basin.assign_coords(basin=field_basin.basin.astype("int32"))
        if 0 not in fb.basin.values:
            fb = fb.reindex(basin=np.r_[0, fb.basin.values])
            fb.loc[dict(basin=0)] = np.nan
        mapped = fb.sel(basin=indexer_ij)
        return mapped.where(valid_mask)

    annual_map   = map_back(P_basin_annual,  labels_i, valid)
    seasonal_map = map_back(P_basin_seasonal, labels_i, valid)

    annual_map.name = f"{variable_name}_annual"
    annual_map.attrs["units"] = annual_units
    seasonal_map.name = f"{variable_name}_season_clim"
    seasonal_map.attrs["units"] = seasonal_units

    if compute:
        annual_map   = annual_map.compute()
        seasonal_map = seasonal_map.compute()

    return annual_map, seasonal_map, P_basin

#----------------------------------------------------------------------------
def ensure_season_index(da,SEAS):
    # make sure we actually have the dim
    if "season" not in da.dims:
        raise ValueError(f"array {da.name!r} lacks a 'season' dim; dims={da.dims}")
    # promote to an index if it's only a coord
    if "season" not in getattr(da, "indexes", {}):
        da = da.assign_coords(season=pd.Index(da["season"].values, name="season"))
    # put seasons in canonical order (keeps only what exists, preserves NaN if missing)
    return da.reindex(season=SEAS)
#----------------------------------------------------------------------------

def pick_year(da, year):
    return da.sel(year=year) if "year" in da.dims else da

#----------------------------------------------------------------------------

def plot_basin_ranked_bar_overlay(
    df,
    basin_col="basin",
    ref_col="Pmb",          # mass-budget precipitation (P_MB)
    prod_cols=None,        # list of other products to overlay
    prod_labels=None,      # pretty labels for legend
    figsize=(12, 5),
):
    import matplotlib.ticker as mticker
    """
    Basin-ranked bar plot: basins sorted by reference precipitation (P_MB),
    with other products overlaid as markers.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain one row per basin with columns for basin ID, P_MB, and other products.
    basin_col : str
        Column with basin identifier (e.g., integer 1–19 or 'A-Ap', etc.).
    ref_col : str
        Column name for the reference precipitation (P_MB).
    prod_cols : list of str
        Column names for other products to overlay (e.g. ["ERA5", "GPCP_v3.3", "RACMO_2.4p1"]).
    prod_labels : list of str
        Labels to use in legend. If None, uses prod_cols.
    figsize : tuple
        Figure size in inches.
    """

    if prod_cols is None:
        prod_cols = ["ERA5", "GPCP_v3.3", "RACMO_2.4p1"]

    if prod_labels is None:
        prod_labels = prod_cols

    # --- Convert basin IDs to int ---
    df[basin_col] = df[basin_col].astype(int)

    # --- 1. Sort by reference precipitation (P_MB) ---
    # Drop rows where ref_col is NaN
    df_plot = df.dropna(subset=[ref_col]).copy()
    df_plot = df_plot.sort_values(ref_col)

    basins = df_plot[basin_col].astype(str).values
    x = np.arange(len(basins))

    # --- 2. Start figure ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- 3. Plot P_MB as bars ---
    ax.bar(
        x,
        df_plot[ref_col].values,
        color="lightgray",
        edgecolor="black",
        linewidth=1.0,
        label=r"$P_{\mathrm{MB}}$",
        zorder=1,
    )

    # --- 4. Overlay other products as markers ---
    markers = ["o", "s", "D", "^", "v"]  # enough variety
    for i, (col, lab) in enumerate(zip(prod_cols, prod_labels)):
        if col not in df_plot.columns:
            continue

        y = df_plot[col].values
        mask = np.isfinite(y)  # avoid NaNs

        ax.plot(
            x[mask],
            y[mask],
            marker=markers[i % len(markers)],
            linestyle="-",
            linewidth=1.5,
            markersize=6,
            label=lab,
            zorder=3,
        )
    
    # ---------- LOG SCALE ----------
    ax.set_yscale("log")
    ax.set_ylim(10, 2000)

    # clean log ticks
    log_ticks = [10, 50, 100, 200, 500, 1000, 1500, 2000]
    ax.set_yticks(log_ticks)
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis='y', labelsize=12)

    # --- 5. Cosmetics ---
    ax.set_xticks(x)
    ax.set_xticklabels(basins, ha="center", fontsize=18)
    ax.set_xlabel("Basin (sorted by " + r"$P_{\mathrm{MB}}$" + ")", fontsize=18)
    ax.set_ylabel("Precipitation [mm/year]", fontsize=18)

    ax.grid(axis="y", linestyle="--", 
            alpha=0.4, zorder=0)

    ax.legend(fontsize=18, ncol=2, frameon=False)
    ax.set_xlim(-0.5, len(basins) - 0.5)
    ax.set_ylim(0,2000)

    plt.tight_layout()
    return fig, ax

#----------------------------------------------------------------
def da_season_to_basin_df(da, basin_name="basin"):
    """
    Convert a seasonal (season, y, x) gridded DataArray into a tidy DataFrame:
    
    Output columns:
        basin   (int)
        season  (str)
        value   (float)  <-- named after da.name
    """

    df_all = []
    for sees in da["season"].values:
        da_sees = da.sel(season=sees)
        df_sees = da_sees.to_dataframe().reset_index()
        cols = df_sees.columns.tolist()
        cols = [c for c in cols if 'precip' in c \
                 or 'pr' in c \
             or 'basin' in c \
            or c in ['season', 'basin']]
        prcp_col = [c for c in cols if 'precip' in c or\
                    'pr' in c][0]
        bsn_col  = [c for c in cols if 'basin' in c][0]
        # name = df.columns[-1]  # last column is the data values

        df_sees = df_sees[cols].copy()

        # Drop dummy basin IDs (0 or NaN) and NaN values
        df_sees = df_sees.dropna(subset=[prcp_col])
        df_sees = df_sees[df_sees[bsn_col] > 1]
        df_sees = df_sees.groupby(["season", bsn_col]).mean().reset_index()

        df_all.append(df_sees)
    df_mean_seas_acc = pd.concat(df_all, ignore_index=True).sort_values(by=bsn_col)

    df_mean_seas_acc[bsn_col] = df_mean_seas_acc[bsn_col].astype(int)

    return df_mean_seas_acc

#----------------------------------------------------------------------------
def _to_monthly_basin_df(df, time_col="time", basin_col="basin", val_col="precipitation"):
    """If df is daily/submonthly, aggregate to monthly per basin. If already monthly, returns as-is."""
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d[basin_col] = d[basin_col].astype(int)

    # detect "already monthly": each basin has one record per year-month
    # (cheap heuristic: if any basin has >1 record in a given month, treat as submonthly)
    d["ym"] = d[time_col].dt.to_period("M")
    dup = d.groupby([basin_col, "ym"]).size().max()

    if dup > 1:
        # aggregate to monthly mean per basin (works for mm/day daily series too
        # as long as you've already converted to mm/month prior to this stage)
        d = (d.groupby([basin_col, "ym"], as_index=False)[val_col]
               .mean())
        d[time_col] = d["ym"].dt.to_timestamp()  # month start
    else:
        # ensure month-start timestamps
        d[time_col] = d["ym"].dt.to_timestamp()

    d["year"] = d[time_col].dt.year
    d["month"] = d[time_col].dt.month
    return d[[time_col, "year", "month", basin_col, val_col]]

def region_monthly_series_from_dict(
    monthly_df_data_mmmonth,
    region_defs,
    basin_weights,
    region_name,
    time_col="time",
    basin_col="basin",
    val_col="precipitation",
):
    """
    Returns a single DataFrame indexed by monthly time with one column per product (mm/month),
    area-weighted over basins in the chosen region.
    """
    basin_ids = dict(region_defs)[region_name]
    basin_ids = set(int(b) for b in basin_ids)

    # normalize weights to region (so they sum to 1 over the region)
    w = {int(k): float(v) for k, v in basin_weights.items() if int(k) in basin_ids}
    wsum = sum(w.values())
    w = {k: v / wsum for k, v in w.items()}

    out = None

    for prod, df in monthly_df_data_mmmonth.items():
        d = _to_monthly_basin_df(df, time_col=time_col, basin_col=basin_col, val_col=val_col)

        # d = d[d[basin_col].isin(basin_ids)].copy()
        # d["w"] = d[basin_col].map(w).astype(float)

        # # weighted mean per month
        # reg = (d.groupby(time_col)
        #          .apply(lambda g: np.sum(g[val_col].values * g["w"].values))
        #          .rename(prod)
        #          .to_frame())

        d = d[d[basin_col].isin(basin_ids)].copy()

        # map weights (some basins may be missing weights)
        d["w"] = d[basin_col].map(w)

        # drop rows where weight is missing
        d = d.dropna(subset=["w"])

        # weighted mean per month with re-normalization (so missing basins don't break everything)
        def _wmean(g):
            ww = g["w"].values.astype(float)
            yy = g[val_col].values.astype(float)
            s = np.nansum(ww)
            if s <= 0:
                return np.nan
            return np.nansum(yy * ww) / s

        reg = d.groupby(time_col).apply(_wmean)

        # reg might be Series or DataFrame depending on pandas; force Series
        if isinstance(reg, pd.DataFrame):
            reg = reg.iloc[:, 0]

        reg = reg.rename(prod).to_frame()

        out = reg if out is None else out.join(reg, how="outer")

    # enforce continuous monthly index (gap-aware)
    # --- enforce continuous monthly index (gap-aware) ---
    if out is None or out.empty:
        raise ValueError(
            f"Region series is empty for '{region_name}'. "
            "Likely: basin IDs mismatch, weights missing, or time parsing failed."
        )

    out = out.sort_index()

    start = pd.to_datetime(out.index.min(), errors="coerce")
    end   = pd.to_datetime(out.index.max(), errors="coerce")

    if pd.isna(start) or pd.isna(end):
        raise ValueError(
            f"Region series has invalid time index for '{region_name}'. "
            f"start={start}, end={end}. Check time column parsing."
        )

    full = pd.date_range(start, end, freq="MS")
    out = out.reindex(full)
    out.index.name = "time"
    return out

#---------------------------------------------------------------------------
def deseasonalize_monthly(df):
    """Subtract monthly climatology (calendar-month mean) from each series."""
    out = df.copy()
    months = out.index.month
    for c in out.columns:
        clim = out[c].groupby(months).mean()
        out[c] = out[c] - months.map(clim).values
    return out

def centered_13mo_rm(df):
    return df.rolling(window=13, center=True, min_periods=7).mean()

#---------------------------------------------------------------------------
from scipy import stats

def trend_with_ar1_correction(y, time_index):
    """
    Linear trend y ~ t with AR(1) effective sample size correction.
    Returns slope per year and an approximate p-value.
    """
    y = np.asarray(y, float)
    mask = np.isfinite(y)
    if mask.sum() < 8:
        return np.nan, np.nan

    # t in years from start
    t = (pd.to_datetime(time_index[mask]) - pd.to_datetime(time_index[mask])[0]).days / 365.25
    yy = y[mask]

    # OLS slope
    slope, intercept, r, p_naive, stderr = stats.linregress(t, yy)

    # residuals
    resid = yy - (intercept + slope * t)

    # lag-1 autocorr
    if len(resid) < 4:
        return slope, p_naive

    r1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    if not np.isfinite(r1):
        r1 = 0.0

    # effective N (Bretherton-like)
    N = len(resid)
    Neff = N * (1 - r1) / (1 + r1)
    Neff = max(3, Neff)

    # recompute t-stat using Neff (approx)
    tstat = slope / stderr if stderr > 0 else np.nan
    dof = Neff - 2
    p_eff = 2 * (1 - stats.t.cdf(np.abs(tstat), df=dof)) if np.isfinite(tstat) else np.nan

    return slope, p_eff
#---------------------------------------------------------------------------

def to_df(da):
    """
    Flatten a (year, basin) DataArray into a DataFrame with columns:
    year, basin, product_name
    """
    
    # df = da.to_dataframe(name).reset_index()
    df = da.to_dataframe().reset_index()
    cols = df.columns.tolist()
    cols = [c for c in cols if 'precip' in c \
             or 'pr' in c \
             or 'basin' in c \
            or c in ['year', 'basin']]
    prcp_col = [c for c in cols if 'precip' in c or\
                'pr' in c][0]
    bsn_col  = [c for c in cols if 'basin' in c][0]
    # name = df.columns[-1]  # last column is the data values

    df = df[cols].copy()

    # Drop dummy basin IDs (0 or NaN) and NaN values
    df = df.dropna(subset=[prcp_col])
    df = df[df[bsn_col] > 1]
    df_grp = df.groupby(["year", bsn_col]).mean().reset_index()
    return df_grp


#%%
# PLOTTING HELPERS
def compare_mean_precp_plot(arr_lst_mean, vmin=0, vmax=300, cbar_tcks=None):
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
    levels = np.linspace(vmin, vmax, 20)  # 27 basins + 1 for boundaries
    norm = BoundaryNorm(levels, cmap.N)

    # Determine the number of rows and columns for the grid
    n_products = len(arr_lst_mean)
    ncols = 4  # Number of columns
    nrows = (n_products + ncols - 1) // ncols  # Calculate rows needed

    # Create a GridSpec with precise control over spacing
    fig = plt.figure(figsize=(28, 10 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.1, hspace=0.15)

    # Loop through each dataset to create the plots
    axes = []
    for i, (product_name, data) in enumerate(arr_lst_mean):
        ax = fig.add_subplot(gs[i], projection=proj)
        ax.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())
        # ax.coastlines(lw=0.25, resolution="110m")
        ax.coastlines(resolution = '110m', color="k", linewidth=0.6)

        # Plot the data
        data.plot(
            ax=ax,
            transform=ccrs.SouthPolarStereo(),
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
        fraction=0.07,  # Fraction of the original axes height
        pad=0.15,  # Distance from the bottom of the subplots
        extend="max"
    )
    cb.set_ticks(cbar_tcks)  # Set integer ticks
    cb.ax.tick_params(labelsize=20)
    cb.set_label("Precipitation [mm/year]", fontsize=20)

    # Show the plot
    plt.tight_layout()

# ---------- IMPROVED LAT-LON LABEL FUNCTION ----------
def add_polar_latlon_labels(ax,
                            lat_rings=(-65, -70, -75, -80), # 
                            lon_spokes=np.arange(-180, 180, 30),
                            label_size=12):
    """
    Adds readable latitude/longitude labels close to their rings/spokes,
    avoiding overlap. Designed for South Polar Stereographic projection.
    """

    # ---- Latitude labels (place at 2° outside the ring) ----
    for lat in lat_rings[1:]:
        ax.text(182, lat, f"{abs(lat)}°S",
        transform=ccrs.PlateCarree(),
        fontsize=label_size,
        va="center", ha="left")


    # ---- Longitude labels (place near outermost latitude ring) ----
    outer_lat = lat_rings[0] + 1.5   # e.g., around -63.5°

    for lon in lon_spokes:
        # Compute label
        if lon == 0:
            lab = "0°"
        elif lon < 0:
            lab = f"{abs(lon)}°W"
        else:
            lab = f"{abs(lon)}°E"

        ax.text(lon, outer_lat,
                lab,
                transform=ccrs.PlateCarree(),
                fontsize=label_size,
                ha="center", va="bottom",
                color="black")
        

# ---------- MAIN UPDATED FUNCTION ----------
def compare_mean_precip_2x2(arr_lst_mean, vmin=0, vmax=300, cbar_tcks=None):

    if len(arr_lst_mean) != 4:
        raise ValueError("compare_mean_precip_2x2 expects exactly 4 datasets.")

    proj = ccrs.SouthPolarStereo()
    cmap = plt.cm.jet
    levels = np.linspace(vmin, vmax, 20)
    norm = BoundaryNorm(levels, cmap.N)

    lat_rings = (-65, -70, -75, -80)
    lon_spokes = np.arange(-180, 180, 30)

    fig, axes = plt.subplots(
        2, 2, subplot_kw={"projection": proj}, 
        figsize=(12, 12)
    )
    axes = axes.ravel()

    fig.subplots_adjust(wspace=0.05, hspace=0.25, bottom=0.20)

    for ax, (product_name, data) in zip(axes, arr_lst_mean):

        # remove bounding box around each subplot
        ax.set_frame_on(False)

        ax.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())
        ax.text(
            0.15, 1.05,              # x<0.5 moves left of center
            product_name,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="bottom"
        )

        # ax.coastlines(resolution="110m", color="k", linewidth=0.55)

        # plot
        data.plot(
            ax=ax,
            transform=ccrs.SouthPolarStereo(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
            add_labels=False, 
        )

        # ---------- IMBIE basin boundaries (bold coast, thin internals) ----------
        # pick the basin-ID DataArray from coords
        if "basin_id" in data.coords:
            basin_da = data["basin_id"]
        elif "basin" in data.coords:
            basin_da = data["basin"]
        else:
            basin_da = None

        if basin_da is not None:
            # ensure 2D (y,x); some variants could carry an extra dim
            if "band" in basin_da.dims:
                basin_da = basin_da.isel(band=0)

            # fill NaNs with 0 so we get a 0–1 boundary at the grounded-coast
            da_for_contour = xr.where(np.isnan(basin_da), 0, basin_da)

            # thin internal basin boundaries
            internal_levels = np.arange(1.5, 19.5, 1.0)
            ax.contour(
                da_for_contour["x"],
                da_for_contour["y"],
                da_for_contour.values,
                levels=internal_levels,
                colors="k",
                linewidths=0.6,
                transform=proj,
                zorder=5,
            )

            # bold coastline (0 vs 1)
            ax.contour(
                da_for_contour["x"],
                da_for_contour["y"],
                da_for_contour.values,
                levels=[0.5],
                colors="k",
                linewidths=1.2,
                transform=proj,
                zorder=6,
            )

        # ax.add_feature(
        #     cfeature.OCEAN,
        #     zorder=1, edgecolor=None, lw=0,
        #     color=None, alpha=0.5
        # )

        # ax.set_title(product_name, fontsize=17, fontweight="bold")

        # gridlines without labels
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=False,
            linewidth=0.55,
            color="gray",
            alpha=0.8,
            linestyle="--",
        )
        gl.ylocator = FixedLocator(lat_rings)
        gl.xlocator = FixedLocator(lon_spokes)

        # improved lat/lon labels
        add_polar_latlon_labels(ax, lat_rings=lat_rings,
                                lon_spokes=lon_spokes,
                                label_size=11)
        
    all_levels = levels  # same 'levels' you used for BoundaryNorm

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  

    # # shared colorbar
    # cb = fig.colorbar(
    #     sm,#ScalarMappable(norm=norm, cmap=cmap),
    #     ax=axes.tolist(),
    #     orientation="horizontal",
    #     fraction=0.045,
    #     pad=0.08,
    #     extend="max",
    #     boundaries=all_levels,   # ensure discrete patches
    #     ticks=all_levels         # tick for every color step
    # )

    # # Now: only label the ticks that are in cbar_tcks; others get ""
    # if cbar_tcks is not None:
    #     labels = []
    #     for val in all_levels:
    #         if any(np.isclose(val, lab) for lab in cbar_tcks):
    #             labels.append(f"{val:.0f}")
    #         else:
    #             labels.append("")   # show tick, no label
    #     cb.set_ticklabels(labels)

    # Create a colorbar at the bottom spanning all subplots
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,  # Attach the colorbar to all axes for better placement
        orientation="horizontal",
        fraction=0.03,  # Fraction of the original axes height
        pad=0.1,  # Distance from the bottom of the subplots
        extend="max"
    )
    cb.set_ticks(cbar_tcks)  # Set integer ticks

    cb.ax.tick_params(labelsize=12)
    cb.ax.minorticks_off()
    cb.set_label("Precipitation [mm/year]", fontsize=14)

    return fig, axes
#-----------------------------------------------------------------------------
def plot_seasonal_heatmaps_by_basin_v2(
    plot_dfs,
    vmin=0, vmax=120,
    seasons=("DJF", "MAM", "JJA", "SON"),
    cmap="jet"
):
    """
    plot_dfs: dict {product_name : df}
        Each df must contain columns including: basin, season, precip variable.
    """

    nprod = len(plot_dfs)
    fig, axes = plt.subplots(
        1, nprod, figsize=(4*nprod, 8),
        sharey=False
    )

    if nprod == 1:
        axes = [axes]

    # --- Collect all images for one colorbar ---
    img_list = []

    for ax, (prod_name, df_prod) in zip(axes, plot_dfs.items()):

        cols = df_prod.columns
        bsn_col  = [c for c in cols if 'basin' in c][0]
        prcp_col = [c for c in cols if 'precip' in c or 'pr' in c][0]

        # Ensure season order
        df_prod = df_prod.copy()
        df_prod["season"] = pd.Categorical(df_prod["season"], seasons, ordered=True)

        # Pivot to matrix (basin × season)
        mat = df_prod.pivot(index=bsn_col, columns="season", values=prcp_col)
        mat = mat.loc[sorted(mat.index), seasons]

        # Plot heatmap
        im = ax.imshow(mat.values, aspect="auto", origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        img_list.append(im)

        # Axes formatting
        ax.set_xticks(np.arange(len(seasons)))
        ax.set_xticklabels(seasons, fontsize=15)
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=15)

        ax.set_title(prod_name, fontsize=15, fontweight="bold")

        # grid lines
        ax.set_xticks(np.arange(-0.5, len(seasons), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.4, color="white")

    axes[0].set_ylabel("Basin ID", fontsize=15)

    # --- COLORBAR OUTSIDE BELOW EVERYTHING ---
    # Add extra space at the bottom
    fig.subplots_adjust(bottom=0.18)

    # cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    cbar_ax = fig.add_axes([0.35, 0.08, 0.30, 0.03])
    cbar = fig.colorbar(img_list[0], cax=cbar_ax, 
                        orientation="horizontal",
                        extend="max",
                        )
    cbar.set_label("Precipitation [mm/season]", fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    plt.show()
#-----------------------------------------------------------------------------
def plot_seasonal_by_season_product(
    plot_dfs,
    basin_order=None,
    vmin=None,
    vmax=None,
    cmap="jet",
):
    """
    Seasonal comparison heatmap.

    Parameters
    ----------
    plot_dfs : dict
        {product_label: df} where each df has columns including:
        - basin column (name contains 'basin')
        - precip column (name contains 'precip' or 'pr')
        - season column ('season')
    basin_order : list or None
        Order of basin IDs on the y-axis. If None, inferred from data.
    vmin, vmax : float or None
        Colorbar limits. If None, computed from all products.
    cmap : str
        Matplotlib colormap name.
    """

    product_labels = list(plot_dfs.keys())

    # ---------- infer basin_order if not given ----------
    if basin_order is None:
        all_basins = []
        for df in plot_dfs.values():
            cols = df.columns.tolist()
            bsn_col = [c for c in cols if "basin" in c][0]
            all_basins.extend(df[bsn_col].dropna().unique())
        # drop 0 if present, sort, cast to int
        basin_order = sorted(int(b) for b in set(all_basins) if b != 0)

    # ---------- infer global vmin / vmax if needed ----------
    if (vmin is None) or (vmax is None):
        vals = []
        for df in plot_dfs.values():
            cols = df.columns.tolist()
            prcp_col = [c for c in cols if ("precip" in c) or ("pr" in c)][0]
            vals.append(df[prcp_col].to_numpy().ravel())
        vals = np.concatenate(vals)
        vals = vals[np.isfinite(vals)]
        if vmin is None:
            vmin = 0.0 if np.nanmin(vals) >= 0 else float(np.nanmin(vals))
        if vmax is None:
            vmax = float(np.nanmax(vals))

    # ---------- make figure: 4 panels, one per season ----------
    fig, axes = plt.subplots(
        1, len(SEAS),
        figsize=(4 * len(SEAS), 6),
        sharey=False,
        sharex=False,
    )

    if len(SEAS) == 1:
        axes = [axes]

    img_list = []

    for ax, season in zip(axes, SEAS):
        # matrix: rows = basins, cols = products
        mat = np.full((len(basin_order), len(product_labels)), np.nan)

        for j, (prod_label, df) in enumerate(plot_dfs.items()):
            cols = df.columns.tolist()
            bsn_col = [c for c in cols if "basin" in c][0]
            prcp_col = [c for c in cols if ("precip" in c) or ("pr" in c)][0]
            seas_col = [c for c in cols if "season" in c.lower()][0]

            # ensure ordered categorical seasons
            df_loc = df.copy()
            df_loc[seas_col] = pd.Categorical(df_loc[seas_col], SEAS, ordered=True)

            # subset this season and aggregate per basin
            df_s = (
                df_loc[df_loc[seas_col] == season]
                .groupby(bsn_col)[prcp_col]
                .mean()
                .reindex(basin_order)
            )

            mat[:, j] = df_s.to_numpy()

        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        img_list.append(im)

        # x / y ticks
        ax.set_xticks(np.arange(len(product_labels)))
        ax.set_xticklabels(product_labels, rotation=45, ha="right", fontsize=12)
        ax.set_yticks(np.arange(len(basin_order)))
        ax.set_yticklabels(basin_order, fontsize=15)

        # panel title = season
        ax.set_title(season, fontsize=15, fontweight="bold")

        # light grid (minor)
        ax.set_xticks(np.arange(-0.5, len(product_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(basin_order), 1), minor=True)
        ax.grid(which="minor", color="w", linewidth=0.6, alpha=0.6)
        ax.tick_params(which="minor", length=0)

    # Axis labels
    axes[0].set_ylabel("Basin ID", fontsize=15)
    # fig.supxlabel("Product", fontsize=13, y=0.10)

    # ---------- colorbar outside, shorter ----------
    fig.subplots_adjust(bottom=0.2, left=0.07, right=0.97, top=0.94, wspace=0.20)

    # [left, bottom, width, height]  → shorter bar centered
    cbar_ax = fig.add_axes([0.35, 0.008, 0.3, 0.04])
    cbar = fig.colorbar(
        img_list[0],
        cax=cbar_ax,
        orientation="horizontal",
        extend="max",
    )
    cbar.set_label("Precipitation [mm/season]", fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    plt.show()

#----------------------------------------------------------------------------
def plot_monthly_cycle_by_basin_products_precomputed(
    plot_dfs,
    basin_order=None,
    figsize=(8, 12),
    vmin_y=None,
    vmax_y=None,
):
    """
    plot_dfs : dict of {product_label : df}
        Each df must already contain monthly-mean-per-basin values.
        Required columns:
           - 'month' (1–12)
           - basin column (detected automatically)
           - precip column (detected automatically)

    No groupby done in this function.
    """

    # ---- Collect basin IDs & colors ----
    all_basins = set()
    for df in plot_dfs.values():
        cols = df.columns.tolist()
        bsn_col = [c for c in cols if "basin" in c][0]
        all_basins.update(df[bsn_col].unique())

    if basin_order is None:
        basin_order = sorted(all_basins)

    n_basins = len(basin_order)
    cmap = get_cmap("tab20", n_basins)
    basin_to_color = {b: cmap(i) for i, b in enumerate(basin_order)}

    # ---- Determine y limits if not provided ----
    if vmin_y is None or vmax_y is None:
        all_vals = []
        for df in plot_dfs.values():
            cols = df.columns.tolist()
            pr_col = [c for c in cols if "precip" in c or "pr" in c][0]
            all_vals.append(df[pr_col].values)
        all_vals = np.concatenate(all_vals)
        if vmin_y is None:
            vmin_y = np.nanmin(all_vals)
        if vmax_y is None:
            vmax_y = np.nanmax(all_vals)

    # ---- Create figure ----
    n_prod = len(plot_dfs)
    fig, axes = plt.subplots(n_prod, 1, figsize=figsize, sharex=True,sharey=False)
    if n_prod == 1:
        axes = [axes]

    months = np.arange(1, 13)

    # ---- Plot each product ----
    for ax, (product_label, df) in zip(axes, plot_dfs.items()):

        cols = df.columns.tolist()
        bsn_col = [c for c in cols if "basin" in c][0]
        pr_col  = [c for c in cols if "precip" in c or "pr" in c][0]

        # Loop basins (df already contains monthly values)
        for basin in basin_order:
            sub = df[df[bsn_col] == basin].sort_values("month")
            y = sub[pr_col].values

            label = str(basin) if ax is axes[0] else None  # legend only once

            ax.plot(months, y, "-o", lw=1.5, ms=3,
                    color=basin_to_color[basin],
                    label=label)

        ax.set_ylim(vmin_y, vmax_y)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylabel("Precipitation [mm/month]", fontsize=11)
        ax.set_title(product_label, fontsize=12, fontweight="bold")

    # ---- Month labels ----
    axes[-1].set_xticks(months)
    axes[-1].set_xticklabels(MONTH_LABELS, fontsize=11)
    axes[-1].set_xlabel("Month", fontsize=12)

    # ---- Shared legend ----
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, title="IMBIE Basin ID",
               loc="lower center", ncol=6,
               fontsize=9, title_fontsize=10,
               frameon=True)

    plt.subplots_adjust(left=0.12, right=0.98,
                        top=0.96, bottom=0.12, hspace=0.25)
    plt.show()

#----------------------------------------------------------------------------
def colors_for_basins(basin_array, default=(0.8, 0.8, 0.8, 1.0)):
    """Return RGBA colors for each basin id using a stable mapping for IDs 2..19."""
    return np.array([ID2COLOR.get(int(b), default) for b in basin_array])

# ---------------------- Main plotting function ----------------------

def plot_pmb_scatter(
    df_mean_yr_acc,
    ref,
    products,
    high_thresh=500.0,
    scale="linear",         # "linear" or "log"
    log_min=5,              # <-- set to 5 to reveal low GPM values
    log_ticks=(5, 10, 20, 50, 100, 200, 500, 1000, 2000),
    ncols=4,                # <-- controls layout; 4 gives 2x4 for 7 products
    figsize_per_col=4.6,    # <-- minimal tuning knobs
    figsize_per_row=4.2,
    share_axes=True,        # keep comparability
    show_ylabel_only_left=True,
):
    """
    Scatter of product vs ref (Pmb) by IMBIE basin with consistent colors (IDs 2..19).

    Improvements:
      - Uses a grid layout for many products (readable)
      - Keeps scale switch: linear or log
      - Allows log_min=5 to show low GPM values
    """

    pretty = {"GPCP": "GPCP v3.3", "RACMO": "RACMO v2.4p1"}

    n = len(products)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per_col, nrows * figsize_per_row),
        sharex=share_axes, sharey=share_axes
    )
    axes = np.array(axes).ravel()

    global_max = 2000.0

    for k, (ax, prod) in enumerate(zip(axes, products)):
        yname = pretty.get(prod, prod)

        # valid rows and arrays
        valid = df_mean_yr_acc[[ref, prod, "basin"]].notnull().all(axis=1)
        sub = df_mean_yr_acc.loc[valid].copy()
        sub["basin"] = sub["basin"].astype(int)

        x_all = sub[ref].to_numpy()
        y_all = sub[prod].to_numpy()
        b_all = sub["basin"].to_numpy()

        # log requires strictly positive values
        if scale == "log":
            pos = (x_all > 0) & (y_all > 0)
        else:
            pos = np.isfinite(x_all) & np.isfinite(y_all)

        x = x_all[pos]
        y = y_all[pos]
        b = b_all[pos]

        # colors by explicit basin mapping
        cols = colors_for_basins(b)

        # scatter
        ax.scatter(
            x, y,
            c=cols, s=110, alpha=0.85,
            edgecolor="k", linewidths=0.6, zorder=2
        )

        # annotate
        diff = np.abs(x - y)
        mask = (diff >= high_thresh) | (((b >= 13) & (b <= 18)) | (x >= 500))
        for xx, yy, bb in zip(x[mask], y[mask], b[mask]):
            ax.annotate(
                f"{int(bb)}",
                xy=(xx, yy), xycoords="data",
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=12, color="black",
                clip_on=False,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                zorder=4
            )

        # limits, scales, and 1:1 line
        if scale == "log":
            lims = (log_min, global_max)
            ax.set_xscale("log"); ax.set_yscale("log")
        else:
            lims = (0.0, global_max)

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_aspect("equal", adjustable="box")

        # stats
        in_box = (x >= lims[0]) & (x <= lims[1]) & (y >= lims[0]) & (y <= lims[1])
        if np.count_nonzero(in_box) >= 2:
            cc = round(p_corr(x[in_box], y[in_box]), 2)
            bias = round(relative_bias(x[in_box], y[in_box]) * 100, 2)
        else:
            cc, bias = np.nan, np.nan

        ax.text(
            0.03, 0.97,
            f"CC={cc:.2f}\nBias={bias:.2f} %",
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=14,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

        # labels: make them less repetitive
        ref_nme = r"$P_{\mathrm{MB}}$" if ref == "Pmb" else ref
        ax.set_title(yname, fontsize=14, fontweight="bold", pad=6)
        ax.set_xlabel(f"{ref_nme} [mm/yr]", fontsize=13)

        if (not show_ylabel_only_left) or (k % ncols == 0):
            ax.set_ylabel(f"{yname} [mm/yr]", fontsize=13)
        else:
            ax.set_ylabel("")

        ax.tick_params(labelsize=12)

        # ticks & grid
        if scale == "log":
            ax.set_xticks(log_ticks)
            ax.set_yticks(log_ticks)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        else:
            step = 500.0
            ax.set_xticks(np.arange(0, global_max + step, step))
            ax.set_yticks(np.arange(0, global_max + step, step))
            ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6)

    # turn off any unused axes
    for ax in axes[len(products):]:
        ax.axis("off")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.03, hspace=0.2)
    return fig, axes

#---------------------------------------------------------------------------

def plot_basin_spread_points(
    df,
    basin_col="basin",
    ref_col="Pmb",
    prod_cols=None,
    prod_labels=None,
    product_styles=None,            # <-- use your product_styles_corr here
    figsize=(13, 5.2),
    log_scale=True,
    ylim=(10, 2000),
    pmb_bar_color="lightgray",
    pmb_edge_color="black",
    legend_ncol=4,
):
    """
    Basin plot with P_MB bars, product points, and per-basin spread annotation.

    Spread per basin:
        (max(products) - min(products)) / mean(products)
    where products = [ref_col] + prod_cols.
    """

    if prod_cols is None:
        prod_cols = ["ERA5", "GPCP v3.3", "RACMO 2.4p1"]
    if prod_labels is None:
        prod_labels = prod_cols

    # styles dict
    product_styles = {} if product_styles is None else product_styles

    # --- clean copy ---
    df_plot = df.copy()
    df_plot[basin_col] = df_plot[basin_col].astype(int)

    # keep only basins where reference exists
    df_plot = df_plot.dropna(subset=[ref_col])

    # sort basins numerically
    df_plot = df_plot.sort_values(basin_col)
    basins = df_plot[basin_col].values
    x = np.arange(len(basins))

    # one row per basin assumption
    df_plot = df_plot.set_index(basin_col).loc[basins]

    # remove ref_col from prod_cols if user accidentally included it
    prod_cols_clean = [c for c in prod_cols if c != ref_col]
    prod_labels_clean = []
    for c in prod_cols:
        if c != ref_col:
            # keep labels aligned
            idx = prod_cols.index(c)
            prod_labels_clean.append(prod_labels[idx] if prod_labels is not None else c)

    prod_cols = prod_cols_clean
    prod_labels = prod_labels_clean

    all_prod_cols = [ref_col] + list(prod_cols)

    # --- figure ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- Pmb bars ---
    ax.bar(
        x,
        df_plot[ref_col].values,
        color=pmb_bar_color,
        edgecolor=pmb_edge_color,
        linewidth=1.0,
        label=r"$P_{\mathrm{MB}}$",
        zorder=1,
    )

    # --- overlay products as points ---
    # fallback cycle if marker not provided
    fallback_markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", ">", "<"]
    fallback_sizes = [8, 7, 7, 7, 7, 7, 7, 8, 7, 7, 7]

    for i, (col, lab) in enumerate(zip(prod_cols, prod_labels)):
        if col not in df_plot.columns:
            continue

        y = df_plot[col].values.astype(float)
        mask = np.isfinite(y)

        st = product_styles.get(col, {})
        color = st.get("color", None)
        marker = st.get("marker", fallback_markers[i % len(fallback_markers)])
        ms = st.get("markersize", fallback_sizes[i % len(fallback_sizes)])

        # if you want specific “hollow” behaviour:
        # make ERA5 hollow only, everything else filled
        is_hollow = (col == "ERA5")
        mfc = "white" if is_hollow else (color if color is not None else None)

        ax.plot(
            x[mask],
            y[mask],
            marker=marker,
            markersize=ms,
            linestyle="None",
            color=color,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.5,
            label=lab,
            zorder=4,
        )

    # --- y scale ---
    if log_scale:
        bottom, top = ylim
        top = top * 1.25
        ax.set_yscale("log")
        ax.set_ylim(bottom, top)

        log_ticks = [10, 20, 50, 100, 200, 500, 1000, 1500, 2000]
        ax.set_yticks([t for t in log_ticks if bottom <= t <= ylim[1]])
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())

    ax.tick_params(axis="y", labelsize=12)

    # --- spread calc ---
    vals_stack = []
    for col in all_prod_cols:
        if col in df_plot.columns:
            vals_stack.append(df_plot[col].values.astype(float))
        else:
            vals_stack.append(np.full(len(basins), np.nan))
    vals_stack = np.vstack(vals_stack)

    vmin = np.nanmin(vals_stack, axis=0)
    vmax = np.nanmax(vals_stack, axis=0)
    vmean = np.nanmean(vals_stack, axis=0)

    spread = np.full_like(vmean, np.nan, dtype=float)
    valid = np.isfinite(vmin) & np.isfinite(vmax) & np.isfinite(vmean) & (vmean != 0)
    spread[valid] = (vmax[valid] - vmin[valid]) / vmean[valid]
    spread_pct = np.round(spread * 100).astype(int)

    # --- annotate spread ---
    y_top_axis = ax.get_ylim()[1]
    y_bot_axis = ax.get_ylim()[0]

    for xi, s, top_val in zip(x, spread_pct, vmax):
        if not np.isfinite(s) or not np.isfinite(top_val) or top_val <= 0:
            continue

        if log_scale:
            y_text = top_val * 1.20
            if y_text > y_top_axis:
                y_text = top_val * 1.05
        else:
            y_text = top_val + 0.05 * (y_top_axis - y_bot_axis)

        ax.text(
            xi, y_text, f"{s}%",
            ha="center", va="bottom",
            fontsize=10, zorder=5,
        )

    # --- cosmetics ---
    ax.set_xticks(x)
    ax.set_xticklabels(basins, ha="center", fontsize=14)
    ax.set_xlabel("Basin", fontsize=16)
    ax.set_ylabel("Precipitation [mm/year]", fontsize=16)

    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    ax.legend(
        fontsize=14,
        ncol=legend_ncol,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig, ax, spread, spread_pct

#-----------------------------------------------------------------
def plot_basin_spread_points_dual(
    df,
    basin_col="basin",
    ref_col="Pmb",                    # P_MB column in df (bar)
    prod_cols=None,                   # products to plot as points (MUST NOT include ref_col)
    prod_labels=None,                 # optional labels (list aligned with prod_cols) or dict col->label
    product_styles=None,              # your product_styles_corr
    non_gpm_group=None,               # spread group 1 (must include ref_col)
    gpm_group=None,                   # spread group 2 (must include ref_col)
    figsize=(13, 5.2),
    log_scale=True,
    ylim=(10, 2000),
    pmb_bar_color="lightgray",
    pmb_edge_color="black",
    annotate_non_gpm_color="black",
    annotate_gpm_color="dimgray",
    annotate_fontsize=10,
    legend_ncol=4,
    place_key=True,
    key_loc=(0.02, 0.98),
    key_fontsize=10,
):
    """
    Basin plot with:
      - P_MB as bars (ref_col)
      - product points for prod_cols using product_styles
      - TWO spread annotations per basin:
          S_nonGPM = spread among non_gpm_group (includes ref_col)
          S_GPM    = spread among gpm_group     (includes ref_col)

    Spread definition:
        S = (max(P_i) - min(P_i)) / mean(P_i) * 100%
    """

    # -----------------------------
    # defaults / safety
    # -----------------------------
    if product_styles is None:
        product_styles = {}

    if prod_cols is None:
        prod_cols = ["ERA5", "GPCP v3.3", "ATMS", "MHS", "DMSP SSMIS", "AMSR2", "GPM Satellites"]

    # HARD RULE: never plot ref_col as marker
    prod_cols = [c for c in prod_cols if c != ref_col]

    # Labels mapping: dict(col->label)
    if prod_labels is None:
        label_map = {c: c for c in prod_cols}
    elif isinstance(prod_labels, dict):
        label_map = {c: prod_labels.get(c, c) for c in prod_cols}
    else:
        # assume list aligned with prod_cols
        if len(prod_labels) != len(prod_cols):
            raise ValueError("prod_labels must have same length as prod_cols (or be a dict col->label).")
        label_map = {c: lab for c, lab in zip(prod_cols, prod_labels)}

    # Spread groups (include ref_col as you wanted)
    if non_gpm_group is None:
        non_gpm_group = [ref_col, "ERA5", "GPCP v3.3"]  # add "RACMO 2.4p1" if present
    if gpm_group is None:
        gpm_group = [ref_col, "ATMS", "MHS", "DMSP SSMIS", "AMSR2"]  # exclude "GPM Satellites" to avoid double-counting

    if ref_col not in non_gpm_group:
        non_gpm_group = [ref_col] + list(non_gpm_group)
    if ref_col not in gpm_group:
        gpm_group = [ref_col] + list(gpm_group)

    # -----------------------------
    # prep df
    # -----------------------------
    df_plot = df.copy()
    df_plot[basin_col] = df_plot[basin_col].astype(int)
    df_plot = df_plot.dropna(subset=[ref_col])
    df_plot = df_plot.sort_values(basin_col)

    basins = df_plot[basin_col].values
    x = np.arange(len(basins))

    # one row per basin expected
    df_plot = df_plot.set_index(basin_col).loc[basins]

    # -----------------------------
    # figure
    # -----------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # P_MB bars (ONLY PMB legend entry)
    bar_container = ax.bar(
        x,
        df_plot[ref_col].values.astype(float),
        color=pmb_bar_color,
        edgecolor=pmb_edge_color,
        linewidth=1.0,
        label=r"$P_{\mathrm{MB}}$",
        zorder=1,
    )

    # -----------------------------
    # points for products (use product_styles_corr)
    # -----------------------------
    fallback_markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", ">", "<"]
    fallback_sizes = [8, 8, 8, 8, 8, 8, 8, 9, 8, 8, 8]

    point_handles = []

    for i, col in enumerate(prod_cols):
        if col not in df_plot.columns:
            continue

        y = df_plot[col].values.astype(float)
        mask = np.isfinite(y)

        st = product_styles.get(col, {})
        color = st.get("color", None)
        marker = st.get("marker", fallback_markers[i % len(fallback_markers)])
        ms = st.get("markersize", fallback_sizes[i % len(fallback_sizes)])

        # Hollow ERA5 only (optional)
        is_hollow = (col == "ERA5")
        mfc = "white" if is_hollow else (color if color is not None else None)

        h = ax.plot(
            x[mask],
            y[mask],
            linestyle="None",
            marker=marker,
            markersize=ms,
            color=color,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.6,
            label=label_map.get(col, col),
            zorder=4,
        )[0]

        point_handles.append(h)

    # -----------------------------
    # y-scale
    # -----------------------------
    if log_scale:
        bottom, top = ylim
        top = top * 1.35  # extra headroom for 2 stacked % labels
        ax.set_yscale("log")
        ax.set_ylim(bottom, top)

        log_ticks = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
        ax.set_yticks([t for t in log_ticks if bottom <= t <= ylim[1]])
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())

    ax.tick_params(axis="y", labelsize=12)

    # -----------------------------
    # spread helpers
    # -----------------------------
    def spread_pct_for_group(df_local, cols):
        arrs = []
        for c in cols:
            if c in df_local.columns:
                arrs.append(df_local[c].values.astype(float))
        if len(arrs) < 2:
            return np.full(len(df_local), np.nan)

        vals = np.vstack(arrs)
        vmin = np.nanmin(vals, axis=0)
        vmax = np.nanmax(vals, axis=0)
        vmean = np.nanmean(vals, axis=0)

        out = np.full_like(vmean, np.nan, dtype=float)
        ok = np.isfinite(vmin) & np.isfinite(vmax) & np.isfinite(vmean) & (vmean != 0)
        out[ok] = (vmax[ok] - vmin[ok]) / vmean[ok] * 100.0
        return out

    spread_non_gpm = spread_pct_for_group(df_plot, non_gpm_group)
    spread_gpm = spread_pct_for_group(df_plot, gpm_group)

    # Use max among all plotted products + ref for label placement
    cols_for_top = sorted(set([ref_col] + list(prod_cols) + list(non_gpm_group) + list(gpm_group)))
    top_stack = np.vstack([df_plot[c].values.astype(float) for c in cols_for_top if c in df_plot.columns])
    top_val_all = np.nanmax(top_stack, axis=0)

    # -----------------------------
    # annotate spreads (better separation)
    # -----------------------------
    y_top_axis = ax.get_ylim()[1]
    y_bot_axis = ax.get_ylim()[0]

    for xi, top_val, s_ref, s_gpm in zip(x, top_val_all, spread_non_gpm, spread_gpm):
        if not np.isfinite(top_val) or top_val <= 0:
            continue

        if log_scale:
            # clearer separation: non-GPM higher, GPM much lower
            y1 = min(top_val * 1.45, y_top_axis * 0.985)  # non-GPM
            y2 = min(top_val * 1.18, y_top_axis * 0.93)   # GPM
        else:
            y1 = top_val + 0.09 * (y_top_axis - y_bot_axis)
            y2 = top_val + 0.03 * (y_top_axis - y_bot_axis)

        if np.isfinite(s_ref):
            ax.text(xi, y1, f"{int(round(s_ref))}%", ha="center", va="bottom",
                    fontsize=annotate_fontsize, color=annotate_non_gpm_color, zorder=6)

        if np.isfinite(s_gpm):
            ax.text(xi, y2, f"{int(round(s_gpm))}%", ha="center", va="bottom",
                    fontsize=annotate_fontsize, color=annotate_gpm_color, zorder=6)

    # -----------------------------
    # cosmetics
    # -----------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(basins, ha="center", fontsize=14)
    ax.set_xlabel("Basin", fontsize=16)
    ax.set_ylabel("[mm/year]", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # ---- tiny key (NO title, not busy) ----
    if place_key:
        # small white box, two colored lines
        ax.text(
            key_loc[0], key_loc[1],
            "% = spread(P_MB, ERA5, GPCP v3.3)",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=key_fontsize,
            color=annotate_non_gpm_color,
            # bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75", alpha=0.95),
            zorder=10,
        )
        ax.text(
            key_loc[0], key_loc[1] - 0.055,
            "% = spread(P_MB, ATMS, MHS, DMSP SSMIS, AMSR2)",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=key_fontsize,
            color=annotate_gpm_color,
            zorder=10,
        )

    # -----------------------------
    # Legend: force correct handles, avoid any accidental "Pmb" marker entry
    # -----------------------------
    handles, labels = ax.get_legend_handles_labels()

    # Deduplicate while preserving order
    seen = set()
    new_handles, new_labels = [], []
    for h, lab in zip(handles, labels):
        if lab in seen:
            continue
        seen.add(lab)
        new_handles.append(h)
        new_labels.append(lab)

    ax.legend(
        new_handles, new_labels,
        fontsize=14,
        ncol=legend_ncol,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig, ax, spread_non_gpm, spread_gpm
#-----------------------------------------------------------------
def _normalize_monthly_df(df):
    """Make df columns consistent: basin, month, precipitation."""
    df = df.copy()

    # Basin column: basin_id -> basin
    if "basin" not in df.columns:
        if "basin_id" in df.columns:
            df = df.rename(columns={"basin_id": "basin"})
        else:
            raise ValueError("DataFrame must have 'basin' or 'basin_id' column.")

    # Precip column: precip_mm_per_month -> precipitation
    if "precipitation" not in df.columns:
        if "precip_mm_per_month" in df.columns:
            df = df.rename(columns={"precip_mm_per_month": "precipitation"})
        else:
            raise ValueError("DataFrame must have 'precipitation' "
                             "or 'precip_mm_per_month' column.")

    # Month: if missing, derive from 'time'
    if "month" not in df.columns:
        if "time" in df.columns:
            df["month"] = pd.to_datetime(df["time"]).dt.month
        else:
            raise ValueError("DataFrame must have 'month' or 'time' column.")

    return df[["month", "basin", "precipitation"]]


def plot_monthly_cycles_regions_3x1(plot_dfs, region_defs):
    """
    plot_dfs: dict mapping product_name -> DataFrame
    region_defs: list of (region_name, [basin_ids])
    """
    # Normalize all dfs first
    norm_dfs = {name: _normalize_monthly_df(df)
                for name, df in plot_dfs.items()}

    months = np.arange(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Pre-compute regional series but no global ymax since y is NOT shared
    region_series = {}

    for region_name, basin_ids in region_defs:
        region_series[region_name] = {}
        for prod_name, df in norm_dfs.items():
            sub = df[df["basin"].isin(basin_ids)]
            s = (sub.groupby("month")["precipitation"]
                     .mean()
                     .reindex(months))
            region_series[region_name][prod_name] = s

    # --- Plotting ---
    n_regions = len(region_defs)
    fig, axes = plt.subplots(n_regions, 1, sharex=True,
                             figsize=(9, 9), constrained_layout=False)
    # Make room on the left for the shared y-label
    fig.subplots_adjust(left=0.18, ) # bottom=0.10, hspace=0.18
    fig.text(
    0.08, 0.5, 
    "Precipitation [mm/month]", 
    va="center", 
    rotation="vertical",
    fontsize=15,
    fontweight="bold"
    )

    if n_regions == 1:
        axes = [axes]

    # consistent colors/markers across products
    product_names = list(plot_dfs.keys())
    color_cycle = ["k", "tab:blue", "tab:orange", "tab:green", "tab:red"]
    marker_cycle = ["o", "s", "D", "^", "v"]

    prod_style = {}
    for i, pname in enumerate(product_names):
        prod_style[pname] = dict(
            color=color_cycle[i % len(color_cycle)],
            marker=marker_cycle[i % len(marker_cycle)]
        )

    # --- Plotting ---
    for i, (ax, (region_name, _)) in enumerate(zip(axes, region_defs)):

        for pname in product_names:
            s = region_series[region_name][pname]
            ax.plot(
                months,
                s.values,
                label=pname,
                linewidth=1.8,
                **prod_style[pname]
            )

        # ---------- REGION-SPECIFIC Y-TICKS ----------
        if region_name == "Antarctica":
            ax.set_yticks([10, 20, 30, 40, 50])
        elif region_name == "East Antarctica":
            ax.set_yticks([0, 5, 10, 15, 20])

        # ---------- AXIS LABELING ----------
        # ax.set_ylabel("Precipitation [mm/month]", fontsize=15)
        ax.set_title(region_name, fontsize=13, fontweight="bold")

        # X-axis on bottom
        axes[-1].set_xticks(months)
        axes[-1].set_xticklabels(month_labels, fontsize=15)

        # ---------- TICKS & GRIDS ----------
        # Major ticks: both axes
        ax.tick_params(axis="both", which="major", direction="out")

        # Minor ticks: ONLY on Y axis
        ax.tick_params(axis="y", which="minor", direction="out")
        ax.minorticks_on()

        # Disable minor ticks on X axis
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)

        # Grid: only major gridlines
        ax.grid(which="major", alpha=0.35)
        ax.grid(which="minor", alpha=0)

    # -------- LEGEND AT BOTTOM --------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center",
               bbox_to_anchor=(0.53, -0.01),
               ncol=len(product_names),
               fontsize=13, frameon=False)

    return fig, axes

#-----------------------------------------------------------------------------

def _normalize_seasonal_df(df):
    """
    Make seasonal df columns consistent: season, basin, precipitation.
    Accepts columns like:
      - basin_id / basin
      - precip_mm_per_month / precipitation_season_clim / pr_season_clim / precipitation
    """
    df = df.copy()

    # Basin column
    if "basin" not in df.columns:
        if "basin_id" in df.columns:
            df = df.rename(columns={"basin_id": "basin"})
        else:
            raise ValueError("DataFrame must have 'basin' or 'basin_id' column.")

    # Precip column
    if "precipitation" not in df.columns:
        for cand in ["precip_mm_per_month",
                     "precipitation_season_clim",
                     "pr_season_clim"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "precipitation"})
                break
        else:
            raise ValueError(
                "DataFrame must have one of "
                "['precipitation', 'precip_mm_per_month', "
                "'precipitation_season_clim', 'pr_season_clim']."
            )

    # Season must exist
    if "season" not in df.columns:
        raise ValueError("DataFrame must have a 'season' column (e.g., DJF, MAM, JJA, SON).")

    return df[["season", "basin", "precipitation"]]

def plot_seasonal_cycles_regions_3x1(plot_dfs, region_defs):
    """
    Seasonal version of the monthly cycles plot.

    plot_dfs: dict mapping product_name -> seasonal DataFrame
              (one of the four you showed for Pmb, ERA5, GPCP, RACMO)
    region_defs: list of (region_name, [basin_ids])
    """
    # --- Normalize all seasonal dfs ---
    norm_dfs = {name: _normalize_seasonal_df(df)
                for name, df in plot_dfs.items()}

    # Order of seasons on x-axis
    seasons = ["DJF", "MAM", "JJA", "SON"]
    x = np.arange(len(seasons))

    # Pre-compute regional series
    region_series = {}
    for region_name, basin_ids in region_defs:
        region_series[region_name] = {}
        for prod_name, df in norm_dfs.items():
            sub = df[df["basin"].isin(basin_ids)]
            s = (sub.groupby("season")["precipitation"]
                     .mean()
                     .reindex(seasons))
            region_series[region_name][prod_name] = s

    # --- Plotting ---
    n_regions = len(region_defs)
    fig, axes = plt.subplots(
        n_regions, 1, sharex=True,
        figsize=(8, 10), constrained_layout=False
    )

    # Make room on the left for shared y-label and at bottom for legend
    fig.subplots_adjust(left=0.18, bottom=0.14, hspace=0.20)

    # Shared y-label
    fig.text(
        0.08, 0.5,
        "Precipitation [mm/season]",
        va="center",
        rotation="vertical",
        fontsize=15,
        fontweight="bold"
    )

    if n_regions == 1:
        axes = [axes]

    # Consistent styles across products
    product_names = list(plot_dfs.keys())
    

    prod_style = {
        pname: dict(
            color=color_cycle[i % len(color_cycle)],
            marker=marker_cycle[i % len(marker_cycle)]
        )
        for i, pname in enumerate(product_names)
    }

    for ax, (region_name, _) in zip(axes, region_defs):

        for pname in product_names:
            s = region_series[region_name][pname]
            ax.plot(
                x,
                s.values,
                label=pname,
                linewidth=1.8,
                **prod_style[pname]
            )

        # Region-specific y-ticks (adapted from your monthly version)
        if region_name == "Antarctica":
            ax.set_yticks([10, 20, 30, 40, 50])
        elif region_name == "East Antarctica":
            ax.set_yticks([0, 5, 10, 15, 20])

        ax.set_title(region_name, fontsize=13, fontweight="bold")

        # X-axis ticks and labels only set once (shared x)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(seasons, fontsize=13)

        # Major ticks on both axes
        ax.tick_params(axis="both", which="major", direction="out")

        # Minor ticks: only on Y
        ax.minorticks_on()
        ax.tick_params(axis="y", which="minor", direction="out")
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)

        # Grid on major ticks only
        ax.grid(which="major", alpha=0.35)
        ax.grid(which="minor", alpha=0.0)

    # Legend at bottom (common)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.02),
        ncol=len(product_names),
        fontsize=12,
        frameon=False
    )

    return fig, axes


#-----------------------------------------------------------------------------
def plot_seasonal_cycles_regions_1x3(plot_dfs, region_defs):
    """
    Seasonal (DJF, MAM, JJA, SON) cycles for:
        Antarctica, West Antarctica, East Antarctica

    Layout: 1 row × 3 columns (no shared y-axis).
    """
    # Normalize all product DataFrames
    norm_dfs = {name: _normalize_seasonal_df(df)
                for name, df in plot_dfs.items()}
    
    SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

    seasons = SEASON_ORDER
    x = np.arange(len(seasons))  # 0..3 for plotting

    # Precompute regional series: region -> product -> seasonal series
    region_series = {}
    for region_name, basin_ids in region_defs:
        region_series[region_name] = {}
        for prod_name, df in norm_dfs.items():
            sub = df[df["basin"].isin(basin_ids)]
            s = (sub.groupby("season")["precipitation"]
                     .mean()
                     .reindex(seasons))
            region_series[region_name][prod_name] = s

    # --- Plotting: 1 row × n_regions (3) ---
    n_regions = len(region_defs)
    fig, axes = plt.subplots(
        1, n_regions,
        figsize=(10, 4.5),
        sharex=False,
        sharey=False,
        constrained_layout=False
    )

    fig.subplots_adjust(left=0.10, right=0.98,
                        bottom=0.25, top=0.88,
                        wspace=0.30)

    # Shared y-label on the left
    fig.text(
        0.03, 0.5,
        "Precipitation [mm/season]",
        va="center",
        rotation="vertical",
        fontsize=13,
        fontweight="bold"
    )

    if n_regions == 1:
        axes = [axes]

    # Consistent color/marker style across products
    product_names = list(plot_dfs.keys())
    color_cycle = ["k", "tab:blue", "tab:orange", "tab:green", "tab:red"]
    marker_cycle = ["o", "s", "D", "^", "v"]

    prod_style = {
        pname: dict(
            color=color_cycle[i % len(color_cycle)],
            marker=marker_cycle[i % len(marker_cycle)]
        )
        for i, pname in enumerate(product_names)
    }

    # --- Draw each region panel ---
    for ax, (region_name, _) in zip(axes, region_defs):

        for pname in product_names:
            s = region_series[region_name][pname]
            ax.plot(
                x,
                s.values,
                label=pname,
                linewidth=1.8,
                **prod_style[pname]
            )

        # Titles
        ax.set_title(region_name, fontsize=13, fontweight="bold")

        # X-axis ticks/labels (DJF–SON)
        ax.set_xticks(x)
        ax.set_xticklabels(seasons, fontsize=12)

        # --- Region-specific y-ticks ---
        if region_name == "West Antarctica":
            ax.set_yticks([20, 40, 60, 80])

        # --- Tick + grid styling ---
        # Major ticks on both axes
        ax.tick_params(axis="both", which="major", direction="out")

        # Minor ticks only on Y axis
        ax.tick_params(axis="y", which="minor", direction="out")
        ax.minorticks_on()
        # Disable minor ticks on X
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)

        # Grid only for major ticks
        ax.grid(which="major", alpha=0.35)
        ax.grid(which="minor", alpha=0.0)

    # Legend at bottom center
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=len(product_names),
        fontsize=12,
        frameon=False
    )

    return fig, axes

#-----------------------------------------------------------------------------
def compute_region_means(df, region_defs, product_cols):
    """
    Returns dictionary:
        {region_name: dataframe with columns [year, products...]}
    """

    region_data = {}

    for region_name, basin_ids in region_defs:

        df_region = df[df["basin"].isin(basin_ids)]

        df_mean = (
            df_region
            .groupby("year")[product_cols]
            .mean()
            .reset_index()
            .sort_values("year")
        )

        region_data[region_name] = df_mean

    return region_data

#-----------------------------------------------------------------------------
def compute_region_means_from_products(prdt_df, region_defs):
    """
    prdt_df: list of tuples like
        [("Pmb", df_pmb), ("ERA5", df_era5), ...]

    region_defs: list like
        [("Antarctica", [2,3,...]), ...]

    Returns:
        dict {region_name: dataframe with year + all products}
    """

    region_data = {}

    for region_name, basin_ids in region_defs:

        region_product_dfs = []

        for prod_name, df in prdt_df:

            # Filter basins
            df_region = df[df["basin"].isin(basin_ids)]

            # Compute regional annual mean
            df_mean = (
                df_region
                .groupby("year")[prod_name]
                .mean()
                .reset_index()
            )

            region_product_dfs.append(df_mean)

        # Merge all product dfs on year
        from functools import reduce

        df_merged = reduce(
            lambda left, right: pd.merge(left, right, on="year", how="outer"),
            region_product_dfs
        ).sort_values("year")

        region_data[region_name] = df_merged

    return region_data

#-----------------------------------------------------------------------------
def compute_region_means_from_maps(
        prdt_df,
        region_defs
    ):
    """
    Compute regional precipitation means directly from annual map DataArrays.

    prdt_df:
        [("Pmb", df_unused, Pmb_annual_map),
         ("ERA5", df_unused, era5_annual_map),
         ...]

    region_defs:
        [("Antarctica", [2,3,...]), ...]

    Returns:
        dict {region_name: dataframe with year + product columns}
    """

    import pandas as pd
    import numpy as np
    from functools import reduce

    region_data = {}

    for region_name, basin_ids in region_defs:

        product_results = []

        for prod_name, _, da in prdt_df:

            # -------------------------
            # 🔎 Smart basin detection
            # -------------------------
            basin_candidates = [
                c for c in da.coords
                if c.lower().startswith("basin")
            ]

            if len(basin_candidates) == 0:
                raise ValueError(
                    f"No basin mask coordinate found in {prod_name}. "
                    f"Available coords: {list(da.coords)}"
                )

            basin_name = basin_candidates[0]  # take first match

            yearly_vals = []

            for yr in da["year"].values:

                da_yr = da.sel(year=yr)

                basin_mask = da_yr.coords[basin_name]

                # Region mask
                region_mask = basin_mask.isin(basin_ids)

                # Apply mask
                masked = da_yr.where(region_mask)

                # Spatial mean (area-weighted since grid uniform)
                mean_val = masked.mean(dim=("y", "x"), skipna=True).item()

                yearly_vals.append((yr, mean_val))

            df_prod = pd.DataFrame(yearly_vals, columns=["year", prod_name])
            product_results.append(df_prod)

        # Merge products
        df_merged = reduce(
            lambda left, right: pd.merge(left, right, on="year", how="outer"),
            product_results
        ).sort_values("year")

        region_data[region_name] = df_merged

    return region_data
#-----------------------------------------------------------------------------


def compute_basin_area_weights_from_mask(
    basins_da,
    basin_ids=None,
    basin_col_name="basin",
):
    """
    Compute basin area weights from a basin-ID raster.

    Parameters
    ----------
    basins_da : xarray.DataArray
        Basin mask raster. Example: your IMBIE basin grid.
        Can have dims like (band, y, x) or (y, x).
    basin_ids : list or None
        If given, restrict to these basin IDs only.
    basin_col_name : str
        Output basin column name.

    Returns
    -------
    weights_df : pd.DataFrame
        Columns:
        - basin
        - n_cells
        - area_m2
        - area_km2
        - weight_global
    """

    da = basins_da.copy()

    # Ensure 2D
    if "band" in da.dims:
        da = da.isel(band=0)

    vals = da.values

    # Valid basin cells
    valid = np.isfinite(vals)
    basin_vals = vals[valid].astype(int)

    if basin_ids is not None:
        basin_ids = set(map(int, basin_ids))
        basin_vals = basin_vals[np.isin(basin_vals, list(basin_ids))]

    unique_ids, counts = np.unique(basin_vals, return_counts=True)

    # Pixel area from transform (projected grid)
    transform = da.rio.transform()
    px_w = abs(transform.a)
    px_h = abs(transform.e)
    pixel_area_m2 = px_w * px_h
    pixel_area_km2 = pixel_area_m2 / 1e6

    area_m2 = counts * pixel_area_m2
    area_km2 = counts * pixel_area_km2
    weight_global = counts / counts.sum()

    weights_df = pd.DataFrame({
        basin_col_name: unique_ids.astype(int),
        "n_cells": counts.astype(int),
        "area_m2": area_m2.astype(float),
        "area_km2": area_km2.astype(float),
        "weight_global": weight_global.astype(float),
    }).sort_values(basin_col_name).reset_index(drop=True)

    return weights_df

#-----------------------------------------------------------------------------
def compute_weighted_region_series_from_basin_df(
    df,
    region_defs,
    basin_weights,
    basin_col="basin",
    value_col="precipitation",
    time_col="time",
    extra_group_cols=("year", "month"),
):
    """
    Compute area-weighted regional monthly series from basin-mean dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least:
        [time, basin, precipitation, year, month]
    region_defs : list
        Example:
        [
            ("Antarctica", [2,3,...]),
            ("West Antarctica", [...]),
            ("East Antarctica", [...]),
        ]
    basin_weights : pd.DataFrame
        Output from compute_basin_area_weights_from_mask()
    basin_col, value_col, time_col : str
        Column names
    extra_group_cols : tuple
        Additional columns to preserve per timestep

    Returns
    -------
    region_dict : dict
        {region_name: regional monthly dataframe}
    """

    df = df.copy()
    df[basin_col] = df[basin_col].astype(int)

    wdf = basin_weights[[basin_col, "area_km2"]].copy()

    region_dict = {}

    group_cols = [time_col] + [c for c in extra_group_cols if c in df.columns]

    for region_name, basin_ids in region_defs:
        basin_ids = set(map(int, basin_ids))

        sub = df[df[basin_col].isin(basin_ids)].copy()
        sub = sub.merge(wdf, on=basin_col, how="left")

        if sub["area_km2"].isna().any():
            missing = sorted(sub.loc[sub["area_km2"].isna(), basin_col].unique())
            raise ValueError(
                f"Missing basin weights for {region_name}: {missing}"
            )

        # Weighted mean within each timestep.
        # Important: renormalize weights using only basins present in that timestep.
        def _weighted_mean(g):
            vals = g[value_col].to_numpy(dtype=float)
            wts = g["area_km2"].to_numpy(dtype=float)

            good = np.isfinite(vals) & np.isfinite(wts)
            if good.sum() == 0:
                return np.nan

            vals = vals[good]
            wts = wts[good]

            return np.sum(vals * wts) / np.sum(wts)

        out = (
            sub.groupby(group_cols, dropna=False)
            .apply(_weighted_mean)
            .reset_index(name=value_col)
            .sort_values(group_cols)
            .reset_index(drop=True)
        )

        region_dict[region_name] = out

    return region_dict

#-----------------------------------------------------------------------------
def compute_weighted_region_monthly_climatologies(
    monthly_df_data,
    region_defs,
    basin_weights,
    basin_col="basin",
    value_col="precipitation",
    time_col="time",
):
    """
    For each product, compute area-weighted regional monthly series,
    then average by month across years to form climatology.

    Parameters
    ----------
    monthly_df_data : dict
        {
            "ERA5": era5_basin_mean,
            "GPCP v3.3": gpcp_df,
            ...
        }

    Returns
    -------
    region_monthly_clim : dict
        {
            "Antarctica": df(month + product columns),
            "West Antarctica": ...,
            "East Antarctica": ...
        }
    """

    from functools import reduce

    per_region_product_tables = {rname: [] for rname, _ in region_defs}

    for product_name, df in monthly_df_data.items():

        reg_series = compute_weighted_region_series_from_basin_df(
            df=df,
            region_defs=region_defs,
            basin_weights=basin_weights,
            basin_col=basin_col,
            value_col=value_col,
            time_col=time_col,
            extra_group_cols=("year", "month"),
        )

        for region_name, reg_df in reg_series.items():
            clim = (
                reg_df.groupby("month", as_index=False)[value_col]
                .mean()
                .rename(columns={value_col: product_name})
            )
            per_region_product_tables[region_name].append(clim)

    region_monthly_clim = {}

    for region_name, tables in per_region_product_tables.items():
        merged = reduce(
            lambda left, right: pd.merge(left, right, on="month", how="outer"),
            tables
        ).sort_values("month").reset_index(drop=True)

        region_monthly_clim[region_name] = merged

    return region_monthly_clim

#-----------------------------------------------------------------------------

def month_to_season(month):
    if month in [12, 1, 2]:
        return "DJF"
    elif month in [3, 4, 5]:
        return "MAM"
    elif month in [6, 7, 8]:
        return "JJA"
    elif month in [9, 10, 11]:
        return "SON"
    return np.nan


def season_order_key(season):
    order = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}
    return order.get(season, 999)


def compute_weighted_region_seasonal_climatologies(
    monthly_df_data,
    region_defs,
    basin_weights,
    basin_col="basin",
    value_col="precipitation",
    time_col="time",
    seasonal_mode="sum",
):
    """
    Compute area-weighted regional seasonal climatologies from basin-level data.

    IMPORTANT:
    Assumes precipitation has already been converted to mm/month.
    Therefore:
      - first collapse to ONE value per year-month
      - then form seasonal totals (sum of 3 months) or seasonal means

    Returns
    -------
    dict of region dataframes with columns:
      season, <product1>, <product2>, ...
    """

    if seasonal_mode not in {"sum", "mean"}:
        raise ValueError("seasonal_mode must be 'sum' or 'mean'")

    per_region_product_tables = {rname: [] for rname, _ in region_defs}

    for product_name, df in monthly_df_data.items():

        # Step 1: area-weighted regional series
        reg_series = compute_weighted_region_series_from_basin_df(
            df=df,
            region_defs=region_defs,
            basin_weights=basin_weights,
            basin_col=basin_col,
            value_col=value_col,
            time_col=time_col,
            extra_group_cols=("year", "month"),
        )

        for region_name, reg_df in reg_series.items():
            tmp = reg_df.copy()

            # -------------------------------------------------
            # CRITICAL FIX:
            # collapse to ONE value per year-month first
            # If there are daily/multiple timestamps within a month,
            # we average them because values are already in mm/month.
            # -------------------------------------------------
            monthly_reg = (
                tmp.groupby(["year", "month"], as_index=False)[value_col]
                .mean()
            )

            # assign season
            monthly_reg["season"] = monthly_reg["month"].apply(month_to_season)

            # Step 2: aggregate 3 monthly values into year-season
            if seasonal_mode == "sum":
                yr_season = (
                    monthly_reg.groupby(["year", "season"], as_index=False)[value_col]
                    .sum()
                )
            else:  # seasonal_mode == "mean"
                yr_season = (
                    monthly_reg.groupby(["year", "season"], as_index=False)[value_col]
                    .mean()
                )

            # Step 3: climatological mean across years
            clim = (
                yr_season.groupby("season", as_index=False)[value_col]
                .mean()
                .rename(columns={value_col: product_name})
            )

            clim["season_order"] = clim["season"].map(season_order_key)
            clim = clim.sort_values("season_order").drop(columns="season_order")

            per_region_product_tables[region_name].append(clim)

    # Merge products by region
    region_seasonal_clim = {}

    for region_name, tables in per_region_product_tables.items():
        merged = reduce(
            lambda left, right: pd.merge(left, right, on="season", how="outer"),
            tables
        )

        merged["season_order"] = merged["season"].map(season_order_key)
        merged = (
            merged.sort_values("season_order")
            .drop(columns="season_order")
            .reset_index(drop=True)
        )

        region_seasonal_clim[region_name] = merged

    return region_seasonal_clim
#----------------------------------------------------------------------------


def compute_weighted_region_annual_totals(
    monthly_df_data_mmmonth,
    region_defs,
    basin_weights,
    basin_col="basin",
    value_col="precipitation",
    time_col="time",
    annual_mode="sum",
):
    """
    Compute area-weighted regional annual precipitation totals from basin-level data.

    Assumes input precipitation is already in mm/month.

    Steps (per product, per region):
      1) area-weighted regional series at (time, year, month)
      2) collapse to ONE value per (year, month) (mean if multiple timestamps in month)
      3) annual total = sum of 12 monthly values  -> mm/year

    Returns
    -------
    region_annual : dict
        {
          "Antarctica": df with columns ["year", prod1, prod2, ...],
          "West Antarctica": ...,
          "East Antarctica": ...
        }
    """

    if annual_mode not in {"sum", "mean"}:
        raise ValueError("annual_mode must be 'sum' or 'mean'")

    per_region_tables = {rname: [] for rname, _ in region_defs}

    for product_name, df in monthly_df_data_mmmonth.items():

        reg_series = compute_weighted_region_series_from_basin_df(
            df=df,
            region_defs=region_defs,
            basin_weights=basin_weights,
            basin_col=basin_col,
            value_col=value_col,
            time_col=time_col,
            extra_group_cols=("year", "month"),
        )

        for region_name, reg_df in reg_series.items():
            tmp = reg_df.copy()

            # collapse to one value per year-month (important if daily/multi-timestamp)
            monthly_reg = (
                tmp.groupby(["year", "month"], as_index=False)[value_col]
                .mean()
            )

            if annual_mode == "sum":
                annual = (
                    monthly_reg.groupby("year", as_index=False)[value_col]
                    .sum()
                )  # mm/year
            else:
                annual = (
                    monthly_reg.groupby("year", as_index=False)[value_col]
                    .mean()
                )  # mean mm/month (not recommended)

            annual = annual.rename(columns={value_col: product_name})
            per_region_tables[region_name].append(annual)

    # Merge all products into one table per region
    region_annual = {}

    for region_name, tables in per_region_tables.items():
        merged = reduce(
            lambda left, right: pd.merge(left, right, on="year", how="outer"),
            tables
        ).sort_values("year").reset_index(drop=True)

        region_annual[region_name] = merged

    return region_annual
#----------------------------------------------------------------------------
def add_scalar_bias_corrected_products_to_region_clim(
    region_monthly_clim,
    reference_col=r"$P_{\mathrm{MB}}$",
    target_products=None,
    suffix=" (corr.)",
    clip_factor=None,
    min_mean=1e-6,
):
    """
    Add scalar bias-corrected versions of selected products to each region monthly climatology.

    For each region and product:
        CF = mean(reference) / mean(product)
        corrected(month) = product(month) * CF

    Parameters
    ----------
    region_monthly_clim : dict
        Output from compute_weighted_region_monthly_climatologies()
    reference_col : str
        Reference column, usually PMB
    target_products : list or None
        Products to correct. If None, all except reference_col.
    suffix : str
        Suffix for corrected product names
    clip_factor : tuple or None
        Optional clipping for correction factor, e.g. (0.25, 10)
    min_mean : float
        Small floor to avoid divide-by-zero
    """

    out_dict = {}
    factor_dict = {}

    for region_name, df in region_monthly_clim.items():
        out = df.copy()
        factor_dict[region_name] = {}

        if reference_col not in out.columns:
            raise ValueError(f"{reference_col} not found in region '{region_name}'")

        ref_vals = out[reference_col].astype(float).to_numpy()
        ref_mean = np.nanmean(ref_vals)
        ref_mean = max(ref_mean, min_mean)

        if target_products is None:
            prods = [c for c in out.columns if c not in ["month", reference_col]]
        else:
            prods = [c for c in target_products if c in out.columns]

        for prod in prods:
            vals = out[prod].astype(float).to_numpy()
            prod_mean = np.nanmean(vals)

            if (not np.isfinite(prod_mean)) or (prod_mean <= 0):
                cf = np.nan
            else:
                cf = ref_mean / prod_mean

            if clip_factor is not None and np.isfinite(cf):
                cf = np.clip(cf, clip_factor[0], clip_factor[1])

            out[f"{prod}{suffix}"] = vals * cf if np.isfinite(cf) else np.nan
            factor_dict[region_name][prod] = cf

        out_dict[region_name] = out

    return out_dict, factor_dict
#----------------------------------------------------------------------------
def add_scalar_bias_corrected_products_to_region_annual(
    region_annual,
    reference_col=r"$P_{\mathrm{MB}}$",
    target_products=None,
    suffix=" (corr.)",
    clip_factor=None,
    min_mean=1e-6,
):
    """
    Add scalar bias-corrected versions of selected products to each region annual table.

    For each region and product:
        CF = mean(reference over years) / mean(product over years)
        corrected(year) = product(year) * CF

    Parameters
    ----------
    region_annual : dict
        Output from compute_weighted_region_annual_totals()
    reference_col : str
        Reference product, usually PMB
    target_products : list or None
        Products to correct
    suffix : str
        Suffix for corrected product names
    clip_factor : tuple or None
        Optional clipping for CF, e.g. (0.25, 10)
    min_mean : float
        Small floor to avoid divide-by-zero
    """

    out_dict = {}
    factor_dict = {}

    for region_name, df in region_annual.items():
        out = df.copy()
        factor_dict[region_name] = {}

        if reference_col not in out.columns:
            raise ValueError(f"{reference_col} not found in region '{region_name}'")

        ref_vals = out[reference_col].astype(float).to_numpy()
        ref_mean = np.nanmean(ref_vals)
        ref_mean = max(ref_mean, min_mean)

        if target_products is None:
            prods = [c for c in out.columns if c != "year" and c != reference_col]
        else:
            prods = [c for c in target_products if c in out.columns]

        for prod in prods:
            vals = out[prod].astype(float).to_numpy()
            prod_mean = np.nanmean(vals)

            if (not np.isfinite(prod_mean)) or (prod_mean <= 0):
                cf = np.nan
            else:
                cf = ref_mean / prod_mean

            if clip_factor is not None and np.isfinite(cf):
                cf = np.clip(cf, clip_factor[0], clip_factor[1])

            out[f"{prod}{suffix}"] = vals * cf if np.isfinite(cf) else np.nan
            factor_dict[region_name][prod] = cf

        out_dict[region_name] = out

    return out_dict, factor_dict
#---------------------------------------------------------------------------
def convert_precip_to_mm_per_month(df, unit, time_col="time", pr_col="precipitation"):
    """
    Convert precipitation column to mm/month.

    unit:
      - "mm/month"
      - "mm/day"
      - "mm/hr"
    """
    out = df.copy()
    t = pd.to_datetime(out[time_col])

    days = t.dt.days_in_month.astype(float)

    if unit == "mm/month":
        factor = 1.0
    elif unit == "mm/day":
        factor = days
    elif unit == "mm/hr":
        factor = 24.0 * days
    else:
        raise ValueError(f"Unsupported unit: {unit}")

    out[pr_col] = out[pr_col].astype(float) * factor
    return out
#----------------------------------------------------------------------------

def _plot_single_polar_precip_panel(
    ax,
    product_name,
    data,
    proj,
    cmap,
    norm,
    lat_rings=(-65, -70, -75, -80),
    lon_spokes=np.arange(-180, 180, 30),
    title_x=0.15,
    title_fs=16,
    title_weight="bold",
    title_prefix=None,
    panel_label=None,
):
    """
    Internal helper to keep styling consistent across all panels.
    Assumes `data` is already on SouthPolarStereo x/y coordinates.
    """

    ax.set_frame_on(False)
    ax.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())

    # Panel title
    if panel_label is not None:
        title_txt = f"({panel_label}) {product_name}"
    else:
        title_txt = product_name

    ax.text(
        title_x, 1.05,
        title_txt,
        transform=ax.transAxes,
        fontsize=title_fs,
        fontweight=title_weight,
        ha="center",
        va="bottom"
    )

    # Avoid zeros / negatives for PowerNorm emphasis
    # (keeps low-end visible but still works if there are zeros)
    data_plot = xr.where(data <= 0, 1e-6, data)

    data_plot.plot(
        ax=ax,
        transform=proj,
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
        add_labels=False,
    )

    # ---------- IMBIE basin boundaries ----------
    if "basin_id" in data.coords:
        basin_da = data["basin_id"]
    elif "basin" in data.coords:
        basin_da = data["basin"]
    else:
        basin_da = None

    if basin_da is not None:
        if "band" in basin_da.dims:
            basin_da = basin_da.isel(band=0)

        da_for_contour = xr.where(np.isnan(basin_da), 0, basin_da)

        # thin internal basin boundaries
        internal_levels = np.arange(1.5, 19.5, 1.0)
        ax.contour(
            da_for_contour["x"],
            da_for_contour["y"],
            da_for_contour.values,
            levels=internal_levels,
            colors="k",
            linewidths=0.6,
            transform=proj,
            zorder=5,
        )

        # bold coastline
        ax.contour(
            da_for_contour["x"],
            da_for_contour["y"],
            da_for_contour.values,
            levels=[0.5],
            colors="k",
            linewidths=1.2,
            transform=proj,
            zorder=6,
        )

    # Gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.55,
        color="gray",
        alpha=0.8,
        linestyle="--",
    )
    gl.ylocator = FixedLocator(lat_rings)
    gl.xlocator = FixedLocator(lon_spokes)

    # Your existing helper
    add_polar_latlon_labels(
        ax,
        lat_rings=lat_rings,
        lon_spokes=lon_spokes,
        label_size=11
    )

#----------------------------------------------------------------------------
def compare_mean_precip_grid_power(
    arr_lst_mean,
    ncols=4,
    figsize=None,
    cmap=None,
    gamma=0.6,
    vmin=0,
    vmax=400,
    cbar_tcks=None,
    cbar_label="Precipitation [mm/year]",
    panel_letters=False,
):
    """
    Generalized version of your PowerNorm ('log-like') comparison plot.

    Parameters
    ----------
    arr_lst_mean : list of tuples
        [(product_name, dataarray), ...]
    ncols : int
        Number of columns in subplot layout.
    figsize : tuple or None
        If None, chosen automatically.
    cmap : matplotlib colormap or None
        Default: plt.cm.jet
    gamma, vmin, vmax : float
        PowerNorm settings
    cbar_tcks : list or None
        Colorbar ticks
    panel_letters : bool
        If True, titles become (a), (b), ...
    """

    if len(arr_lst_mean) == 0:
        raise ValueError("arr_lst_mean is empty.")

    proj = ccrs.SouthPolarStereo()
    cmap = plt.cm.jet if cmap is None else cmap
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    n = len(arr_lst_mean)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    if figsize is None:
        figsize = (4.0 * ncols, 4.2 * nrows)

    fig, axes = plt.subplots(
        nrows, ncols,
        subplot_kw={"projection": proj},
        figsize=figsize
    )

    axes = np.atleast_1d(axes).ravel()

    # keep similar spacing feel to your original
    fig.subplots_adjust(
        left=0.04, right=0.90,
        top=0.96, bottom=0.10,
        wspace=0.05, hspace=0.20
    )

    letters = list("abcdefghijklmnopqrstuvwxyz")

    for i, (ax, (product_name, data)) in enumerate(zip(axes, arr_lst_mean)):
        panel_label = letters[i] if panel_letters and i < len(letters) else None

        _plot_single_polar_precip_panel(
            ax=ax,
            product_name=product_name,
            data=data,
            proj=proj,
            cmap=cmap,
            norm=norm,
            panel_label=panel_label,
        )

    # Hide unused axes
    for ax in axes[len(arr_lst_mean):]:
        ax.set_visible(False)

    # Shared colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cb = fig.colorbar(
        sm,
        ax=[ax for ax in axes if ax.get_visible()],
        orientation="horizontal",
        fraction=0.03,
        pad=0.06,
    )

    if cbar_tcks is not None:
        cb.set_ticks(cbar_tcks)

    cb.ax.tick_params(labelsize=12)
    cb.ax.minorticks_off()
    cb.set_label(cbar_label, fontsize=14)

    return fig, axes

#----------------------------------------------------------------------------
def compare_mean_precip_grid_power_dual_cbar(
    arr_lst_mean,
    group1_idx,
    group2_idx,
    ncols=4,
    figsize=None,
    cmap=None,
    # group 1 norm (higher-range products)
    gamma1=0.6,
    vmin1=0,
    vmax1=300,
    cbar_tcks1=None,
    cbar_label1="Precipitation [mm/year]",
    # group 2 norm (lower-range GPM-like products)
    gamma2=0.6,
    vmin2=0,
    vmax2=80,
    cbar_tcks2=None,
    cbar_label2="Precipitation [mm/year]",
    panel_letters=False,
):
    """
    Multi-panel polar plot with TWO PowerNorm colorbars.

    Parameters
    ----------
    arr_lst_mean : list of tuples
        [(product_name, dataarray), ...]
    group1_idx : list of int
        Indices in arr_lst_mean that use norm1 / colorbar1
    group2_idx : list of int
        Indices in arr_lst_mean that use norm2 / colorbar2
    """

    if len(arr_lst_mean) == 0:
        raise ValueError("arr_lst_mean is empty.")

    idx_all = set(range(len(arr_lst_mean)))
    if set(group1_idx).union(set(group2_idx)) != idx_all:
        raise ValueError("group1_idx and group2_idx must cover all panels exactly.")
    if set(group1_idx).intersection(set(group2_idx)):
        raise ValueError("group1_idx and group2_idx must not overlap.")

    proj = ccrs.SouthPolarStereo()
    cmap = plt.cm.jet if cmap is None else cmap

    norm1 = PowerNorm(gamma=gamma1, vmin=vmin1, vmax=vmax1)
    norm2 = PowerNorm(gamma=gamma2, vmin=vmin2, vmax=vmax2)

    n = len(arr_lst_mean)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    if figsize is None:
        figsize = (4.0 * ncols + 1.2, 4.2 * nrows)

    fig, axes = plt.subplots(
        nrows, ncols,
        subplot_kw={"projection": proj},
        figsize=figsize
    )
    axes = np.atleast_1d(axes).ravel()

    # leave room at right for two stacked vertical colorbars
    fig.subplots_adjust(
        left=0.04, right=0.87,
        top=0.96, bottom=0.06,
        wspace=0.05, hspace=0.20
    )

    letters = list("abcdefghijklmnopqrstuvwxyz")

    for i, (ax, (product_name, data)) in enumerate(zip(axes, arr_lst_mean)):
        panel_label = letters[i] if panel_letters and i < len(letters) else None
        norm = norm1 if i in group1_idx else norm2

        _plot_single_polar_precip_panel(
            ax=ax,
            product_name=product_name,
            data=data,
            proj=proj,
            cmap=cmap,
            norm=norm,
            panel_label=panel_label,
        )

    # Hide unused axes
    for ax in axes[len(arr_lst_mean):]:
        ax.set_visible(False)

    # ---- two vertical colorbars on the right ----
    # upper colorbar
    cax1 = fig.add_axes([0.89, 0.54, 0.016, 0.28])  # [left, bottom, width, height]
    sm1 = ScalarMappable(norm=norm1, cmap=cmap)
    sm1.set_array([])

    cb1 = fig.colorbar(sm1, cax=cax1, orientation="vertical")
    if cbar_tcks1 is not None:
        cb1.set_ticks(cbar_tcks1)
    cb1.ax.tick_params(labelsize=11)
    cb1.ax.minorticks_off()
    cb1.ax.set_title("mm/year", fontsize=12, pad=12)
    cb1.set_label(cbar_label1, fontsize=0)  # keep clean; title already says mm/year

    # lower colorbar
    cax2 = fig.add_axes([0.89, 0.12, 0.016, 0.28])
    sm2 = ScalarMappable(norm=norm2, cmap=cmap)
    sm2.set_array([])

    cb2 = fig.colorbar(sm2, cax=cax2, orientation="vertical")
    if cbar_tcks2 is not None:
        cb2.set_ticks(cbar_tcks2)
    cb2.ax.tick_params(labelsize=11)
    cb2.ax.minorticks_off()
    cb2.ax.set_title("mm/year", fontsize=12, pad=12)
    cb2.set_label(cbar_label2, fontsize=0)

    return fig, axes, cb1, cb2


#-----------------------------------------------------------------------------

def plot_weighted_region_monthly_climatology(
    region_monthly_clim,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=None,
    product_styles=None,
    ylabel="[mm/month]",
    figsize=(9, 10),
    sharex=True,
):
    """
    Plot stacked monthly climatologies for regions.

    Parameters
    ----------
    region_monthly_clim : dict
        Output of compute_weighted_region_monthly_climatologies()
    region_order : tuple/list
        Region subplot order
    product_order : list or None
        Order of products in legend / plotting
    product_styles : dict or None
        Example:
        {
            "ERA5": {"color": "tab:blue", "marker": "s", "lw": 1.8},
            ...
        }
    """

    nrows = len(region_order)
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=figsize,
        sharex=sharex
    )

    axes = np.atleast_1d(axes)

    if product_styles is None:
        product_styles = {}

    month_labels = [calendar.month_abbr[i] for i in range(1, 13)]

    legend_handles = []
    legend_labels = []

    for ax, region_name in zip(axes, region_order):
        df = region_monthly_clim[region_name].copy()

        if product_order is None:
            prod_cols = [c for c in df.columns if c != "month"]
        else:
            prod_cols = [c for c in product_order if c in df.columns]

        for prod in prod_cols:
            style = product_styles.get(prod, {})
            line, = ax.plot(
                df["month"],
                df[prod],
                label=prod,
                **style
            )

            if prod not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(prod)

        ax.set_title(region_name, fontsize=16, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 12)
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(month_labels, fontsize=14)

    fig.supylabel(ylabel, fontsize=16, fontweight="bold")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=min(len(legend_labels), 5),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.03)
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig, axes


#-----------------------------------------------------------------------------
def plot_region_time_series(region_data, product_cols):

    fig, axes = plt.subplots(
        nrows=len(region_data),
        ncols=1,
        figsize=(8, 10),
        sharex=True
    )

    if len(region_data) == 1:
        axes = [axes]

    prod_style = {
        pname: dict(
            color=color_cycle[i % len(color_cycle)],
            marker=marker_cycle[i % len(marker_cycle)],
            linewidth=2,
            markersize=5
        )
        for i, pname in enumerate(product_cols)
    }

    # Store legend handles
    handles = None

    for ax, (region_name, df_region) in zip(axes, region_data.items()):

        # ---- Plot lines ----
        for prod in product_cols:

            if prod == "GPCP":
                prod_lb = "GPCP v3.3"
            elif prod == "RACMO":
                prod_lb = "RACMO 2.4p1"
            else:
                prod_lb = prod

            line = ax.plot(
                df_region["year"],
                df_region[prod],
                label=prod_lb,
                **prod_style[prod]
            )

        # Save handles once
        if handles is None:
            handles = ax.get_legend_handles_labels()

        ax.set_title(region_name)
        ax.set_ylabel("Precipitation (mm/yr)")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(which="major", alpha=0.3)

        # ---- Metrics vs Pmb ----
        metric_text = ""
        for prod in product_cols:
            if prod == "Pmb":
                continue

            metrics = calculate_metrics(df_region, "Pmb", prod)

            metric_text += (
                f"{prod}: "
                f"CC={metrics['CC']}  "
                f"Bias={metrics['Bias']}%\n"
            )

        ax.text(
            0.02, 0.98,
            metric_text.strip(),
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )

    axes[-1].set_xlabel("Year")

    # ---- Legend outside bottom ----
    fig.legend(
        handles[0],
        handles[1],
        loc="lower center",
        ncol=len(product_cols),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()

#----------------------------------------------------------------------------

def compare_mean_precip_2x2_log(arr_lst_mean,
                                vmin=1,
                                vmax=400,
                                cbar_tcks=None):

    if len(arr_lst_mean) != 4:
        raise ValueError("compare_mean_precip_2x2 expects exactly 4 datasets.")

    proj = ccrs.SouthPolarStereo()
    cmap = plt.cm.jet

    # ---- Log normalization ----
    norm = PowerNorm(gamma=0.6, vmin=0, vmax=450)

    lat_rings = (-65, -70, -75, -80)
    lon_spokes = np.arange(-180, 180, 30)

    fig, axes = plt.subplots(
        2, 2,
        subplot_kw={"projection": proj},
        figsize=(12, 12)
    )
    axes = axes.ravel()

    fig.subplots_adjust(wspace=0.05, hspace=0.25, bottom=0.20)

    for ax, (product_name, data) in zip(axes, arr_lst_mean):

        ax.set_frame_on(False)
        ax.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())

        ax.text(
            0.15, 1.05,
            product_name,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="bottom"
        )

        # ---- Avoid zeros for log scale ----
        data_log = xr.where(data <= 0, 1e-3, data)

        data_log.plot(
            ax=ax,
            transform=ccrs.SouthPolarStereo(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
            add_labels=False,
        )

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=False,
            linewidth=0.55,
            color="gray",
            alpha=0.8,
            linestyle="--",
        )
        gl.ylocator = FixedLocator(lat_rings)
        gl.xlocator = FixedLocator(lon_spokes)

        add_polar_latlon_labels(
            ax,
            lat_rings=lat_rings,
            lon_spokes=lon_spokes,
            label_size=11
        )

    # ---- Colorbar ----
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cb = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.03,
        pad=0.1
    )

    if cbar_tcks is not None:
        cb.set_ticks(cbar_tcks)

    cb.ax.tick_params(labelsize=12)
    cb.set_label("Precipitation [mm/year] (log scale)", fontsize=14)

    return fig, axes

#----------------------------------------------------------------------------

def plot_weighted_region_seasonal_climatology(
    region_seasonal_clim,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=None,
    product_styles=None,
    ylabel="Precipitation [mm/season]",
    figsize=(8, 9),
    sharex=True,
):
    """
    Plot seasonal climatology by region.

    Parameters
    ----------
    region_seasonal_clim : dict
        Output from compute_weighted_region_seasonal_climatologies()
    product_order : list or None
        Plotting/legend order
    product_styles : dict or None
        Style dict, same concept as monthly plotting
    """

    nrows = len(region_order)
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=figsize,
        sharex=sharex
    )

    axes = np.atleast_1d(axes)

    if product_styles is None:
        product_styles = {}

    season_labels = ["DJF", "MAM", "JJA", "SON"]
    x = np.arange(len(season_labels))

    legend_handles = []
    legend_labels = []

    for ax, region_name in zip(axes, region_order):
        df = region_seasonal_clim[region_name].copy()

        # Ensure season order
        df["season_order"] = df["season"].map(season_order_key)
        df = df.sort_values("season_order").drop(columns="season_order")

        if product_order is None:
            prod_cols = [c for c in df.columns if c != "season"]
        else:
            prod_cols = [c for c in product_order if c in df.columns]

        for prod in prod_cols:
            style = product_styles.get(prod, {})
            line, = ax.plot(
                x,
                df[prod].to_numpy(dtype=float),
                label=prod,
                **style
            )

            if prod not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(prod)

        ax.set_title(region_name, fontsize=18, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.15, len(season_labels) - 0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(season_labels, fontsize=15, fontweight="bold")

    fig.supylabel(ylabel, fontsize=18, fontweight="bold")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=min(len(legend_labels), 4),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.05)
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    return fig, axes

#----------------------------------------------------------------------------

def plot_weighted_region_interannual(
    region_annual,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=None,
    product_styles=None,
    ylabel="Precipitation [mm/year]",
    figsize=(10, 10),
    sharex=True,
):
    """
    Plot interannual variability (annual totals) for each region.
    region_annual: dict of dataframes from compute_weighted_region_annual_totals()
    """

    nrows = len(region_order)
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=sharex)
    axes = np.atleast_1d(axes)

    if product_styles is None:
        product_styles = {}

    legend_handles = []
    legend_labels = []

    for ax, region_name in zip(axes, region_order):
        df = region_annual[region_name].copy()

        if product_order is None:
            prod_cols = [c for c in df.columns if c != "year"]
        else:
            prod_cols = [c for c in product_order if c in df.columns]

        for prod in prod_cols:
            style = product_styles.get(prod, {})
            line, = ax.plot(df["year"], df[prod], label=prod, **style)

            if prod not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(prod)

        ax.set_title(region_name, fontsize=18, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xticks(sorted(region_annual[region_order[0]]["year"].unique()))
    axes[-1].tick_params(axis="x", labelsize=12)

    fig.supylabel(ylabel, fontsize=18, fontweight="bold")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=min(len(legend_labels), 4),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    return fig, axes

#----------------------------------------------------------------------------
def plot_region_trend_panels(
    monthly_df_data_mmmonth,
    region_defs,
    basin_weights,
    product_order,
    product_styles,
    regions=("Antarctica", "West Antarctica", "East Antarctica"),
    use_running_mean=True,
    show_pmb_trend_only=True,
    pmb_name=r"$P_{\mathrm{MB}}$",
):
    fig, axes = plt.subplots(len(regions), 1, figsize=(12, 10), sharex=True)

    for ax, region in zip(axes, regions):
        ts = region_monthly_series_from_dict(
            monthly_df_data_mmmonth, region_defs, basin_weights, region
        )

        # deseasonalize then optionally smooth
        ts_anom = deseasonalize_monthly(ts)
        ts_plot = centered_13mo_rm(ts_anom) if use_running_mean else ts_anom

        # plot lines
        for prod in product_order:
            if prod not in ts_plot.columns:
                continue
            st = product_styles.get(prod, {})
            ax.plot(ts_plot.index, ts_plot[prod], label=prod, **{k:v for k,v in st.items() if k in ["color","lw","ls"]})

        # dashed trend line for PMB only (on anomalies, NOT smoothed, safer)
        if show_pmb_trend_only and pmb_name in ts_anom.columns:
            slope, p = trend_with_ar1_correction(ts_anom[pmb_name].values, ts_anom.index)
            # trend line in anomaly space
            t_years = (ts_anom.index - ts_anom.index[0]).days / 365.25
            intercept = np.nanmean(ts_anom[pmb_name].values)  # just center-ish
            trend_line = intercept + slope * t_years

            ax.plot(ts_anom.index, trend_line, "k--", lw=2.0, label=f"{pmb_name} trend")

        ax.set_title(region, fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Deseasonalized P \nanomalies (mm/month)")

    axes[-1].set_xlabel("Year")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig, axes
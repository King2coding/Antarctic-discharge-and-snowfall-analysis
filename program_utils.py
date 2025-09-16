#%%
# packages
import warnings
warnings.filterwarnings("ignore")

import gc
import os

import pandas as pd
import numpy as np
import datetime

from datetime import datetime, timedelta
from scipy.stats import linregress
import math
from collections import Counter

import matplotlib.pyplot as plt

import subprocess
from scipy.interpolate import griddata
import xarray as xr
# import xesmf as xe
from rasterio.warp import Resampling
from pyproj import CRS, Transformer

from osgeo import gdal, osr

import h5py

import cartopy.crs as ccrs
import cartopy.crs as ccrs

from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.cm import ScalarMappable

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import cm
import matplotlib as mpl


#%%
# floating variables
misc_out = r'/ra1/pubdat/AVHRR_CloudSat_proj/miscelaneous_outs'

crs = "+proj=longlat +datum=WGS84 +no_defs"  
crs_format = 'proj4'

crs_stereo = "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84"

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#%%
# fucntions
# function to read and process ERA5 precip file
def process_era5_file_to_basin(era5_xr_data, er_tme, basin, fle_svnme):

        er_time = pd.to_datetime(er_tme).strftime('%Y%m%d')

        er_precip_dat = era5_xr_data.sel(time=er_tme).sel(lat=slice(-55, -90))

        yshp, xshp = er_precip_dat.shape

        minx = er_precip_dat['lon'].min().item()
        maxy = er_precip_dat['lat'].max().item()
        px_sz = round(er_precip_dat['lon'].diff('lon').mean().item(), 2)

        dest_flnme = os.path.join(misc_out, os.path.basename(fle_svnme).replace('.nc', '_sh.nc'))

        gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, er_precip_dat.data)

        output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('_sh.nc', '_sh_stere.nc'))

        gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'

        subprocess.run(gdalwarp_command, shell=True)

        # Read the stereographic projection file
        er_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

        os.remove(dest_flnme)
        os.remove(output_file_stereo)

        # Clip the data to the bounds of the basin dataset
        er_xrr_clip = er_xrr_sh_stereo.sel(
            x=slice(-3333250, 3333250),
            y=slice(-3333250, 3333250)
        ).squeeze()

        ccrs = CRS.from_proj4('+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84')

        # Explicitly set the CRS before reprojecting
        er_xrr_clip.rio.write_crs(ccrs.to_string(), inplace=True)

        er_xrr_clip_res = er_xrr_clip.rio.reproject(
            er_xrr_clip.rio.crs,
            shape=basin.shape,  # set the shape as the basin data shape
            resampling=Resampling.nearest,
            transform=basin.rio.transform()
        )

        # Add the time coordinate to the reprojected DataArray
        er_xrr_clip_res_arr = er_xrr_clip_res['Band1'].values
        er_xrr_clip_res_arr = np.where(basin.values > 0, er_xrr_clip_res_arr, np.nan)
        er_xrr_clip_res = xr.DataArray(
            np.expand_dims(er_xrr_clip_res_arr, axis=0),  # Expand dimensions to match ['time', 'y', 'x']
            dims=['time', 'y', 'x'],
            coords={'time': [np.datetime64(pd.to_datetime(er_time, format='%Y%m%d'), 'D')], 
                    'y': er_xrr_clip_res.coords['y'], 
                    'x': er_xrr_clip_res.coords['x']},
            name='precipitation'  # Change the variable name to 'precipitation'
        )

        return er_xrr_clip_res
#----------------------------------------------------------------------------
# function to read and process AIRS precip file
def process_airs_file_to_basin(airs_xrr_data, ai_tme, basin, fle_svnme):
        
        airs_time = pd.to_datetime(ai_tme).strftime('%Y%m%d')

        airs_precip_dat = airs_xrr_data.sel(time=ai_tme).sel(lat=slice(-55, -90))


        yshp, xshp = airs_precip_dat.shape

        minx = airs_precip_dat['lon'].min().item()
        maxy = airs_precip_dat['lat'].max().item()
        px_sz = round(airs_precip_dat['lon'].diff('lon').mean().item(), 2)

        dest_flnme = os.path.join(misc_out, os.path.basename(fle_svnme).replace('.nc', '_sh.nc'))

        gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, airs_precip_dat.data)

        output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('_sh.nc', '_sh_stere.nc'))

        gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'

        subprocess.run(gdalwarp_command, shell=True)

        # Read the stereographic projection file
        ai_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

        os.remove(dest_flnme)
        os.remove(output_file_stereo)

        # Clip the data to the bounds of the basin dataset
        ai_xrr_clip = ai_xrr_sh_stereo.sel(
            x=slice(-3333250, 3333250),
            y=slice(-3333250, 3333250)
        ).squeeze()

        ccrs = CRS.from_proj4('+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84')

        # Explicitly set the CRS before reprojecting
        ai_xrr_clip.rio.write_crs(ccrs.to_string(), inplace=True)
        ai_xrr_clip_res = ai_xrr_clip.rio.reproject(
            ai_xrr_clip.rio.crs,
            shape=basin.shape,  # set the shape as the basin data shape
            resampling=Resampling.nearest,
            transform=basin.rio.transform()
        )

        # Add the time coordinate to the reprojected DataArray
        ai_xrr_clip_res_arr = ai_xrr_clip_res['Band1'].values
        ai_xrr_clip_res_arr = np.where(basin.values > 0, ai_xrr_clip_res_arr, np.nan)
        ai_xrr_clip_res = xr.DataArray(
            np.expand_dims(ai_xrr_clip_res_arr, axis=0),  # Expand dimensions to match ['time', 'y', 'x']
            dims=['time', 'y', 'x'],
            coords={'time': [np.datetime64(pd.to_datetime(airs_time, format='%Y%m%d'), 'D')], 'y': ai_xrr_clip_res.coords['y'], 'x': ai_xrr_clip_res.coords['x']},
            name='precipitation'  # Change the variable name to 'precipitation'
        )

        return ai_xrr_clip_res
#----------------------------------------------------------------------------
def process_ssmis_file_to_basin(ss, basin):

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

        gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'

        subprocess.run(gdalwarp_command, shell=True)

        # Read the stereographic projection file
        ss_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

        os.remove(dest_flnme)
        os.remove(output_file_stereo)

        # Clip the data to the bounds of the basin dataset
        ss_xrr_clip = ss_xrr_sh_stereo.sel(
            x=slice(-3333250, 3333250),
            y=slice(-3333250, 3333250)
        ).squeeze()

        ccrs = CRS.from_proj4('+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84')

        # Explicitly set the CRS before reprojecting
        ss_xrr_clip.rio.write_crs(ccrs.to_string(), inplace=True)
        ss_xrr_clip_res = ss_xrr_clip.rio.reproject(
            ss_xrr_clip.rio.crs,
            shape=basin.shape,  # set the shape as the basin data shape
            resampling=Resampling.nearest,
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
def process_avhrr_file_to_basin(file,yr,basin):
    av_x, av_y = return_sim_cords(file)
    av_arr, av_time = read_tif_sim_file(file,yr)

    av_arr = np.expand_dims(av_arr, axis=0) if av_arr.ndim == 2 else av_arr  # Ensure img_arr has 3 dimensions
    av_xrr = xr.DataArray(av_arr, dims=['time', 'lat', 'lon'], coords={'time': [av_time], 'lat': av_y, 'lon': av_x})
    av_xrr_sh = av_xrr.sel(lat=slice(-55, -90)).squeeze()
    yshp, xshp = av_xrr_sh.shape

    minx = av_xrr_sh['lon'].min().item()
    maxy = av_xrr_sh['lat'].max().item()
    px_sz = round(av_xrr_sh['lon'].diff('lon').mean().item(), 2)

    dest_flnme = os.path.join(misc_out, os.path.basename(file).replace('.tif', '_sh.tif'))

    gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, crs, crs_format, av_xrr_sh.data)

    output_file_stereo = os.path.join(misc_out, os.path.basename(dest_flnme).replace('_sh.tif', '_sh_stere.tif'))

    gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'

    subprocess.run(gdalwarp_command, shell=True)

    # Read the stereographic projection file
    av_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

    os.remove(dest_flnme)
    os.remove(output_file_stereo)

    # Clip the data to the bounds of the basin dataset
    av_xrr_clip = av_xrr_sh_stereo.sel(
        x=slice(-3333250, 3333250),
        y=slice(3333250, -3333250)
    ).squeeze()

    av_xrr_clip_res = av_xrr_clip.rio.reproject(
        av_xrr_clip.rio.crs,
        shape=basin.shape,  # set the shape as the basin data shape
        resampling=Resampling.nearest,
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
    # av_xrr_clip_res = av_xrr_clip_res.rename({'band_data': 'precipitation'})

    return av_xrr_clip_res
#----------------------------------------------------------------------------
# function to process IMERG file to basin
def process_imerg_file_to_basin(im, misc_out, basin):
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
    gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'
    subprocess.run(gdalwarp_command, shell=True)

    # Read the stereographic projection file
    img_xrr_sh_stereo = xr.open_dataset(output_file_stereo)

    os.remove(dest_flnme)
    os.remove(output_file_stereo)

    # Clip the data to the bounds of the second dataset
    img_xrr_clip = img_xrr_sh_stereo.sel(
        x=slice(-3333250, 3333250),
        y=slice(3333250, -3333250)
    ).squeeze()

    img_xrr_clip_res = img_xrr_clip.rio.reproject(
        img_xrr_clip.rio.crs,
        shape=basin.shape,  # set the shape as the autosnow data shape (1800, 3600)
        resampling=Resampling.nearest,
        transform=basin.rio.transform()
    )
    # Add the time coordinate to the reprojected DataArray
    img_xrr_clip_res_arr = img_xrr_clip_res['band_data'].values
    img_xrr_clip_res_arr = np.where(basin.values > 0, img_xrr_clip_res_arr, np.nan)
    img_xrr_clip_res = xr.DataArray(
        np.expand_dims(img_xrr_clip_res_arr, axis=0),  # Expand dimensions to match ['time', 'y', 'x']
        dims=['time', 'y', 'x'],
        coords={'time': [img_time], 'y': img_xrr_clip_res.coords['y'], 'x': img_xrr_clip_res.coords['x']},
        name='precipitation'  # Change the variable name to 'precipitation'
    )       

    return img_xrr_clip_res

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
        for basin_id in range(1, 20):
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

#%%
# plot

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
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.025, hspace=0.15)

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
        fraction=0.04,  # Fraction of the original axes height
        pad=0.15,  # Distance from the bottom of the subplots
        extend="max"
    )
    cb.set_ticks(cbar_tcks)  # Set integer ticks
    cb.ax.tick_params(labelsize=20)
    cb.set_label("Precipitation [mm/year]", fontsize=20)

    # Show the plot
    plt.tight_layout()

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

def _stack_space(da):
    if "stacked_y_x" in da.dims:
        return da.rename({"stacked_y_x": "space"})
    return da.stack(space=("y", "x"))

def _basin_labels_from(da):
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
    data   = _stack_space(da)
    labels = _basin_labels_from(da)
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
    data   = _stack_space(da)
    labels = _basin_labels_from(da)
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

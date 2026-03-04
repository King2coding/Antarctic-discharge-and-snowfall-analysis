#%%
'''
This code creates GPM products of precipitation over Antarctic basins
GPM constellation satelite categories: DMSP SSMIS, ATMS, GCOM-W1 AMSR2, MHS
DMSP: F16, 17, 18, 19
ATMS: SNPP, NOAA 20
MHS: NOAA (18,19), METOP (A, B, C)
'''
#%%
# import packages

from program_utils import *

#%%
# define paths
basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'

gpm_satellites_path = r'/ra1/pubdat/GPM-Constellation-Satellites_MI_and_Sounders'

# paths to put satellite precip over basins data
path_to_put_precip_over_basins = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/gpm_constellation_satellites'
#%%
# floating variables
SATELLITE_CATEGORIES = {
    "ATMS": ["SNPP", "NOAA-20"],
    "DMSP-SSMIS": ["F16", "F17", "F18", "F19"],
    "MHS": [("NOAA",("NOAA-18", "NOAA-19")), ("METOP", ("METOP-A", "METOP-B", "METOP-C"))],
    "GCOM-W1_AMSR2": ["AMSR2"],
}

# %%
SATELLITE_CATEGORY_FILES = {}

for satellite in os.listdir(gpm_satellites_path):

    monthly_path = os.path.join(gpm_satellites_path, satellite, "Monthly")

    if not os.path.isdir(monthly_path):
        continue

    files = glob.glob(
        os.path.join(monthly_path, "**", "*.nc4"),
        recursive=True
    )

    # rename AMSR2 key for convenience
    key = "AMSR2" if satellite == "GCOM-W1_AMSR2" else satellite

    SATELLITE_CATEGORY_FILES[key] = sorted(files)
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
SATELLITE_CATEGORY_DATA = []

for sat, files in SATELLITE_CATEGORY_FILES.items():

    print(f"{sat} has {len(files)} files")

    fname = os.path.basename(files[0]).split('_',1)[1]
    key = '_'.join(fname.split('.')[:4])
    sv_path = os.path.join(path_to_put_precip_over_basins, sat)

    ds = xr.open_mfdataset(
        sorted(files),
        preprocess=preprocess,
        combine="nested",
        concat_dim="time",
        parallel=True
    )

    da = ds["surfacePrecipitation"].transpose("time", "lat", "lon")

    # average duplicate timestamps
    if not sat == 'AMSR2':
        da = da.groupby("time").mean(skipna=True)
    # da = da.groupby("time").mean(skipna=True)

    for idx, fle_tme in enumerate(da.time.values):
        
        fle_tme = pd.to_datetime(fle_tme).strftime('%Y%m%d')
        fle_svnme = os.path.join(sv_path, f'{key}_{fle_tme}.nc')

        basin_precip = process_gpm_precip_file(da.sel(time=fle_tme), fle_tme, basins, fle_svnme)

        encoding = {basin_precip.name: {"zlib": True, "complevel": 9}}
        basin_precip.to_netcdf(os.path.join(sv_path, fle_svnme), 
                            mode='w', 
                            format='NETCDF4', 
                            encoding=encoding)
        SATELLITE_CATEGORY_DATA.append(basin_precip)

        if idx % 25 == 0:
            print(f"Processed {idx} files: {os.path.basename(fle_svnme)}")
    ds.close()

all_satellite_precip = xr.concat(SATELLITE_CATEGORY_DATA, dim='time')
all_satellite_precip = all_satellite_precip.groupby("time").mean(skipna=True)
encoding = {all_satellite_precip.name: {"zlib": True, "complevel": 9}}
all_satellite_precip.to_netcdf(os.path.join(path_to_put_precip_over_basins, 
                                            f'all_gpm_satellite_precip_mean_{cde_run_dte}.nc'), 
                            mode='w', 
                            format='NETCDF4', 
                            encoding=encoding)


#%% Sanity checks
# Open the dataset and select the land-sea mask data
land_sea_mask_path = '/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/ancillary_imerg_data/GPM_IMERG_LandSeaMask.2.nc4'
lsm_ds = xr.open_dataset(land_sea_mask_path)
lsm_arr = lsm_ds['landseamask']

# Transpose the data to get longitude on the x-axis
lsm_transposed = lsm_arr.transpose('lat', 'lon')

# Flip the latitude axis so that latitude is displayed south to north
lsm_flipped = lsm_transposed.isel(lat=slice(None, None, -1))

# Apply the land-sea mask condition
lsm = xr.where(lsm_flipped < 25, 1, 0)

cc = CRS.from_authority(code=4326,auth_name='EPSG')

lsm.rio.write_crs(cc.to_string(), inplace=True)

lsm = lsm.rio.reproject(lsm.rio.crs, 
                        shape=da.shape[1:],
                        resampling=Resampling.mode,)

lsm = lsm.rename({'x': 'lon', 'y': 'lat'})

ant_lsm = lsm.sel(lat=slice(-55, -90))

ant_precip = da[0].sel(lat=slice(-90, -55)).where(ant_lsm == 1)


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.path as mpath

def plot_antarctica_polar(da2d, title=None, vmin=None, vmax=None):
    """
    da2d: xarray DataArray with dims (lat, lon) or (y, x) but coords lat/lon in degrees.
    """
    # Ensure (lat, lon) for plotting
    if "lat" in da2d.dims and "lon" in da2d.dims:
        plot_da = da2d
    elif "y" in da2d.dims and "x" in da2d.dims:
        plot_da = da2d.rename({"y": "lat", "x": "lon"})
    else:
        plot_da = da2d.transpose("lat", "lon")

    proj = ccrs.SouthPolarStereo()
    pc = ccrs.PlateCarree()

    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection=proj)

    # Antarctica extent (lon W/E, lat S/N) in PlateCarree coords
    ax.set_extent([-180, 180, -90, -55], crs=pc)

    # Add coastlines for context
    ax.coastlines(linewidth=0.8)

    # Plot (pcolormesh is usually safest)
    im = ax.pcolormesh(
        plot_da["lon"].values,
        plot_da["lat"].values,
        plot_da.values,
        transform=pc,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    # Circular boundary (gives the “round” polar plot look)
    theta = np.linspace(0, 2*np.pi, 200)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Gridlines (optional)
    gl = ax.gridlines(crs=pc, draw_labels=False, linewidth=0.5, linestyle="--", alpha=0.5)

    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.05)
    cb.set_label(f"{da2d.name or 'precip'} ({da2d.attrs.get('units','')})")

    if title:
        ax.set_title(title)

    plt.show()

# Example use (pick one time if da has time dim)
plot_antarctica_polar(
    (ant_precip*24*30),
    title="Antarctica precipitation (polar stereographic)",vmax=80
)


plot_antarctica_polar(
    ((basin_precip.isel(time=0)*24)*30) if "time" in basin_precip.dims else basin_precip,
    title="Antarctica precipitation (polar stereographic)",vmax=80
)
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

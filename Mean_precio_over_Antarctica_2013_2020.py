#%%
from program_utils import *
from Extra_util_functions import *

#%%
# path_precip_files
path_to_gpcp_v3pt3 = r'/ra1/pubdat/Satellite_eval_over_Oceans/data/GPCP/GPCP_v3_pnt_3_1998_2024'
path_to_era5_tp = r'/ra1/pubdat/ECMWF/ERA5/daily'

all_gpcp_v3pt3_files = sorted([os.path.join(path_to_gpcp_v3pt3, f) for f in os.listdir(path_to_gpcp_v3pt3) if f.endswith('.nc4')])
all_gpcp_v3pt3_files_2013_2020 = [f for f in all_gpcp_v3pt3_files if \
                                  2013 <= int(os.path.basename(f).split('_')[2][:4]) <= 2020]

all_era5_tp_files = sorted([os.path.join(path_to_era5_tp, f) for f in os.listdir(path_to_era5_tp) if f'era5_tp_' in f and f.endswith('.nc')])
all_era5_tp_files_2013_2020 = [f for f in all_era5_tp_files if \
                              2013 <= int(os.path.basename(f).split('_')[2]) <= 2020]


#%% Flaoting variables

mask_path = "/ra1/pubdat/mask_land_ocean/mask50km.mat"
mask = loadmat(mask_path)["mask50"][-60:, :].swapaxes(0, 1)
mask = np.flip(mask, axis=1)
mask_binary = mask < 75
new_mask = mask_binary.copy()
new_mask[:360, :] = mask_binary[360:, :].copy()
new_mask[360:, :] = mask_binary[:360, :].copy()
new_mask = np.flip(new_mask.T, axis=0)
new_mask_ = new_mask.astype(int)
#------------------------------------------------------------------
basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'

basins  = xr.open_dataarray(os.path.join(basins_path,'bedmap3_basins_0.1deg.tif'))
# Mask out invalid values (0 or NaN)
# zwally_data = basins_zwally.where((basins_zwally > 0) & (basins_zwally.notnull()))
basins = basins.where((basins > 1) & (basins.notnull()))
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
#%% load data
gpcp_ds_v3pt2_xr = xr.open_mfdataset(all_gpcp_v3pt3_files_2013_2020,
                                    combine="nested",              # files are time-sequenced
                                    concat_dim="time",             # concatenate along time                                               
                                    coords="minimal",
                                    compat="override",
                                    parallel=True,
                                    engine="netcdf4",
                                    chunks={"time": 120, "lat": 180, "lon": 360},  # <<< important
                                    cache=False
                                    )

gpcp_ds_v3pt2_xr = ds_swaplon(gpcp_ds_v3pt2_xr)
gpcp_ds_v3pt2_xr_ant = gpcp_ds_v3pt2_xr['precip'].mean(dim='time').compute()  # Convert from mm/day to mm/month (approximate)
gpcp_ds_v3pt2_xr_ant = gpcp_ds_v3pt2_xr_ant.isel(lat=slice(-60, None)).where(new_mask_ == 1)

#--- ERA5 loading with multiprocessing ---

era5_ds_xr_list = []
for er5 in all_era5_tp_files_2013_2020:
    if not os.path.exists(er5):
        raise FileNotFoundError(f"ERA5 file not found: {er5}")
    era5_data = process_era5_file(er5)#xr.open_dataarray(er5)
    era5_ds_xr_list.append(era5_data)
# Combine all processed batches into a single xarray dataset - simple version
if era5_ds_xr_list:
    era5_ds_xr = xr.concat(era5_ds_xr_list, dim="valid_time")
    print("✅ ERA5 loading complete")
print("-" * 30 + "\n")
del(era5_ds_xr_list)
gc.collect() 

era5_clean = era5_ds_xr.mean(dim="valid_time", skipna=True)
print(era5_clean.dims)
print(era5_clean.shape)

era5_ant = era5_clean.sel(latitude=slice(-60, -90)).mean(dim='time')

target_lons = np.arange(-180.0, 180.0, 0.5)
target_lats = np.arange(-60.0, -90.0, -0.5)

era5_ant_mean_05 = era5_ant.sel(
    latitude=xr.DataArray(target_lats, dims="latitude"),
    longitude=xr.DataArray(target_lons, dims="longitude"),
    method="nearest"
)

era5_ant_masked = era5_ant_mean_05.where(new_mask_ == 1)

small_id_offsets = {
    11: (-4.0e5, -1.0e5),
    13: (-4.8e5,  1.3e5),
    14: (-8.4e5,  1.9e5),
    15: (-4.8e5,  2.5e5),
    16: (-0.8e5,  3.7e5),
    17: (-0.5e5,  4.2e5),
}

# levels = [0.5, 5, 10, 20, 40, 80, 120, 180, 260, 400]
levels = [0.1, 2.5, 5,10, 50, 100, 150, 250, 500, 750, 1000, 1500, 2000]

fig, axes, cb = plot_antarctic_precip_dual_discrete(
    arr_lst=[
        ("ERA5 (2013–2020 mean)", era5_ant_masked*365),
        ("GPCP v3.3 (2013–2020 mean)", gpcp_ds_v3pt2_xr_ant*365),
    ],
    basins_da=basins,
    basin_label_ids=True,
    basin_id_offsets=small_id_offsets,
    figsize=(12, 6),
    dpi=300,
    cmap="turbo",
    levels=levels,
    norm_type="log",      # <- key for peninsula compression
    vmin=0.1,
    vmax=2000,
    cbar_ticks=[0.1, 2.5,5, 10, 50, 100, 150, 250, 500, 1000, 1500, 2000],
    cbar_label="mm/year",
    panel_letters=True,
    show_panel_mean=True,
    mean_fmt="{:.0f}",
    mean_xy=(0.08, 0.90),
    mean_fontsize=16,
    basin_linecolor="gray",
    basin_linewidth=0.7,
)

plt.show()
gc.collect()
#--------------------------------------------------------------------------
mean_annual_plot_arrs = [                    
                    (f'ERA5', era5_ant_masked*365), 
                    (f'GPCP v3.3', gpcp_ds_v3pt2_xr_ant*365),
                    
                   ]


fig, axes = compare_mean_precip_grid_power(
    mean_annual_plot_arrs,
    ncols=4,
    gamma=0.6,
    vmin=5,
    vmax=400,
    cbar_tcks=[5, 25, 50, 100,  200, 300, 400],
    cbar_label="Precipitation [mm/year]",
    panel_letters=True,
)
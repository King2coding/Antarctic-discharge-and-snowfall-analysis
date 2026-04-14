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
gpcp_ds_v3pt2_xr_ant_yrly = gpcp_ds_v3pt2_xr_ant * 365
cos_weighted_mean_gpcp3pt3 = get_zonal(gpcp_ds_v3pt2_xr_ant_yrly, new_mask_, gpcp_ds_v3pt2_xr_ant.lat.values)

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

era5_ant_mask_yrly = era5_ant_masked*365
cosine_weighted_mean_era5 = get_zonal(era5_ant_mask_yrly, new_mask_, era5_ant_masked.latitude.values)



#%%
small_id_offsets = {
    11: (-4.0e5, -1.0e5),
    13: (-4.8e5,  1.3e5),
    14: (-8.4e5,  1.9e5),
    15: (-4.8e5,  2.5e5),
    16: (-0.8e5,  3.7e5),
    17: (-0.5e5,  4.2e5),
}

# levels = [0.5, 5, 10, 20, 40, 80, 120, 180, 260, 400]
levels = [0.1, 10, 50, 100, 150, 250, 500, 750, 1000, 2000]

fig, axes, cb = plot_antarctic_precip_dual_discrete(
    arr_lst=[
        ("ERA5 (2013–2020 mean)", era5_ant_mask_yrly, cosine_weighted_mean_era5),
        ("GPCP v3.3 (2013–2020 mean)", gpcp_ds_v3pt2_xr_ant_yrly, cos_weighted_mean_gpcp3pt3),
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
    cbar_ticks=[0.1, 10, 50, 100, 150, 250, 500, 1000, 2000],
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
    vmin=0,
    vmax=400,
    cbar_tcks=[5, 25, 50, 100,  200, 300, 400],
    cbar_label="Precipitation [mm/year]",
    panel_letters=True,
)



#%% The cosine weighted routine test

gpcp_template = prepare_latlon_template(
    gpcp_ds_v3pt2_xr_ant,
    lat_name="lat",
    lon_name="lon",
    crs="EPSG:4326"
)

basins_gpcp = reproject_basin_ids_to_match(basins, gpcp_template)
#------------------------------------------------------------------

era5_template = prepare_latlon_template(
    era5_ant_mean_05,
    lat_name="latitude",
    lon_name="longitude",
    crs="EPSG:4326"
)

basins_era5 = reproject_basin_ids_to_match(basins, era5_template)

WAIS_BASINS = [10, 11, 12, 13, 14, 15, 16, 17]
EAIS_BASINS = [2, 3, 4, 5, 6, 7, 8, 9, 18, 19]
AIS_BASINS  = WAIS_BASINS + EAIS_BASINS
#------------------------------------------------------------------

mask_gpcp_wais = np.isin(basins_gpcp.values, WAIS_BASINS)
mask_gpcp_eais = np.isin(basins_gpcp.values, EAIS_BASINS)
mask_gpcp_ais  = np.isin(basins_gpcp.values, AIS_BASINS)
#------------------------------------------------------------------

mask_era5_wais = np.isin(basins_era5.values, WAIS_BASINS)
mask_era5_eais = np.isin(basins_era5.values, EAIS_BASINS)
mask_era5_ais  = np.isin(basins_era5.values, AIS_BASINS)

# -----------------------------
# GPCP DAILY FIELD
# -----------------------------
gpcp_daily = ds_swaplon(gpcp_ds_v3pt2_xr)["precip"]

# subset Antarctica and apply Antarctica land mask
gpcp_daily_ant = gpcp_daily.sel(lat=slice(-60, -90))
gpcp_daily_ant = gpcp_daily_ant.where(new_mask_ == 1)

print("GPCP daily Antarctica shape:", gpcp_daily_ant.shape)


# -----------------------------
# ERA5 DAILY FIELD
# -----------------------------
# era5_clean already reconstructed from valid_time
# expected dims: (time, latitude, longitude)

# subset Antarctica
era5_daily_ant = era5_clean.sel(latitude=slice(-60, -90))

# regrid from 0.25° -> 0.5° using nearest neighbor to match mask/product grid
target_lons = np.arange(-180.0, 180.0, 0.5)
target_lats = np.arange(-60.0, -90.0, -0.5)

era5_daily_ant_05 = era5_daily_ant.sel(
    latitude=xr.DataArray(target_lats, dims="latitude"),
    longitude=xr.DataArray(target_lons, dims="longitude"),
    method="nearest"
)

# apply Antarctica mask
era5_daily_ant_05 = era5_daily_ant_05.where(new_mask_ == 1)

print("ERA5 daily Antarctica shape:", era5_daily_ant_05.shape)

#-----------------------------
region_masks_gpcp = {
    "Antarctica": mask_gpcp_ais,
    "West Antarctica": mask_gpcp_wais,
    "East Antarctica": mask_gpcp_eais,
}

region_masks_era5 = {
    "Antarctica": mask_era5_ais,
    "West Antarctica": mask_era5_wais,
    "East Antarctica": mask_era5_eais,
}

#-----------------------------
# GPCP annual regional means [mm/year]
gpcp_regional_annual = annual_regional_means_from_daily_xr(
    da_daily=gpcp_daily_ant,
    region_masks=region_masks_gpcp,
    lat_name="lat",
    lon_name="lon",
    annual_mode="mean"
)
gpcp_regional_annual["product"] = "GPCP v3.3"

# gp = (gpcp_daily_ant.mean(dim="time", skipna=True).compute()) * 365
#------------------------------------------------------------

# ERA5 annual regional means [mm/year]
era5_regional_annual = annual_regional_means_from_daily_xr(
    da_daily=era5_daily_ant_05,
    region_masks=region_masks_era5,
    lat_name="latitude",
    lon_name="longitude",
    annual_mode="mean"
)
era5_regional_annual["product"] = "ERA5"


regional_annual_df = pd.concat(
    [gpcp_regional_annual, era5_regional_annual],
    ignore_index=True
)

regional_annual_df.head()


#------------------------------
fig, axes = plot_regional_annual_timeseries(regional_annual_df)
plt.show()

#-----------------------------
multi_year_mean_df = (
    regional_annual_df
    .groupby(["region", "product"], as_index=False)["precipitation"]
    .mean()
)

multi_year_mean_df
#----------------------------
fig, ax = plot_multiyear_mean_bar_by_region(multi_year_mean_df)
plt.show()

new_diff_df = compute_relative_differences(
    regional_annual_df,
    ref_product="GPCP v3.3",
    target_products=["ERA5"]
)

new_diff_df.head()

diff_product_styles_old = {
    "ERA5 - $P_{\\mathrm{MB}}$": dict(color="blue", marker="s", lw=2.3),
    "GPCP v3.3 - $P_{\\mathrm{MB}}$": dict(color="orange", marker="D", lw=2.3),
}

fig, axes = plot_regional_annual_timeseries(
    new_diff_df,
    product_order=tuple(new_diff_df["product"].unique()),
    product_styles={
        new_diff_df["product"].unique()[0]: dict(color="blue", marker="s", lw=2.3),
    },
    ylabel="[mm/year]"
)
plt.show()


new_diff_mean_df = (
    new_diff_df
    .groupby(["region", "product"], as_index=False)["precipitation"]
    .mean()
)

fig, ax = plot_multiyear_mean_bar_by_region(
    new_diff_mean_df,
    product_order=tuple(new_diff_mean_df["product"].unique()),
    colors={
        new_diff_mean_df["product"].unique()[0]: "blue",
    },
    title="2013–2020 mean bias relative to GPCP v3.3",
    ylabel="[mm/year]"
)
plt.show()


#%%% Cosine-weighted mean test over Antarctica spatial distribution
era5_ds_xr_yrly = era5_clean.sel(latitude=slice(-60, -90)).groupby('time.year').mean(skipna=True)
era5_ds_xr_yrly_05 = era5_ds_xr_yrly.sel(
    latitude=xr.DataArray(target_lats, dims="latitude"),
    longitude=xr.DataArray(target_lons, dims="longitude"),
    method="nearest"
)
era5_ds_xr_yrly_05 = era5_ds_xr_yrly_05 * 365
era5_ds_xr_yrly_avg = era5_ds_xr_yrly_05.mean(dim='year', skipna=True)
era5_ds_xr_yrly_avg_bsn = era5_ds_xr_yrly_avg.where(mask_era5_ais)
cosine_weighted_mean_era5_yrly_avg_bsn = get_zonal(era5_ds_xr_yrly_avg_bsn,
                                                mask_era5_ais, 
                                                era5_ds_xr_yrly_avg_bsn.latitude.values) 


# the gpcp version
gpcp_ds_v3pt2_xr_ant = gpcp_ds_v3pt2_xr['precip'].copy()
gpcp_ds_v3pt2_xr_ant = gpcp_ds_v3pt2_xr_ant.sel(lat=slice(-60, -90))
gpcp_ds_v3pt2_xr_ant = gpcp_ds_v3pt2_xr_ant.compute()
gpcp_ds_v3pt2_xr_ant_yrly = gpcp_ds_v3pt2_xr_ant.groupby('time.year').mean(skipna=True)
gpcp_ds_v3pt2_xr_ant_yrly_avg = gpcp_ds_v3pt2_xr_ant_yrly * 365
gpcp_ds_v3pt2_xr_ant_yrly_avg = gpcp_ds_v3pt2_xr_ant_yrly_avg.mean(dim='year', skipna=True)
gpcp_ds_v3pt2_xr_ant_bsn = gpcp_ds_v3pt2_xr_ant_yrly_avg.where(mask_gpcp_ais)
cos_weighted_mean_gpcp3pt3_yrly_avg_bns = get_zonal(gpcp_ds_v3pt2_xr_ant_bsn, 
                                       mask_gpcp_ais, 
                                       gpcp_ds_v3pt2_xr_ant_bsn.lat.values)

levels = [0.1, 10, 50, 100, 150, 250, 500, 750, 1000, 2000]

fig, axes, cb = plot_antarctic_precip_dual_discrete(
    arr_lst=[
        ("ERA5 (2013–2020 mean)", 
         era5_ds_xr_yrly_avg_bsn, 
         cosine_weighted_mean_era5_yrly_avg_bsn),
        ("GPCP v3.3 (2013–2020 mean)", 
         gpcp_ds_v3pt2_xr_ant_bsn, 
         cos_weighted_mean_gpcp3pt3_yrly_avg_bns),
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
    cbar_ticks=[0.1, 10, 50, 100, 150, 250, 500, 1000, 2000],
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


#%% PMb cosine weighted version

Pmb_mm_fle = os.path.join(basins_path, 'Monthly_mass_budget_precip_RignotBasin_in_mm_20260226.nc')
# 'Monthly_mass_budget_precip_RignotBasin_in_mm.nc'

P_mm_mnth = xr.open_dataarray(Pmb_mm_fle)

# Example target lat-lon DataArray
# Use the exact grid you used for the cosine-weighted ERA5/GPCP plots
target_template = prepare_latlon_template(
    era5_template,   # or any lat-lon DataArray on the desired target grid
    lat_name="y",
    lon_name="x",
    crs="EPSG:4326"
)

# Prepare P_MB
# Rename date -> time first
pmb_native = P_mm_mnth.rename({"date": "time"})

# Drop 2D / auxiliary coords that break rioxarray reprojection
drop_coords = [c for c in ["basin_id", "mapping", "band"] if c in pmb_native.coords]
pmb_native = pmb_native.drop_vars(drop_coords, errors="ignore")


# Set spatial dims + CRS
pmb_native = pmb_native.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
pmb_native = pmb_native.rio.write_crs(P_mm_mnth.mapping.crs_wkt, inplace=False)
# Reproject to lat-lon grid
pmb_latlon = pmb_native.rio.reproject_match(
    target_template,
    resampling=Resampling.average
)

# rename back if needed
pmb_latlon = pmb_latlon.rename({"y": "lat", "x": "lon"})

basins_native = basins.squeeze(drop=True)
basins_native = basins_native.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
basins_native = basins_native.rio.write_crs(basins_native.mapping.crs_wkt, inplace=False)

basin_mask_latlon = basins_native.rio.reproject_match(
    pmb_latlon.isel(time=0),
    resampling=Resampling.nearest
).rename({"y": "lat", "x": "lon"})



pmb_cos_annual_df = compute_pmb_cosine_weighted_annual(
    pmb_latlon,
    basin_mask_latlon,
    time_name="time",
    lat_name="lat",
    lon_name="lon",
    annual_mode="sum",
)

pmb_cos_annual_df = pmb_cos_annual_df[pmb_cos_annual_df['year'].isin(YEARS)]

print(pmb_cos_annual_df.head())

pmb_cos_annual_df_plot = (
    pmb_cos_annual_df
    .rename(columns={"P_MB": "precipitation"})
    .assign(product=r"$P_{\mathrm{MB}}$")
)


regional_annual_df = pd.concat(
    [pmb_cos_annual_df_plot, gpcp_regional_annual, era5_regional_annual, ],
    ignore_index=True
)

regional_annual_df.head()


#------------------------------
fig, axes = plot_regional_annual_timeseries(regional_annual_df)
plt.show()
gc.collect() 


#-----------------------------
multi_year_mean_df = (
    regional_annual_df
    .groupby(["region", "product"], as_index=False)["precipitation"]
    .mean()
)

multi_year_mean_df.head(2)
#----------------------------
fig, ax = plot_multiyear_mean_bar_by_region(multi_year_mean_df)
plt.show()


new_diff_df = compute_relative_differences(
    regional_annual_df,
    ref_product=r"$P_{\mathrm{MB}}$",
    target_products=["ERA5", "GPCP v3.3"]
)

new_diff_df.head()

diff_product_styles_old = {
    "ERA5 - $P_{\\mathrm{MB}}$": dict(color="blue", marker="s", lw=2.3),
    "GPCP v3.3 - $P_{\\mathrm{MB}}$": dict(color="orange", marker="D", lw=2.3),
}

fig, axes = plot_regional_annual_timeseries(
    new_diff_df,
    product_order=tuple(new_diff_df["product"].unique()),
    product_styles={
        new_diff_df["product"].unique()[0]: dict(color="blue", marker="s", lw=2.3),
        new_diff_df["product"].unique()[1]: dict(color="orange", marker="D", lw=2.3)
    },
    ylabel="[mm/year]"
)
plt.show()



new_diff_mean_df = (
    new_diff_df
    .groupby(["region", "product"], as_index=False)["precipitation"]
    .mean()
)

fig, ax = plot_multiyear_mean_bar_by_region(
    new_diff_mean_df,
    product_order=tuple(new_diff_mean_df["product"].unique()),
    colors={
        new_diff_mean_df["product"].unique()[0]: "blue",
        new_diff_mean_df["product"].unique()[1]: "orange",
    },
    title="2013–2020 mean bias relative to GPCP v3.3",
    ylabel="[mm/year]"
)
plt.show()
#%% old data plots
old_region_dict = {
    'Antarctica': pd.DataFrame({
        'year': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        r'$P_{\mathrm{MB}}$': [131.044600, 136.756067, 139.833796, 164.295035, 139.039149, 121.798221, 133.630473, 173.243846],
        'ERA5': [185.611645, 173.331736, 191.736134, 204.218646, 185.429898, 184.731090, 184.694323, 205.732051],
        'GPCP v3.3': [151.802211, 158.822192, 158.586049, 157.290001, 159.954636, 162.770769, 159.475971, 156.580907],
    }),
    'West Antarctica': pd.DataFrame({
        'year': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        r'$P_{\mathrm{MB}}$': [286.914225, 371.049067, 366.777750, 430.768415, 384.076351, 293.945817, 362.722854, 494.729250],
        'ERA5': [362.599814, 420.463579, 410.895804, 484.389699, 403.309828, 390.491277, 436.621470, 464.999217],
        'GPCP v3.3': [321.158921, 412.060840, 370.385468, 378.848135, 386.465079, 353.113183, 420.125661, 375.717831],
    }),
    'East Antarctica': pd.DataFrame({
        'year': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        r'$P_{\mathrm{MB}}$': [93.575348, 80.434744, 85.279098, 100.237922, 80.135045, 80.415934, 78.559318, 95.962480],
        'ERA5': [143.065740, 113.924104, 139.052689, 136.868771, 133.054087, 135.268724, 124.133956, 143.407228],
        'GPCP v3.3': [111.090820, 97.946555, 107.671922, 104.029993, 105.504150, 117.014658, 96.818806, 103.902930],
    }),
}



old_regional_annual_df = regional_dict_to_tidy_df(old_region_dict)
old_regional_annual_df.head()

product_styles_old = {
    r'$P_{\mathrm{MB}}$': dict(color="black", marker="o", lw=2.3),
    "ERA5": dict(color="blue", marker="s", lw=2.3),
    "GPCP v3.3": dict(color="orange", marker="D", lw=2.3),
}

fig, axes = plot_regional_annual_timeseries(
    old_regional_annual_df,
    product_order=(r'$P_{\mathrm{MB}}$', "ERA5", "GPCP v3.3"),
    product_styles=product_styles_old,
)
plt.show()

old_multi_year_mean_df = (
    old_regional_annual_df
    .groupby(["region", "product"], as_index=False)["precipitation"]
    .mean()
)

fig, ax = plot_multiyear_mean_bar_by_region(
    old_multi_year_mean_df,
    product_order=(r'$P_{\mathrm{MB}}$', "ERA5", "GPCP v3.3"),
    colors={
        r'$P_{\mathrm{MB}}$': "black",
        "ERA5": "blue",
        "GPCP v3.3": "orange",
    }
)
plt.show()


old_diff_df = compute_relative_differences(
    old_regional_annual_df,
    ref_product=r'$P_{\mathrm{MB}}$',
    target_products=["ERA5", "GPCP v3.3"]
)

old_diff_df.head()

fig, axes = plot_regional_annual_timeseries(
    old_diff_df,
    product_order=tuple(old_diff_df["product"].unique()),
    product_styles={
        old_diff_df["product"].unique()[0]: dict(color="blue", marker="s", lw=2.3),
        old_diff_df["product"].unique()[1]: dict(color="orange", marker="D", lw=2.3),
    },
    ylabel="[mm/year]"
)
plt.show()

old_diff_mean_df = (
    old_diff_df
    .groupby(["region", "product"], as_index=False)["precipitation"]
    .mean()
)


fig, ax = plot_multiyear_mean_bar_by_region(
    old_diff_mean_df,
    product_order=tuple(old_diff_mean_df["product"].unique()),
    colors={
        old_diff_mean_df["product"].unique()[0]: "blue",
        old_diff_mean_df["product"].unique()[1]: "orange",
    },
    title="2013–2020 mean bias relative to $P_{\\mathrm{MB}}$",
    ylabel="[mm/year]"
)
plt.show()
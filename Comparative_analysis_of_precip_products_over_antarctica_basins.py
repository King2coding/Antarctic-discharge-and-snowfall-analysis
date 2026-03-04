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
basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'

# paths to put satellite precip over basins data
imerg_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/imerg_precip'

era5_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/era5_precip'
gpcpv3pt3_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/gpcpv3pt3'
racmo_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/RACMO2pt4p1'

atms_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/gpm_constellation_satellites/ATMS'
dmsp_ssmis_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/gpm_constellation_satellites/DMSP-SSMIS'
amsr2_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/gpm_constellation_satellites/AMSR2'
mhs_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/gpm_constellation_satellites/MHS'
all_gpm_basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/gpm_constellation_satellites'
annual_precip_in_basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/precip_in_basins/annual'
seasonal_precip_in_basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/precip_in_basins/seasonal'

# path to put outs e.g. plots, dfs
path_to_plots = r'/home/kkumah/Projects/Antarctic_discharge_work/plots'
path_to_dfs = r'/home/kkumah/Projects/Antarctic_discharge_work/dfs'
#%%
# floating variables

crs = "+proj=longlat +datum=WGS84 +no_defs"  
crs_format = 'proj4' 

batch_size = 10

cde_run_dte = str(date.today().strftime('%Y%m%d'))

#----------------------------------------------------------------------------------

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

# basins_zwally = basins['zwally']

# basins_imbie = basins['imbie']

# %%
colors = plt.cm.gist_ncar(np.linspace(0, 1, 19))

# Give Basin 19 a unique neutral color not used elsewhere
colors[-1] = np.array([0.60, 0.60, 0.60, 1.0])   # medium gray

cmap = mcolors.ListedColormap(colors)
cmap.set_bad(color='white')

vmin, vmax = 1, 19
levels = np.linspace(vmin, vmax, vmax - vmin + 2)
norm = mcolors.BoundaryNorm(levels, cmap.N)

# work with a 2D slice (drop the band dimension)
da = basins.isel(band=0)

proj = ccrs.SouthPolarStereo()
fig, ax = plt.subplots(figsize=(12, 8), dpi=300,
                       subplot_kw={'projection': proj})
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

p = da.plot(
    ax=ax,
    transform=proj,
    cmap=cmap,
    norm=norm,
    add_colorbar=False,
    add_labels=False,
)

ax.set_facecolor('white')

# --- boundaries: use a copy with ocean=0 instead of NaN ---
da_for_contour = da.fillna(0)          # 0 outside basins, 1..19 inside

# coastline (0–1) + internal boundaries (1–19)
boundary_levels = np.arange(0.5, 19.5, 1.0)

ax.contour(
    da["x"].values,
    da["y"].values,
    da_for_contour.values,   # 2D (y, x) with 0/1..19
    levels=boundary_levels,
    colors="k",
    linewidths=0.8,
    transform=proj,
    zorder=5,
)

# --- label offsets for small WAIS basins (same as your name-plot) ---
small_id_offsets = {
    11: (-4.0e5, -1.0e5),  # F-G
    13: (-4.8e5,  1.3e5),  # H-Hp
    14: (-8.4e5,  1.9e5),  # Hp-I
    15: (-4.8e5,  2.5e5),  # I-Ipp
    16: (-0.8e5,  3.7e5),  # Ipp-J
    17: (-0.5e5,  4.2e5),  # J-Jpp
    # add 12 or others if you want them outside too
}

# Annotate each basin
for basin_id in range(1, 20):
    mask = (da == basin_id)
    y, x = np.where(mask.values)
    if len(x) == 0:
        continue

    cx = da["x"].values[x].mean()
    cy = da["y"].values[y].mean()
    label = str(basin_id)

    if basin_id in small_id_offsets:
        # place label outside with a leader line
        dx, dy = small_id_offsets[basin_id]
        lx, ly = cx + dx, cy + dy

        ax.annotate(
            label,
            xy=(cx, cy),      # centroid (tail of line)
            xytext=(lx, ly),  # label position
            textcoords='data',
            xycoords='data',
            ha='center',
            va='center',
            fontsize=15,
            transform=proj,
            arrowprops=dict(
                arrowstyle="-",   # simple line
                lw=0.8,
                color="k"
            ),
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.7
            ),
        )
    else:
        # “normal” in-basin label
        ax.text(
            cx, cy, label,
            color="black",
            fontsize=15,
            ha="center",
            va="center",
            transform=proj,
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.6
            ),
        )

ax.axis("off")
plt.tight_layout()
# plt.show()
# Save the imbie basin plot
output_path = os.path.join(path_to_plots, 'imbie_basins_with_ids.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
gc.collect()
#%%
Pmb_mm_fle = os.path.join(basins_path, 'Monthly_mass_budget_precip_RignotBasin_in_mm_20260226.nc')
# 'Monthly_mass_budget_precip_RignotBasin_in_mm.nc'

P_mm_mnth = xr.open_dataarray(Pmb_mm_fle)
p_mm_df = P_mm_mnth.to_dataframe().reset_index().dropna(axis=0)
p_mm_df = p_mm_df.dropna(axis=0, subset=["precip_mm_per_month"])
p_mm_df = p_mm_df[['date','basin_id','precip_mm_per_month']].copy()
p_mm_df['year'] = p_mm_df['date'].dt.year
p_mm_df['month'] = p_mm_df['date'].dt.month 
p_mm_mean_df = p_mm_df.groupby(['year','month','basin_id'])['precip_mm_per_month'].mean().reset_index()
p_mm_mean_df['time'] = pd.to_datetime(dict(year=p_mm_mean_df['year'], month=p_mm_mean_df['month'], day=1))
p_mm_mean_df['basin_id'] = p_mm_mean_df['basin_id'].astype(int)

Pmb_annual = xr.open_dataarray(os.path.join(basins_path, "Pmb_annual_2013_2022_mm.nc"))
Pmb_seasonal = xr.open_dataarray(os.path.join(basins_path, "Pmb_seasonal_mm_2013_2022.nc"))
img_fle_lst = sorted([os.path.join(imerg_basin_path, x) for x in os.listdir(imerg_basin_path) if 'imbie_basin' in x])
era5_fle_lst = sorted([os.path.join(era5_basin_path, x) for x in os.listdir(era5_basin_path) if 'imbie_basin' in x])
gpcpv3pt3_fle_lst = sorted([os.path.join(gpcpv3pt3_basin_path, x) for x in os.listdir(gpcpv3pt3_basin_path) if 'imbie_basin' in x])
racmo_pr = xr.open_dataarray(os.path.join(racmo_path,'pr_monthlyS_ANT11_RACMO2.4p1_ERA5_2013_2022.nc'))

atms_fle_lst = sorted([os.path.join(atms_basin_path, x) for x in os.listdir(atms_basin_path) if x.endswith('.nc')])
dmsp_ssmis_fle_lst = sorted([os.path.join(dmsp_ssmis_basin_path, x) for x in os.listdir(dmsp_ssmis_basin_path) if x.endswith('.nc')])
amsr2_fle_lst = sorted([os.path.join(amsr2_basin_path, x) for x in os.listdir(amsr2_basin_path) if x.endswith('.nc')])
mhs_fle_lst = sorted([os.path.join(mhs_basin_path, x) for x in os.listdir(mhs_basin_path) if x.endswith('.nc')])
all_gpm_fle = os.path.join(all_gpm_basin_path, "all_gpm_satellite_precip_mean_20260303.nc")

# 1) drop the dummy band dimension
racmo_pr = racmo_pr.squeeze("band", drop=True)          # now (time, y, x)

# 2) remove non-essential coordinates that are hitchhiking on the array
#    (keep x,y,time; drop the CRS helpers and the basin mask)
racmo_pr = racmo_pr.reset_coords(names=["mapping", "basin_id"], drop=True)

# 3) tidy attributes / name
racmo_pr = racmo_pr.assign_attrs({
    "standard_name": "precipitation_flux",
    "long_name": "Precipitation",
    "units": "kg m-2"     # 1 kg m-2 == 1 mm w.e.
}).rename("precipitation")

gc.collect()
#%%
# read and process satellite precipitation data
# IMERG
print('Processing IMERG data')

# imerg_annual_mean, imerg_seasonal_mean = process_precipitation_data(img_fle_lst, basins)
# imerg_annual_mean = imerg_annual_mean * 365
gc.collect()

#----------------------------------------------------------------------------------

# read and process era5 data
print('Processing ERA5 data')

era5_annual_mean, era5_seasonal_mean,era5_b_mean = process_precipitation_data(era5_fle_lst, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)
era5_annual_mean = era5_annual_mean * 365

era5_basin_mean = era5_b_mean.to_dataframe().reset_index()
era5_basin_mean['year'] = era5_basin_mean['time'].dt.year
era5_basin_mean['month'] = era5_basin_mean['time'].dt.month

era5_basin_mnth_mean = era5_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
era5_basin_mnth_mean['time'] = pd.to_datetime(dict(year=era5_basin_mnth_mean['year'], month=era5_basin_mnth_mean['month'], day=1))


gc.collect()


#----------------------------------------------------------------------------------

print('Processing GPCP v3.3 data')

gpcpv3pt3_annual_mean, gpcpv3pt3_seasonal_mean, gpcpv3pt3_b_mean = process_precipitation_data(gpcpv3pt3_fle_lst, 
                                                                            basins,
                                                                            'precipitation',
                                                                            False,)
gpcpv3pt3_annual_mean = gpcpv3pt3_annual_mean * 365

gpcpv3pt3_basin_mean = gpcpv3pt3_b_mean.to_dataframe().reset_index()
gpcpv3pt3_basin_mean['year'] = gpcpv3pt3_basin_mean['time'].dt.year
gpcpv3pt3_basin_mean['month'] = gpcpv3pt3_basin_mean['time'].dt.month

gpcpv3pt3_basin_mnth_mean = gpcpv3pt3_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
gpcpv3pt3_basin_mnth_mean['time'] = pd.to_datetime(dict(year=gpcpv3pt3_basin_mnth_mean['year'], month=gpcpv3pt3_basin_mnth_mean['month'], day=1))


gc.collect()

#----------------------------------------------------------------------------------
print('processing RACMO pr data')
racmo_pr_annual_mean, racmo_pr_seasonal_mean, racmo_pr_b_mean = process_precipitation_data(racmo_pr, 
                                                                          basins, 
                                                                          'pr',
                                                                          True,)
racmo_basin_mnth_mean = racmo_pr_b_mean.to_dataframe().reset_index()
racmo_basin_mnth_mean['year'] = racmo_basin_mnth_mean['time'].dt.year
racmo_basin_mnth_mean['month'] = racmo_basin_mnth_mean['time'].dt.month
# racmo_pr_annual_mean = racmo_pr_annual_mean * 365
gc.collect()

#----------------------------------------------------------------------------------

# read and process era5 data
print('Processing GPM Constellation data')
print('processing ATMS data')
atms_annual_mean, atms_seasonal_mean,atms_b_mean = process_precipitation_data(atms_fle_lst, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)
atms_annual_mean = (atms_annual_mean * 24) * 365

atms_basin_mean = atms_b_mean.to_dataframe().reset_index()
atms_basin_mean['year'] = atms_basin_mean['time'].dt.year
atms_basin_mean['month'] = atms_basin_mean['time'].dt.month

atms_basin_mnth_mean = atms_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
atms_basin_mnth_mean['time'] = pd.to_datetime(dict(year=atms_basin_mnth_mean['year'], month=atms_basin_mnth_mean['month'], day=1))

print('Processing MSH data')
msh_annual_mean, msh_seasonal_mean,msh_b_mean = process_precipitation_data(msh_fle_lst, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)
msh_annual_mean = (msh_annual_mean * 24) * 365
gc.collect()

#----------------------------------------------------------------------------------
Pmb_annual_mean = Pmb_annual.mean(dim='year')
era5_annual_mean_mean = era5_annual_mean.mean(dim='year')
gpcpv3pt3_annual_mean_mean = gpcpv3pt3_annual_mean.mean(dim='year')
racmo_pr_annual_mean_mean = racmo_pr_annual_mean.mean(dim='year')

#%% make some plots ('IMERG', imerg_annual_mean[0])
# annual plots
# calculate and plot the mean across years
mean_annual_plot_arrs = [
                    (r"P$_{MB}$", Pmb_annual_mean),
                    (f'ERA5', era5_annual_mean_mean), 
                    (f'GPCP v3.3', gpcpv3pt3_annual_mean_mean),
                    (f'RACMO 2.4p1', racmo_pr_annual_mean_mean),                    
                   ]

compare_mean_precip_2x2(mean_annual_plot_arrs, 
                 vmin=0, vmax=400,
                 cbar_tcks=[0, 50, 100, 150, 200, 250, 300, 350, 400])

# save plot to disk
svnme = os.path.join(path_to_plots, f'annual_snowfall_accumulation_over_imbie_basins_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()


compare_mean_precip_2x2_log(
    mean_annual_plot_arrs,
    vmin=0,        # or 0.5 depending on your minimum values
    vmax=450,
    cbar_tcks=[0, 10, 25, 50, 100, 200, 300, 400,]
)
svnme = os.path.join(path_to_plots, f'annual_snowfall_accumulation_over_imbie_basins_log_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()
#----------------------------------------------------------------------------------
# make scatter plots comparison

# --- Convert each product to tidy DF ---
df_pmb   = to_df(Pmb_annual)
df_pmb.rename(columns={'basin_id': 'basin',
                       'precip_mm_per_month': 'Pmb'}, inplace=True)
df_era5  = to_df(era5_annual_mean)
df_era5.rename(columns={'precipitation_annual': 'ERA5'}, inplace=True)
df_gpcp  = to_df(gpcpv3pt3_annual_mean)
df_gpcp.rename(columns={'precipitation_annual': 'GPCP'}, inplace=True)
df_racmo = to_df(racmo_pr_annual_mean)
df_racmo.rename(columns={'pr_annual': 'RACMO'}, inplace=True)

# --- Merge all together on (year, basin) ---
df = df_pmb.merge(df_era5, on=["year","basin"])
df = df.merge(df_gpcp, on=["year","basin"])
df = df.merge(df_racmo, on=["year","basin"])

# add a "year-basin" key if you like
df["year_basin"] = df["year"].astype(str) + "-" + df["basin"].astype(str)

df_mean_yr_acc = df.groupby("basin")[["Pmb", "ERA5", "GPCP", "RACMO"]].mean().reset_index()

f, ax = plt.subplots(figsize=(8, 6))
width = 0.2  # Width of each bar
basins = df_mean_yr_acc['basin']
x = np.arange(len(basins))  # X positions for the bars

# Plot each product as a separate bar group
for i, col in enumerate(["Pmb", "ERA5", "GPCP", "RACMO"]):
    ax.bar(x + i * width, df_mean_yr_acc[col], width=width, label=col)

ax.set_xticks(x + width * 1.5)  # Center the ticks
ax.set_xticklabels(basins)
ax.set_xlabel('Basin')
ax.set_ylabel('Precipitation (mm/yr)')
ax.legend()
plt.tight_layout()
plt.show()

# Save the DataFrame to a CSV file
df_mean_yr_acc.round(2).to_csv(os.path.join(path_to_dfs, f'df_mean_yr_acc_{cde_run_dte}.csv'), index=False)


#%%
# plot monthly cycles

Pmb_mnth_cycle = p_mm_mean_df.groupby(['month','basin_id'])['precip_mm_per_month'].mean().reset_index()
era5_mnth_cycle = era5_basin_mnth_mean.groupby(['month','basin'])['precipitation'].mean().reset_index()
gpcp_v3pt3_mnth_cycle = gpcpv3pt3_basin_mnth_mean.groupby(['month','basin'])['precipitation'].mean().reset_index()
racmo_mnth_cycle = racmo_basin_mnth_mean.groupby(['month','basin'])['precipitation'].mean().reset_index()

plot_dfs = {
    r"$P_{\mathrm{MB}}$": Pmb_mnth_cycle,
    "ERA5": era5_mnth_cycle,
    "GPCP v3.3": gpcp_v3pt3_mnth_cycle,
    "RACMO 2.4p1": racmo_mnth_cycle,
}


#----------------------------------------------------------------------------------
# id2name must already exist, e.g.
# id2name = {1: "F-G", 2: "A-Ap", 3: "Ap-B", 4: "B-C", ...}

name2id = {v: k for k, v in id2name.items()}

# --- Region definitions in terms of basin NAMES ---

# West Antarctica proper (RACMO R24)
wais_core_names = ["H-Hp", "J-Jpp", "G-H", "F-G", "Ep-F"]

# Antarctic Peninsula (RACMO R24) – here we MERGE into WAIS for the 1×4 plot
ap_names = ["I-Ipp", "Ipp-J", "Hp-I"]

wais_all_names = wais_core_names + ap_names

# East Antarctica = all remaining IMBIE basins (EAIS)
eais_names = [nm for nm in id2name.values() if nm not in wais_all_names]

# Inland: basins dominated by the interior plateau (approximation)
inland_names = ["Jpp-K", "K-A", "A-Ap", "Ap-B"]

# Safety: keep inland subset of EAIS
inland_names = [nm for nm in inland_names if nm in eais_names]

# --- Convert names → IDs ---

WAIS_IDS   = [name2id[nm] for nm in wais_all_names]
EAIS_IDS   = [name2id[nm] for nm in eais_names]
INLAND_IDS = [name2id[nm] for nm in inland_names]

print("WAIS basin IDs:", WAIS_IDS)
print("EAIS basin IDs:", EAIS_IDS)
print("Inland basin IDs:", INLAND_IDS)

# ---- Region definitions (from our agreed mapping) ----
REGION_DEFS = [
    ("Antarctica",      [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
    ("West Antarctica", [13, 17, 12, 11, 10, 15, 16, 14]),
    ("East Antarctica", [2, 3, 4, 5, 6, 7, 8, 9, 18, 19]),
]

fig, axes = plot_monthly_cycles_regions_3x1(plot_dfs, REGION_DEFS)
# plt.show()
# Save the plot to disk
svnme = os.path.join(path_to_plots, f'monthly_cycles_precip_over_imbie_basins_regions_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()
# For the continent we’ll just use “all basins we see in the dataframe”
#%%

fig, ax = plot_basin_ranked_bar_overlay(
    df_mean_yr_acc,
    basin_col="basin",
    ref_col="Pmb",
    prod_cols=["ERA5", "GPCP", "RACMO"],
    prod_labels=["ERA5", "GPCP v3.3", "RACMO 2.4p1"],
    figsize=(12, 5),
)
plt.show()

#----------------------------------------------------------------------------------

fig, ax, spread, spread_pct = plot_basin_spread_points(
    df_mean_yr_acc,
    basin_col="basin",
    ref_col="Pmb",
    prod_cols=["ERA5", "GPCP", "RACMO"],
    prod_labels=["ERA5", "GPCP v3.3", "RACMO 2.4p1"],
)
svnme = os.path.join(path_to_plots, f'basin_spread_points_precip_over_imbie_basins_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()
#%%

Pmb_mean_seas_df = da_season_to_basin_df(Pmb_seasonal, basin_name="basin_id")

era5_mean_df = da_season_to_basin_df(era5_seasonal_mean*30, basin_name="basin")

gpcp_mean_df = da_season_to_basin_df(gpcpv3pt3_seasonal_mean*30, basin_name="basin")
racmo_mean_df = da_season_to_basin_df(racmo_pr_seasonal_mean, basin_name="basin")

#----------------------------------------------------------------------------------
plot_dfs = {
    r"$P_{\mathrm{MB}}$": Pmb_mean_seas_df,
    "ERA5": era5_mean_df,
    "GPCP v3.3": gpcp_mean_df,
    "RACMO 2.4p1": racmo_mean_df,
}

#----------------------------------------------------------------------------------
fig, axes = plot_seasonal_cycles_regions_3x1(
            plot_dfs, 
            REGION_DEFS)
svnme = os.path.join(path_to_plots, f'seasonal_cycles_precip_over_imbie_basins_regions_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()

#----------------------------------------------------------------------------------
fig, axes = plot_seasonal_cycles_regions_1x3(
            plot_dfs, 
            REGION_DEFS)
# plt.show()
gc.collect()
#----------------------------------------------------------------------------------
plot_seasonal_heatmaps_by_basin_v2(
    plot_dfs,
    vmin=0,
    vmax=100  # or whatever range you want
)

gc.collect()

#----------------------------------------------------------------------------------
plot_seasonal_by_season_product(
    plot_dfs,
    basin_order=list(range(2, 20)),  # or None to infer
    vmin=0,
    vmax=100,
    cmap="jet",
)

gc.collect()

#----------------------------------------------------------------------------------
# Yearly cycle
product_cols = ["Pmb", "ERA5", "GPCP", "RACMO"]

prdt_df = [("Pmb",df_pmb, Pmb_annual), ("ERA5",df_era5, era5_annual_mean), 
           ("GPCP",df_gpcp, gpcpv3pt3_annual_mean), ("RACMO",df_racmo, racmo_pr_annual_mean)]

# region_data = compute_region_means(df, REGION_DEFS, product_cols)
# region_data = compute_region_means_from_products(prdt_df, REGION_DEFS)
region_data = compute_region_means_from_maps(
    prdt_df,
    REGION_DEFS
)

plot_region_time_series(region_data, product_cols)
svnme = os.path.join(path_to_plots, f'annual_mean_time_series_precip_over_imbie_basins_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()
# df_mean_yr_acc.plot(kind='box')
#%%

# ----------- Scatter plot -------------

# Example usage:
products = ["ERA5", "GPCP", "RACMO"]
plot_pmb_scatter(df_mean_yr_acc, "Pmb", products, high_thresh=500.0, scale="linear")

plot_pmb_scatter(df_mean_yr_acc, "Pmb", products, high_thresh=500.0, scale="log")
svnme = os.path.join(path_to_plots, f'log_scatterplot_precip_over_imbie_basins_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()

products = ["GPCP", "RACMO"]
plot_pmb_scatter(df_mean_yr_acc, "ERA5", products, high_thresh=500.0, scale="linear")
plot_pmb_scatter(df_mean_yr_acc, "ERA5", products, high_thresh=500.0, scale="log")

products = ["GPCP", "ERA5"]
plot_pmb_scatter(df_mean_yr_acc, "RACMO", products, high_thresh=500.0, scale="linear")
plot_pmb_scatter(df_mean_yr_acc, "RACMO", products, high_thresh=500.0, scale="log")



#%%

# --- Scatterplots ---
# Non-log-scaled version (commented out for future reference)
# colors = {2019: "tab:blue", 2020: "tab:red"}
# products = ["ERA5", "GPCP", "RACMO"]
# fig, axes = plt.subplots(1, len(products), figsize=(15, 5), sharey=True)

# for ax, prod in zip(axes, products):
#     for yr in [2019, 2020]:
#         m = df["year"] == yr
#         ax.scatter(df.loc[m, "Pmb"], df.loc[m, prod], color=colors[yr], 
#                    alpha=0.7, s=50, edgecolor='k', label=str(yr))

#     lims = (0, 2000)  # Adjust limits for non-log scale
#     ax.plot([lims[0], lims[1]], [lims[0], lims[1]], "k--", lw=1)
#     ax.set_xlim(lims[0], lims[1])
#     ax.set_ylim(lims[0], lims[1])
#     ax.set_aspect('equal', adjustable='box')

#     x, y = df["Pmb"], df[prod]
#     valid = (x > 0) & (y > 0)  # Ensure valid values
#     cc = np.corrcoef(x[valid], y[valid])[0, 1]
#     bias = np.nanmean(y[valid]) / np.nanmean(x[valid])
#     ax.text(0.03, 0.97, f"CC={cc:.2f}\nBias={bias:.2f}", transform=ax.transAxes,
#             va='top', ha='left', fontsize=15)
#     ax.set_xlabel("Pmb (mm/yr)", fontsize=15)
#     if prod == 'GPCP':
#         prod = 'GPCP v3.3'
#     elif prod == 'RACMO':
#         prod = 'RACMO v2.4'
#     ax.set_ylabel(f"{prod} (mm/yr)", fontsize=15)

#     # Set major and minor ticks
#     ax.set_xticks(np.arange(0, 2001, 500))
#     ax.set_yticks(np.arange(0, 2001, 500))
#     ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)

# # Legend on the first axis only (to avoid duplicates)
# axes[0].legend(title="Year", loc='lower right', ncol=2,
#                fontsize=15, frameon=False)

# plt.tight_layout()
# plt.show()

# Log-scaled version
# colors = {2019: "tab:blue", 2020: "tab:red"}
# products = ["ERA5", "GPCP", "RACMO"]
# fig, axes = plt.subplots(1, len(products), figsize=(15, 5), sharey=True)

# for ax, prod in zip(axes, products):
#     for yr in [2019, 2020]:
#         m = df["year"] == yr
#         ax.scatter(df.loc[m, "Pmb"], df.loc[m, prod], color=colors[yr], 
#                    alpha=0.7, s=50, edgecolor='k', label=str(yr))

#     lims = (10, 2000)  # Adjust limits to start from 10
#     ax.plot([lims[0], lims[1]], [lims[0], lims[1]], "k--", lw=1)
#     ax.set_xlim(lims[0], lims[1])
#     ax.set_ylim(lims[0], lims[1])
#     ax.set_aspect('equal', adjustable='box')

#     x, y = df["Pmb"], df[prod]
#     valid = (x > 0) & (y > 0)  # Ensure valid values for log scale
#     cc = np.corrcoef(x[valid], y[valid])[0, 1]
#     bias = np.nanmean(y[valid]) / np.nanmean(x[valid])
#     ax.text(0.03, 0.97, f"CC={cc:.2f}\nBias={bias:.2f}", transform=ax.transAxes,
#             va='top', ha='left', fontsize=15)
#     ax.set_xlabel("Pmb (mm/yr)", fontsize=15)
#     if prod == 'GPCP':
#         prod = 'GPCP v3.3'
#     elif prod == 'RACMO':
#         prod = 'RACMO v2.4'
#     ax.set_ylabel(f"{prod} (mm/yr)", fontsize=15)

#     # Set major and minor ticks
#     ax.set_xticks([100, 200, 800, 1000, 2000])
#     ax.set_yticks([100, 200, 800, 1000, 2000])
#     ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)
#     ax.set_xscale('log')
#     ax.set_yscale('log')

# # Legend on the first axis only (to avoid duplicates)
# axes[0].legend(title="Year", loc='lower right', ncol=2,
#                fontsize=15, frameon=False)

# plt.tight_layout()
# plt.show()

# Log-scaled version

colors = {2019: "tab:blue", 2020: "tab:red"}
products = ["ERA5", "GPCP", "RACMO"]

fig, axes = plt.subplots(1, len(products), figsize=(15, 5), sharey=True)

# Choose log limits from the actual positive data so points are not clipped.
# You can also hard-code (10, 2000) if you prefer.
xmin = 10#np.nanmin(df["Pmb"][df["Pmb"] > 0])
xmax = 2000#np.nanmax(df["Pmb"])
# pad a little so markers aren't on the frame
lims = (max(10, xmin*0.9), xmax*1.1)

for ax, prod in zip(axes, products):
    x_all = df["Pmb"].values
    y_all = df[prod].values

    # mask to ONLY points within the plotted box (so linear/log include the same dots)
    in_box = (x_all > 0) & (y_all > 0) & (x_all >= lims[0]) & (x_all <= lims[1]) \
             & (y_all >= lims[0]) & (y_all <= lims[1])

    # scatter
    for yr in [2019, 2020]:
        m = (df["year"] == yr).values & in_box
        ax.scatter(x_all[m], y_all[m], color=colors[yr], alpha=0.8, s=50,
                   edgecolor='k', linewidth=0.5, label=str(yr))

    # log scale + limits
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lims); ax.set_ylim(lims)

    # equal aspect so 1:1 slope is visually 45°
    ax.set_aspect('equal', adjustable='box')

    # 1:1 line
    ax.plot(lims, lims, 'k--', lw=1)

    # stats computed on the same points that are actually shown
    cc = np.corrcoef(x_all[in_box], y_all[in_box])[0, 1]
    bias = np.nanmean(y_all[in_box]) / np.nanmean(x_all[in_box])
    ax.text(0.03, 0.97, f"CC={cc:.2f}\nBias={bias:.2f}", transform=ax.transAxes,
            va='top', ha='left', fontsize=15)

    # labeling
    label = {'GPCP':'GPCP v3.3', 'RACMO':'RACMO v2.4'}.get(prod, prod)
    ax.set_xlabel("Pmb (mm/yr)", fontsize=15)
    ax.set_ylabel(f"{label} (mm/yr)", fontsize=15)

    # log ticks/format
    from matplotlib.ticker import LogLocator, ScalarFormatter
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10)*0.1))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10)*0.1))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)

    from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator, FuncFormatter

    # 1) Grid-friendly tick locations (unlabeled minor ticks)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # 2) Clean major labels: only a few handpicked values
    labels_to_show = [10, 100, 500,  1000, 2000]
    ax.xaxis.set_major_locator(FixedLocator(labels_to_show))
    ax.yaxis.set_major_locator(FixedLocator(labels_to_show))

    fmt = FuncFormatter(lambda v, _: f"{int(v):d}" if v in labels_to_show else "")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    # Optional: avoid crowding on the left
    ax.tick_params(axis="x", which="major", pad=4)
    ax.tick_params(axis="y", which="major", pad=4)

# Rotate x-axis tick labels
for ax in axes:
    ax.tick_params(axis="x", labelrotation=30)  # try 30°, 45°, or 60°

# one legend
axes[0].legend(title="Year", loc='lower right', ncol=2, fontsize=15, frameon=False)
plt.tight_layout(); plt.show()

# bring CloudSat into the discusiion
# read and process CloudSat data
# ant_data_path = r"/ra1/pubdat/AVHRR_CloudSat_proj/CS_Antartica_analysis_kkk/miscellaneous"  # Replace with your actual path
# cs_ant_filename = os.path.join(ant_data_path,"CS_seasonal_climatology-2007-2010.nc")
# cs_ant = xr.open_dataarray(cs_ant_filename)

# cs_ant_annual_clim = cs_ant.mean(dim='season',skipna=True)#.where(new_mask_ == 1)

# we will fill nan areas in cs ant with era5 data so create that data
era5_data = xr.open_dataarray(r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_0.25deg/ERA5_to_netcdf_files/ERA5_daily_precipitation_2010.nc')
if 'lon' in era5_data.coords and 'lat' in era5_data.coords:
    era5_data = era5_data.rename({'lon': 'x', 'lat': 'y'})
era5_data_ant = era5_data.sel(y=slice(-55, -90), x=slice(-180, 180))
era5_data_ant = era5_data_ant.mean(dim='time',skipna=True)

era5_data_ant.rio.write_crs(CRS.from_proj4(crs).to_string(), inplace=True)
# era5_trns = Affine(round(np.unique(np.diff(era5_data_ant['x'].values))[0],2),
#                    0.0,
#                    era5_data_ant['x'].min().item(),
#                    0.0,
#                    round(np.unique(np.diff(era5_data_ant['y'].values))[0],2),
#                    era5_data_ant['y'].max().item())

era5_data_ant = era5_data_ant.rio.reproject(
                                    dst_crs=era5_data_ant.rio.crs,
                                    shape=(70, 720),                                    
                                    resampling=Resampling.nearest
)

# Set the fill value to NaN
# era5_data_ant = era5_data_ant.where(era5_data_ant != era5_data_ant.attrs['_FillValue'], np.nan)

cs_annual_path = r'/ra1/pubdat/Reza_archive/CS_2022_Maps/annual'
cs_annual_filename = os.path.join(cs_annual_path,"2007_2010.npy")
raw = np.load(cs_annual_filename, allow_pickle=True)
cs_annual = np.flip(raw[:,:].transpose(), axis=0)
cs_annual = xr.DataArray(cs_annual,
                         dims=['lat', 'lon'],
                         coords={'lat': np.arange(90,-89.75, -0.5),
                          'lon': np.arange(-179.75, 180, 0.5)},
                          name = 'precipitation')
cs_annual_ant = cs_annual.sel(lat=slice(-55, -90), lon=slice(-180, 180))
cs_annual_ant = cs_annual_ant*24
# set areas in cs with na to era5 values
cs_annual_ant = cs_annual_ant.fillna(era5_data_ant.values)

yshp, xshp = cs_annual_ant.shape

minx = cs_annual_ant['lon'].min().item()
maxy = cs_annual_ant['lat'].max().item()
px_sz = round(cs_annual_ant['lon'].diff('lon').mean().item(), 2)

dest_flnme = os.path.join(misc_out, os.path.basename(cs_annual_filename).replace('.npy', '.tif'))

gdal_based_save_array_to_disk(dest_flnme, xshp, yshp, px_sz, minx, maxy, 
                              crs, crs_format, cs_annual_ant.data)

output_file_stereo = os.path.join(misc_out, os.path.basename(cs_annual_filename).replace('.npy', '_stere.tif'))

gdalwarp_command = f'gdalwarp -t_srs "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84" -r near {dest_flnme} {output_file_stereo}'

subprocess.run(gdalwarp_command, shell=True)

# Read the stereographic projection file
cs_ant_xrr_sh_stereo = xr.open_dataset(output_file_stereo)['band_data']

os.remove(dest_flnme)
os.remove(output_file_stereo)

# Clip the data to the bounds of the basin dataset
cs_ant_xrr_clip = cs_ant_xrr_sh_stereo.sel(
    x=slice(-3333250, 3333250),
    y=slice(3333250, -3333250)
).squeeze()


# Explicitly set the CRS before reprojecting
cs_ant_xrr_clip.rio.write_crs(CRS.from_proj4(crs_stereo).to_string(), inplace=True)

cs_ant_xrr_clip_res = cs_ant_xrr_clip.rio.reproject(
    cs_ant_xrr_clip.rio.crs,
    shape=basins.shape,  # set the shape as the basin data shape
    resampling=Resampling.nearest,
    transform=basins.rio.transform()
)

cs_ant_xrr_clip_res_arr = cs_ant_xrr_clip_res.values
cs_ant_xrr_clip_res_arr = np.where(basins.values > 0, cs_ant_xrr_clip_res_arr, np.nan)
cs_ant_xrr_clip_res = xr.DataArray(
    cs_ant_xrr_clip_res_arr,  # Use the 2D numpy array directly
    dims=['y', 'x'],  # Define dimensions
    coords={'y': cs_ant_xrr_clip_res.coords['y'], 
            'x': cs_ant_xrr_clip_res.coords['x']},
    name='precipitation'  # Rename the DataArray to 'precipitation'
)

# Create an empty DataArray to store mean precipitation mapped to basins
cs_ant_precip_xrr_basin_mapped = xr.full_like(basins, np.nan, dtype=float)

# Loop through each basin ID (Zwally basins are numbered from 1 to 27)
for basin_id in range(1, 20):
    # print(f"Processing basin {basin_id}")
    # Create a mask for the current basin
    basin_mask = basins == basin_id

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

# compare_mean_precp_plot(mean_annual_plot_arrs, 
#                     vmin=0, vmax=350, 
#                     cbar_tcks=[0,  50, 100, 150, 200, 250, 300, 350])
# gc.collect()

# for idx, yr in (enumerate(['2019', '2020'])): #[0, 1]:
       
#     annual_plot_arrs = [
#                     (f'Pmb annual\n accumulation_{yr}', Pmb_annual[idx]),
#                     (f'ERA5 annual\n accumulation_{yr}', era5_annual_mean[idx]), 
#                     (f'GPCP_v3.3 annual\n accumulation_{yr}', gpcpv3pt3_annual_mean[idx]),
#                     (f'RACMO_2.4p1 annual\n accumulation_{yr}', racmo_pr_annual_mean[idx]),                    
#                    ]
#     compare_mean_precp_plot(annual_plot_arrs, 
#                         vmin=0, vmax=400, 
#                         cbar_tcks=[0,  50, 100, 200,300, 400])
# gc.collect()

#---------------------------------------------------------------------------------
# compute and plot differences with Pmb
# diff_annual_plot_arrs = [(r"ERA5 – $P_{MB}$", era5_annual_mean.mean(dim='year') - Pmb_annual.mean(dim='year')),
#                          (r"GPCP_v3.3 – $P_{MB}$", gpcpv3pt3_annual_mean.mean(dim='year') - Pmb_annual.mean(dim='year')),
#                          (r"RACMO_2.4p1 – $P_{MB}$", racmo_pr_annual_mean.mean(dim='year') - Pmb_annual.mean(dim='year')),
#                         ]
    
# compare_mean_precp_plot(diff_annual_plot_arrs, 
#                     vmin=-10, vmax=100, 
#                     cbar_tcks=[-10, 0, 10, 25, 50, 75, 100])
# gc.collect()
# #---------------------------------------------------------------------------------

# SEAS = ["DJF", "MAM", "JJA", "SON"]


# year = 2019  # or whatever your idx represents

# products_for_year = [
#     (r"P$_{MB}$",       ensure_season_index(pick_year(Pmb_seasonal, year),SEAS)),
#     ("ERA5",      ensure_season_index(pick_year(era5_seasonal_mean*30, year),SEAS)),
#     ("GPCP_v3.3", ensure_season_index(pick_year(gpcpv3pt3_seasonal_mean*30, year),SEAS)),
#     ("RACMO_2.4p1", ensure_season_index(pick_year(racmo_pr_seasonal_mean, year),SEAS)),

# ]
#----------------------------------------------------------------------------------

# seasonal_plot_arrs = [
#     (f"{name} — {s} mean", da.sel(season=s))
#     for name, da in products_for_year
#     for s in SEAS
# ]

# # compare_mean_precp_plot(seasonal_plot_arrs, 
# #                         vmin=0, vmax=30, 
# #                         cbar_tcks=[0, 2.5, 5, 7.5, 10, 15, 20, 25, 30])
# # gc.collect()

# # SEAS = ("DJF", "MAM", "JJA", "SON")

# fig, axes = plot_seasonal_precip_maps(
#     products_for_year,
#     seasons=SEAS,
#     vmin=0,
#     vmax=30,
#     cbar_ticks=[0, 2.5, 5, 7.5, 10, 15, 20, 25, 30]
# )

# gc.collect()
#----------------------------------------------------------------------------------
# plot_monthly_cycle_by_basin_products_precomputed(plot_dfs)

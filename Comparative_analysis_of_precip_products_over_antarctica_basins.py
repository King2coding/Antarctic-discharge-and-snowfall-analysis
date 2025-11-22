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

# Set up colormap and norm for 27 discrete basins
colors = plt.cm.gist_ncar(np.linspace(0, 1, 19))
cmap = mcolors.ListedColormap(colors)
cmap.set_bad(color='white')  # Set background (masked or NaN) to white

# Use min and max values for levels
vmin, vmax = 1,19 #1, 27
levels = np.linspace(vmin, vmax, vmax - vmin + 2)  # 27 basins + 1 for boundaries
norm = mcolors.BoundaryNorm(levels, cmap.N)


# Plot
proj = ccrs.SouthPolarStereo()
fig, ax = plt.subplots(figsize=(12, 8), dpi=300, subplot_kw={'projection': proj})

# Set extent for Antarctica
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

# Plot the data
p = basins.plot(
    ax=ax,
    transform=proj,
    cmap=cmap,
    norm=norm,
    add_colorbar=False
)

# Add white background
ax.set_facecolor('white')

# Annotate each basin with its ID
for basin_id in range(1, 20):
    # Create a mask for the current basin
    basin_mask = basins == basin_id

    # Get the centroid of the basin
    y, x = np.where(basin_mask[0])
    if len(x) > 0 and len(y) > 0:
        centroid_x = basins['x'].values[x].mean()
        centroid_y = basins['y'].values[y].mean()
        ax.text(
            centroid_x, centroid_y, str(basin_id),
            color='black', fontsize=15, ha='center', va='center', zorder=5,
            transform=ccrs.SouthPolarStereo()
        )

# Add coastlines
ax.coastlines(resolution='110m', color='black', linewidth=0.5)

# Remove axis
ax.axis('off')

# # Final cleanup
# ax.set_title("IMBIE Basins with IDs ", fontsize=18)
# plt.tight_layout()
# # plt.show()
# # Save the imbie basin plot
# output_path = os.path.join(path_to_plots, 'imbie_basins_with_ids.png')
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
Pmb_annual = xr.open_dataarray(os.path.join(basins_path, "Pmb_annual_2019_2020_mm.nc"))
Pmb_seasonal = xr.open_dataarray(os.path.join(basins_path, "Pmb_seasonal_mm_2019_2020.nc"))
img_fle_lst = sorted([os.path.join(imerg_basin_path, x) for x in os.listdir(imerg_basin_path) if 'imbie_basin' in x])
era5_fle_lst = sorted([os.path.join(era5_basin_path, x) for x in os.listdir(era5_basin_path) if 'imbie_basin' in x])
gpcpv3pt3_fle_lst = sorted([os.path.join(gpcpv3pt3_basin_path, x) for x in os.listdir(gpcpv3pt3_basin_path) if 'imbie_basin' in x])
racmo_pr = xr.open_dataarray(os.path.join(racmo_path,'pr_monthlyS_ANT11_RACMO2.4p1_ERA5_2019_2022.nc'))
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
#%%
# read and process satellite precipitation data
# IMERG
print('Processing IMERG data')

imerg_annual_mean, imerg_seasonal_mean = process_precipitation_data(img_fle_lst, basins)
# imerg_annual_mean = imerg_annual_mean * 365
gc.collect()

#----------------------------------------------------------------------------------

# read and process era5 data
print('Processing ERA5 data')

era5_annual_mean, era5_seasonal_mean = process_precipitation_data(era5_fle_lst, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)
era5_annual_mean = era5_annual_mean * 365
gc.collect()

#----------------------------------------------------------------------------------

print('Processing GPCP v3.3 data')

gpcpv3pt3_annual_mean, gpcpv3pt3_seasonal_mean = process_precipitation_data(gpcpv3pt3_fle_lst, 
                                                                            basins,
                                                                            'precipitation',
                                                                            False,)
gpcpv3pt3_annual_mean = gpcpv3pt3_annual_mean * 365
gc.collect()

#----------------------------------------------------------------------------------
print('processing RACMO pr data')
racmo_pr_annual_mean, racmo_pr_seasonal_mean = process_precipitation_data(racmo_pr, 
                                                                          basins, 
                                                                          'pr',
                                                                          True,)
# racmo_pr_annual_mean = racmo_pr_annual_mean * 365
gc.collect()

#-------------------------------------------------------------------------------
# make some plots ('IMERG', imerg_annual_mean[0])
# annual plots
for idx, yr in (enumerate(['2019', '2020'])): #[0, 1]:
       
    annual_plot_arrs = [
                    (f'Pmb annual\n accumulation_{yr}', Pmb_annual[idx]),
                    (f'ERA5 annual\n accumulation_{yr}', era5_annual_mean[idx]), 
                    (f'GPCP_v3.3 annual\n accumulation_{yr}', gpcpv3pt3_annual_mean[idx]),
                    (f'RACMO_2.4p1 annual\n accumulation_{yr}', racmo_pr_annual_mean[idx]),                    
                   ]
    compare_mean_precp_plot(annual_plot_arrs, 
                        vmin=0, vmax=400, 
                        cbar_tcks=[0,  50, 100, 200,300, 400])
gc.collect()

# calculate and plot the mean across years
mean_annual_plot_arrs = [
                    (r"P$_{MB}$", Pmb_annual.mean(dim='year')),
                    (f'ERA5', era5_annual_mean.mean(dim='year')), 
                    (f'GPCP v3.3', gpcpv3pt3_annual_mean.mean(dim='year')),
                    (f'RACMO 2.4p1', racmo_pr_annual_mean.mean(dim='year')),                    
                   ]

compare_mean_precp_plot(mean_annual_plot_arrs, 
                    vmin=0, vmax=350, 
                    cbar_tcks=[0,  50, 100, 150, 200, 250, 300, 350])
gc.collect()

compare_mean_precip_2x2(mean_annual_plot_arrs, 
                 vmin=0, vmax=400,
                 cbar_tcks=[0, 50, 100, 150, 200, 250, 300, 350, 400])
gc.collect()
#---------------------------------------------------------------------------------
# compute and plot differences with Pmb
diff_annual_plot_arrs = [(r"ERA5 – $P_{MB}$", era5_annual_mean.mean(dim='year') - Pmb_annual.mean(dim='year')),
                         (r"GPCP_v3.3 – $P_{MB}$", gpcpv3pt3_annual_mean.mean(dim='year') - Pmb_annual.mean(dim='year')),
                         (r"RACMO_2.4p1 – $P_{MB}$", racmo_pr_annual_mean.mean(dim='year') - Pmb_annual.mean(dim='year')),
                        ]
    
compare_mean_precp_plot(diff_annual_plot_arrs, 
                    vmin=-10, vmax=100, 
                    cbar_tcks=[-10, 0, 10, 25, 50, 75, 100])
gc.collect()
#---------------------------------------------------------------------------------

SEAS = ["DJF", "MAM", "JJA", "SON"]


year = 2019  # or whatever your idx represents

products_for_year = [
    (r"P$_{MB}$",       ensure_season_index(pick_year(Pmb_seasonal, year),SEAS)),
    ("ERA5",      ensure_season_index(pick_year(era5_seasonal_mean*30, year),SEAS)),
    ("GPCP_v3.3", ensure_season_index(pick_year(gpcpv3pt3_seasonal_mean*30, year),SEAS)),
    ("RACMO_2.4p1", ensure_season_index(pick_year(racmo_pr_seasonal_mean, year),SEAS)),

]

seasonal_plot_arrs = [
    (f"{name} — {s} mean", da.sel(season=s))
    for name, da in products_for_year
    for s in SEAS
]

# compare_mean_precp_plot(seasonal_plot_arrs, 
#                         vmin=0, vmax=30, 
#                         cbar_tcks=[0, 2.5, 5, 7.5, 10, 15, 20, 25, 30])
# gc.collect()

# SEAS = ("DJF", "MAM", "JJA", "SON")

fig, axes = plot_seasonal_precip_maps(
    products_for_year,
    seasons=SEAS,
    vmin=0,
    vmax=30,
    cbar_ticks=[0, 2.5, 5, 7.5, 10, 15, 20, 25, 30]
)

gc.collect()
#----------------------------------------------------------------------------------
# make scatter plots comparison
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

# df_mean_yr_acc.round(2).to_csv(os.path.join(path_to_dfs, 'df_mean_yr_acc.csv'), index=False)

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

# ---------------------- Stable basin palette (IDs 2..19) ----------------------
BASIN_IDS = np.arange(2, 20)  # [2, 3, ..., 19]
PALETTE   = plt.cm.gist_ncar(np.linspace(0, 1, len(BASIN_IDS)))
ID2COLOR  = dict(zip(BASIN_IDS, PALETTE))

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
    log_min=10,             # lower bound for log plots
    log_ticks=(10, 50, 100, 200, 500, 1000, 1500, 2000)
):
    """
    Scatter of product vs ref (Pmb) by IMBIE basin with consistent colors (IDs 2..19).
    scale="linear" keeps your original look; scale="log" switches axes to log with clean ticks.
    """

    # nice product display names
    pretty = {"GPCP": "GPCP v3.3", "RACMO": "RACMO v2.4"}

    fig, axes = plt.subplots(1, len(products), figsize=(18, 6), sharey=True)
    if len(products) == 1:
        axes = [axes]

    # FIXED max at 2000 for both linear and log
    global_max = 2000.0

    for ax, prod in zip(axes, products):
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
        ax.scatter(x, y, c=cols, s=110, alpha=0.85,
                   edgecolor="k", linewidths=0.6, zorder=2)

        # annotate
        diff = np.abs(x - y)
        mask = (diff >= high_thresh) | (((b >= 13) & (b <= 18)) | (x >= 500))
        for xx, yy, bb in zip(x[mask], y[mask], b[mask]):
            ax.annotate(f"{int(bb)}", xy=(xx, yy), xycoords="data",
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=12, color="black",
                        clip_on=True, path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                        zorder=4)

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
            cc = np.corrcoef(x[in_box], y[in_box])[0, 1]
            bias = np.nanmean(y[in_box]) / np.nanmean(x[in_box])
        else:
            cc, bias = np.nan, np.nan

        ax.text(0.03, 0.97, f"CC={cc:.2f}\nBias={bias:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=14,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        # labels
        ax.set_xlabel(f"{ref} (mm/yr)", fontsize=14)
        ax.set_ylabel(f"{yname} (mm/yr)", fontsize=14)
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

    plt.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.1, wspace=0.15)
    plt.show()

# Example usage:
products = ["ERA5", "GPCP", "RACMO"]
plot_pmb_scatter(df_mean_yr_acc, "Pmb", products, high_thresh=500.0, scale="linear")
products = ["ERA5", "GPCP", "RACMO"]
plot_pmb_scatter(df_mean_yr_acc, "Pmb", products, high_thresh=500.0, scale="log")


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
# work on the reference data

imbie_basin_discharge_params = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins/antarctic_discharge_2013-2022_imbie.xlsx'
# xls = pd.ExcelFile(imbie_basin_discharge_params)
# print("Sheets:", xls.sheet_names)


# # 2) Preview each sheet
# for sheet in xls.sheet_names:
#     df = xls.parse(sheet)
#     print(f"\n--- {sheet} (first 5 rows) ---")
#     print(df.head())

df_sum = pd.read_excel(
    imbie_basin_discharge_params,
    sheet_name="Summary",
    engine="openpyxl"
)

# 2) Build lookup but only for the first 19 basins
lookup_gt = {
    i+1: gt
    for i, gt in enumerate(df_sum["SMB total 2013-2022 Gt/yr"].values[:19])
}
# 3) Convert Gt/yr → mm/yr on the uniform 500 m grid
cell_area = 500.0 * 500.0   # m² per grid cell (500 m × 500 m)
P_MB_mm = xr.full_like(basins_imbie, np.nan, dtype=float)

# 4) Loop over each basin code
for code, smb_gt in lookup_gt.items():
    mask = (basins_imbie == code)               # DataArray of True/False
    n_cells = int(mask.sum(dim=("y","x")).values)  # extract integer count
    area = n_cells * cell_area
    # Convert: Gt/yr → m³/yr → m/yr depth → mm/yr
    depth_mm = (smb_gt * 1e9) / area * 1000.0
    P_MB_mm = P_MB_mm.where(~mask, other=depth_mm)

# 2) For a proper map projection (optional)
ax = plt.axes(projection=ccrs.SouthPolarStereo())
P_MB_mm.plot.imshow(
    x="x", y="y",
    transform=ccrs.SouthPolarStereo(),
    cmap="jet",
    origin="lower",
    yincrease=False,
    ax=ax,
    vmin=0,
    vmax=300
)
ax.coastlines(color="k", linewidth=0.5)
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
# plt.title("SMB Total Mapped to IMBIE Basins (Polar Stereo)")
plt.show()

# save imbie basins 
encoding = {P_MB_mm.name:{"zlib": True, "complevel": 9}}
svename = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins/P_MB_2013-2022_imbie_basin_annual_precip.nc'
P_MB_mm.to_netcdf(svename, mode='w', format='NETCDF4', encoding=encoding)

gc.collect()
#%%
print('Plotting')

plot_arras = [('P$_{MB}$', P_MB_mm),
              ('AVHRR', stereo_avhrr_xrr_basin_mm_per_year['imbie']), 
              ('ERA5', stereo_era5_xrr_basin_mm_per_year['imbie']),
              ('AIRS', stereo_airs_xrr_basin_mm_per_year['imbie']),
              ('CS', cs_ant_precip_xrr_basin_mapped_mm_per_year) 
             ]
            #   ('IMERG', stereo_img_xrr_basin_mm_per_year_5km),
            #   
            #   ] #

svnme = os.path.join(path_to_plots, 'annual_snowfall_accumulation_over_imbie_basins.png')
compare_mean_precp_plot(plot_arras, 
                        vmin=0, vmax=300, 
                        cbar_tcks=[0, 50, 100, 150, 200, 250, 300])
plt.savefig(svnme,  dpi=1000, bbox_inches='tight')
gc.collect()


plot_arras = [ ('SSMIS-F17', stereo_ssmi_17_xrr_basin_mm_per_year['imbie']),
               ('IMERG', stereo_img_xrr_basin_mm_per_year['imbie']), ] #

svnme = os.path.join(path_to_plots, 'IMERG-SSMIS-F17-annual_snowfall_accumulation_over_imbie_basins.png')
compare_mean_precp_plot(plot_arras, 
                        vmin=0, vmax=100, 
                        cbar_tcks=[0, 10 ,25, 50, 75, 85 ,100])

plt.savefig(svnme,  dpi=1000, bbox_inches='tight')
gc.collect()

# Example usage
# single_precp_plot(stereo_img_xrr_basin_mm_per_year['imbie'], 'IMERG', vmin=0, vmax=100)
# single_precp_plot(stereo_avhrr_xrr_basin_mm_per_year['imbie'], 'AVHRR', vmin=0, vmax=350)
# single_precp_plot(stereo_era5_xrr_basin_mm_per_year['imbie'], 'ERA5', vmin=0, vmax=350)
# single_precp_plot(stereo_ssmi_17_xrr_basin_mm_per_year['imbie'], 'SSMIS-F17', vmin=0, vmax=100)
# single_precp_plot(stereo_airs_xrr_basin_mm_per_year['imbie'], 'AIRS', vmin=0, vmax=350)
# single_precp_plot(cs_ant_precip_xrr_basin_mapped_mm_per_year, 'CloudSat', vmin=0, vmax=350)


#%%
print('Making a table of annual mean precipitation for each basin')
arras = [('P$_{MB}$', P_MB_mm),
        ('IMERG', stereo_img_xrr_basin_mm_per_year), 
        ('AVHRR', stereo_avhrr_xrr_basin_mm_per_year), 
        ('ERA5', stereo_era5_xrr_basin_mm_per_year),
        ('SSMIS-F17', stereo_ssmi_17_xrr_basin_mm_per_year), 
        ('AIRS', stereo_airs_xrr_basin_mm_per_year),
        ('CS', cs_ant_precip_xrr_basin_mapped_mm_per_year)] #
# make a table of the mean precipitation for each basin
annual_mean_df = pd.DataFrame(columns=list(range(1, 20)), index=[x[0] for x in arras])
for product_name, data in arras:
    print(product_name)

    for basin_id in range(1, 20):
        # Create a mask for the current basin
        basin_mask = basins_zwally == basin_id

        # Mask the precipitation data for the current basin
        # Ensure 'data' is a DataArray by selecting the first variable if it's a Dataset
        if isinstance(data, xr.Dataset):
            data = list(data.data_vars.values())[0]

        basin_precip = np.unique(data.where(basin_mask.data).values[~np.isnan(data.where(basin_mask.data).values)])[0]

        annual_mean_df.loc[product_name, basin_id] = basin_precip

annual_mean_df = annual_mean_df.applymap(lambda x: round(x, 2) if pd.notnull(x) else x)

# Save the DataFrame to a CSV file
annual_mean_df.to_csv(os.path.join(path_to_dfs, f'annual_mean_precip_over_imbie_basins_{cde_run_dte}.csv'))

# make a scatter plot of all products agianst P_MB
svename = os.path.join(path_to_plots, f'scatter_compare_annual_mean_precip_over_imbie_basins_{cde_run_dte}.png')

x = annual_mean_df.loc['P_MB'].astype(float).values
products = [p for p in annual_mean_df.index if p != 'P_MB']

rows, cols = 2, 3
fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True, sharey=True)
axes_flat = axes.flatten()

for ax, prod in zip(axes_flat, products):
    y = annual_mean_df.loc[prod].astype(float).values
    valid = (~np.isnan(x)) & (~np.isnan(y))
    cc = np.corrcoef(x[valid], y[valid])[0,1]
    bias = np.nanmean(y) / np.nanmean(x)
    ax.scatter(x, y, alpha=0.7)
    lims = [min(np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y))]
    ax.plot(lims, lims, 'k--', linewidth=1)
    ax.text(0.05, 0.95, f"CC={cc:.2f}\nBias={bias:.2f}", transform=ax.transAxes, fontsize=20, 
            verticalalignment='top', horizontalalignment='left')
    ax.set_xlabel("P$_{MB}$ (mm/yr)", fontsize=20)
    ax.set_ylabel(f"{prod} (mm/yr)", fontsize=20)

for ax in axes_flat[len(products):]:
    ax.axis('off')

plt.tight_layout()
plt.savefig(svename,  dpi=1000, bbox_inches='tight')
# plt.show()

#------------------------------------------------------------------------------
df = annual_mean_df.copy().astype(float)

# 2) Products list (excluding P_MB)
products = [p for p in df.index if p != 'P_MB']

# 3) References
refs = ['CS', 'ERA5']

# 4) Prepare results table
cols = [f"CC_vs_{ref}" for ref in refs] + [f"Bias_vs_{ref}" for ref in refs]
results = pd.DataFrame(index=products, columns=cols)

# 5) Compute CC and Bias for each scenario
for ref in refs:
    x_ref = df.loc[ref].values
    for prod in products:
        if prod == ref:
            results.loc[prod, f"CC_vs_{ref}"] = 1.0
            results.loc[prod, f"Bias_vs_{ref}"] = 1.0
        else:
            y = df.loc[prod].values
            valid = (~np.isnan(x_ref)) & (~np.isnan(y))
            cc = np.corrcoef(x_ref[valid], y[valid])[0,1]
            bias = np.nanmean(y) / np.nanmean(x_ref)
            results.loc[prod, f"CC_vs_{ref}"] = round(cc, 2)
            results.loc[prod, f"Bias_vs_{ref}"] = round(bias, 2)

svnme = os.path.join(path_to_dfs, f'metrics_compare_annual_mean_precip_over_imbie_basins_{cde_run_dte}.csv')

results.to_csv(svnme)

#%%
from scipy.io import loadmat

mask_path = "/ra1/pubdat/mask_land_ocean/mask50km.mat"
mask = loadmat(mask_path)["mask50"][-60:, :].swapaxes(0, 1)
mask = np.flip(mask, axis=1)
mask_binary = mask < 75
new_mask = mask_binary.copy()
new_mask[:360, :] = mask_binary[360:, :].copy()
new_mask[360:, :] = mask_binary[:360, :].copy()
new_mask = np.flip(new_mask.T, axis=0)
new_mask_ = new_mask.astype(int)


cs_annual_path = r'/home/kkumah/CS_2022_Maps/annual'
cs_annual_filename = os.path.join(cs_annual_path,"2007_2010.npy")
raw = np.load(cs_annual_filename, allow_pickle=True)
cs_annual = np.flip(raw[:,:].transpose(), axis=0)
cs_annual = xr.DataArray(raw)
cs_antarctica_data = cs_annual.where(new_mask_ == 1)


#%%

# ncolors = 15

# # 2) Get discrete ‘jet’ colormap
# cmap = plt.get_cmap('jet', ncolors)

# # 3) Define bin edges
# levels = np.linspace(0,
#                      500,
#                      ncolors + 1)
# norm = BoundaryNorm(levels, ncolors)

# # 4) Plot
# plt.figure(figsize=(12, 4))
# im = plt.imshow(annual_mean_df, aspect='auto', cmap=cmap, norm=norm)
# cbar = plt.colorbar(im, ticks=levels, spacing='proportional')
# cbar.set_label('Annual Precipitation (mm)')
# plt.xticks(ticks=np.arange(annual_mean_df.shape[1]),
#            labels=annual_mean_df.columns, rotation=90)
# plt.yticks(ticks=np.arange(annual_mean_df.shape[0]),
#            labels=annual_mean_df.index)
# plt.xlabel('Basin ID')
# plt.ylabel('Satellite Product')
# plt.title('Annual Precipitation by Basin (Discrete Jet Colormap)')
# plt.tight_layout()
# plt.show()
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
# fig, ax = plt.su<truncated__content/>
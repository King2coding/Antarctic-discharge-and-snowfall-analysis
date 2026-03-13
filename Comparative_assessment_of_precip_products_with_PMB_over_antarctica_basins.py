#%%
# packages

from program_utils import *


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
#----------------------------------------------------------------------------------

product_order_corr = [
    r"$P_{\mathrm{MB}}$",
    "ERA5",
    "GPCP v3.3",
    "ATMS",
    "ATMS (corr.)",
    "MHS",
    "MHS (corr.)",
    "DMSP SSMIS",
    "DMSP SSMIS (corr.)",
    "AMSR2",
    "AMSR2 (corr.)",
    "GPM Satellites",
    "GPM Satellites (corr.)",
]

product_styles_corr = {
    r"$P_{\mathrm{MB}}$": {"color": "k", "marker": "o", "lw": 2.5},

    "ERA5": {"color": "blue", "marker": "s", "lw": 2.5},
    "GPCP v3.3": {"color": "orange", "marker": "D", "lw": 2.5},

    "ATMS": {"color": "tab:blue", "lw": 1.5},
    "ATMS (corr.)": {"color": "tab:blue", "ls": "--", "lw": 2},

    "MHS": {"color": "lime", "lw": 1.5},
    "MHS (corr.)": {"color": "lime", "ls": "--", "lw": 2},

    "DMSP SSMIS": {"color": "green", "lw": 1.5},
    "DMSP SSMIS (corr.)": {"color": "green", "ls": "--", "lw": 2},

    "AMSR2": {"color": "red", "lw": 1.5},
    "AMSR2 (corr.)": {"color": "red", "ls": "--", "lw": 2},

    "GPM Satellites": {"color": "cyan", "lw": 1.5},
    "GPM Satellites (corr.)": {"color": "cyan", "ls": "--", "lw": 2},
}

corr_targets = [
    "ATMS",
    "MHS",
    "DMSP SSMIS",
    "AMSR2",
    "GPM Satellites",
]

product_order = [
    r"$P_{\mathrm{MB}}$",
    "ERA5",
    "GPCP v3.3",
    # "RACMO 2.4p1",
    "ATMS",
    "MHS",
    "DMSP SSMIS",
    "AMSR2",
    "GPM Satellites",
]

product_styles = {
    r"$P_{\mathrm{MB}}$": {"color": "k", "marker": "o", "lw": 2.0},
    "ERA5": {"color": "blue", "marker": "s", "lw": 1.8},
    "GPCP v3.3": {"color": "tab:orange", "marker": "D", "lw": 1.8},
    # "RACMO 2.4p1": {"color": "tab:green", "marker": "^", "lw": 1.8},
    "ATMS": {"lw": 1.5},
    "MHS": {"lw": 1.5},
    "DMSP SSMIS": {"lw": 1.5},
    "AMSR2": {"lw": 1.5},
    "GPM Satellites": {"lw": 1.5},
}

#----------------------------------------------------------------------------------
# basin grid

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
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
gc.collect()

all_basin_ids = sorted({
    bid
    for _, ids in REGION_DEFS
    for bid in ids
})

basin_weights = compute_basin_area_weights_from_mask(
    basins,
    basin_ids=all_basin_ids
)
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
all_gpm_fle = os.path.join(all_gpm_basin_path, "all_gpm_satellite_precip_mean_20260304.nc")

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
gc.collect()

print('Processing MHS data')
mhs_annual_mean, mhs_seasonal_mean,mhs_b_mean = process_precipitation_data(mhs_fle_lst, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)
mhs_annual_mean = (mhs_annual_mean * 24) * 365

mhs_basin_mean = mhs_b_mean.to_dataframe().reset_index()
mhs_basin_mean['year'] = mhs_basin_mean['time'].dt.year
mhs_basin_mean['month'] = mhs_basin_mean['time'].dt.month

mhs_basin_mnth_mean = mhs_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
mhs_basin_mnth_mean['time'] = pd.to_datetime(dict(year=mhs_basin_mnth_mean['year'], month=mhs_basin_mnth_mean['month'], day=1))

gc.collect()

print('Processing DMSP-SSMIS data')
dmsp_ssmis_annual_mean, dmsp_ssmis_seasonal_mean,dmsp_ssmis_b_mean = process_precipitation_data(dmsp_ssmis_fle_lst, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)
dmsp_ssmis_annual_mean = (dmsp_ssmis_annual_mean * 24) * 365

dmsp_ssmis_basin_mean = dmsp_ssmis_b_mean.to_dataframe().reset_index()
dmsp_ssmis_basin_mean['year'] = dmsp_ssmis_basin_mean['time'].dt.year
dmsp_ssmis_basin_mean['month'] = dmsp_ssmis_basin_mean['time'].dt.month

dmsp_ssmis_basin_mnth_mean = dmsp_ssmis_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
dmsp_ssmis_basin_mnth_mean['time'] = pd.to_datetime(dict(year=dmsp_ssmis_basin_mnth_mean['year'], month=dmsp_ssmis_basin_mnth_mean['month'], day=1))

gc.collect()

print('Processing AMSR2 data')
amsr2_annual_mean, amsr2_seasonal_mean,amsr2_b_mean = process_precipitation_data(amsr2_fle_lst, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)
amsr2_annual_mean = (amsr2_annual_mean * 24) * 365

amsr2_basin_mean = amsr2_b_mean.to_dataframe().reset_index()
amsr2_basin_mean['year'] = amsr2_basin_mean['time'].dt.year
amsr2_basin_mean['month'] = amsr2_basin_mean['time'].dt.month

amsr2_basin_mnth_mean = amsr2_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
amsr2_basin_mnth_mean['time'] = pd.to_datetime(dict(year=amsr2_basin_mnth_mean['year'], month=amsr2_basin_mnth_mean['month'], day=1))

gc.collect()

print('Processing Mean GPM Satellite Data')
gpm_sat_annual_mean, gpm_sat_seasonal_mean,gpm_sat_b_mean = process_precipitation_data(all_gpm_fle, 
                                                                  basins, 
                                                                  'precipitation',
                                                                  False,)

gpm_sat_annual_mean = (gpm_sat_annual_mean * 24) * 365

gpm_sat_basin_mean = gpm_sat_b_mean.to_dataframe().reset_index()
gpm_sat_basin_mean['year'] = gpm_sat_basin_mean['time'].dt.year
gpm_sat_basin_mean['month'] = gpm_sat_basin_mean['time'].dt.month

gpm_sat_basin_mnth_mean = gpm_sat_basin_mean.groupby(['year','month','basin'])['precipitation'].sum().reset_index()
gpm_sat_basin_mnth_mean['time'] = pd.to_datetime(dict(year=gpm_sat_basin_mnth_mean['year'], month=gpm_sat_basin_mnth_mean['month'], day=1))

gc.collect()


#----------------------------------------------------------------------------------
Pmb_annual_mean = Pmb_annual.mean(dim='year')
era5_annual_mean_mean = era5_annual_mean.mean(dim='year')
gpcpv3pt3_annual_mean_mean = gpcpv3pt3_annual_mean.mean(dim='year')
racmo_pr_annual_mean_mean = racmo_pr_annual_mean.mean(dim='year')
atms_annual_mean_mean = atms_annual_mean.mean(dim='year')
mhs_annual_mean_mean = mhs_annual_mean.mean(dim='year')
dmsp_ssmis_annual_mean_mean = dmsp_ssmis_annual_mean.mean(dim='year')
amsr2_annual_mean_mean = amsr2_annual_mean.mean(dim='year')
gpm_sat_annual_mean_mean = gpm_sat_annual_mean.mean(dim='year')

#%% Annual Means - Spatial
# calculate and plot the mean across years
mean_annual_plot_arrs = [
                    (r"P$_{MB}$", Pmb_annual_mean),
                    (f'ERA5', era5_annual_mean_mean), 
                    (f'GPCP v3.3', gpcpv3pt3_annual_mean_mean),
                    # (f'RACMO 2.4p1', racmo_pr_annual_mean_mean),
                    (f'ATMS', atms_annual_mean_mean),
                    (f'MHS', mhs_annual_mean_mean),
                    (f'DMSP-SSMIS', dmsp_ssmis_annual_mean_mean),
                    (f'AMSR2', amsr2_annual_mean_mean),                    
                   (f'GPM Satellites', gpm_sat_annual_mean_mean),
                   ]


fig, axes = compare_mean_precip_grid_power(
    mean_annual_plot_arrs,
    ncols=4,
    gamma=0.6,
    vmin=0,
    vmax=400,
    cbar_tcks=[0, 25, 50, 100,  200, 300, 400],
    cbar_label="Precipitation [mm/year]",
    panel_letters=True,
)
svnme = f'annual_snowfall_accumulation_over_imbie_basins_{cde_run_dte}_sharedplot.png'
# save plot to disk
svnme = os.path.join(path_to_plots, svnme)
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()
#----------------------------------------------------------------------------------
fig, axes, cb1, cb2 = compare_mean_precip_grid_power_dual_cbar(
    mean_annual_plot_arrs,
    group1_idx=[0, 1, 2],          # P_MB, ERA5, GPCP
    group2_idx=[3, 4, 5, 6, 7],    # ATMS, MHS, DMSP-SSMIS, AMSR2, GPM Satellite
    ncols=4,
    # upper / high-range products
    gamma1=0.6,
    vmin1=0,
    vmax1=400,
    cbar_tcks1=[0, 25, 50, 100, 200, 300, 400],
    # lower / low-range products
    gamma2=0.6,
    vmin2=0,
    vmax2=80,
    cbar_tcks2=[0, 5, 10, 20, 40, 60, 80],
    panel_letters=True,
)
svnme = os.path.join(path_to_plots, 
                     f'annual_snowfall_accumulation_over_imbie_basins_log_{cde_run_dte}_diff_cbar.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()


#%% COMPUTE AND PLOT AREA WEIGHTED MEAN MONTHLY CYCLES
# monthly_df_data = {
#     r"$P_{\mathrm{MB}}$": p_mm_mean_df.rename(columns={"basin_id": "basin",
#                                                        "precip_mm_per_month": "precipitation"}),
#     "ERA5": era5_basin_mean,
#     "GPCP v3.3": gpcpv3pt3_basin_mean,
#     # "RACMO 2.4p1": racmo_basin_mnth_mean,
#     "ATMS": atms_basin_mean,
#     "MHS": mhs_basin_mean,
#     "DMSP SSMIS": dmsp_ssmis_basin_mean,
#     "AMSR2": amsr2_basin_mean,
#     "GPM Satellites": gpm_sat_basin_mean,
# }

monthly_df_data_mmmonth = {
    r"$P_{\mathrm{MB}}$": convert_precip_to_mm_per_month(
        p_mm_mean_df.rename(columns={"basin_id": "basin", "precip_mm_per_month": "precipitation"}),
        unit="mm/month",
    ),
    "ERA5": convert_precip_to_mm_per_month(era5_basin_mean, unit="mm/day"),
    "GPCP v3.3": convert_precip_to_mm_per_month(gpcpv3pt3_basin_mean, unit="mm/day"),
    "ATMS": convert_precip_to_mm_per_month(atms_basin_mean, unit="mm/hr"),
    "MHS": convert_precip_to_mm_per_month(mhs_basin_mean, unit="mm/hr"),
    "DMSP SSMIS": convert_precip_to_mm_per_month(dmsp_ssmis_basin_mean, unit="mm/hr"),
    "AMSR2": convert_precip_to_mm_per_month(amsr2_basin_mean, unit="mm/hr"),
    "GPM Satellites": convert_precip_to_mm_per_month(gpm_sat_basin_mean, unit="mm/hr"),
}

region_monthly_clim = compute_weighted_region_monthly_climatologies(
    monthly_df_data=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights,
    basin_col="basin",
    value_col="precipitation",
    time_col="time",
)

product_order = [
    r"$P_{\mathrm{MB}}$",
    "ERA5",
    "GPCP v3.3",
    # "RACMO 2.4p1",
    "ATMS",
    "MHS",
    "DMSP SSMIS",
    "AMSR2",
    "GPM Satellites",
]

product_styles = {
    r"$P_{\mathrm{MB}}$": {"color": "k", "marker": "o", "lw": 2.0},
    "ERA5": {"color": "tab:blue", "marker": "s", "lw": 1.8},
    "GPCP v3.3": {"color": "tab:orange", "marker": "D", "lw": 1.8},
    # "RACMO 2.4p1": {"color": "tab:green", "marker": "^", "lw": 1.8},
    "ATMS": {"lw": 1.5},
    "MHS": {"lw": 1.5},
    "DMSP SSMIS": {"lw": 1.5},
    "AMSR2": {"lw": 1.5},
    "GPM Satellites": {"lw": 1.5},
}

fig, axes = plot_weighted_region_monthly_climatology(
    region_monthly_clim,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=product_order,
    product_styles=product_styles,
    ylabel="Precipitation [mm/month]",
    figsize=(10, 10),
)
#-----------------------------------------------------------------------------------

region_monthly_clim_corr, correction_factors = add_scalar_bias_corrected_products_to_region_clim(
    region_monthly_clim,
    reference_col=r"$P_{\mathrm{MB}}$",
    target_products=corr_targets,
    suffix=" (corr.)",
    clip_factor=None,   # or e.g. (0.25, 10.0)
)


fig, axes = plot_weighted_region_monthly_climatology(
    region_monthly_clim_corr,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=product_order_corr,
    product_styles=product_styles_corr,
    ylabel="[mm/month]",
    figsize=(11, 10),
)
# plt.show()
# Save the plot to disk
svnme = os.path.join(path_to_plots, f'basin_area_weighted_monthly_cycles_precip_over_imbie_basins_regions_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()

#%% SEASONAL CYCLES - Basin Area Weighted
region_seasonal_clim = compute_weighted_region_seasonal_climatologies(
    monthly_df_data=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights,
    basin_col="basin",
    value_col="precipitation",
    time_col="time",
    seasonal_mode="sum",   # gives mm/season
)

fig, axes = plot_weighted_region_seasonal_climatology(
    region_seasonal_clim,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=product_order,
    product_styles=product_styles,
    ylabel="[mm/season]",
    figsize=(10, 8),
)


#----------------------------------------------------------------------------------
region_seasonal_clim_corr, seasonal_corr_factors = add_scalar_bias_corrected_products_to_region_clim(
    region_seasonal_clim,
    reference_col=r"$P_{\mathrm{MB}}$",
    target_products=corr_targets,
    suffix=" (corr.)",
    clip_factor=None,   # or e.g. (0.25, 10.0) if you want to prevent extreme scaling
)

fig, axes = plot_weighted_region_seasonal_climatology(
    region_seasonal_clim_corr,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=product_order_corr,
    product_styles=product_styles_corr,
    ylabel="[mm/season]",
    figsize=(10, 9),
)

svnme = os.path.join(path_to_plots, f'basin_area_weighted_seasonal_cycles_precip_over_imbie_basins_regions_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')

gc.collect()

#%% Year to Year Variability - Basin Area Weighted
region_annual = compute_weighted_region_annual_totals(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights,
    annual_mode="sum",  # mm/year
)

fig, axes = plot_weighted_region_interannual(
    region_annual,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=product_order_corr,
    product_styles=product_styles_corr,
    ylabel="[mm/year]",
    figsize=(11, 10),
)

#----------------------------------------------------------------------------------
region_annual_corr, annual_corr_factors = add_scalar_bias_corrected_products_to_region_annual(
    region_annual,
    reference_col=r"$P_{\mathrm{MB}}$",
    target_products=corr_targets,
    suffix=" (corr.)",
    clip_factor=None,   # or e.g. (0.25, 10.0)
)

fig, axes = plot_weighted_region_interannual(
    region_annual_corr,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=product_order_corr,
    product_styles=product_styles_corr,
    ylabel="[mm/year]",
    figsize=(11, 10),
)
svnme = os.path.join(path_to_plots, f'basin_area_weighted_year_to_year_variability_precip_over_imbie_basins_regions_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()
# For the continent we’ll just use “all basins we see in the dataframe”
#%% BAR CHART WITH SPREAD

# --- Convert each product to tidy DF ---
df_pmb   = to_df(Pmb_annual)
df_pmb.rename(columns={'basin_id': 'basin',
                       'precip_mm_per_month': 'Pmb'}, inplace=True)
df_era5  = to_df(era5_annual_mean)
df_era5.rename(columns={'precipitation_annual': 'ERA5'}, inplace=True)
df_gpcp  = to_df(gpcpv3pt3_annual_mean)
df_gpcp.rename(columns={'precipitation_annual': 'GPCP v3.3'}, inplace=True)
# df_racmo = to_df(racmo_pr_annual_mean)
# df_racmo.rename(columns={'pr_annual': 'RACMO'}, inplace=True)
df_atms = to_df(atms_annual_mean)
df_atms.rename(columns={'precipitation_annual': 'ATMS'}, inplace=True)
df_mhs = to_df(mhs_annual_mean)
df_mhs.rename(columns={'precipitation_annual': 'MHS'}, inplace=True)
df_dmsp_ssmis = to_df(dmsp_ssmis_annual_mean)
df_dmsp_ssmis.rename(columns={'precipitation_annual': 'DMSP SSMIS'}, inplace=True)
df_amsr2 = to_df(amsr2_annual_mean)
df_amsr2.rename(columns={'precipitation_annual': 'AMSR2'}, inplace=True)

df_gpm_sat = to_df(gpm_sat_annual_mean)
df_gpm_sat.rename(columns={'precipitation_annual': 'GPM Satellites'}, inplace=True)

# --- Merge all together on (year, basin) ---
df = df_pmb.merge(df_era5, on=["year","basin"])
df = df.merge(df_gpcp, on=["year","basin"])
# df = df.merge(df_racmo, on=["year","basin"])
df = df.merge(df_atms, on=["year","basin"])
df = df.merge(df_mhs, on=["year","basin"])
df = df.merge(df_dmsp_ssmis, on=["year","basin"])
df = df.merge(df_amsr2, on=["year","basin"])
df = df.merge(df_gpm_sat, on=["year","basin"])

# add a "year-basin" key if you like
df["year_basin"] = df["year"].astype(str) + "-" + df["basin"].astype(str)
cols = ["Pmb", "ERA5", "GPCP v3.3", "ATMS", "MHS", "DMSP SSMIS", "AMSR2", "GPM Satellites"]
df_mean_yr_acc = df.groupby("basin")[cols].mean().reset_index()

fig, ax = plot_basin_ranked_bar_overlay(
    df_mean_yr_acc,
    basin_col="basin",
    ref_col="Pmb",
    prod_cols=cols,
    prod_labels=cols,
    figsize=(12, 5),
)
plt.show()

#----------------------------------------------------------------------------------
cols = ["ERA5", "GPCP v3.3", "ATMS", "MHS", "DMSP SSMIS", "AMSR2", "GPM Satellites"]

non_gpm_group = ["Pmb", "ERA5", "GPCP v3.3"]
gpm_group     = ["Pmb", "ATMS", "MHS", "DMSP SSMIS", "AMSR2"]  # exclude "GPM Satellites" from spread

fig, ax, spread_non_gpm, spread_gpm = plot_basin_spread_points_dual(
    df_mean_yr_acc,
    basin_col="basin",
    ref_col="Pmb",
    prod_cols=cols,
    prod_labels=cols,                 # or omit; it will auto-use col names
    product_styles=product_styles_corr,
    non_gpm_group=non_gpm_group,
    gpm_group=gpm_group,
    log_scale=True,
    ylim=(5, 2000),
    annotate_non_gpm_color="black",
    annotate_gpm_color="dimgray",
    place_key=True,
)
svnme = os.path.join(path_to_plots, f'basin_spread_points_precip_over_imbie_basins_{cde_run_dte}.png')
plt.savefig(svnme,  dpi=500, bbox_inches='tight')
gc.collect()

#%%  ----------- Scatter plot -------------

# Example usage:
products = ["ERA5", "GPCP v3.3", "ATMS", "MHS", "DMSP SSMIS", "AMSR2", "GPM Satellites"]

fig, axes = plot_pmb_scatter(
    df_mean_yr_acc,
    "Pmb",
    products,
    high_thresh=500.0,
    scale="log",
    log_min=5,                   # <-- show low GPM values
    ncols=4                      # 2 rows for 7 products
)

svnme = os.path.join(path_to_plots, f'log_scatterplot_precip_over_imbie_basins_{cde_run_dte}.png')
plt.savefig(svnme, dpi=500, bbox_inches="tight")
# plt.close(fig)
gc.collect()


#%% TREND ANALYSIS
# if basin_weights is a pandas Series with basin IDs as index:

basin_weights_ = dict(zip(basin_weights["basin"].astype(int),
                         basin_weights["weight_global"].astype(float)))

region_ts_wais = region_monthly_series_from_dict(
    monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="West Antarctica",
)

region_ts_eais = region_monthly_series_from_dict(
    monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="East Antarctica",
)

region_ts_ais = region_monthly_series_from_dict(
    monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="Antarctica",
)

product_order = [
    r"$P_{\mathrm{MB}}$",
    "ERA5",
    "GPCP v3.3",
    "ATMS",
    "MHS",
    "DMSP SSMIS",
    "AMSR2",
    "GPM Satellites",
]

fig, axes = plot_region_trend_panels(
    monthly_df_data_mmmonth,
    REGION_DEFS,
    basin_weights_,
    product_order=product_order,
    product_styles=product_styles_corr,
    use_running_mean=True,   # 13-mo RM for clean plot
    show_pmb_trend_only=True
)
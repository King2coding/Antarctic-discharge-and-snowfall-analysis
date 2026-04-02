#%%
# packages
# from codes.Monthly_mass_budget_precip_RignotBasin import YEARS
from program_utils import *

#%% Paths
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

#%%
# floating variables
YEARS = np.arange(2013,2021)#[2019, 2020]

S_tier1 = pd.read_pickle(os.path.join(basins_path, 'DataCombo_RignotBasins_LI_tier1_20260325.pkl'))

rignot_deltaS = pd.read_excel(os.path.join(basins_path, 'DataCombo_RignotBasins.xlsx'), sheet_name='Basin_Timeseries (Gt)')

# rignot_deltaS["Date"] = rignot_deltaS["Time"].apply(decimal_year_to_date).dt.strftime('%Y-%m-%d')
# rignot_deltaS['Date'] = pd.to_datetime(rignot_deltaS['Date'])
# rignot_deltaS["Year"] = pd.to_datetime(rignot_deltaS["Date"]).dt.year
# rignot_deltaS["Month"] = pd.to_datetime(rignot_deltaS["Date"]).dt.month

# 1) build month-start timestamps directly
rignot_deltaS["date"] = rignot_deltaS["Time"].apply(
    lambda x: decimal_year_to_month_start(x, mode="nearest")   # or mode="floor"
)

# 2) derive Year/Month from the date (robust)
rignot_deltaS["Year"]  = rignot_deltaS["date"].dt.year
rignot_deltaS["Month"] = rignot_deltaS["date"].dt.month

basin_cols = [c for c in rignot_deltaS.columns if c not in ("Time","date","Year","Month")]

# ---------------------------------------------------------
# FULL MONTHLY STATE SERIES (2013–2020)
# ---------------------------------------------------------
start_date = "2013-01-01"
end_date   = "2020-12-01"
full_index = pd.date_range(start=start_date, end=end_date, freq="MS")

# Group to monthly mean first (in case multiple entries per month)
dfm = (
    rignot_deltaS
    .groupby(["Year","Month"], as_index=False)[basin_cols]
    .mean()
)

dfm["date"] = pd.to_datetime(dict(year=dfm["Year"], month=dfm["Month"], day=1))
dfm = dfm.set_index("date").sort_index()

# Reindex to full monthly grid
S_full = dfm[basin_cols].reindex(full_index)
S_full.index.name = "date"

print("Total months expected:", len(full_index))
print("Observed months:", dfm.shape[0])
print("Missing months:", S_full.isna().any(axis=1).sum())

cde_run_dte = str(date.today().strftime('%Y%m%d'))

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


all_basin_ids = sorted({
    bid
    for _, ids in REGION_DEFS
    for bid in ids
})

basin_weights = compute_basin_area_weights_from_mask(
    basins,
    basin_ids=all_basin_ids
)


basin_weights_ = dict(zip(basin_weights["basin"].astype(int),
                         basin_weights["weight_global"].astype(float)))



#%%
Pmb_mm_fle = os.path.join(basins_path, 'Monthly_mass_budget_precip_RignotBasin_in_mm_20260325.nc')
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

Pmb_annual = xr.open_dataarray(os.path.join(basins_path, "Pmb_annual_2013_2020_mm_20260325.nc"))
Pmb_seasonal = xr.open_dataarray(os.path.join(basins_path, "Pmb_seasonal_mm_2013_2022_20260325.nc"))
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


#----------------------------------------------------------------------------------
era5_mm = convert_precip_to_mm_per_month(era5_basin_mean, unit="mm/day")
era5_mm = era5_mm[era5_mm['year'].isin(YEARS)].copy()

gpcpv33_mm = convert_precip_to_mm_per_month(gpcpv3pt3_basin_mean, unit="mm/day")
gpcpv33_mm = gpcpv33_mm[gpcpv33_mm['year'].isin(YEARS)].copy()

atms_mm = convert_precip_to_mm_per_month(atms_basin_mean, unit="mm/hr")
atms_mm = atms_mm[atms_mm['year'].isin(YEARS)].copy()

mhs_mm = convert_precip_to_mm_per_month(mhs_basin_mean, unit="mm/hr")
mhs_mm = mhs_mm[mhs_mm['year'].isin(YEARS)].copy()

dmsp_ssmis_mm = convert_precip_to_mm_per_month(dmsp_ssmis_basin_mean, unit="mm/hr")
dmsp_ssmis_mm = dmsp_ssmis_mm[dmsp_ssmis_mm['year'].isin(YEARS)].copy()

amsr2_mm = convert_precip_to_mm_per_month(amsr2_basin_mean, unit="mm/hr")
amsr2_mm = amsr2_mm[amsr2_mm['year'].isin(YEARS)].copy()

gpm_sat_mm = convert_precip_to_mm_per_month(gpm_sat_basin_mean, unit="mm/hr")
gpm_sat_mm = gpm_sat_mm[gpm_sat_mm['year'].isin(YEARS)].copy()

pmb_mm = convert_precip_to_mm_per_month(
        p_mm_mean_df.rename(columns={"basin_id": "basin", "precip_mm_per_month": "precipitation"}),
        unit="mm/month",
    )

# do seasonal mean


common_year_mnth_idx = (
    get_unique_year_month_index(era5_mm)
    .intersection(get_unique_year_month_index(gpcpv33_mm))
    .intersection(get_unique_year_month_index(atms_mm))
    .intersection(get_unique_year_month_index(mhs_mm))
    .intersection(get_unique_year_month_index(dmsp_ssmis_mm))
    .intersection(get_unique_year_month_index(amsr2_mm))
    .intersection(get_unique_year_month_index(gpm_sat_mm))
    .intersection(get_unique_year_month_index(pmb_mm))
)

era5_mm       = subset_to_common_year_month(era5_mm, common_year_mnth_idx)
gpcpv33_mm    = subset_to_common_year_month(gpcpv33_mm, common_year_mnth_idx)
atms_mm       = subset_to_common_year_month(atms_mm, common_year_mnth_idx)
mhs_mm        = subset_to_common_year_month(mhs_mm, common_year_mnth_idx)
dmsp_ssmis_mm = subset_to_common_year_month(dmsp_ssmis_mm, common_year_mnth_idx)
amsr2_mm      = subset_to_common_year_month(amsr2_mm, common_year_mnth_idx)
gpm_sat_mm    = subset_to_common_year_month(gpm_sat_mm, common_year_mnth_idx)
pmb_mm        = subset_to_common_year_month(pmb_mm, common_year_mnth_idx)

# align the dates by the coomon year months


monthly_df_data_mmmonth = {
    r"$P_{\mathrm{MB}}$": pmb_mm,
    "ERA5": era5_mm,
    "GPCP v3.3": gpcpv33_mm,
    "ATMS": atms_mm,
    "MHS": mhs_mm,
    "DMSP SSMIS": dmsp_ssmis_mm,
    "AMSR2": amsr2_mm,
    "GPM Satellites": gpm_sat_mm,
}

# ============================================================
# STEP 1: build raw monthly regional series once per region
# ============================================================

ts_eais_raw = build_region_monthly_series(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="East Antarctica",
)

ts_wais_raw = build_region_monthly_series(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="West Antarctica",
)

ts_ais_raw = build_region_monthly_series(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="Antarctica",
)

#%% # ============================================================
# APPROACH A: conventional seasonal anomaly
# ============================================================

# EAIS
ts_eais_seasonal_conv = build_conventional_seasonal_series_from_region_monthly(
    ts_region_monthly=ts_eais_raw,
    seasonal_mode="mean",
    drop_incomplete=True,
)

ts_eais_seasonal_conv_anom = deseasonalize_seasonal_series(ts_eais_seasonal_conv)

fig, ax = plot_region_input_timeseries(
    ts_input=ts_eais_seasonal_conv,
    region_name="East Antarctica",
    method="conventional",
    product_order=product_order,
    product_styles=product_styles_corr,
    title="EAIS — conventional seasonal input",
    ylabel="Precipitation [mm/month]",
)

fig, ax, _ = plot_region_seasonal_anomaly_timeseries(
    ts_seasonal=ts_eais_seasonal_conv,
    region_name="East Antarctica",
    method="conventional",
    product_order=product_order,
    product_styles=product_styles_corr,
    title="EAIS — conventional seasonal anomalies",
)

fig, axes, stats_dict, df_used = plot_anomaly_scatter_multi_product(
    df_in=ts_eais_seasonal_conv_anom,
    ref_col=r"$P_{\mathrm{MB}}$",
    target_cols=["ERA5", "GPCP v3.3", "ATMS", "MHS", "DMSP SSMIS", "AMSR2", "GPM Satellites"],
    compute_anomaly_inside=False,
    ncols=3,
    region_name="East Antarctica",
    share_lims=False,
)

plot_anomaly_scatter_single(
    ts_eais_seasonal_conv_anom,
    ref_col=r"ERA5",
    target_col="GPCP v3.3",
    method="seasonal_clim_monthly",
    compute_anomaly_inside=True,
    ax=None,
    title=None,
    xlabel=None,
    ylabel=None,
    lims=[-2, 2],
    equal_axes=True,
    marker_size=18,
    alpha=0.75,
)

#-------------------------------------------------------------------------------------------------------
# WAIS
ts_wais_seasonal_conv = build_conventional_seasonal_series_from_region_monthly(
    ts_region_monthly=ts_wais_raw,
    seasonal_mode="mean",
    drop_incomplete=True,
)

ts_wais_seasonal_conv_anom = deseasonalize_seasonal_series(ts_wais_seasonal_conv)

fig, ax = plot_region_input_timeseries(
    ts_input=ts_wais_seasonal_conv,
    region_name="West Antarctica",
    method="conventional",
    product_order=product_order,
    product_styles=product_styles_corr,
    title="WAIS — conventional seasonal input",
    ylabel="Precipitation [mm/month]",
)

fig, ax, _ = plot_region_seasonal_anomaly_timeseries(
    ts_seasonal=ts_wais_seasonal_conv,
    region_name="West Antarctica",
    method="conventional",
    product_order=product_order,
    product_styles=product_styles_corr,
    title="WAIS — conventional seasonal anomalies",
)
fig, axes, stats_dict, df_used = plot_anomaly_scatter_multi_product(
    df_in=ts_wais_seasonal_conv_anom,
    ref_col=r"$P_{\mathrm{MB}}$",
    target_cols=["ERA5", "GPCP v3.3", "ATMS", "MHS", "DMSP SSMIS", "AMSR2", "GPM Satellites"],
    compute_anomaly_inside=False,
    ncols=3,
    region_name="West Antarctica",
    share_lims=False,
    lims=[-10, 10],
)

#-------------------------------------------------------------------------------------------------------
# AIS
ts_ais_seasonal_conv = build_conventional_seasonal_series_from_region_monthly(
    ts_region_monthly=ts_ais_raw,
    seasonal_mode="mean",
    drop_incomplete=True,
)
ts_ais_seasonal_conv_anom = deseasonalize_seasonal_series(ts_ais_seasonal_conv)

fig, ax = plot_region_input_timeseries(
    ts_input=ts_ais_seasonal_conv,
    region_name="Antarctica",
    method="conventional",
    product_order=product_order,
    product_styles=product_styles_corr,
    title="AIS — conventional seasonal input",
    ylabel="Precipitation [mm/month]",
)

fig, ax, _ = plot_region_seasonal_anomaly_timeseries(
    ts_seasonal=ts_ais_seasonal_conv,
    region_name="Antarctica",
    method="conventional",
    product_order=product_order,
    product_styles=product_styles_corr,
    title="AIS — conventional seasonal anomalies",
)

fig, axes, stats_dict, df_used = plot_anomaly_scatter_multi_product(
    df_in=ts_ais_seasonal_conv_anom,
    ref_col=r"$P_{\mathrm{MB}}$",
    target_cols=["ERA5", "GPCP v3.3", "ATMS", "MHS", "DMSP SSMIS", "AMSR2", "GPM Satellites"],
    compute_anomaly_inside=False,
    ncols=3,
    region_name="Antarctica",
    share_lims=False,
    lims=[-3, 3],
)

#%% ============================================================
# APPROACH B: monthly anomaly relative to seasonal climatology
# ============================================================

ts_eais_raw_scm, ts_eais_scm_anom = build_region_monthly_anomalies(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="East Antarctica",
    method="seasonal_clim_monthly",
)

fig, ax = plot_region_input_timeseries(
    ts_input=ts_eais_raw_scm,
    region_name="East Antarctica",
    method="monthly",
    product_order=product_order,
    product_styles=product_styles_corr,
    title="EAIS — monthly input",
    ylabel="Precipitation [mm/month]",
)

fig, ax = plot_region_monthly_anomaly_timeseries(
    ts_anom=ts_eais_scm_anom,
    region_name="East Antarctica",
    method="conventional",
    product_order=product_order,
    product_styles=product_styles_corr,
    title="EAIS — monthly anomalies relative to seasonal climatology",
)

fig, axes, stats_dict, df_used = plot_anomaly_scatter_multi_product(
    df_in=ts_eais_scm_anom,
    ref_col=r"$P_{\mathrm{MB}}$",
    target_cols=["ERA5", "GPCP v3.3", "ATMS", "MHS", "DMSP SSMIS", "AMSR2", "GPM Satellites"],
    compute_anomaly_inside=False,
    ncols=3,
    region_name="East Antarctica",
    share_lims=False,
)
gc.collect()

#%% # ============================================================
# APPROACH C: rolling seasonal-mean monthly anomaly
# ============================================================

ts_eais_raw_roll, ts_eais_roll_anom = build_region_monthly_anomalies(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    region_defs=REGION_DEFS,
    basin_weights=basin_weights_,
    region_name="East Antarctica",
    method="rolling",
    min_periods=3,
)

fig, ax = plot_region_input_timeseries(
    ts_input=ts_eais_raw_roll,
    region_name="East Antarctica",
    method="rolling",
    product_order=product_order,
    product_styles=product_styles_corr,
    title="EAIS — monthly input for rolling method",
    ylabel="Precipitation [mm/month]",
)

fig, ax = plot_region_monthly_anomaly_timeseries(
    ts_anom=ts_eais_roll_anom,
    region_name="East Antarctica",
    method="rolling",
    product_order=product_order,
    product_styles=product_styles_corr,
    title="EAIS — rolling seasonal-mean monthly anomalies",
)

fig, axes, stats_dict, df_used = plot_anomaly_scatter_multi_product(
    df_in=ts_eais_roll_anom,
    ref_col=r"$P_{\mathrm{MB}}$",
    target_cols=["ERA5", "GPCP v3.3", "ATMS", "MHS", "DMSP SSMIS", "AMSR2", "GPM Satellites"],
    compute_anomaly_inside=False,
    ncols=3,
    region_name="East Antarctica",
    share_lims=False,
)

gc.collect()

#%% Diagnostic 1 — Count “filled GRACE basin-months” per region and overlay on anomaly plot

# 1A) Build filled-mask counts from your existing S_full and S_tier1

# --- inputs you already have ---
# S_full, S_tier1 (or S_tier2)
# REGION_DEFS, id2name
# monthly_df_data_mmmonth, basin_weights_  (dict basin->weight)

# 1) compute filled counts
filled_count_df, filled_mask = compute_filled_counts_by_region(
    S_full=S_full,
    S_filled=S_tier1,          # or S_tier2
    REGION_DEFS=REGION_DEFS,
    id2name=id2name,
)

# 1b) compute gap-spanning dS counts (often more diagnostic)
gapspan_df = compute_gapspanning_dS_counts_by_region(
    S_full=S_full,
    REGION_DEFS=REGION_DEFS,
    id2name=id2name,
)

# 2) compute region TS (your existing function)
ts_wais = region_monthly_series_from_dict(monthly_df_data_mmmonth, REGION_DEFS, basin_weights_, "West Antarctica")
ts_eais = region_monthly_series_from_dict(monthly_df_data_mmmonth, REGION_DEFS, basin_weights_, "East Antarctica")
ts_ais  = region_monthly_series_from_dict(monthly_df_data_mmmonth, REGION_DEFS, basin_weights_, "Antarctica")

# 3) plot (choose either filled_count_df or gapspan_df)
fig, _ = plot_region_anom_with_gapcounts(
    ts_region_mmmonth=ts_ais,
    region_name="Antarctica",
    gap_counts=gapspan_df["Antarctica"],
    product_order=product_order,
    product_styles=product_styles_corr,
    show_13mo_rm=True,
    plot_trend_products=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP v3.3"),
)

fig, _ = plot_region_anom_with_gapcounts(
    ts_region_mmmonth=ts_wais,
    region_name="West Antarctica",
    gap_counts=gapspan_df["West Antarctica"],
    product_order=product_order,
    product_styles=product_styles_corr,
    show_13mo_rm=True,
    plot_trend_products=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP v3.3"),
)

fig, _ = plot_region_anom_with_gapcounts(
    ts_region_mmmonth=ts_eais,
    region_name="East Antarctica",
    gap_counts=gapspan_df["East Antarctica"],
    product_order=product_order,
    product_styles=product_styles_corr,
    show_13mo_rm=True,
    plot_trend_products=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP v3.3"),
)

# ---------------------------------------------------------
# 3) Run it for PMB, GPCP, ERA5 across Antarctica/WAIS/EAIS
# ---------------------------------------------------------
trend_df = compute_region_trends_bars(
    monthly_df_data_mmmonth=monthly_df_data_mmmonth,
    REGION_DEFS=REGION_DEFS,
    basin_weights=basin_weights_,
    regions=("Antarctica", "West Antarctica", "East Antarctica"),
    products=(r"$P_{\mathrm{MB}}$", "GPCP v3.3", "ERA5"),
)

fig, ax = plot_region_trend_barchart(
    trend_df,
    product_styles=product_styles_corr,  # <-- uses your colors
    title="Trend comparison (2013–2022): Antarctica vs WAIS vs EAIS",
    ylabel="Trend (mm/year)",
)

#%% Diagnostic -2

# basin_weights_ is your dict(basin_id -> weight_global) already created
# dS_basin, D_basin, BM_basin, SUB_basin must be xr.DataArray(date, basin_id)

dst = os.path.join(basins_path, 'bedmap3_basins_0.1deg.tif')

# Read the remapped file
basins_imbie = xr.open_dataarray(dst)
if not basins_imbie.rio.crs:
    basins_imbie = basins_imbie.rio.write_crs(CRS.from_proj4(crs_stereo))
basin_transform = basins_imbie.rio.transform()
height, width = basins_imbie.shape[1:]
xmin, ymax = basin_transform.c, basin_transform.f
xres, yres = basin_transform.a, -basin_transform.e
xmax = xmin + width * xres
ymin = ymax - height * yres
print(f"Basin grid: width={width}, height={height}, xres={xres}, yres={yres}")
basin_bounds = (xmin, xmax, ymin, ymax)  # (minx, maxx, miny, maxy)
print(f"Basin bounds: {basin_bounds}")
basins_imbie = basins_imbie.where((basins_imbie > 1) & (basins_imbie.notnull()))

# --- Helper: carry mapping in DataArray coords (nice for groupby) ---

# Create a labelled copy where 'basin_name' is a coordinate matching each pixel
name_lookup = xr.DataArray(
    np.vectorize(lambda v: id2name.get(int(v), "NA") if np.isfinite(v) else "NA")(basins_imbie.values),
    coords=basins_imbie.coords, dims=basins_imbie.dims
)
name_lookup.name = "basin_name"
basin_imbie_with_name_map = xr.Dataset(dict(basin_id=basins_imbie, basin_name=name_lookup))

# 0) Squeeze band, mask invalid, and (optionally) drop Islands (ID==1)
basin_id = basin_imbie_with_name_map['basin_id']
basin_name = basin_imbie_with_name_map['basin_name']

rignot_deltaS = pd.read_pickle(os.path.join(basins_path, 'DataCombo_RignotBasins_LI_tier1_20260226.pkl'))

# rignot_deltaS["Date"] = rignot_deltaS["Time"].apply(decimal_year_to_date).dt.strftime('%Y-%m-%d')
# rignot_deltaS['Date'] = pd.to_datetime(rignot_deltaS['Date'])
rignot_deltaS["Year"] = rignot_deltaS.index.year
# pd.to_datetime(rignot_deltaS["Date"]).dt.year
rignot_deltaS["Month"] = rignot_deltaS.index.month
# pd.to_datetime(rignot_deltaS["Date"]).dt.month

# Keep only 2013–2020 rows
# rignot_deltaS = rignot_deltaS[rignot_deltaS["Year"].isin(YEARS)].copy()

# Identify basin columns
basin_cols = [c for c in rignot_deltaS.columns if c not in ("Time","Date","Year","Month")]

# ΔS(m) = S(t_{m+1}) - S(t_m) (per basin)
# First, ensure monthly step by grouping per (Year, Month); if multiple epochs per month, average
dfm = rignot_deltaS.groupby(["Year","Month"], as_index=False)[basin_cols].mean()
# Create a "month index" for easy shift
dfm["date"] = pd.to_datetime(dict(year=dfm["Year"], month=dfm["Month"], day=1))
dfm = dfm.sort_values("date")

# Compute difference forward in time per basin
# ΔS for a given month is next_month_state - current_month_state
dfm_shift = dfm.copy()
dfm_shift[basin_cols] = dfm[basin_cols].shift(-1)
dS = dfm_shift[basin_cols] - dfm[basin_cols]
dS["date"] = dfm["date"]  # assign ΔS to the current month index

# Keep only months that have a valid "next month" within the range
dS = dS.dropna(subset=basin_cols, how="all")

# Melt to long format
dS_long = dS.melt(id_vars="date", var_name="basin", value_name="dS_Gt")
dS_long = dS_long.dropna(subset=["dS_Gt"]).reset_index(drop=True)

print(f"[David] ΔS computed for months {dS_long['date'].min().date()} to {dS_long['date'].max().date()}")

# create an xarray data based on the df above mapping basin mae to the GT values
# ds_long_xr = dS_long.set_index(["date", "basin"]).to_xarray()


# Mask out non-basin pixels
basin_id = basin_id.where(np.isfinite(basin_id))
basin_name = basin_name.where(basin_name != 'NA')

# If you want to exclude islands up front:
basin_id = basin_id.where(basin_id != 1)
basin_name = basin_name.where(basin_id.notnull())
# Carry the basin_id and basin_name into the dataframe for mapping
dS_long_df = generate_basin_id_mapping(basin_id, basin_name, dS_long)

# Build the time-by-space raster

attributes = {
    'description': 'Monthly basin-total ΔS painted to all pixels of each basin (not areal density). ',
    'long_name': 'Monthly basin mass anomaly change',
    'units': 'Gt/month',
    'source': 'Computed from David Rignot basin Excel (DataCombo_RignotBasins.xlsx)',
    'note': 'Islands (ID=1) excluded; names matched via modal name per ID from the basin grid.'
}

dS_raster = create_basin_xrr(basin_id[0], basin_name[0], dS_long_df, 
                             'dS_Gt', attributes, "deltaS_Gt_per_month")

#--- Discharge ---
discharge_data = pd.read_excel(os.path.join(basins_path, 'antarctic_discharge_2013-2022_imbie.xlsx'), sheet_name=['Discharge (Gt yr^-1)', 'Summary'])
basin_discharge = discharge_data['Discharge (Gt yr^-1)']
basin_cols_ = basin_cols.copy()
basin_cols_ = [b.replace('-f', '-F') if b == 'Ep-f' else b for b in basin_cols_]

basin_discharge = basin_discharge.loc[basin_discharge['IMBIE basin'].isin(basin_cols_), ['IMBIE basin'] + [str(yr) for yr in YEARS]].rename(columns={'IMBIE basin': 'basin'})

# --- Basal melt ---
basal_melt = discharge_data['Summary']
basal_melt = basal_melt.loc[basal_melt['IMBIE basin'].isin(basin_cols_), ['IMBIE basin', 'Basal melt total Gt/yr']].rename(columns={'IMBIE basin': 'basin'})


# Convert annual (Gt/yr) → monthly (Gt/mo), then expand to months
D_month = annual_to_monthly_long(basin_discharge,YEARS, "discharge_Gt")
D_month_df = generate_basin_id_mapping(basin_id, basin_name, D_month)

# Convert annual to monthly
basal_melt["basal_Gt_per_month"] = basal_melt["Basal melt total Gt/yr"] / 12.0

# Expand to months for requested years
rows = []
for _, row in basal_melt.iterrows():
    for y in YEARS:
        for m in range(1, 13):
            rows.append({
                "date": pd.to_datetime(f"{y}-{m:02d}-01"),
                "basin": row["basin"],
                "basal_Gt": row["basal_Gt_per_month"]
            })
B_month = pd.DataFrame(rows)
B_month_df = generate_basin_id_mapping(basin_id, basin_name, B_month)

Q_month = D_month.merge(B_month, on=["date","basin"], how="outer")
print(f"[Chad] Qnet monthly rows: {len(Q_month)}")

# create x array for discharge and basal melt separately
attributes = {
    'description': 'Monthly basin-total discharge + basal melt painted to all pixels of each basin (not areal density). ',
    'long_name': 'Monthly basin discharge ',
    'units': 'Gt/month ',
    'source': 'Computed from Chad’s discharge Excel (antarctic_discharge_2013-2022_imbie.xlsx)',
    'note': 'Islands (ID=1) excluded; names matched via modal name per ID from the basin grid.'
}
discharge_raster = create_basin_xrr(basin_id[0], basin_name[0], 
                                    D_month_df, 'discharge_Gt', 
                                    attributes, "discharge_Gt_per_month")



attributes = {
    'description': 'Monthly basin-total basal melt painted to all pixels of each basin (not areal density). ',
    'long_name': 'Monthly basin basal melt ',
    'units': 'Gt/month ',
    'source': 'Computed from Chad’s discharge Excel (antarctic_discharge_2013-2022_imbie.xlsx)',
    'note': 'Islands (ID=1) excluded; names matched via modal name per ID from the basin grid.'
}
# plot one of the time stamps
basal_melt_raster = create_basin_xrr(basin_id[0], basin_name[0], B_month_df, 
                                     'basal_Gt', attributes, "basal_Gt_per_month")

#---- Subblim -----
subblm_data = os.path.join(racmo_path, "subltot_monthlyS_ANT11_RACMO2.4p1_ERA5_2013_2022.nc")

racmo_on_imbie = xr.open_dataarray(subblm_data)


#----------------------------------------------------------------------------

# dS_basin = dS_raster.copy()
# D_basin = discharge_raster.copy()
# BM_basin = basal_melt_raster.copy()
# SUB_basin = racmo_on_imbie.copy()

# --- 1) Convert rasters -> basin series (date, basin_id) --------------------

# dS_raster, discharge_raster, basal_melt_raster are (date, y, x) with basin_id(y,x)
dS_basin = painted_to_basin_series(dS_raster)          # (date, basin_id)
D_basin  = painted_to_basin_series(discharge_raster)   # (date, basin_id)
BM_basin = painted_to_basin_series(basal_melt_raster)  # (date, basin_id)

# --- SUB: RACMO subltot is areal field (kg m-2 per month) -------------------
SUB = racmo_on_imbie.copy()

# make time axis month-start and rename to 'date'
# rename and force month-start timestamps safely
SUB = racmo_on_imbie.rename(time="date")
SUB = SUB.assign_coords(date=pd.to_datetime(SUB["date"].values).to_period("M").to_timestamp(how="start"))
# mask to basins and carry basin_id coord
SUBm = mask_to_basins(SUB, dS_raster["basin_id"])  # now SUBm has basin_id(y,x) coord

# convert areal field -> basin totals (Gt/month) -> (date, basin_id)
SUB_basin = areal_to_basin_series(
    SUBm,
    pixel_area_m2=10000.0 * 10000.0,   # your grid is 10 km × 10 km
    units="kg m-2"                     # matches your SUB attrs
)
SUB_basin.name = "subl_Gt_per_month"


#--- 2) Plot budget terms for Antarctica, West Antarctica, and East Antarctica:
fig, _, df_ais_terms = plot_budget_terms_region(
    "Antarctica", REGION_DEFS, basin_weights_,
    dS_basin=dS_basin, D_basin=D_basin, BM_basin=BM_basin, SUB_basin=SUB_basin,
    smooth_13mo=False
)

fig, _, df_wais_terms = plot_budget_terms_region(
    "West Antarctica", REGION_DEFS, basin_weights_,
    dS_basin=dS_basin, D_basin=D_basin, BM_basin=BM_basin, SUB_basin=SUB_basin
)

fig, _, df_eais_terms = plot_budget_terms_region(
    "East Antarctica", REGION_DEFS, basin_weights_,
    dS_basin=dS_basin, D_basin=D_basin, BM_basin=BM_basin, SUB_basin=SUB_basin
)

#----------------
# basin_weights_ should be dict: {basin_id: weight_global}
# REGION_DEFS as you defined earlier

for reg in ["Antarctica", "West Antarctica", "East Antarctica"]:
    fig, axes, df_terms = plot_budget_terms_region_3x1(
        reg, REGION_DEFS, basin_weights_,
        dS_basin=dS_basin, D_basin=D_basin, BM_basin=BM_basin, SUB_basin=SUB_basin,
        smooth_13mo=False, figsize=(14, 8)
    )
    # optional save
    # fig.savefig(f"{reg.replace(' ', '_')}_Pmb_terms_3x1.png", dpi=300, bbox_inches="tight")

for reg in ["Antarctica", "West Antarctica", "East Antarctica"]:
    fig, axes, df_terms, gc = plot_budget_terms_region_3x1_with_gapcounts(
        region_name=reg,
        REGION_DEFS=REGION_DEFS,
        basin_weights_dict=basin_weights_,
        dS_basin=dS_basin,
        D_basin=D_basin,
        BM_basin=BM_basin,
        SUB_basin=SUB_basin,
        gap_counts=gapspan_df[reg],     # ✅ best for ΔS
        smooth_13mo=False,
        figsize=(14, 8),
    )

#%%
# Unsmooothed (recommended for “real” stats)
plot_region_anom_scatter_panels_consistent(ts_ais[ts_ais.index.year <= 2020],  "Antarctica",
    ref_col=r"$P_{\mathrm{MB}}$", target_cols=("GPCP v3.3","ERA5"),
    colors=("orange","blue"), smooth_13mo=False, lims=(-10,10))

_ = plot_region_anom_scatter_panels_consistent(ts_wais[ts_wais.index.year <= 2020], "West Antarctica",
    ref_col=r"$P_{\mathrm{MB}}$", target_cols=("GPCP v3.3","ERA5"),
    colors=("orange","blue"), smooth_13mo=False, lims=(-10,10))

plot_region_anom_scatter_panels_consistent(ts_eais[ts_eais.index.year <= 2020], "East Antarctica",
    ref_col=r"$P_{\mathrm{MB}}$", target_cols=("GPCP v3.3","ERA5"),
    colors=("orange","blue"), smooth_13mo=False, lims=(-5,5))

# Optional “smoothed view” version
_ = plot_region_anom_scatter_panels_consistent(ts_ais[ts_ais.index.year <= 2020], "Antarctica",
    ref_col=r"$P_{\mathrm{MB}}$", target_cols=("GPCP v3.3","ERA5"),
    colors=("orange","blue"), smooth_13mo=True, lims=(-2,2))

_ = plot_region_anom_scatter_panels_consistent(ts_wais[ts_wais.index.year <= 2020], "West Antarctica",
    ref_col=r"$P_{\mathrm{MB}}$", target_cols=("GPCP v3.3","ERA5"),
    colors=("orange","blue"), smooth_13mo=True, lims=(-2,2))

_ = plot_region_anom_scatter_panels_consistent(ts_eais[ts_eais.index.year <= 2020], "East Antarctica",
    ref_col=r"$P_{\mathrm{MB}}$", target_cols=("GPCP v3.3","ERA5"),
    colors=("orange","blue"), smooth_13mo=True, lims=(-2,2))

#%%
# ts_ais, ts_wais, ts_eais already computed
scatter_seasonal_panels_raw(
    ts_ais, "Antarctica",
    ref_col=r"$P_{\mathrm{MB}}$",
    targets=("GPCP v3.3", "ERA5"),
    colors=("orange", "blue"),
    lims=None  # or set e.g. (-20, 20) if you want fixed bounds
)

scatter_seasonal_panels_raw(ts_wais, "West Antarctica",
                            ref_col=r"$P_{\mathrm{MB}}$",
                            targets=("GPCP v3.3","ERA5"),
                            colors=("orange","blue"))

scatter_seasonal_panels_raw(ts_eais, "East Antarctica",
                            ref_col=r"$P_{\mathrm{MB}}$",
                            targets=("GPCP v3.3","ERA5"),
                            colors=("orange","blue"))


#%% Diagnosing anamaly issues: EAIS perspective
#%% =========================================================
# Diagnose which EAIS basins may be driving the regional jump
# ---------------------------------------------------------
# What this block does:
# 1) loads the corrected Tier-1 GRACE state series
# 2) computes monthly ΔS
# 3) builds EAIS basin-level ΔS anomalies
# 4) builds the summed EAIS ΔS anomaly
# 5) computes each basin's fractional contribution to EAIS area/weight
# 6) plots:
#    - EAIS regional ΔS anomaly
#    - basin-level ΔS anomalies
#    - basin contributions to EAIS ΔS anomaly through time
#    - bar plot of basin weights
# 7) prints a ranked table of which basins contribute most around 2016–2018
# =========================================================


# ---------------------------------------------------------
# USER PATHS / INPUTS
# ---------------------------------------------------------
basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'

# Use the newly corrected Tier-1 pickle you just created
tier1_file = os.path.join(basin_path, "DataCombo_RignotBasins_LI_tier1_20260325.pkl")

# EAIS basin labels based on your basin map / previous code
eais_basins = ["A-Ap", "Ap-B", "B-C", "C-Cp", "Cp-D", "D-Dp", "Dp-E", "E-Ep", "Jpp-K", "K-A"]

# Optional: focus window for the suspicious period
focus_start = "2016-01-01"
focus_end   = "2018-12-01"


#%% ---------------------------------------------------------
# STEP 1 — LOAD CORRECTED TIER-1 STATE SERIES
# ---------------------------------------------------------
S_tier1 = pd.read_pickle(tier1_file).copy()

# Keep only the basins that truly exist in the dataframe
eais_basins = [b for b in eais_basins if b in S_tier1.columns]
print("EAIS basins found:", eais_basins)

# ---------------------------------------------------------
# OPTIONAL — BASIN-LEVEL RAW ΔS
# matching the PMB raw basin layout/style
# ---------------------------------------------------------
dS_eais_raw = S_tier1[eais_basins].copy()

n = len(eais_basins)
ncols = 2
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(16, 3.8 * nrows),
    sharex=True,
    # sharey=True
)

axes = np.atleast_1d(axes).ravel()

# common y-limits
# ymin, ymax = -250, 250

# optional auto:
# vals = dS_eais_raw[eais_basins].to_numpy().ravel()
# vals = vals[np.isfinite(vals)]
# ymin = np.floor(vals.min() / 10) * 10
# ymax = np.ceil(vals.max() / 10) * 10

for ax, basin in zip(axes, eais_basins):
    ax.plot(
        dS_eais_raw.index,
        dS_eais_raw[basin],
        color="tab:blue",
        lw=1.4
    )
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvspan(
        pd.Timestamp(focus_start),
        pd.Timestamp(focus_end),
        color="red",
        alpha=0.08
    )
    ax.set_title(basin, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    # ax.set_ylim([ymin, ymax])

for ax in axes[len(eais_basins):]:
    ax.set_visible(False)

fig.suptitle(
    "EAIS basin-level raw monthly ΔS (corrected Tier-1)",
    fontsize=16,
    fontweight="bold",
    y=1.02
)
fig.supxlabel("Date", fontsize=14, fontweight="bold")
fig.supylabel(r"$\Delta S$ (Gt/month)", fontsize=14, fontweight="bold")

for ax in axes:
    if ax.get_visible():
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------
# STEP 2 — COMPUTE MONTHLY ΔS
# ---------------------------------------------------------
# ΔS(t) = S(t) - S(t-1)
# After diff(), the first time step becomes NaN and is dropped
dS_tier1 = S_tier1.diff().dropna()


#%% ---------------------------------------------------------
# STEP 3 — BUILD BASIN-LEVEL MONTHLY ANOMALIES
# ---------------------------------------------------------
# For each basin, remove its monthly climatology so we can focus
# on departures from normal seasonal behavior.
dS_eais_anom = dS_tier1[eais_basins].copy()

for basin in eais_basins:
    monthly_clim = dS_eais_anom[basin].groupby(dS_eais_anom.index.month).transform("mean")
    dS_eais_anom[basin] = dS_eais_anom[basin] - monthly_clim


#%% ---------------------------------------------------------
# STEP 4 — BUILD THE SUMMED EAIS ΔS SERIES AND ANOMALY
# ---------------------------------------------------------
# This gives us the total EAIS ΔS from the selected basins only.
eais_dS = dS_tier1[eais_basins].sum(axis=1)

# Remove the EAIS monthly climatology to get the regional anomaly
eais_dS_anom = eais_dS - eais_dS.groupby(eais_dS.index.month).transform("mean")


#%% ---------------------------------------------------------
# STEP 5 — COMPUTE STATIC BASIN WEIGHTS FOR EAIS
# ---------------------------------------------------------
# Here we use the basin mask directly to compute basin areas/weights,
# then convert the global basin weights into EAIS-normalized weights.
# ---------------------------------------------------------

# Load basin mask
basins = xr.open_dataarray(os.path.join(basins_path, 'bedmap3_basins_0.1deg.tif'))

# Keep only valid basin IDs
basins = basins.where((basins > 1) & (basins.notnull()))

# Basin IDs used across all defined regions
all_basin_ids = sorted({
    bid
    for _, ids in REGION_DEFS
    for bid in ids
})

# Compute basin weights from basin mask
# Expected output columns include:
# basin, n_cells, area_m2, area_km2, weight_global
basin_weights = compute_basin_area_weights_from_mask(
    basins,
    basin_ids=all_basin_ids
)

# Basin ID -> basin name mapping
basin_id_to_name = {
    2: "A-Ap",
    3: "Ap-B",
    4: "B-C",
    5: "C-Cp",
    6: "Cp-D",
    7: "D-Dp",
    8: "Dp-E",
    9: "E-Ep",
    10: "Ep-f",
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

# Make working copy
weights_df = basin_weights.copy()

# Ensure basin IDs are integer
weights_df["basin"] = weights_df["basin"].astype(int)

# Add basin-name column
weights_df["basin_name"] = weights_df["basin"].map(basin_id_to_name)

# Keep only EAIS basins
weights_eais = weights_df[weights_df["basin_name"].isin(eais_basins)].copy()

# Normalize weights so they sum to 1 within EAIS only
weights_eais["weight_eais"] = (
    weights_eais["weight_global"] / weights_eais["weight_global"].sum()
)

# Sort for readability
weights_eais = weights_eais.sort_values("weight_eais", ascending=False).reset_index(drop=True)

print(weights_eais[["basin", "basin_name", "area_km2", "weight_global", "weight_eais"]])


#%% ---------------------------------------------------------
# STEP 6 — QUANTIFY WHICH BASINS DRIVE THE 2016–2018 PERIOD
# ---------------------------------------------------------

focus_mask = (dS_eais_anom.index >= focus_start) & (dS_eais_anom.index <= focus_end)
focus_df = dS_eais_anom.loc[focus_mask, eais_basins].copy()

summary_rows = []

for basin in eais_basins:
    w = weights_eais.loc[weights_eais["basin_name"] == basin, "weight_eais"].values
    w = w[0] if len(w) else np.nan

    s = focus_df[basin].dropna()

    summary_rows.append({
        "basin": basin,
        "weight_eais": w,
        "mean_abs_anom_2016_2018": np.mean(np.abs(s)),
        "max_abs_anom_2016_2018": np.max(np.abs(s)),
        "var_anom_2016_2018": np.var(s),
        "weighted_mean_abs_anom": w * np.mean(np.abs(s)) if np.isfinite(w) else np.nan
    })

summary_df = pd.DataFrame(summary_rows).sort_values(
    "weighted_mean_abs_anom", ascending=False
).reset_index(drop=True)

print("\nRanked basin influence summary for 2016–2018:")
print(summary_df.round(3))


#%% ---------------------------------------------------------
# STEP 7 — TIME-VARYING FRACTIONAL CONTRIBUTION TO EAIS ANOMALY
# ---------------------------------------------------------
# This helps show which basin is responsible at each time.
# We define contribution fraction relative to the total absolute EAIS anomaly.
#
# contribution_frac = basin_anom / sum(abs(all basin anomalies))
#
# This is not an area weight; it is a diagnostic measure of which basin
# dominates the regional anomaly at a given month.
# ---------------------------------------------------------
denom = dS_eais_anom.abs().sum(axis=1).replace(0, np.nan)
contrib_frac = dS_eais_anom.divide(denom, axis=0)


#%% ---------------------------------------------------------
# STEP 8 — PLOT 1: SUMMED EAIS ΔS ANOMALY
# ---------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(eais_dS_anom.index, eais_dS_anom, color="k", lw=2, label="EAIS summed ΔS anomaly")
plt.axhline(0, color="gray", ls="--", lw=1)
plt.axvspan(pd.Timestamp(focus_start), pd.Timestamp(focus_end), color="red", alpha=0.08,
            label="Focus period (2016–2018)")
plt.title("EAIS summed ΔS anomaly from corrected Tier-1 GRACE series")
plt.ylabel("ΔS anomaly (Gt/month)")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------
# ---------------------------------------------------------
# STEP 9 — PLOT 2: BASIN-LEVEL ΔS ANOMALIES
# matching the PMB basin anomaly layout/style
# ---------------------------------------------------------
n = len(eais_basins)
ncols = 2
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(16, 3.8 * nrows),
    sharex=True,
    sharey=True
)

axes = np.atleast_1d(axes).ravel()

# choose one common y-limit across all basins
# manual:
ymin, ymax = -40, 50

# optional automatic version instead:
# vals = dS_eais_anom[eais_basins].to_numpy().ravel()
# vals = vals[np.isfinite(vals)]
# ymin = np.floor(vals.min() / 10) * 10
# ymax = np.ceil(vals.max() / 10) * 10

for ax, basin in zip(axes, eais_basins):
    ax.plot(
        dS_eais_anom.index,
        dS_eais_anom[basin],
        color="tab:blue",
        lw=1.4
    )
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvspan(
        pd.Timestamp(focus_start),
        pd.Timestamp(focus_end),
        color="red",
        alpha=0.08
    )
    ax.set_title(basin, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_ylim([ymin, ymax])

# hide unused panels
for ax in axes[len(eais_basins):]:
    ax.set_visible(False)

fig.suptitle(
    "EAIS basin-level ΔS anomalies (corrected Tier-1)",
    fontsize=16,
    fontweight="bold",
    y=1.02
)
fig.supxlabel("Date", fontsize=14, fontweight="bold")
fig.supylabel(r"$\Delta S$ anomaly (Gt/month)", fontsize=14, fontweight="bold")

# rotate x tick labels
for ax in axes:
    if ax.get_visible():
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------
# STEP 10 — PLOT 3: TIME-VARYING BASIN CONTRIBUTIONS
# ---------------------------------------------------------
# This stacked-area view shows which basins dominate the anomaly budget over time.
# Positive and negative values are preserved.
fig, ax = plt.subplots(figsize=(13, 6))

for basin in eais_basins:
    ax.plot(contrib_frac.index, contrib_frac[basin], lw=1.2, label=basin)

ax.axhline(0, color="gray", ls="--", lw=1)
ax.axvspan(pd.Timestamp(focus_start), pd.Timestamp(focus_end), color="red", alpha=0.08)
ax.set_title("Relative basin contributions to EAIS ΔS anomaly through time")
ax.set_ylabel("Contribution fraction")
ax.set_xlabel("Date")
ax.legend(ncol=5, fontsize=9, frameon=False)
plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------
# STEP 11 — PLOT 4: STATIC BASIN WEIGHTS WITHIN EAIS
# ---------------------------------------------------------
# These are the EAIS-normalized basin weights, i.e. they sum to 1
# across only the EAIS basins.
# ---------------------------------------------------------

weights_plot = weights_eais.copy().sort_values("weight_eais", ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(weights_plot["basin_name"], weights_plot["weight_eais"])
plt.title("Static basin weights within EAIS")
plt.ylabel("EAIS-normalized weight")
plt.xlabel("Basin")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------
# STEP 12 — PLOT 5: WEIGHTED BASIN INFLUENCE IN 2016–2018
# ---------------------------------------------------------
# This helps combine anomaly strength and basin importance in one view.
summary_plot = summary_df.sort_values("weighted_mean_abs_anom", ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(summary_plot["basin"], summary_plot["weighted_mean_abs_anom"])
plt.title("Weighted basin anomaly influence during 2016–2018")
plt.ylabel("Weight × mean absolute anomaly")
plt.xlabel("Basin")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------
# STEP 13 — OPTIONAL: PRINT TOP BASINS ONLY
# ---------------------------------------------------------
top_n = 5
print(f"\nTop {top_n} candidate basins driving EAIS anomaly behavior (2016–2018):")
print(summary_df.head(top_n).round(3))



#%% =========================================================
# EAIS PMB basin-by-basin precipitation and anomaly diagnostics
# =========================================================
# Assumes you already have:
#   pmb_mm  -> DataFrame with columns:
#              ['year', 'month', 'basin', 'precipitation', 'time']
#
# This block will:
# 1) subset EAIS basins
# 2) pivot to time x basin table
# 3) plot raw monthly PMB precipitation for each EAIS basin
# 4) compute basin-level monthly anomalies
# 5) plot basin-level anomaly panels
# 6) compute summed EAIS anomaly
# 7) rank basin influence during a suspicious focus period
# =========================================================


# ---------------------------------------------------------
# STEP 1 — USER SETTINGS
# ---------------------------------------------------------
# EAIS basin IDs from your agreed mapping
EAIS_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 18, 19]

# Basin ID -> name
basin_id_to_name = {
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

# Suspicious period to inspect
focus_start = "2015-07-01"
focus_end   = "2018-12-01"

# Optional static EAIS-normalized weights from earlier work
# If you already have weights_eais with columns ['basin','basin_name','weight_eais'],
# this block will use it. Otherwise it will compute equal weights as fallback.
use_weights_if_available = True


# ---------------------------------------------------------
# STEP 2 — SUBSET PMB TO EAIS BASINS
# ---------------------------------------------------------
pmb_eais = pmb_mm.copy()

# make sure time is datetime
pmb_eais["time"] = pd.to_datetime(pmb_eais["time"])

# keep only EAIS basins
pmb_eais = pmb_eais[pmb_eais["basin"].isin(EAIS_IDS)].copy()

# add basin names
pmb_eais["basin_name"] = pmb_eais["basin"].map(basin_id_to_name)

# keep only required columns
pmb_eais = pmb_eais[["time", "year", "month", "basin", "basin_name", "precipitation"]].copy()

# sort for safety
pmb_eais = pmb_eais.sort_values(["time", "basin"]).reset_index(drop=True)

print("EAIS PMB rows:", len(pmb_eais))
print("EAIS basins found:", sorted(pmb_eais["basin"].unique()))


# ---------------------------------------------------------
# STEP 3 — BUILD TIME x BASIN TABLE FOR RAW PMB PRECIP
# ---------------------------------------------------------
# rows = monthly times
# cols = basin names
pmb_eais_wide = (
    pmb_eais
    .pivot_table(index="time", columns="basin_name", values="precipitation", aggfunc="mean")
    .sort_index()
)

# order columns consistently by basin ID order
eais_basin_names = [basin_id_to_name[i] for i in EAIS_IDS if basin_id_to_name[i] in pmb_eais_wide.columns]
pmb_eais_wide = pmb_eais_wide[eais_basin_names]

print(pmb_eais_wide.head())


# ---------------------------------------------------------
# STEP 4 — PLOT RAW MONTHLY PMB PRECIPITATION PER EAIS BASIN
# ---------------------------------------------------------
# GRID PLOT OF EAIS BASIN-LEVEL RAW MONTHLY PMB PRECIPITATION
# ---------------------------------------------------------
# Why this version:
# 1) puts basins in a 5x2 grid so comparisons are easier
# 2) uses the same y-limits across all panels
# 3) keeps the suspicious focus period shaded
# 4) removes unused axes automatically if needed

# --- choose grid shape
n_basins = len(eais_basin_names)
ncols = 2
nrows = math.ceil(n_basins / ncols)

# --- make figure
fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(18, 3.8 * nrows),
    sharex=True,
    sharey=True
)

# flatten axes for easy looping
axes = np.array(axes).ravel()

# --- common y-limits across all basins
# you can keep your manual choice, or compute from data
ymin = -50
ymax = 80

# alternative automatic option:
# ymin = np.floor(np.nanmin(pmb_eais_wide[eais_basin_names].values) / 10) * 10
# ymax = np.ceil(np.nanmax(pmb_eais_wide[eais_basin_names].values) / 10) * 10

for i, basin_name in enumerate(eais_basin_names):
    ax = axes[i]

    ax.plot(
        pmb_eais_wide.index,
        pmb_eais_wide[basin_name],
        color="tab:blue",
        lw=1.5
    )

    # highlight suspicious period
    ax.axvspan(
        pd.Timestamp(focus_start),
        pd.Timestamp(focus_end),
        color="red",
        alpha=0.08
    )

    # zero line
    ax.axhline(0, color="gray", ls="--", lw=0.8)

    # panel title
    ax.set_title(basin_name, fontsize=13, fontweight="bold")

    # common y-limits
    ax.set_ylim([ymin, ymax])

    # light grid
    ax.grid(True, alpha=0.25)

# --- hide unused panels
for j in range(n_basins, len(axes)):
    axes[j].set_visible(False)

# --- common labels
fig.suptitle(
    "EAIS basin-level raw monthly PMB precipitation",
    fontsize=16,
    fontweight="bold",
    y=1.02
)

fig.supxlabel("Date", fontsize=14, fontweight="bold")
fig.supylabel("PMB precipitation (mm/month)", fontsize=14, fontweight="bold")

# rotate x tick labels a bit for readability
for ax in axes:
    if ax.get_visible():
        ax.tick_params(axis="x", labelrotation=30)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# STEP 5 — COMPUTE BASIN-LEVEL MONTHLY ANOMALIES
# ---------------------------------------------------------
# Remove each basin's monthly climatology
pmb_eais_anom = pmb_eais_wide.copy()

for basin_name in eais_basin_names:
    monthly_clim = (
        pmb_eais_anom[basin_name]
        .groupby(pmb_eais_anom.index.month)
        .transform("mean")
    )
    pmb_eais_anom[basin_name] = pmb_eais_anom[basin_name] - monthly_clim


# ---------------------------------------------------------
# STEP 6 — PLOT BASIN-LEVEL PMB ANOMALIES
# ---------------------------------------------------------
n = len(eais_basin_names)
ncols = 2
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(16, 3.8 * nrows),
    sharex=True,
    sharey=True
)

axes = np.atleast_1d(axes).ravel()

for ax, basin_name in zip(axes, eais_basin_names):
    ax.plot(pmb_eais_anom.index, pmb_eais_anom[basin_name], color="tab:blue", lw=1.4)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvspan(pd.Timestamp(focus_start), pd.Timestamp(focus_end), color="red", alpha=0.08)
    ax.set_title(basin_name, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_ylim([-40, 50])

# hide unused panels
for ax in axes[len(eais_basin_names):]:
    ax.set_visible(False)

# common labels
fig.suptitle("EAIS basin-level PMB precipitation anomalies", fontsize=16, fontweight="bold", y=1.02)
fig.supxlabel("Date", fontsize=14, fontweight="bold")
fig.supylabel("PMB precipitation anomaly (mm/month)", fontsize=14, fontweight="bold")

# rotate x tick labels a bit for readability
for ax in axes:
    if ax.get_visible():
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# STEP 7 — BUILD SUMMED EAIS PMB SERIES AND ANOMALY
# ---------------------------------------------------------
# Option A: simple unweighted sum across EAIS basins
eais_pmb_sum = pmb_eais_wide.sum(axis=1)
eais_pmb_sum_anom = eais_pmb_sum - eais_pmb_sum.groupby(eais_pmb_sum.index.month).transform("mean")

# Plot summed anomaly
plt.figure(figsize=(12, 4))
plt.plot(eais_pmb_sum_anom.index, eais_pmb_sum_anom, color="k", lw=2, label="EAIS summed PMB anomaly")
plt.axhline(0, color="gray", ls="--", lw=0.8)
plt.axvspan(pd.Timestamp(focus_start), pd.Timestamp(focus_end), color="red", alpha=0.08,
            label="Focus period")
plt.title("EAIS summed PMB anomaly from basin-level PMB precipitation", fontsize=15, fontweight="bold")
plt.ylabel("PMB anomaly")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# STEP 8 — PREPARE EAIS WEIGHTS
# ---------------------------------------------------------
# If you already created weights_eais earlier, use them.
# Otherwise fall back to equal weights.
if use_weights_if_available and "weights_eais" in globals():
    weights_use = weights_eais.copy()

    # Expect columns:
    # ['basin', 'basin_name', 'weight_eais', ...]
    weights_use = weights_use[["basin", "basin_name", "weight_eais"]].copy()

else:
    weights_use = pd.DataFrame({
        "basin": EAIS_IDS,
        "basin_name": [basin_id_to_name[i] for i in EAIS_IDS],
        "weight_eais": np.repeat(1 / len(EAIS_IDS), len(EAIS_IDS))
    })

print(weights_use.sort_values("weight_eais", ascending=False))


# ---------------------------------------------------------
# STEP 9 — QUANTIFY WHICH BASINS DRIVE THE FOCUS PERIOD
# ---------------------------------------------------------
focus_mask = (pmb_eais_anom.index >= focus_start) & (pmb_eais_anom.index <= focus_end)
focus_df = pmb_eais_anom.loc[focus_mask, eais_basin_names].copy()

summary_rows = []

for basin_name in eais_basin_names:
    w = weights_use.loc[weights_use["basin_name"] == basin_name, "weight_eais"].values
    w = w[0] if len(w) else np.nan

    s = focus_df[basin_name].dropna()

    summary_rows.append({
        "basin": basin_name,
        "weight_eais": w,
        "mean_abs_anom_focus": np.mean(np.abs(s)),
        "max_abs_anom_focus": np.max(np.abs(s)),
        "var_anom_focus": np.var(s),
        "weighted_mean_abs_anom": w * np.mean(np.abs(s)) if np.isfinite(w) else np.nan
    })

summary_df = pd.DataFrame(summary_rows).sort_values(
    "weighted_mean_abs_anom", ascending=False
).reset_index(drop=True)

print("\nRanked PMB basin influence summary:")
print(summary_df.round(3))


# ---------------------------------------------------------
# STEP 10 — TIME-VARYING FRACTIONAL CONTRIBUTION OF EACH BASIN
# ---------------------------------------------------------
# diagnostic contribution fraction relative to sum of abs anomalies
denom = pmb_eais_anom.abs().sum(axis=1).replace(0, np.nan)
contrib_frac = pmb_eais_anom.divide(denom, axis=0)

plt.figure(figsize=(13, 6))
for basin_name in eais_basin_names:
    plt.plot(contrib_frac.index, contrib_frac[basin_name], lw=1.2, label=basin_name)

plt.axhline(0, color="gray", ls="--", lw=0.8)
plt.axvspan(pd.Timestamp(focus_start), pd.Timestamp(focus_end), color="red", alpha=0.08)
plt.title("Relative basin contributions to EAIS PMB anomaly through time", fontsize=15, fontweight="bold")
plt.ylabel("Contribution fraction")
plt.xlabel("Date")
plt.legend(ncol=5, fontsize=9, frameon=False)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# STEP 11 — STATIC EAIS BASIN WEIGHTS
# ---------------------------------------------------------
weights_plot = weights_use.sort_values("weight_eais", ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(weights_plot["basin_name"], weights_plot["weight_eais"])
plt.title("Static basin weights within EAIS", fontsize=15, fontweight="bold")
plt.ylabel("EAIS-normalized weight")
plt.xlabel("Basin")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# STEP 12 — WEIGHTED BASIN INFLUENCE BAR PLOT
# ---------------------------------------------------------
summary_plot = summary_df.sort_values("weighted_mean_abs_anom", ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(summary_plot["basin"], summary_plot["weighted_mean_abs_anom"])
plt.title("Weighted PMB basin anomaly influence during focus period", fontsize=15, fontweight="bold")
plt.ylabel("Weight × mean absolute anomaly")
plt.xlabel("Basin")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# STEP 13 — OPTIONAL: PRINT TOP BASINS ONLY
# ---------------------------------------------------------
top_n = 5
print(f"\nTop {top_n} candidate basins driving EAIS PMB anomaly behavior:")
print(summary_df.head(top_n).round(3))

#%%
# =========================================================
# QUANTIFY HOW MUCH EAIS PMB ANOMALY IS EXPLAINED BY ΔS
# =========================================================
# Assumes you already have:
#   pmb_eais_wide        -> EAIS basin-level PMB precip (mm/month), index=time, cols=basin names
#   pmb_eais_anom        -> EAIS basin-level PMB precip anomalies (mm/month)
#   dS_tier1             -> monthly basin ΔS state differences (Gt/month), index=time, cols=basin names
#   dS_eais_anom         -> EAIS basin-level ΔS anomalies (Gt/month)
#   weights_eais         -> DataFrame with columns ['basin_name','weight_eais']
#   eais_basin_names     -> ordered list of EAIS basin names
#   focus_start, focus_end
#
# Goal:
# 1) reconstruct EAIS PMB anomaly from basin-level PMB anomalies using EAIS weights
# 2) reconstruct EAIS ΔS anomaly from basin-level ΔS anomalies using same EAIS weights
# 3) compare the two series directly
# 4) diagnose which basins contribute most to the covariance / mismatch
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# STEP 1 — BUILD A CLEAN WEIGHT VECTOR
# ---------------------------------------------------------
w_eais = (
    weights_eais.set_index("basin_name")["weight_eais"]
    .reindex(eais_basin_names)
    .astype(float)
)

# safety normalization
w_eais = w_eais / w_eais.sum()

print("EAIS weights used:")
print(w_eais.round(4))


# ---------------------------------------------------------
# STEP 2 — ALIGN BASIN-LEVEL PMB ANOMALY AND ΔS ANOMALY TABLES
# ---------------------------------------------------------
common_time = pmb_eais_anom.index.intersection(dS_eais_anom.index)

pmb_anom_aligned = pmb_eais_anom.loc[common_time, eais_basin_names].copy()
ds_anom_aligned  = dS_eais_anom.loc[common_time, eais_basin_names].copy()

# ensure same basin order
pmb_anom_aligned = pmb_anom_aligned[eais_basin_names]
ds_anom_aligned  = ds_anom_aligned[eais_basin_names]

print("Aligned monthly samples:", len(common_time))


# ---------------------------------------------------------
# STEP 3 — RECONSTRUCT EAIS REGIONAL ANOMALIES FROM BASIN PANELS
# ---------------------------------------------------------
# PMB anomaly in mm/month
eais_pmb_anom_from_basins = pmb_anom_aligned.mul(w_eais, axis=1).sum(axis=1)
eais_pmb_anom_from_basins.name = "EAIS PMB anomaly"

# ΔS anomaly in Gt/month
eais_ds_anom_from_basins = ds_anom_aligned.mul(w_eais, axis=1).sum(axis=1)
eais_ds_anom_from_basins.name = "EAIS ΔS anomaly (weighted)"

# optional standardized versions for shape-only comparison
eais_pmb_std = (eais_pmb_anom_from_basins - eais_pmb_anom_from_basins.mean()) / eais_pmb_anom_from_basins.std(ddof=1)
eais_ds_std  = (eais_ds_anom_from_basins  - eais_ds_anom_from_basins.mean())  / eais_ds_anom_from_basins.std(ddof=1)


# ---------------------------------------------------------
# STEP 4 — PLOT RAW REGIONAL ANOMALY COMPARISON
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(13, 4.8))

ax1.plot(
    eais_pmb_anom_from_basins.index,
    eais_pmb_anom_from_basins,
    color="k", lw=2.2, label="EAIS PMB anomaly"
)
ax1.axhline(0, color="gray", ls="--", lw=0.8)
ax1.axvspan(pd.Timestamp(focus_start), pd.Timestamp(focus_end), color="red", alpha=0.08)
ax1.set_ylabel("PMB anomaly (mm/month)", fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.25)

ax2 = ax1.twinx()
ax2.plot(
    eais_ds_anom_from_basins.index,
    eais_ds_anom_from_basins,
    color="tab:blue", lw=2.0, label="EAIS ΔS anomaly (weighted basins)"
)
ax2.set_ylabel("ΔS anomaly (Gt/month)", fontsize=13, fontweight="bold")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)

ax1.set_title("EAIS regional anomaly: PMB vs weighted ΔS contribution", fontsize=15, fontweight="bold")
ax1.set_xlabel("Date", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# STEP 5 — PLOT STANDARDIZED SHAPE COMPARISON
# ---------------------------------------------------------
plt.figure(figsize=(13, 4.5))
plt.plot(eais_pmb_std.index, eais_pmb_std, color="k", lw=2.2, label="EAIS PMB anomaly (standardized)")
plt.plot(eais_ds_std.index,  eais_ds_std,  color="tab:blue", lw=2.0, label="EAIS ΔS anomaly (standardized)")
plt.axhline(0, color="gray", ls="--", lw=0.8)
plt.axvspan(pd.Timestamp(focus_start), pd.Timestamp(focus_end), color="red", alpha=0.08)
plt.title("Standardized shape comparison: EAIS PMB anomaly vs EAIS ΔS anomaly", fontsize=15, fontweight="bold")
plt.ylabel("Standardized anomaly", fontsize=13, fontweight="bold")
plt.xlabel("Date", fontsize=13, fontweight="bold")
plt.legend(frameon=False)
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# STEP 6 — SIMPLE CORRELATION / FOCUS-PERIOD CORRELATION
# ---------------------------------------------------------
overall_corr = eais_pmb_anom_from_basins.corr(eais_ds_anom_from_basins)

focus_mask = (
    (eais_pmb_anom_from_basins.index >= pd.Timestamp(focus_start)) &
    (eais_pmb_anom_from_basins.index <= pd.Timestamp(focus_end))
)

focus_corr = (
    eais_pmb_anom_from_basins.loc[focus_mask]
    .corr(eais_ds_anom_from_basins.loc[focus_mask])
)

print(f"Overall correlation (PMB anomaly vs weighted ΔS anomaly): {overall_corr:.3f}")
print(f"Focus-period correlation ({focus_start} to {focus_end}): {focus_corr:.3f}")


# ---------------------------------------------------------
# STEP 7 — BASIN-BY-BASIN CONTRIBUTION TO REGIONAL PMB AND ΔS
# ---------------------------------------------------------
# Weighted basin contributions
pmb_contrib = pmb_anom_aligned.mul(w_eais, axis=1)
ds_contrib  = ds_anom_aligned.mul(w_eais, axis=1)

# For each basin:
#   - corr with EAIS PMB anomaly
#   - corr between basin PMB contrib and basin ΔS contrib
#   - mean absolute contribution in focus period
summary_rows = []

for basin in eais_basin_names:
    pmb_b = pmb_contrib[basin]
    ds_b  = ds_contrib[basin]

    # correlation with total EAIS PMB anomaly
    corr_to_total = pmb_b.corr(eais_pmb_anom_from_basins)

    # how similarly this basin behaves in PMB vs ΔS contribution space
    corr_pmb_vs_ds = pmb_b.corr(ds_b)

    # focus period strength
    pmb_focus_meanabs = pmb_b.loc[focus_mask].abs().mean()
    ds_focus_meanabs  = ds_b.loc[focus_mask].abs().mean()

    # mismatch metric during focus period
    mismatch_focus = (pmb_b.loc[focus_mask] - ds_b.loc[focus_mask].mean()).abs().mean()

    summary_rows.append({
        "basin": basin,
        "weight_eais": w_eais.loc[basin],
        "corr_basin_PMB_to_total_PMB": corr_to_total,
        "corr_basin_PMB_to_basin_dS": corr_pmb_vs_ds,
        "mean_abs_PMB_contrib_focus": pmb_focus_meanabs,
        "mean_abs_dS_contrib_focus": ds_focus_meanabs,
        "focus_strength_ratio_dS_over_PMB": ds_focus_meanabs / pmb_focus_meanabs if pmb_focus_meanabs != 0 else np.nan,
        "focus_mismatch_proxy": mismatch_focus,
    })

summary_contrib_df = pd.DataFrame(summary_rows)

print("\nBasin contribution summary:")
print(summary_contrib_df.sort_values("mean_abs_PMB_contrib_focus", ascending=False).round(3))


# ---------------------------------------------------------
# STEP 8 — RANK BASINS BY PMB CONTRIBUTION DURING FOCUS PERIOD
# ---------------------------------------------------------
rank_pmb = (
    summary_contrib_df
    .sort_values("mean_abs_PMB_contrib_focus", ascending=False)
    .reset_index(drop=True)
)

rank_ds = (
    summary_contrib_df
    .sort_values("mean_abs_dS_contrib_focus", ascending=False)
    .reset_index(drop=True)
)

rank_mismatch = (
    summary_contrib_df
    .sort_values("focus_mismatch_proxy", ascending=False)
    .reset_index(drop=True)
)

print("\nTop basins by PMB contribution strength in focus period:")
print(rank_pmb[["basin", "weight_eais", "mean_abs_PMB_contrib_focus", "corr_basin_PMB_to_total_PMB"]].head(6).round(3))

print("\nTop basins by ΔS contribution strength in focus period:")
print(rank_ds[["basin", "weight_eais", "mean_abs_dS_contrib_focus", "corr_basin_PMB_to_basin_dS"]].head(6).round(3))

print("\nTop basins by PMB-vs-ΔS mismatch proxy:")
print(rank_mismatch[["basin", "weight_eais", "focus_mismatch_proxy", "corr_basin_PMB_to_basin_dS"]].head(6).round(3))


# ---------------------------------------------------------
# STEP 9 — BAR PLOTS FOR QUICK DIAGNOSIS
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# PMB contribution
tmp = rank_pmb.copy()
axes[0].bar(tmp["basin"], tmp["mean_abs_PMB_contrib_focus"])
axes[0].set_title("Focus-period PMB contribution", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Mean |weighted contribution|", fontsize=12)
axes[0].tick_params(axis="x", rotation=45)

# ΔS contribution
tmp = rank_ds.copy()
axes[1].bar(tmp["basin"], tmp["mean_abs_dS_contrib_focus"], color="tab:blue")
axes[1].set_title("Focus-period ΔS contribution", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Mean |weighted contribution|", fontsize=12)
axes[1].tick_params(axis="x", rotation=45)

# mismatch
tmp = rank_mismatch.copy()
axes[2].bar(tmp["basin"], tmp["focus_mismatch_proxy"], color="tab:red")
axes[2].set_title("PMB vs ΔS mismatch proxy", fontsize=13, fontweight="bold")
axes[2].set_ylabel("Mean absolute mismatch", fontsize=12)
axes[2].tick_params(axis="x", rotation=45)

for ax in axes:
    ax.grid(True, axis="y", alpha=0.25)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# STEP 10 — OPTIONAL: PANEL PLOT OF BASIN WEIGHTED CONTRIBUTIONS
# ---------------------------------------------------------
top_basins = rank_pmb["basin"].head(6).tolist()

fig, axes = plt.subplots(len(top_basins), 1, figsize=(12, 2.4 * len(top_basins)), sharex=True)

if len(top_basins) == 1:
    axes = [axes]

for ax, basin in zip(axes, top_basins):
    ax.plot(pmb_contrib.index, pmb_contrib[basin], color="k", lw=1.5, label=f"{basin} PMB contrib")
    ax.plot(ds_contrib.index,  ds_contrib[basin],  color="tab:blue", lw=1.3, label=f"{basin} ΔS contrib")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvspan(pd.Timestamp(focus_start), pd.Timestamp(focus_end), color="red", alpha=0.08)
    ax.set_title(basin, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=10, loc="upper left")

axes[-1].set_xlabel("Date", fontsize=12, fontweight="bold")
fig.suptitle("Top-basin weighted contributions: PMB vs ΔS", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()
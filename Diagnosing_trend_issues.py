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
YEARS = np.arange(2013,2023)#[2019, 2020]

S_tier1 = pd.read_pickle(os.path.join(basins_path, 'DataCombo_RignotBasins_LI_tier1_20260226.pkl'))

rignot_deltaS = pd.read_excel(os.path.join(basins_path, 'DataCombo_RignotBasins.xlsx'), sheet_name='Basin_Timeseries (Gt)')

rignot_deltaS["Date"] = rignot_deltaS["Time"].apply(decimal_year_to_date).dt.strftime('%Y-%m-%d')
rignot_deltaS['Date'] = pd.to_datetime(rignot_deltaS['Date'])
rignot_deltaS["Year"] = pd.to_datetime(rignot_deltaS["Date"]).dt.year
rignot_deltaS["Month"] = pd.to_datetime(rignot_deltaS["Date"]).dt.month
# 1) build month-start timestamps directly
# rignot_deltaS["date"] = rignot_deltaS["Time"].apply(
#     lambda x: decimal_year_to_month_start(x, mode="nearest")   # or mode="floor"
# )

# # 2) derive Year/Month from the date (robust)
# rignot_deltaS["Year"]  = rignot_deltaS["date"].dt.year
# rignot_deltaS["Month"] = rignot_deltaS["date"].dt.month

basin_cols = [c for c in rignot_deltaS.columns if c not in ("Time","date","Year","Month")]

# ---------------------------------------------------------
# FULL MONTHLY STATE SERIES (2013–2022)
# ---------------------------------------------------------
start_date = "2013-01-01"
end_date   = "2022-12-01"
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

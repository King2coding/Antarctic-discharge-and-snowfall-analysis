"""
Monthly Mass-Budget Precipitation (2019–2020, Rignot/IMBIE basins)

Inputs
------
1) David (GRACE/altimetry): Excel with sub-annual basin mass anomalies (Gt)
   - Sheet: "Basin_Timeseries (Gt)" (typical)
   - Columns: Time (decimal year), then one column per basin (e.g., "A-Ap", "Ap-B", ...)

2) Chad (discharge & basal melt): Excel with annual values per basin (Gt/yr)
   - Provide sheet/column names in CONFIG below.

3) RACMO2 sublimation: NetCDF monthly ANT11 domain (RACMO2.4p1)
   - Variable: "subltot" (mm water equivalent per month) [common name]
   - Requires cell areas to convert to Gt. Tries variable "cell_area" (m^2) or "AREA" (km^2).
   - Basin mask on the same grid: tries to find "basin" integer IDs in the NetCDF; if not present,
     set BASIN_MASK_PATH to a co-registered mask (same shape/projection as RACMO grid).

Output
------
- CSV with monthly P_MB per basin (Gt) for 2019–2020, and (if basin areas known) mm/month.
- Columns: date, basin, dS_Gt, discharge_Gt, basal_Gt, subl_Gt, Pmb_Gt, (optional) basin_area_km2, Pmb_mm

Author: (K. K. Kumah)
Date: 2025-09-10
"""
#%%
# import libraries
from program_utils import *

#%%
# define paths
basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'
racmo_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/RACMO2pt4p1'

#%%
# floating variables
bedmap3_basins = xr.open_dataset(os.path.join(basin_path, 'bedmap3_basins.nc'))
basins_imbie = bedmap3_basins['imbie'].copy()
basins_imbie = basins_imbie.where((basins_imbie > 0) & (basins_imbie.notnull()))


rignot_deltaS_err = pd.read_excel(os.path.join(basin_path, 'DataCombo_RignotBasins.xlsx'), sheet_name='1-sigma_Error(Gt)')

# - - - - - - - - - - - - - - - - - - - - - - - -- - - -- - - - - - - -- - - - - 
# Year window
YEARS = [2019, 2020]
#%% 1) Read David's Excel and compute ΔS (Gt/month) for 2019–2020
rignot_deltaS = pd.read_excel(os.path.join(basin_path, 'DataCombo_RignotBasins.xlsx'), sheet_name='Basin_Timeseries (Gt)')

rignot_deltaS["Date"] = rignot_deltaS["Time"].apply(decimal_year_to_date).dt.strftime('%Y-%m-%d')
rignot_deltaS['Date'] = pd.to_datetime(rignot_deltaS['Date'])
rignot_deltaS["Year"] = pd.to_datetime(rignot_deltaS["Date"]).dt.year
rignot_deltaS["Month"] = pd.to_datetime(rignot_deltaS["Date"]).dt.month

# Keep only 2019–2020 rows
rignot_deltaS = rignot_deltaS[rignot_deltaS["Year"].isin(YEARS)].copy()

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

#%% 2) Read Chad's discharge & basal melt (annual Gt/yr) and spread to months
discharge_data = pd.read_excel(os.path.join(basin_path, 'antarctic_discharge_2013-2022_imbie.xlsx'), sheet_name=['Discharge (Gt yr^-1)', 'Summary'])
basin_discharge = discharge_data['Discharge (Gt yr^-1)']
basin_cols_ = basin_cols.copy()
basin_cols_ = [b.replace('-f', '-F') if b == 'Ep-f' else b for b in basin_cols_]

basin_discharge = basin_discharge.loc[basin_discharge['IMBIE basin'].isin(basin_cols_), ['IMBIE basin', '2019', '2020']].rename(columns={'IMBIE basin': 'basin'})

# --- Basal melt ---
basal_melt = discharge_data['Summary']
basal_melt = basal_melt.loc[basal_melt['IMBIE basin'].isin(basin_cols_), ['IMBIE basin', 'Basal melt total Gt/yr']].rename(columns={'IMBIE basin': 'basin'})


# Convert annual (Gt/yr) → monthly (Gt/mo), then expand to months
D_month = annual_to_monthly_long(basin_discharge,YEARS, "discharge_Gt")

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

Q_month = D_month.merge(B_month, on=["date","basin"], how="outer")
print(f"[Chad] Qnet monthly rows: {len(Q_month)}")

#%% 3) RACMO: integrate subltot (mm/month) to basin Gt/month
sublimation_data = xr.open_dataset(os.path.join(racmo_path, 'subltot_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc'))

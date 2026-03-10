#%% Packages
from program_utils import *

#%% Paths
basin_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'


#%% Floating Variables

start_date = "2013-01-01"
end_date   = "2022-12-01"

#%% Step A — Prepare Full Monthly State Series
rignot_deltaS = pd.read_excel(os.path.join(basin_path, 'DataCombo_RignotBasins.xlsx'), sheet_name='Basin_Timeseries (Gt)')

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
# FULL MONTHLY STATE SERIES (2013–2022)
# ---------------------------------------------------------

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


#%% Step B — Deseasonalize
# ---------------------------------------------------------
# TIER 1: Deseasonalized Linear Interpolation
# ---------------------------------------------------------

S_tier1 = S_full.copy()

for basin in basin_cols:

    series = S_full[basin]

    # Compute climatology from available data
    climatology = series.groupby(series.index.month).mean()

    # Remove seasonal cycle
    deseason = series - series.index.month.map(climatology)

    # Linear interpolation (only inside bounds)
    deseason_interp = deseason.interpolate(method="linear", limit_area="inside")
    # deseason.interpolate(method="linear", limit_direction="both")

    # Add seasonal cycle back
    S_tier1[basin] = deseason_interp + deseason_interp.index.month.map(climatology)

# save df to disk
S_tier1.to_pickle(os.path.join(basin_path, f"DataCombo_RignotBasins_LI_tier1_{cde_run_dte}.pkl"))
#%%
# ---------------------------------------------------------
# TIER 2: Harmonic Trend + Annual Cycle Fit
# ---------------------------------------------------------

S_tier2 = pd.DataFrame(index=full_index)

# Convert time to fractional years
t = np.arange(len(full_index)) / 12.0  # in years

for basin in basin_cols:

    y = S_full[basin].values
    mask = np.isfinite(y)

    t_obs = t[mask]
    y_obs = y[mask]

    # Design matrix: [1, t, cos(2πt), sin(2πt)]
    X = np.column_stack([
        np.ones_like(t_obs),
        t_obs,
        np.cos(2*np.pi*t_obs),
        np.sin(2*np.pi*t_obs)
    ])

    beta, *_ = np.linalg.lstsq(X, y_obs, rcond=None)

    # Predict full series
    X_full = np.column_stack([
        np.ones_like(t),
        t,
        np.cos(2*np.pi*t),
        np.sin(2*np.pi*t)
    ])

    S_tier2[basin] = X_full @ beta

S_tier2.index = full_index

# save df to disk
S_tier2.to_pickle(os.path.join(basin_path, f"DataCombo_RignotBasins_Harmonic_tier2_{cde_run_dte}.pkl"))

#%%
# ---------------------------------------------------------
# Compute ΔS (monthly difference)
# ---------------------------------------------------------

dS_tier1 = S_tier1.diff().dropna()
dS_tier2 = S_tier2.diff().dropna()

#%% Visual Comparison of Tiers
test_basins = ["H-Hp", "Ep-f", "A-Ap", "Jpp-K", "C-Cp", "I-Ipp"]  # example names
for basin in test_basins:

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # -------- Panel 1: State --------
    axes[0].plot(S_full.index, S_full[basin], color= 'k', ls = '--', lw = 4, label="Observed", markersize=8)
    axes[0].plot(S_tier1.index, S_tier1[basin], label="Tier 1 (Interp)",color = 'tab:blue')
    axes[0].plot(S_tier2.index, S_tier2[basin], label="Tier 2 (Harmonic)", color='tab:orange')
    axes[0].set_ylabel("Mass Anomaly (Gt)")
    axes[0].set_title(f"GRACE State Reconstruction — {basin}")
    axes[0].legend()

    # -------- Panel 2: ΔS --------
    dS_obs = S_full[basin].diff()

    axes[1].plot(dS_obs.index, dS_obs, color='k', ls = '--', lw = 4, label="Observed ΔS", markersize=8)
    axes[1].plot(dS_tier1.index, dS_tier1[basin], label="Tier 1 ΔS", color='tab:blue')
    axes[1].plot(dS_tier2.index, dS_tier2[basin], label="Tier 2 ΔS", color='tab:orange')
    axes[1].set_ylabel("ΔS (Gt/month)")
    # axes[1].legend()

    plt.tight_layout()
    plt.show()
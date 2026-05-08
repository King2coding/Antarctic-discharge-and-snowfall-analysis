import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from program_utils import *
import math
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, LogNorm, PowerNorm
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import matplotlib.dates as mdates

#%%
REGION_NEG_THRESHOLDS = {
    "West Antarctica": -50.0,
    "East Antarctica": -18, # -25.0
}

#%%


def make_basin_annual_from_monthly_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns expected:
        year, month, basin, precipitation, time
    Returns annual totals per basin:
        year, basin, annual_precip
    """
    out = (
        df.groupby(["year", "basin"], as_index=False)["precipitation"]
          .sum()
          .rename(columns={"precipitation": "annual_precip"})
    )
    return out

def plot_eais_basin_interannual(
    monthly_df_data_mmmonth,
    basin_list,
    products_to_plot,
    basin_weights=None,
    ncols=2,
    figsize=(14, 18),
    ylabel="[mm/year]",
):
    """
    Plot annual basin accumulation for selected EAIS basins.

    Parameters
    ----------
    monthly_df_data_mmmonth : dict[str, pd.DataFrame]
        Dict mapping product name -> monthly basin dataframe
    basin_list : list[int]
        Basin IDs to plot
    products_to_plot : list[str]
        Product names to include
    basin_weights : dict or pd.Series, optional
        Basin weights/areas for optional titles
    """
    nbasins = len(basin_list)
    nrows = math.ceil(nbasins / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        sharex=True,
        sharey=False
    )
    axes = axes.flatten()

    # simple styles
    style_map = {
        r"$P_{\mathrm{MB}}$": dict(color="k", marker="o", lw=2.2),
        "ERA5": dict(color="blue", marker="s", lw=2.2),
        "GPCP v3.3": dict(color="orange", marker="D", lw=2.2),
        "GPM PMW V07": dict(color="cyan", marker=None, lw=2.2),
    }

    annual_cache = {}
    for prod in products_to_plot:
        if prod not in monthly_df_data_mmmonth:
            continue
        annual_cache[prod] = make_basin_annual_from_monthly_df(monthly_df_data_mmmonth[prod])

    for ax, basin in zip(axes, basin_list):
        for prod in products_to_plot:
            if prod not in annual_cache:
                continue

            df_ann = annual_cache[prod]
            sub = (
                df_ann[df_ann["basin"] == basin]
                .sort_values("year")
            )

            if sub.empty:
                continue

            style = style_map.get(prod, dict(lw=2.0))
            ax.plot(
                sub["year"],
                sub["annual_precip"],
                label=prod,
                **style
            )

        title = f"Basin {basin}"
        if basin_weights is not None:
            try:
                area_val = basin_weights[basin]
                title += f" (w={area_val:.3g})"
            except Exception:
                pass

        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # remove unused axes
    for j in range(len(basin_list), len(axes)):
        fig.delaxes(axes[j])

    # common labels
    # fig.supxlabel("Year", fontsize=16, fontweight="bold")
    fig.supylabel(ylabel, fontsize=16, fontweight="bold")

    # one legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(labels), 4),
        frameon=False,
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig, axes

def ensure_monthly_basin_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure basin dataframe is monthly total precipitation with columns:
    year, month, basin, precipitation, time

    If input is daily (multiple rows per basin-month), sum to monthly totals.
    If already monthly (one row per basin-month), return sorted copy.
    """
    req = {"year", "month", "basin", "precipitation"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # count rows per basin-month
    counts = (
        df.groupby(["year", "month", "basin"])
          .size()
          .reset_index(name="n")
    )

    is_monthly = counts["n"].max() == 1

    if is_monthly:
        out = df.copy()
    else:
        out = (
            df.groupby(["year", "month", "basin"], as_index=False)["precipitation"]
              .sum()
        )

    out["time"] = pd.to_datetime(
        dict(year=out["year"], month=out["month"], day=1)
    )
    out = out.sort_values(["year", "month", "basin"]).reset_index(drop=True)
    return out


def basin_mean_bias_table(monthly_df_data_mmmonth, ref_name, test_name, basin_list):
    ref_ann = make_basin_annual_from_monthly_df(monthly_df_data_mmmonth[ref_name])
    tst_ann = make_basin_annual_from_monthly_df(monthly_df_data_mmmonth[test_name])

    merged = ref_ann.merge(
        tst_ann,
        on=["year", "basin"],
        suffixes=("_ref", "_test")
    )

    merged["bias"] = merged["annual_precip_test"] - merged["annual_precip_ref"]

    tab = (
        merged[merged["basin"].isin(basin_list)]
        .groupby("basin", as_index=False)
        .agg(
            mean_ref=("annual_precip_ref", "mean"),
            mean_test=("annual_precip_test", "mean"),
            mean_bias=("bias", "mean"),
        )
        .sort_values("mean_bias", ascending=False)
    )
    return tab

#------------------------------------------------------------------------------

def convert_outlier_basin_names_to_ids(outlier_list_by_name, basin_name_to_id):
    """
    Convert outlier basin/month definitions from basin names to basin IDs.
    """

    out = []

    for item in outlier_list_by_name:
        year = int(item["year"])
        month = int(item["month"])
        basin_names = item["basins"]

        basin_ids = []
        for name in basin_names:
            if name not in basin_name_to_id:
                raise ValueError(f"Basin name not found in mapping: {name}")
            basin_ids.append(basin_name_to_id[name])

        out.append({
            "year": year,
            "month": month,
            "basins": basin_ids,
            "basin_names": basin_names,
        })

    return out

#------------------------------------------------------------------------------

def mask_specific_basin_months_in_field(
    da,
    basin_mask,
    outlier_basin_months,
    time_name="time",
):
    """
    Mask selected basin/month combinations in a monthly gridded DataArray.
    """

    da_masked = da.copy()

    da_masked = da_masked.assign_coords(
        {time_name: pd.to_datetime(da_masked[time_name].values)}
    )

    for item in outlier_basin_months:
        year = int(item["year"])
        month = int(item["month"])
        basins_to_mask = item["basins"]
        basin_names = item.get("basin_names", basins_to_mask)

        tmask = (
            (da_masked[time_name].dt.year == year) &
            (da_masked[time_name].dt.month == month)
        )

        matching_times = da_masked[time_name].where(tmask, drop=True)

        if matching_times.size == 0:
            print(f"Warning: no matching time found for {year}-{month:02d}")
            continue

        bmask = basin_mask.isin(basins_to_mask)

        for tt in matching_times.values:
            da_masked.loc[{time_name: tt}] = (
                da_masked.sel({time_name: tt}).where(~bmask)
            )

        print(
            f"Masked PMB for {year}-{month:02d}; "
            f"basins={basin_names}; basin_ids={basins_to_mask}"
        )

    return da_masked

#-----------------------------------------------------------------------------
def compare_original_vs_sensitivity_values(
    seasonal_orig_df,
    seasonal_sens_df,
    regions=("West Antarctica", "East Antarctica"),
    product=r"$P_{\mathrm{MB}}$",
    years=(2014, 2017),
):
    """
    Compare original and sensitivity PMB seasonal values.
    """

    orig = seasonal_orig_df[
        (seasonal_orig_df["product"] == product) &
        (seasonal_orig_df["region"].isin(regions)) &
        (seasonal_orig_df["time"].dt.year.isin(years))
    ].copy()

    sens = seasonal_sens_df[
        (seasonal_sens_df["product"] == product) &
        (seasonal_sens_df["region"].isin(regions)) &
        (seasonal_sens_df["time"].dt.year.isin(years))
    ].copy()

    keep_cols = ["region", "time", "season", "precipitation"]

    orig = orig[keep_cols].rename(columns={"precipitation": "PMB_original"})
    sens = sens[keep_cols].rename(columns={"precipitation": "PMB_sensitivity"})

    out = orig.merge(
        sens,
        on=["region", "time", "season"],
        how="outer"
    )

    out["difference_sens_minus_orig"] = (
        out["PMB_sensitivity"] - out["PMB_original"]
    )

    return out.sort_values(["region", "time"]).reset_index(drop=True)
#-----------------------------------------------------------------------------
def basin_month_pmb_from_grid(
    pmb_da,
    basin_mask,
    basin_ids,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
):
    """
    Compute cosine-weighted basin-month PMB values from gridded PMB.
    Returns tidy dataframe: time, year, month, basin, pmb_mm_month.
    """

    rows = []

    # Make sure basin mask is on same lat/lon ordering as PMB
    basin_mask = basin_mask.reindex_like(pmb_da.isel({time_name: 0}), method=None)

    # 1D latitude weights
    lat = pmb_da[lat_name]
    weights_lat = np.cos(np.deg2rad(lat))

    # Broadcast to 2D lat/lon
    weights_2d = xr.ones_like(pmb_da.isel({time_name: 0})) * weights_lat

    for bid in basin_ids:
        bmask = basin_mask == bid

        vals = pmb_da.where(bmask)

        # xarray weighted() cannot accept NaN weights
        w = weights_2d.where(bmask).fillna(0)

        # If the basin has no valid weights, skip
        if float(w.sum()) == 0:
            print(f"Warning: no valid weights for basin {bid}")
            continue

        basin_ts = vals.weighted(w).mean(
            dim=(lat_name, lon_name),
            skipna=True,
        )

        df = basin_ts.to_dataframe(name="pmb_mm_month").reset_index()
        df["basin"] = bid
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)

    out["time"] = pd.to_datetime(out[time_name])
    out["year"] = out["time"].dt.year
    out["month"] = out["time"].dt.month

    return out[["time", "year", "month", "basin", "pmb_mm_month"]]
#-----------------------------------------------------------------------------
def get_problem_season_basin_months(
    pmb_basin_month_df,
    region_name,
    basin_ids,
    year,
    season_months,
    sort_ascending=True,
):
    """
    Pull basin/month PMB values for a specific region/year/season.

    season_months should be a list of months.
    Example:
        DJF 2017 may need [1, 2, 12] depending on your season convention.
        SON 2014 = [9, 10, 11].
    """

    sub = pmb_basin_month_df[
        (pmb_basin_month_df["basin"].isin(basin_ids)) &
        (pmb_basin_month_df["year"] == year) &
        (pmb_basin_month_df["month"].isin(season_months))
    ].copy()

    sub = sub.sort_values("pmb_mm_month", ascending=sort_ascending)

    print(f"\n{region_name} {year}, months={season_months}")
    print(sub[["time", "month", "basin", "basin_name", "pmb_mm_month"]].head(20))

    return sub
#-----------------------------------------------------------------------------

def compute_basin_month_series_from_grid(
    da,
    basin_mask,
    basin_ids,
    value_name="value",
    lat_name="lat",
    lon_name="lon",
    time_name="time",
):
    """
    Compute cosine-weighted basin-month values from a gridded monthly DataArray.

    Returns
    -------
    DataFrame with:
        time, year, month, basin, value_name
    """

    rows = []

    da = da.assign_coords({time_name: pd.to_datetime(da[time_name].values)})

    # Ensure basin mask aligns with one time slice
    basin_mask_2d = basin_mask.reindex_like(da.isel({time_name: 0}), method=None)

    # Latitude cosine weights
    lat = da[lat_name]
    weights_lat = np.cos(np.deg2rad(lat))
    weights_2d = xr.ones_like(da.isel({time_name: 0})) * weights_lat

    for bid in basin_ids:
        bmask = basin_mask_2d == bid

        vals = da.where(bmask)
        w = weights_2d.where(bmask).fillna(0)

        if float(w.sum()) == 0:
            print(f"Warning: no valid weights for basin {bid}")
            continue

        ts = vals.weighted(w).mean(
            dim=(lat_name, lon_name),
            skipna=True,
        )

        df = ts.to_dataframe(name=value_name).reset_index()
        df["basin"] = int(bid)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out["time"] = pd.to_datetime(out[time_name])
    out["year"] = out["time"].dt.year
    out["month"] = out["time"].dt.month

    return out[["time", "year", "month", "basin", value_name]]
#-------------------------------------------------------------------------------

def mask_basin_months_from_negative_pmb_table(
    da,
    basin_mask,
    negative_table,
    basin_col="basin",
    time_col="time",
    lat_name="lat",
    lon_name="lon",
    time_name="time",
):
    """
    Mask selected basin/months in a gridded monthly product.

    The negative_table is usually derived from PMB and contains
    time + basin combinations to exclude.

    This function applies those exclusions to any product so that all
    products use the same basin/month support.
    """

    da_out = da.copy()
    da_out = da_out.assign_coords({time_name: pd.to_datetime(da_out[time_name].values)})

    basin_mask_2d = basin_mask.reindex_like(da_out.isel({time_name: 0}), method=None)

    tbl = negative_table[[time_col, basin_col]].copy()
    tbl[time_col] = pd.to_datetime(tbl[time_col])
    tbl["year"] = tbl[time_col].dt.year
    tbl["month"] = tbl[time_col].dt.month

    grouped = tbl.groupby(["year", "month"])[basin_col].apply(list).reset_index()

    for _, row in grouped.iterrows():
        yy = int(row["year"])
        mm = int(row["month"])
        basins_to_remove = [int(b) for b in row[basin_col]]

        tmask = (
            (da_out[time_name].dt.year == yy) &
            (da_out[time_name].dt.month == mm)
        )

        matching_times = da_out[time_name].where(tmask, drop=True)

        if matching_times.size == 0:
            print(f"Warning: no matching product time for {yy}-{mm:02d}")
            continue

        bmask = basin_mask_2d.isin(basins_to_remove)

        for tt in matching_times.values:
            da_out.loc[{time_name: tt}] = (
                da_out.sel({time_name: tt}).where(~bmask)
            )

        print(
            f"Masked common PMB-negative support for {yy}-{mm:02d}; "
            f"basins={basins_to_remove}"
        )

    return da_out
#-------------------------------------------------------------------------------
def diagnose_grace_storage_neighbors(
    S_df,
    flagged_table,
    basin_col="basin",
    time_col="time",
):
    """
    Diagnose whether problematic PMB basin-months are linked to
    current, previous, or next GRACE storage anomaly values.

    S_df should be wide:
        index = monthly date
        columns = basin names or basin IDs
        values = storage anomaly S [Gt]

    flagged_table should contain:
        time, basin
    """

    S = S_df.copy()
    S.index = pd.to_datetime(S.index)
    S = S.sort_index()

    rows = []

    flagged = flagged_table.copy()
    flagged[time_col] = pd.to_datetime(flagged[time_col])

    for _, r in flagged.iterrows():
        t = pd.Timestamp(r[time_col]).to_period("M").to_timestamp()
        basin = r[basin_col]

        if basin not in S.columns:
            continue
        if t not in S.index:
            continue

        prev_t = t - pd.DateOffset(months=1)
        next_t = t + pd.DateOffset(months=1)

        S_prev = S.loc[prev_t, basin] if prev_t in S.index else np.nan
        S_curr = S.loc[t, basin]
        S_next = S.loc[next_t, basin] if next_t in S.index else np.nan

        dS_curr = S_curr - S_prev if np.isfinite(S_prev) and np.isfinite(S_curr) else np.nan
        dS_next = S_next - S_curr if np.isfinite(S_next) and np.isfinite(S_curr) else np.nan

        rows.append({
            "time": t,
            "basin": basin,
            "S_prev": S_prev,
            "S_curr": S_curr,
            "S_next": S_next,
            "dS_curr": dS_curr,
            "dS_next": dS_next,
        })

    return pd.DataFrame(rows)

#-------------------------------------------------------------------------------

def replace_flagged_storage_with_monthly_climatology(
    S_df,
    flagged_storage_table,
    clim_stat="mean",
    exclude_flagged_from_clim=True,
):
    """
    Replace selected GRACE storage anomaly S values with basin-specific
    monthly climatological S values.

    Parameters
    ----------
    S_df : pd.DataFrame
        Wide monthly storage anomaly dataframe.
        index = monthly dates
        columns = basin names
        values = storage anomaly [Gt]

    flagged_storage_table : pd.DataFrame
        Must contain:
            time, basin

        These are the actual storage anomaly months selected for replacement.

    clim_stat : str
        "mean" or "median".

    exclude_flagged_from_clim : bool
        If True, flagged storage months are excluded from the monthly
        climatology calculation.

    Returns
    -------
    S_corr : pd.DataFrame
        Corrected storage anomaly dataframe.

    correction_log : pd.DataFrame
        Diagnostic table showing original and replacement values.
    """

    S = S_df.copy()
    S.index = pd.to_datetime(S.index).to_period("M").to_timestamp()
    S = S.sort_index()

    flagged = flagged_storage_table.copy()
    flagged["time"] = (
        pd.to_datetime(flagged["time"])
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    flagged["basin"] = flagged["basin"].astype(str)

    # Wide to long
    S_long = (
        S
        .reset_index(names="time")
        .melt(id_vars="time", var_name="basin", value_name="S_Gt")
    )

    S_long["basin"] = S_long["basin"].astype(str)
    S_long["month"] = S_long["time"].dt.month

    # Mark flagged rows
    flagged_key = flagged[["time", "basin"]].copy()
    flagged_key["_flagged"] = True

    S_long = S_long.merge(
        flagged_key,
        on=["time", "basin"],
        how="left"
    )

    S_long["_flagged"] = S_long["_flagged"].fillna(False)

    # Compute monthly climatology
    clim_source = S_long.copy()

    if exclude_flagged_from_clim:
        clim_source = clim_source[~clim_source["_flagged"]].copy()

    if clim_stat == "mean":
        clim = (
            clim_source
            .groupby(["basin", "month"], as_index=False)["S_Gt"]
            .mean()
            .rename(columns={"S_Gt": "S_clim_Gt"})
        )
    elif clim_stat == "median":
        clim = (
            clim_source
            .groupby(["basin", "month"], as_index=False)["S_Gt"]
            .median()
            .rename(columns={"S_Gt": "S_clim_Gt"})
        )
    else:
        raise ValueError("clim_stat must be 'mean' or 'median'")

    S_long = S_long.merge(
        clim,
        on=["basin", "month"],
        how="left"
    )

    # Replace flagged values
    S_long["S_original_Gt"] = S_long["S_Gt"]

    mask = S_long["_flagged"]

    S_long.loc[mask, "S_Gt"] = S_long.loc[mask, "S_clim_Gt"]

    S_long["S_correction_Gt"] = (
        S_long["S_Gt"] - S_long["S_original_Gt"]
    )

    correction_log = (
        S_long[S_long["_flagged"]]
        [
            [
                "time",
                "basin",
                "month",
                "S_original_Gt",
                "S_clim_Gt",
                "S_Gt",
                "S_correction_Gt",
            ]
        ]
        .sort_values(["time", "basin"])
        .reset_index(drop=True)
    )

    # Long back to wide
    S_corr = (
        S_long
        .pivot(index="time", columns="basin", values="S_Gt")
        .sort_index()
    )

    # Preserve original basin-column order
    S_corr = S_corr.reindex(columns=S.columns)

    return S_corr, correction_log


#-------------------------------------------------------------------------------
def normalize_basin_name_for_grace(name):
    """
    Normalize basin names to match David's GRACE storage dataframe.
    Currently David's file uses 'Ep-f' rather than 'Ep-F'.
    """
    name_map = {
        "Ep-F": "Ep-f",
    }
    return name_map.get(name, name)

#-------------------------------------------------------------------------------
# =============================================================================
# Diagnostic: inspect storage anomaly S around flagged months
# =============================================================================

def inspect_storage_neighbors(S_df, flagged_table):
    rows = []

    S = S_df.copy()
    S.index = pd.to_datetime(S.index).to_period("M").to_timestamp()
    S = S.sort_index()

    ft = flagged_table.copy()
    ft["time"] = pd.to_datetime(ft["time"]).dt.to_period("M").dt.to_timestamp()

    for _, r in ft.iterrows():
        t = r["time"]
        b = r["basin"]

        prev_t = t - pd.DateOffset(months=1)
        next_t = t + pd.DateOffset(months=1)

        S_prev = S.loc[prev_t, b] if prev_t in S.index else np.nan
        S_curr = S.loc[t, b] if t in S.index else np.nan
        S_next = S.loc[next_t, b] if next_t in S.index else np.nan

        rows.append({
            "time": t,
            "basin": b,
            "S_prev": S_prev,
            "S_curr": S_curr,
            "S_next": S_next,
            "dS_curr": S_curr - S_prev if np.isfinite(S_prev) and np.isfinite(S_curr) else np.nan,
            "dS_next": S_next - S_curr if np.isfinite(S_next) and np.isfinite(S_curr) else np.nan,
        })

    return pd.DataFrame(rows)


#-------------------------------------------------------------------------------
def replace_flagged_deltaS_with_monthly_climatology(
    dS_df,
    flagged_deltaS_table,
    clim_stat="mean",
    exclude_flagged_from_clim=True,
):
    """
    Replace selected monthly ΔS basin values with basin-specific
    calendar-month ΔS climatology.

    Parameters
    ----------
    dS_df : pd.DataFrame
        Wide monthly ΔS dataframe.
        index = monthly dates
        columns = basin names
        values = ΔS [Gt/month]

    flagged_deltaS_table : pd.DataFrame
        Must contain:
            time, basin

        These are the ΔS basin-months selected for replacement.

    clim_stat : str
        "mean" or "median".

    exclude_flagged_from_clim : bool
        If True, flagged ΔS months are excluded from climatology calculation.

    Returns
    -------
    dS_corr : pd.DataFrame
        Corrected ΔS dataframe.

    correction_log : pd.DataFrame
        Diagnostic table showing original and replacement ΔS values.
    """

    dS = dS_df.copy()
    dS.index = pd.to_datetime(dS.index).to_period("M").to_timestamp()
    dS = dS.sort_index()

    flagged = flagged_deltaS_table.copy()
    flagged["time"] = (
        pd.to_datetime(flagged["time"])
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    flagged["basin"] = flagged["basin"].astype(str)

    dS_long = (
        dS
        .reset_index(names="time")
        .melt(id_vars="time", var_name="basin", value_name="dS_Gt")
    )

    dS_long["basin"] = dS_long["basin"].astype(str)
    dS_long["month"] = dS_long["time"].dt.month

    flagged_key = flagged[["time", "basin"]].copy()
    flagged_key["_flagged"] = True

    dS_long = dS_long.merge(
        flagged_key,
        on=["time", "basin"],
        how="left"
    )

    dS_long["_flagged"] = dS_long["_flagged"].fillna(False)

    clim_source = dS_long.copy()

    if exclude_flagged_from_clim:
        clim_source = clim_source[~clim_source["_flagged"]].copy()

    if clim_stat == "mean":
        clim = (
            clim_source
            .groupby(["basin", "month"], as_index=False)["dS_Gt"]
            .mean()
            .rename(columns={"dS_Gt": "dS_clim_Gt"})
        )
    elif clim_stat == "median":
        clim = (
            clim_source
            .groupby(["basin", "month"], as_index=False)["dS_Gt"]
            .median()
            .rename(columns={"dS_Gt": "dS_clim_Gt"})
        )
    else:
        raise ValueError("clim_stat must be 'mean' or 'median'")

    dS_long = dS_long.merge(
        clim,
        on=["basin", "month"],
        how="left"
    )

    dS_long["dS_original_Gt"] = dS_long["dS_Gt"]

    mask = dS_long["_flagged"]
    dS_long.loc[mask, "dS_Gt"] = dS_long.loc[mask, "dS_clim_Gt"]

    dS_long["dS_correction_Gt"] = (
        dS_long["dS_Gt"] - dS_long["dS_original_Gt"]
    )

    correction_log = (
        dS_long[dS_long["_flagged"]]
        [
            [
                "time",
                "basin",
                "month",
                "dS_original_Gt",
                "dS_clim_Gt",
                "dS_Gt",
                "dS_correction_Gt",
            ]
        ]
        .sort_values(["time", "basin"])
        .reset_index(drop=True)
    )

    dS_corr = (
        dS_long
        .pivot(index="time", columns="basin", values="dS_Gt")
        .sort_index()
    )

    dS_corr = dS_corr.reindex(columns=dS.columns)

    return dS_corr, correction_log


#%% PLOT MONTHLY PMB TIME SERIES BY BASIN FOR SELECTED REGION/YEAR
# =============================================================================
# Purpose:
# Plot basin-level monthly PMB values for the affected years/seasons.
# This is the PMB-based diagnostic used to identify outlier basin/month values.
# =============================================================================

def plot_monthly_pmb_by_basin(
    pmb_basin_month_df,
    year,
    basin_ids,
    region_name,
    basin_name_col="basin_name",
    value_col="pmb_mm_month",
    figsize=(14, 4.8),
    ylim=None,
    highlight_months=None,
    title=None,
    legend_ncol=4,
):
    """
    Plot monthly PMB time series by basin for one region/year.

    Parameters
    ----------
    pmb_basin_month_df : DataFrame
        Must contain: time, year, month, basin, basin_name, pmb_mm_month.
    year : int
        Year to plot.
    basin_ids : list
        Basin IDs for the region.
    region_name : str
        Region name for title.
    highlight_months : list or None
        Months to lightly highlight, e.g. [11] or [1, 2].
    """

    sub = pmb_basin_month_df[
        (pmb_basin_month_df["year"] == year) &
        (pmb_basin_month_df["basin"].isin(basin_ids))
    ].copy()

    if sub.empty:
        raise ValueError(f"No PMB basin-month data found for {region_name}, {year}")

    if title is None:
        title = f"{year} {region_name} monthly PMB by basin"

    fig, ax = plt.subplots(figsize=figsize)

    for basin, ss in sub.groupby("basin"):
        ss = ss.sort_values("month")

        label = (
            ss[basin_name_col].iloc[0]
            if basin_name_col in ss.columns
            else str(basin)
        )

        ax.plot(
            ss["month"],
            ss[value_col],
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            label=label,
        )

    # Zero line
    ax.axhline(0, color="black", linewidth=1.0)

    # Highlight affected months
    if highlight_months is not None:
        for m in highlight_months:
            ax.axvspan(
                m - 0.5,
                m + 0.5,
                color="gray",
                alpha=0.12,
                zorder=0,
            )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Month", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$P_{\mathrm{MB}}$ (mm/month)", fontsize=11, fontweight="bold")

    ax.set_xticks(np.arange(1, 13))
    ax.set_xlim(0.5, 12.5)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(
        frameon=False,
        ncol=legend_ncol,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )

    plt.tight_layout()

    return fig, ax
#-----------------------------------------------------------------------------
def plot_basin_bias_ranked(
    bias_tab,
    interior_basins=None,
    coastal_basins=None,
    figsize=(8, 6),
    xcol="mean_bias",
    title="ERA5 - P_MB mean annual bias by basin",
    xlabel="Bias [mm/year]",
):
    df = bias_tab.copy()

    if interior_basins is None:
        interior_basins = []
    if coastal_basins is None:
        coastal_basins = []

    def basin_group(b):
        if b in interior_basins:
            return "Interior-like"
        elif b in coastal_basins:
            return "Coastal-like"
        return "Other"

    df["group"] = df["basin"].map(basin_group)

    color_map = {
        "Interior-like": "tab:blue",
        "Coastal-like": "tab:orange",
        "Other": "gray",
    }
    df["color"] = df["group"].map(color_map)

    df = df.sort_values(xcol, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        df["basin"].astype(str),
        df[xcol],
        color=df["color"],
        edgecolor="black",
        alpha=0.9
    )

    for i, v in enumerate(df[xcol].values):
        ax.text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=10)

    ax.axvline(0, color="k", lw=1.0)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Basin", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    # simple legend
    handles = [
        plt.Line2D([0], [0], color=color_map["Interior-like"], lw=8, label="Interior-like"),
        plt.Line2D([0], [0], color=color_map["Coastal-like"], lw=8, label="Coastal-like"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")

    plt.tight_layout()
    return fig, ax


def plot_basin_metric_ranked(
    df,
    metric_col,
    interior_basins=None,
    coastal_basins=None,
    figsize=(8, 6),
    title=None,
    xlabel=None,
    annotate=True,
):
    import matplotlib.pyplot as plt

    d = df.copy()

    if interior_basins is None:
        interior_basins = []
    if coastal_basins is None:
        coastal_basins = []

    def basin_group(b):
        if b in interior_basins:
            return "Interior-like"
        elif b in coastal_basins:
            return "Coastal-like"
        return "Other"

    d["group"] = d["basin"].map(basin_group)

    color_map = {
        "Interior-like": "tab:blue",
        "Coastal-like": "tab:orange",
        "Other": "gray",
    }
    d["color"] = d["group"].map(color_map)

    d = d.sort_values(metric_col, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        d["basin"].astype(str),
        d[metric_col],
        color=d["color"],
        edgecolor="black",
        alpha=0.9
    )

    if annotate:
        for i, v in enumerate(d[metric_col].values):
            ax.text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=10)

    ax.axvline(0, color="k", lw=1.0)
    ax.set_title(title if title else metric_col, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel if xlabel else metric_col, fontsize=12)
    ax.set_ylabel("Basin", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color=color_map["Interior-like"], lw=8, label="Interior-like"),
        plt.Line2D([0], [0], color=color_map["Coastal-like"], lw=8, label="Coastal-like"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")

    plt.tight_layout()
    return fig, ax


def plot_antarctic_precip_dual_discrete(
    arr_lst,
    basins_da=None,
    basin_label_ids=True,
    basin_id_offsets=None,
    extent=(-180, 180, -90, -60),
    figsize=(12, 6),
    dpi=300,
    cmap="Spectral_r",
    levels=None,
    norm_type="log",         # "log", "power", or "linear"
    gamma=0.6,               # used if norm_type="power"
    vmin=None,
    vmax=None,
    cbar_ticks=None,
    cbar_label="mm/year",
    title_fontsize=14,
    label_fontsize=12,
    basin_linecolor="gray",
    basin_linewidth=0.7,
    panel_letters=False,
    letter_fontsize=14,
    show_panel_mean=True,
    mean_xy=(0.04, 0.96),
    mean_fontsize=12,
    mean_fmt="{:.0f}",
    mean_box_alpha=0.70,
    add_coastline=False,
    facecolor="white",
):
    """
    Plot side-by-side Antarctic polar precipitation maps with:
      - discrete colormap
      - discrete colorbar
      - optional log / power / linear normalization
      - optional basin boundaries + basin IDs

    Parameters
    ----------
    arr_lst : list of tuples
        [(title, dataarray), ...]
        Each dataarray should already be 2D with lat/lon coords or x/y coords.
    basins_da : xarray.DataArray, optional
        Basin mask on SouthPolarStereo x/y grid (e.g. IMBIE basins).
        Used for boundaries and basin labels.
    basin_label_ids : bool
        Whether to annotate basin IDs.
    basin_id_offsets : dict, optional
        Offsets for small basins, e.g. {11: (-4e5, -1e5), ...}
    levels : sequence, optional
        Discrete colorbar bin edges. Example:
        [1, 5, 10, 20, 40, 80, 120, 160, 220, 300]
    norm_type : str
        "log", "power", or "linear"
    """

    if len(arr_lst) == 0:
        raise ValueError("arr_lst is empty.")

    proj = ccrs.SouthPolarStereo()
    pc = ccrs.PlateCarree()
    n = len(arr_lst)

    if levels is None:
        raise ValueError("Please provide discrete `levels` for the precipitation bins.")

    levels = np.asarray(levels, dtype=float)
    if np.any(np.diff(levels) <= 0):
        raise ValueError("`levels` must be strictly increasing.")

    if vmin is None:
        vmin = levels[0]
    if vmax is None:
        vmax = levels[-1]

    cmap_obj = plt.get_cmap(cmap, len(levels) - 1)
    cmap_disc = mcolors.ListedColormap(cmap_obj(np.arange(cmap_obj.N)))
    cmap_disc.set_bad("white")

    # ---- norm ----
    if norm_type == "log":
        if vmin <= 0:
            raise ValueError("For norm_type='log', vmin must be > 0.")
        base_norm = LogNorm(vmin=vmin, vmax=vmax)
    elif norm_type == "power":
        base_norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    elif norm_type == "linear":
        base_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        raise ValueError("norm_type must be one of: 'log', 'power', 'linear'")

    # discrete bins mapped through the chosen scaling
    level_positions = base_norm(levels)
    norm_disc = BoundaryNorm(level_positions, cmap_disc.N, clip=True)

    ncols = n
    fig, axes = plt.subplots(
        1, ncols,
        figsize=figsize,
        dpi=dpi,
        subplot_kw={"projection": proj}
    )
    axes = np.atleast_1d(axes).ravel()

    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.08, top=0.92, wspace=0.03)

    letters = list("abcdefghijklmnopqrstuvwxyz")
    basin_id_offsets = {} if basin_id_offsets is None else basin_id_offsets

    for i, (ax, (title, da, arr_mean)) in enumerate(zip(axes, arr_lst)):
        ax.set_extent(extent, crs=pc)
        ax.set_facecolor(facecolor)

        # ---- choose transform based on coords ----
        if {"latitude", "longitude"}.issubset(set(da.coords)):
            x = da["longitude"].values
            y = da["latitude"].values
            transform = pc
        elif {"lat", "lon"}.issubset(set(da.coords)):
            x = da["lon"].values
            y = da["lat"].values
            transform = pc
        elif {"x", "y"}.issubset(set(da.coords)):
            x = da["x"].values
            y = da["y"].values
            transform = proj
        else:
            raise ValueError(
                f"Could not determine coordinates for panel '{title}'. "
                "Expected coords like (lat, lon), (latitude, longitude), or (x, y)."
            )

        arr = da.values.astype(float)
        arr = np.where(arr <= 0, np.nan, arr) if norm_type == "log" else arr

        # map scaled values to discrete bins
        scaled_arr = base_norm(arr)

        ax.pcolormesh(
            x, y, scaled_arr,
            cmap=cmap_disc,
            norm=norm_disc,
            shading="auto",
            transform=transform,
            zorder=1,
        )

        if add_coastline:
            ax.coastlines(linewidth=0.5, color="black")

        # ---- basin boundaries ----
        if basins_da is not None:
            bda = basins_da
            if "band" in bda.dims:
                bda = bda.isel(band=0)

            bvals = bda.fillna(0).values
            bx = bda["x"].values
            by = bda["y"].values

            boundary_levels = np.arange(0.5, np.nanmax(bvals) + 0.5, 1.0)

            ax.contour(
                bx, by, bvals,
                levels=boundary_levels,
                colors=basin_linecolor,
                linewidths=basin_linewidth,
                transform=proj,
                zorder=5,
            )

            # ---- basin labels ----
            if basin_label_ids:
                max_basin = int(np.nanmax(bvals))
                for basin_id in range(1, max_basin + 1):
                    mask = (bda == basin_id)
                    yy, xx = np.where(mask.values)
                    if len(xx) == 0:
                        continue

                    cx = bx[xx].mean()
                    cy = by[yy].mean()
                    label = str(basin_id)

                    if basin_id in basin_id_offsets:
                        dx, dy = basin_id_offsets[basin_id]
                        lx, ly = cx + dx, cy + dy
                        ax.annotate(
                            label,
                            xy=(cx, cy),
                            xytext=(lx, ly),
                            textcoords="data",
                            xycoords="data",
                            ha="center",
                            va="center",
                            fontsize=label_fontsize,
                            transform=proj,
                            arrowprops=dict(
                                arrowstyle="-",
                                lw=0.8,
                                color=basin_linecolor
                            ),
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                fc="white",
                                ec="none",
                                alpha=0.75
                            ),
                            zorder=8,
                        )
                    else:
                        ax.text(
                            cx, cy, label,
                            color="black",
                            fontsize=label_fontsize,
                            ha="center",
                            va="center",
                            transform=proj,
                            zorder=8,
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                fc="white",
                                ec="none",
                                alpha=0.65
                            ),
                        )

        # ---- panel title ----
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=10)

        # ---- panel letter ----
        if panel_letters and i < len(letters):
            ax.text(
                0.02, 0.98, f"({letters[i]})",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=letter_fontsize,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.70
                ),
                zorder=20,
            )

        # ---- panel mean ----
        if show_panel_mean:
            # panel_mean = float(np.nanmean(arr))
            ax.text(
                mean_xy[0], mean_xy[1],
                f"Mean = {mean_fmt.format(arr_mean)}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=mean_fontsize,
                fontweight="bold",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    facecolor="white",
                    edgecolor="none",
                    alpha=mean_box_alpha
                ),
                zorder=20,
            )

        ax.axis("off")

    # ---- discrete colorbar ----
    cax = fig.add_axes([0.92, 0.18, 0.018, 0.62])
    sm = ScalarMappable(norm=norm_disc, cmap=cmap_disc)
    sm.set_array([])

    cb = fig.colorbar(
        sm,
        cax=cax,
        orientation="vertical",
        boundaries=level_positions,
        spacing="proportional",
    )

    if cbar_ticks is None:
        tick_vals = levels
    else:
        tick_vals = np.asarray(cbar_ticks, dtype=float)

    cb.set_ticks(base_norm(tick_vals))
    cb.set_ticklabels([f"{t:g}" for t in tick_vals])
    cb.ax.tick_params(labelsize=11)
    cb.ax.minorticks_off()
    cb.ax.set_title(cbar_label, fontsize=12, pad=10)

    return fig, axes, cb

def get_zonal(spatial_product, mask, y, axis=(0, 1)):
    cosines = np.cos(np.radians(y))[:, np.newaxis]  # Broadcast to match mask shape
    cosines = np.where(mask, cosines, 0)
    weights = cosines / np.nansum(cosines)

    return np.nansum(weights * spatial_product, axis=axis)

def cosine_weighted_mean_2d(data, region_mask, lat_vals):
    """
    Cosine-weighted mean over a 2D lat-lon field.

    Parameters
    ----------
    data : 2D ndarray (lat, lon)
        Field to average.
    region_mask : 2D boolean ndarray (lat, lon)
        True where region is included.
    lat_vals : 1D ndarray
        Latitude values matching data.shape[0].

    Returns
    -------
    float
        Cosine-weighted regional mean.
    """
    lat_weights = np.cos(np.deg2rad(lat_vals)).reshape(-1, 1)
    weights = lat_weights * np.ones_like(data)

    valid = np.isfinite(data) & region_mask
    if valid.sum() == 0:
        return np.nan

    return np.nansum(data * weights * valid) / np.nansum(weights * valid)

def cosine_weighted_mean_timeseries(data, region_mask, lat_vals):
    """
    Cosine-weighted mean over a 3D field (time, lat, lon).

    Returns
    -------
    1D ndarray
        Regional mean time series.
    """
    out = []
    for t in range(data.shape[0]):
        # out.append(cosine_weighted_mean_2d(data[t], region_mask, lat_vals))
        out.append(get_zonal(data[t], region_mask, lat_vals))
    return np.array(out)


def make_region_mask_from_basin_ids(basin_id_grid, basin_ids):
    """
    basin_id_grid : 2D ndarray of basin IDs
    basin_ids : list of ints

    Returns
    -------
    2D boolean mask
    """
    return np.isin(basin_id_grid, basin_ids)


def regrid_basin_ids_to_product_grid(basin_da, target_lat, target_lon,
                                     basin_lat_name="y", basin_lon_name="x"):
    """
    Regrid categorical basin IDs to product lat-lon grid using nearest neighbor.
    """
    if basin_lat_name != "lat" or basin_lon_name != "lon":
        basin_da = basin_da.rename({basin_lat_name: "lat", basin_lon_name: "lon"})

    basin_on_target = basin_da.sel(
        lat=xr.DataArray(target_lat, dims="lat"),
        lon=xr.DataArray(target_lon, dims="lon"),
        method="nearest"
    )
    return basin_on_target


import xarray as xr
import numpy as np
import rioxarray
from rasterio.enums import Resampling

def prepare_latlon_template(da, lat_name="lat", lon_name="lon", crs="EPSG:4326"):
    """
    Prepare a 2D lat-lon DataArray as a template for rioxarray reproject_match.
    """
    if lat_name != "y" or lon_name != "x":
        da = da.rename({lat_name: "y", lon_name: "x"})
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    da = da.rio.write_crs(crs, inplace=False)
    return da

def reproject_basin_ids_to_match(basin_da, target_da):
    """
    Reproject categorical basin IDs to the target grid using nearest-neighbor.
    """
    basin2d = basin_da.squeeze(drop=True)

    if not basin2d.rio.crs:
        raise ValueError("Source basin raster has no CRS.")

    basin_on_target = basin2d.rio.reproject_match(
        target_da,
        resampling=Resampling.nearest
    )
    return basin_on_target

#---------------------------------------------------------------------------------

def build_region_specific_negative_pmb_table(

    pmb_basin_month_df,

    region_basins,

    region_thresholds,

    value_col="pmb_mm_month",

):

    """

    Build a basin/month exclusion table using region-specific PMB thresholds.

    Parameters

    ----------

    pmb_basin_month_df : DataFrame

        Must include time, year, month, basin, and value_col.

    region_basins : dict

        Region -> list of basin IDs.

    region_thresholds : dict

        Region -> threshold in mm/month.

        Example: {"West Antarctica": -50, "East Antarctica": -25}

    value_col : str

        PMB monthly value column.

    Returns

    -------

    exclusion_table : DataFrame

        Basin/month rows where PMB is below the region-specific threshold.

    """

    rows = []

    for region, threshold in region_thresholds.items():

        basins = region_basins[region]

        sub = pmb_basin_month_df[

            (pmb_basin_month_df["basin"].isin(basins)) &

            (pmb_basin_month_df[value_col] < threshold)

        ].copy()

        sub["region"] = region

        sub["threshold_mm_month"] = threshold

        rows.append(sub)

    exclusion_table = (
        pd.concat(rows, ignore_index=True)

        .sort_values(["time", "basin"])

        .reset_index(drop=True)
    )

    return exclusion_table
#---------------------------------------------------------------------------------
def outlier_id_list_to_table(outlier_list, pmb_basin_month_df):
    """
    Convert manually specified outlier basin/month list into a dataframe
    compatible with mask_basin_months_from_negative_pmb_table().
    """

    rows = []

    for item in outlier_list:
        yy = int(item["year"])
        mm = int(item["month"])
        basins = [int(b) for b in item["basins"]]

        sub = pmb_basin_month_df[
            (pmb_basin_month_df["year"] == yy) &
            (pmb_basin_month_df["month"] == mm) &
            (pmb_basin_month_df["basin"].isin(basins))
        ].copy()

        sub["region"] = "East Antarctica"
        sub["threshold_mm_month"] = "targeted_EAIS_DJF_2017"

        rows.append(sub)

    return pd.concat(rows, ignore_index=True)


#-  ----------------------------------------------------------------------------
def annual_regional_means_from_daily_xr(
    da_daily,
    region_masks,
    lat_name="lat",
    lon_name="lon",
    annual_mode="sum"
):
    """
    Compute annual regional means from daily lat-lon xarray data.

    Parameters
    ----------
    da_daily : xr.DataArray
        Daily precipitation field with dims (time, lat, lon) or (time, latitude, longitude)
    region_masks : dict
        {"Antarctica": mask2d, "West Antarctica": mask2d, "East Antarctica": mask2d}
    annual_mode : str
        "sum" -> annual accumulation [mm/year]
        "mean" -> annual mean daily precipitation [mm/day]

    Returns
    -------
    pd.DataFrame with columns [year, region, precipitation]
    """
    if lat_name not in da_daily.dims or lon_name not in da_daily.dims:
        raise ValueError(f"Expected dims including {lat_name} and {lon_name}")
    # if annual_mode == "sum":
    #     da_annual = da_daily.groupby("time.year").sum("time", skipna=True)
    # elif annual_mode == "mean":
    #     da_annual = da_daily.groupby("time.year").mean("time", skipna=True)
    # else:
    #     raise ValueError("annual_mode must be 'sum' or 'mean'")  

    rows = []
    yrs = np.unique(da_daily.time.dt.year.values)
    for yr in yrs:

        arr = da_daily.where(da_daily.time.dt.year == yr, drop=True)

        if annual_mode == "sum":
            arr = arr.sum(dim="time", skipna=True)
        elif annual_mode == "mean":
            arr = arr.mean(dim="time", skipna=True)
        else:
            raise ValueError("annual_mode must be 'sum' or 'mean'")
        lat_vals = arr[lat_name].values
        # arr = da_annual.sel(year=yr).values
        for region_name, region_mask in region_masks.items():
            # mean_val = cosine_weighted_mean_2d(arr, region_mask, lat_vals)
            mean_val = get_zonal(arr, region_mask, lat_vals)
            if annual_mode == "mean":
                mean_val *= 365
            rows.append({
                "year": int(yr),
                "region": region_name,
                "precipitation": float(mean_val)
            })

    return pd.DataFrame(rows)


def plot_regional_annual_timeseries(
    df,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=("ERA5", "GPCP v3.3", r"$P_{\mathrm{MB}}$"),
    product_styles=None,
    figsize=(11, 12),
    ylabel="[mm/year]"
):
    if product_styles is None:
        product_styles = {
            "ERA5": dict(color="blue", marker="s", lw=2.3),
            "GPCP v3.3": dict(color="orange", marker="D", lw=2.3),
            r'$P_{\mathrm{MB}}$': dict(color="black", marker="o", lw=2.3),
        }

    fig, axes = plt.subplots(len(region_order), 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)

    for ax, region in zip(axes, region_order):
        sub = df[df["region"] == region]

        for prod in product_order:
            s = sub[sub["product"] == prod].sort_values("year")
            if len(s) == 0:
                continue
            style = product_styles.get(prod, {})
            ax.plot(
                s["year"], s["precipitation"],
                label=prod,
                **style
            )

        ax.set_title(region, fontsize=18, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=12)

    axes[-1].set_xlabel("Year", fontsize=16, fontweight="bold")
    fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical",
             fontsize=18, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(product_order),
               frameon=False, fontsize=14)

    plt.tight_layout(rect=[0.06, 0.06, 1, 1])
    return fig, axes


def plot_multiyear_mean_bar_by_region(
    df_mean,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(r"$P_{\mathrm{MB}}$","ERA5", "GPCP v3.3"),
    colors=None,
    figsize=(9, 6),
    ylabel="[mm/year]",
    title="2013–2020 mean annual precipitation"
):
    if colors is None:
        colors = {
            "ERA5": "blue",
            "GPCP v3.3": "orange",
            r"$P_{\mathrm{MB}}$": "black",
        }

    x = np.arange(len(region_order))
    nprod = len(product_order)
    width = 0.35 if nprod == 2 else 0.8 / nprod

    fig, ax = plt.subplots(figsize=figsize)

    for i, prod in enumerate(product_order):
        vals = []
        for reg in region_order:
            sub = df_mean[(df_mean["region"] == reg) & (df_mean["product"] == prod)]
            vals.append(sub["precipitation"].iloc[0] if len(sub) else np.nan)

        xpos = x + (i - (nprod - 1) / 2) * width
        bars = ax.bar(xpos, vals, width=width, color=colors.get(prod, None), label=prod)

        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 1.8,
                    f"{v:.0f}",
                    ha="center", va="bottom", fontsize=10
                )

    ax.set_xticks(x)
    ax.set_xticklabels(region_order, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=12)

    plt.tight_layout()
    return fig, ax


def regional_dict_to_tidy_df(region_dict):
    rows = []
    for region, df in region_dict.items():
        product_cols = [c for c in df.columns if c != "year"]
        for _, row in df.iterrows():
            for product in product_cols:
                rows.append({
                    "year": int(row["year"]),
                    "region": region,
                    "product": product,
                    "precipitation": float(row[product]),
                })
    return pd.DataFrame(rows)


def compute_relative_differences(df, ref_product, target_products):
    """
    Compute target - reference by year and region.

    Parameters
    ----------
    df : tidy DataFrame
        Columns: year, region, product, precipitation
    ref_product : str
        Reference product name
    target_products : list of str
        Products to subtract from reference

    Returns
    -------
    diff_df : tidy DataFrame
        Columns: year, region, product, precipitation
        where 'product' is e.g. 'ERA5 - PMB'
    """
    rows = []

    for region in df["region"].unique():
        for year in sorted(df["year"].unique()):
            sub = df[(df["region"] == region) & (df["year"] == year)]

            ref_row = sub[sub["product"] == ref_product]
            if len(ref_row) == 0:
                continue
            ref_val = ref_row["precipitation"].iloc[0]

            for prod in target_products:
                prod_row = sub[sub["product"] == prod]
                if len(prod_row) == 0:
                    continue
                prod_val = prod_row["precipitation"].iloc[0]

                rows.append({
                    "year": int(year),
                    "region": region,
                    "product": f"{prod} - {ref_product}",
                    "precipitation": float(prod_val - ref_val),
                })

    return pd.DataFrame(rows)


# --------------------------------------------------------
# 4. Compute annual regional P_MB using cosine weighting
# --------------------------------------------------------
REGION_BASINS = {
    "Antarctica": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "West Antarctica": [10, 11, 12, 13, 14, 15, 16, 17],
    "East Antarctica": [2, 3, 4, 5, 6, 7, 8, 9, 18, 19],
}

def make_region_mask_from_basin_ids_xr(basin_mask, basin_ids):
    return xr.DataArray(
        np.isin(basin_mask.values, basin_ids),
        coords=basin_mask.coords,
        dims=basin_mask.dims,
    )

def cosine_weighted_mean_timeseries_xr(da, region_mask, lat_name="lat", lon_name="lon"):
    """
    da         : xr.DataArray(date, lat, lon)
    region_mask: xr.DataArray(lat, lon) boolean
    returns    : xr.DataArray(date)
    """
    lat_weights = xr.DataArray(
        np.cos(np.deg2rad(da[lat_name].values)),
        coords={lat_name: da[lat_name].values},
        dims=(lat_name,),
    )

    # broadcast lat weights to 2D
    w2d = lat_weights.broadcast_like(region_mask)

    valid = region_mask & da.notnull()

    num = (da.where(valid) * w2d.where(valid)).sum(dim=(lat_name, lon_name), skipna=True)
    den = w2d.where(valid).sum(dim=(lat_name, lon_name), skipna=True)

    return num / den

def get_zonal_timeseries(data_3d, mask, y):
    """
    Apply get_zonal over time.

    Parameters
    ----------
    data_3d : 3D ndarray
        Shape (time, lat, lon)
    mask : 2D boolean ndarray
        Shape (lat, lon)
    y : 1D ndarray
        Latitude values in degrees

    Returns
    -------
    1D ndarray
        Cosine-weighted regional mean time series
    """
    out = np.full(data_3d.shape[0], np.nan, dtype=float)

    for t in range(data_3d.shape[0]):
        out[t] = get_zonal(data_3d[t], mask, y)

    return out
REGION_BASINS = {
    "Antarctica": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "West Antarctica": [10, 11, 12, 13, 14, 15, 16, 17],
    "East Antarctica": [2, 3, 4, 5, 6, 7, 8, 9, 18, 19],
}

def compute_pmb_cosine_weighted_annual(
    pmb_monthly_latlon,
    basin_mask_latlon,
    region_basins=REGION_BASINS,
    time_name="date",
    lat_name="lat",
    lon_name="lon",
    annual_mode="sum",
):
    """
    Compute regional annual P_MB using the same cosine-weighted logic as get_zonal.

    Parameters
    ----------
    pmb_monthly_latlon : xr.DataArray
        Monthly P_MB on lat-lon grid, dims (time/date, lat, lon), units mm/month
    basin_mask_latlon : xr.DataArray
        Basin IDs on same lat-lon grid
    annual_mode : str
        'sum' -> annual total [mm/year]
        'mean' -> mean monthly value [mm/month]

    Returns
    -------
    pd.DataFrame
        Columns: year, region, P_MB
    """
    if time_name != "time":
        da = pmb_monthly_latlon.rename({time_name: "time"})
    else:
        da = pmb_monthly_latlon.copy()

    y = da[lat_name].values
    data = da.values
    basin_ids_2d = basin_mask_latlon.values

    out = []

    for region_name, basin_ids in region_basins.items():
        region_mask = np.isin(basin_ids_2d, basin_ids)

        monthly_ts = get_zonal_timeseries(data, region_mask, y)

        ts_df = pd.DataFrame({
            "time": pd.to_datetime(da["time"].values),
            "P_MB": monthly_ts
        })
        ts_df["year"] = ts_df["time"].dt.year
        ts_df["region"] = region_name

        if annual_mode == "sum":
            ann = (
                ts_df.groupby(["year", "region"], as_index=False)["P_MB"]
                .sum()
            )
        elif annual_mode == "mean":
            ann = (
                ts_df.groupby(["year", "region"], as_index=False)["P_MB"]
                .mean()
            )
        else:
            raise ValueError("annual_mode must be 'sum' or 'mean'")

        out.append(ann)

    return pd.concat(out, ignore_index=True)
#---------------------------------------------------------------------------------
def monthly_regional_df_to_conventional_seasonal(

    df,

    time_col="time",

    region_col="region",

    product_col="product",

    value_col="precipitation",

    require_complete_season=True,

):

    """

    Convert regional monthly precipitation dataframe to conventional seasonal means.

    Input dataframe format:

        time | region | product | precipitation

    Output dataframe format:

        time | season_year | season | region | product | precipitation | n_months

    DJF is assigned to the year of January/February.

    Example:

        Dec 2013 + Jan 2014 + Feb 2014 -> DJF 2014

    """

    out = df.copy()

    out[time_col] = pd.to_datetime(out[time_col])

    out["year"] = out[time_col].dt.year

    out["month"] = out[time_col].dt.month

    month_to_season = {

        12: "DJF", 1: "DJF", 2: "DJF",

        3: "MAM", 4: "MAM", 5: "MAM",

        6: "JJA", 7: "JJA", 8: "JJA",

        9: "SON", 10: "SON", 11: "SON",

    }

    season_mid_month = {

        "DJF": 1,

        "MAM": 4,

        "JJA": 7,

        "SON": 10,

    }

    season_order = {

        "DJF": 1,

        "MAM": 2,

        "JJA": 3,

        "SON": 4,

    }

    out["season"] = out["month"].map(month_to_season)

    # Assign December to the following DJF year

    out["season_year"] = out["year"]

    out.loc[out["month"] == 12, "season_year"] += 1

    grouped = (

        out

        .groupby([region_col, product_col, "season_year", "season"], as_index=False)

        .agg(

            precipitation=(value_col, "mean"),

            n_months=(value_col, "count"),

        )

    )

    if require_complete_season:

        grouped = grouped[grouped["n_months"] == 3].copy()

    grouped["season_order"] = grouped["season"].map(season_order)

    grouped["time"] = [

        pd.Timestamp(

            year=int(y),

            month=season_mid_month[s],

            day=1

        )

        for y, s in zip(grouped["season_year"], grouped["season"])

    ]

    grouped = grouped.sort_values(

        [region_col, product_col, "season_year", "season_order"]

    ).reset_index(drop=True)

    grouped = grouped[

        ["time", "season_year", "season", region_col, product_col, "precipitation", "n_months"]

    ]

    return grouped
#---------------------------------------------------------------------------------
def plot_seasonal_timeseries_regions(

    seasonal_df,

    region_order=("Antarctica", "West Antarctica", "East Antarctica"),

    product_order=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP v3.3"),

    product_styles=None,

    figsize=(12, 8),

    ylabel="Precipitation [mm/month]",

    y_nbins=4,

    legend_ncol=4,

    title_suffix="conventional seasonal input",

    x_major_year_interval=1,

):

    """

    Plot conventional seasonal-mean precipitation time series

    for multiple regions in stacked panels.

    Expected seasonal_df columns:

        time | season_year | season | region | product | precipitation

    Notes

    -----

    This function assumes that the input dataframe has already been converted

    from monthly to conventional seasonal means.

    """

    df = seasonal_df.copy()

    df["time"] = pd.to_datetime(df["time"])

    fig, axes = plt.subplots(

        len(region_order),

        1,

        figsize=figsize,

        sharex=True

    )

    if len(region_order) == 1:

        axes = [axes]

    for ax, region in zip(axes, region_order):

        sub = df[df["region"] == region].copy()

        for prod in product_order:

            ss = sub[sub["product"] == prod].copy()

            if ss.empty:

                continue

            ss = ss.sort_values("time")

            style = {} if product_styles is None else product_styles.get(prod, {}).copy()

            ax.plot(

                ss["time"],

                ss["precipitation"],

                label=prod,

                **style

            )

        ax.set_title(

            f"{region} — {title_suffix}",

            fontweight="bold",

            fontsize=18

        )

        ax.grid(True, alpha=0.3)

        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=y_nbins))

    fig.supylabel(ylabel, x=0.06, fontweight="bold", fontsize=18)

    # Clean yearly x-axis

    axes[-1].xaxis.set_major_locator(

        mdates.YearLocator(base=x_major_year_interval)

    )

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(

        handles,

        labels,

        loc="lower center",

        bbox_to_anchor=(0.5, -0.03),

        ncol=legend_ncol,

        fontsize=15,

        frameon=False

    )

    plt.tight_layout(rect=[0.05, 0.06, 1, 1])

    return fig, axes


#--------------------------------------------------------------------------------
def plot_seasonal_climatology_1x3(
    clim_df,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(
        r"$P_{\mathrm{MB}}$",
        "ERA5",
        "GPCP V3.3",
        "GPM PMW V07",
        "GPM PMW V07 (corr.)",
    ),
    product_styles=None,
    figsize=(15, 4.8),
    ylabel="mm/season",
    y_nbins=4,
    legend_ncol=5,
):
    season_labels = ["DJF", "MAM", "JJA", "SON"]

    fig, axes = plt.subplots(
        1, len(region_order),
        figsize=figsize,
        sharex=True,
        sharey=False
    )

    if len(region_order) == 1:
        axes = [axes]

    for ax, region in zip(axes, region_order):

        sub = clim_df[clim_df["region"] == region].copy()

        for prod in product_order:
            ss = sub[sub["product"] == prod].copy()

            if ss.empty:
                continue

            ss["season"] = pd.Categorical(
                ss["season"],
                categories=season_labels,
                ordered=True
            )
            ss = ss.sort_values("season")

            style = {} if product_styles is None else product_styles.get(prod, {})

            ax.plot(
                ss["season"],
                ss["precipitation"],
                label=prod,
                color=style.get("color", None),
                marker=style.get("marker", None),
                linestyle=style.get("linestyle", style.get("ls", "-")),
                linewidth=style.get("lw", style.get("linewidth", 2.5)),
                markersize=style.get("ms", style.get("markersize", 6)),
            )

        ax.set_title(region, fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))

        if ax == axes[0]:
            ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")

        ax.tick_params(axis="both", labelsize=13)

    # one shared legend below all panels
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=legend_ncol,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    return fig, axes
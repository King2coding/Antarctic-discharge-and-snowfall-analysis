import math
import pandas as pd
import matplotlib.pyplot as plt

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


import pandas as pd

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



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
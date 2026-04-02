import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
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


    import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, LogNorm, PowerNorm
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs


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

    for i, (ax, (title, da)) in enumerate(zip(axes, arr_lst)):
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
            panel_mean = float(np.nanmean(arr))
            ax.text(
                mean_xy[0], mean_xy[1],
                f"Mean = {mean_fmt.format(panel_mean)}",
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
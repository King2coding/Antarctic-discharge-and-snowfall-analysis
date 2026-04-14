import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from program_utils import *
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


import xarray as xr

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


import pandas as pd
import numpy as np

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
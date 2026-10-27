#%%
"""Functions for a strict coincident PMB/product comparison including UA-HIPA/AIRS.

These functions are intentionally independent of file paths. After the existing
workflow has prepared every product on the common 0.1-degree basin grid, pass
the monthly DataArrays to ``prepare_all_product_common_period``. The current
intersection is expected to be January-December 2018; additional complete years
will be included automatically as UA-HIPA expands.
"""

#%%
# =============================================================================
# SECTION 1. IMPORTS AND DISPLAY SETTINGS
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from uahipa_airs_monthly_utils import subset_products_to_common_months
from program_utile_13Apr2026 import product_styles_corr


# One shared project style dictionary.  Keeping this alias avoids a second,
# potentially divergent color definition in the UA-HIPA/AIRS helper module.
PRODUCT_STYLES = product_styles_corr

SEASON_MONTHS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5),
                 "JJA": (6, 7, 8), "SON": (9, 10, 11)}


#%%
# =============================================================================
# SECTION 2. STRICT TEMPORAL COINCIDENCE
# =============================================================================

def prepare_all_product_common_period(
    pmb: xr.DataArray,
    era5: xr.DataArray,
    gpcp_v33: xr.DataArray,
    gpm_pmw_v08: xr.DataArray,
    uahipa_q75: xr.DataArray,
    airs: xr.DataArray,
    require_complete_calendar_years: bool = False,
) -> tuple[dict[str, xr.DataArray], pd.DatetimeIndex]:
    """Return all six products on exact coincident months.

    Set ``require_complete_calendar_years=True`` only for analyses that must
    discard partial years. The main workflow retains partial common years for
    monthly and seasonal work, then subsets complete years for annual figures.
    """

    products = {
        r"$P_{\mathrm{MB}}$": pmb,
        "ERA5": era5,
        "GPCP V3.3": gpcp_v33,
        "GPM PMW V08": gpm_pmw_v08,
        "UA-HIPA q75": uahipa_q75,
        "AIRS": airs,
    }
    common_products, common_months = subset_products_to_common_months(
        products,
        require_complete_calendar_years=require_complete_calendar_years,
    )
    print(
        "Strict common comparison period:",
        common_months.min().strftime("%Y-%m"),
        "to",
        common_months.max().strftime("%Y-%m"),
        f"({len(common_months)} months)",
    )
    return common_products, common_months


#%%
# =============================================================================
# SECTION 3. REGIONAL SERIES AND METRICS
# =============================================================================

def cosine_weighted_regional_series(
    products: Mapping[str, xr.DataArray],
    region_masks: Mapping[str, xr.DataArray],
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> pd.DataFrame:
    """Build tidy monthly cosine-area-weighted regional means."""

    rows = []
    for product_name, field in products.items():
        latitude_weights = np.cos(np.deg2rad(field[lat_name]))
        for region_name, mask in region_masks.items():
            region_field = field.where(mask)
            series = region_field.weighted(latitude_weights).mean((lat_name, lon_name))
            for timestamp, value in zip(pd.to_datetime(series.time.values), series.values):
                rows.append(
                    {
                        "time": pd.Timestamp(timestamp),
                        "year": int(timestamp.year),
                        "month": int(timestamp.month),
                        "region": region_name,
                        "product": product_name,
                        "precipitation_mm_month": float(value),
                    }
                )
    return pd.DataFrame(rows)


def paired_product_metrics(
    regional_monthly: pd.DataFrame,
    reference_product: str = r"$P_{\mathrm{MB}}$",
) -> pd.DataFrame:
    """Calculate paired monthly bias, MAE, RMSE, and correlation against PMB."""

    rows = []
    for region, region_frame in regional_monthly.groupby("region", sort=False):
        pivot = region_frame.pivot(index="time", columns="product", values="precipitation_mm_month")
        if reference_product not in pivot:
            raise KeyError(f"Reference product {reference_product!r} missing for {region}")
        reference = pivot[reference_product]
        for product in pivot.columns:
            if product == reference_product:
                continue
            paired = pd.concat([reference, pivot[product]], axis=1, keys=["reference", "product"]).dropna()
            error = paired["product"] - paired["reference"]
            rows.append(
                {
                    "region": region,
                    "product": product,
                    "reference": reference_product,
                    "n_common_months": int(len(paired)),
                    "start_month": paired.index.min(),
                    "end_month": paired.index.max(),
                    "mean_bias_mm_month": float(error.mean()),
                    "mae_mm_month": float(error.abs().mean()),
                    "rmse_mm_month": float(np.sqrt(np.mean(error**2))),
                    "pearson_r": float(paired.corr().iloc[0, 1]) if len(paired) >= 2 else np.nan,
                }
            )
    return pd.DataFrame(rows)


#%%
# =============================================================================
# SECTION 4. COINCIDENT-PERIOD PLOTS
# =============================================================================

def plot_regional_monthly_comparison(
    regional_monthly: pd.DataFrame,
    output_path: str | Path | None = None,
):
    """Plot one monthly comparison panel per region for the strict common period."""

    regions = regional_monthly["region"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(regions),
        1,
        figsize=(13, 3.8 * len(regions)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for axis, region in zip(axes, regions):
        region_frame = regional_monthly[regional_monthly["region"] == region]
        for product, product_frame in region_frame.groupby("product", sort=False):
            style = PRODUCT_STYLES.get(product, {})
            axis.plot(
                product_frame["time"],
                product_frame["precipitation_mm_month"],
                label=product,
                **style,
            )
        axis.set_title(region)
        axis.set_ylabel("Precipitation [mm month$^{-1}$]")
        axis.grid(alpha=0.25)

    axes[0].legend(ncol=3, frameon=False)
    axes[-1].set_xlabel("Month")

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, bbox_inches="tight")
    return figure, axes


def plot_regional_mean_annual_comparison(
    regional_monthly: pd.DataFrame,
    output_path: str | Path | None = None,
):
    """Plot mean annual totals from complete common calendar years only."""

    annual = (
        regional_monthly.groupby(["region", "product", "year"], as_index=False)
        ["precipitation_mm_month"]
        .sum()
        .rename(columns={"precipitation_mm_month": "precipitation_mm_year"})
    )
    climatology = (
        annual.groupby(["region", "product"], as_index=False)["precipitation_mm_year"]
        .mean()
    )

    regions = climatology["region"].drop_duplicates().tolist()
    products = climatology["product"].drop_duplicates().tolist()
    x = np.arange(len(regions))
    width = 0.82 / len(products)
    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)

    for index, product in enumerate(products):
        product_values = (
            climatology[climatology["product"] == product]
            .set_index("region")
            .reindex(regions)["precipitation_mm_year"]
        )
        color = PRODUCT_STYLES.get(product, {}).get("color")
        axis.bar(
            x - 0.41 + width / 2 + index * width,
            product_values.values,
            width,
            label=product,
            color=color,
        )

    axis.set_xticks(x, regions)
    axis.set_ylabel("Mean annual precipitation [mm yr$^{-1}$]")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, frameon=False)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, bbox_inches="tight")
    return figure, axis, annual, climatology


#%%
# =============================================================================
# SECTION 5. MANUSCRIPT-ANALOGOUS TABLES
# =============================================================================

def annual_and_mean_annual(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum monthly values by calendar year, requiring all 12 months."""
    annual = (monthly.groupby(["region", "product", "year"], as_index=False)
              ["precipitation_mm_month"].sum(min_count=12)
              .rename(columns={"precipitation_mm_month": "precipitation_mm_year"}))
    mean_annual = (annual.groupby(["region", "product"], as_index=False)
                   ["precipitation_mm_year"].mean())
    return annual, mean_annual


def seasonal_climatology(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return seasonal totals averaged across the strict common years.

    DJF follows the manuscript climatology convention: all December, January,
    and February monthly climatological values are summed. With one available
    year this is Dec 2018 + Jan 2018 + Feb 2018, not a contiguous seasonal-year
    DJF; the figure subtitle states this limitation.
    """
    rows = []
    for (region, product), frame in monthly.groupby(["region", "product"], sort=False):
        month_clim = frame.groupby("month")["precipitation_mm_month"].mean()
        for season, months in SEASON_MONTHS.items():
            values = month_clim.reindex(months)
            rows.append({"region": region, "product": product, "season": season,
                         "precipitation_mm_season": values.sum(min_count=len(months))})
    return pd.DataFrame(rows)


def seasonal_climatology_with_common_month_fallback(
    regional_monthly: pd.DataFrame,
    regional_pmb_uncertainty: pd.DataFrame,
    min_common_months: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build seasonal totals, using a scaled partial season only as fallback.

    Complete conventional seasons use their ordinary three-month sum. If a
    named season has no complete season anywhere in the common record, a
    season-year with at least ``min_common_months`` jointly valid months across
    every product is estimated as ``3 * mean(available monthly totals)``.
    PMB uncertainty is propagated as ``(3 / n) * sqrt(sum(sigma_month**2))``.
    """
    if not 2 <= min_common_months <= 3:
        raise ValueError("min_common_months must be 2 or 3")

    data = regional_monthly.copy()
    data["time"] = pd.to_datetime(data["time"])
    data["year"] = data["time"].dt.year
    data["month"] = data["time"].dt.month
    month_to_season = {month: season for season, months in SEASON_MONTHS.items() for month in months}
    data["season"] = data["month"].map(month_to_season)
    data["season_year"] = data["year"]
    data.loc[data["month"] == 12, "season_year"] += 1

    value_col = "precipitation_mm_month"
    availability = data.pivot_table(
        index=["region", "season", "season_year", "time", "month"],
        columns="product", values=value_col, aggfunc="first",
    )
    common = availability.notna().all(axis=1).rename("jointly_valid").reset_index()
    common = common[common["jointly_valid"]].drop(columns="jointly_valid")

    selected_rows = []
    for (region, season), season_months in common.groupby(["region", "season"], sort=False):
        groups = list(season_months.groupby("season_year", sort=True))
        complete = [(year, frame) for year, frame in groups if frame["month"].nunique() == 3]
        chosen = complete if complete else [
            (year, frame) for year, frame in groups
            if frame["month"].nunique() >= min_common_months
        ]
        for season_year, frame in chosen:
            months = sorted(frame["month"].unique().tolist())
            selected_rows.append({
                "region": region, "season": season, "season_year": int(season_year),
                "n_months_used": len(months),
                "months_used": ",".join(str(month) for month in months),
                "is_partial_season": len(months) < 3,
            })
    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        raise RuntimeError("No seasons meet the common-month coverage requirement")

    seasonal_rows = []
    for record in selected.itertuples(index=False):
        months = [int(month) for month in record.months_used.split(",")]
        subset = data[
            (data["region"] == record.region)
            & (data["season"] == record.season)
            & (data["season_year"] == record.season_year)
            & (data["month"].isin(months))
        ]
        scale = 3.0 / record.n_months_used
        for product, frame in subset.groupby("product", sort=False):
            seasonal_rows.append({
                "region": record.region, "product": product,
                "season": record.season, "season_year": record.season_year,
                "precipitation": scale * frame[value_col].sum(min_count=record.n_months_used),
                "n_months_used": record.n_months_used,
                "months_used": record.months_used,
                "is_partial_season": record.is_partial_season,
            })
    seasonal_by_year = pd.DataFrame(seasonal_rows)
    seasonal_clim = (
        seasonal_by_year.groupby(["region", "product", "season"], as_index=False)
        .agg(
            precipitation=("precipitation", "mean"),
            n_seasons=("season_year", "nunique"),
            minimum_months_used=("n_months_used", "min"),
            used_partial_season=("is_partial_season", "any"),
        )
    )

    uncertainty = regional_pmb_uncertainty.copy()
    uncertainty["time"] = pd.to_datetime(uncertainty["time"])
    uncertainty["year"] = uncertainty["time"].dt.year
    uncertainty["month"] = uncertainty["time"].dt.month
    uncertainty["season"] = uncertainty["month"].map(month_to_season)
    uncertainty["season_year"] = uncertainty["year"]
    uncertainty.loc[uncertainty["month"] == 12, "season_year"] += 1
    uncertainty_rows = []
    for record in selected.itertuples(index=False):
        months = [int(month) for month in record.months_used.split(",")]
        values = uncertainty[
            (uncertainty["region"] == record.region)
            & (uncertainty["season"] == record.season)
            & (uncertainty["season_year"] == record.season_year)
            & (uncertainty["month"].isin(months))
        ]["pmb_uncertainty"].dropna().to_numpy(float)
        sigma = ((3.0 / record.n_months_used) * np.sqrt(np.sum(values ** 2))
                 if len(values) == record.n_months_used else np.nan)
        uncertainty_rows.append({
            "region": record.region, "season": record.season,
            "season_year": record.season_year, "pmb_uncertainty": sigma,
            "n_months_used": record.n_months_used,
            "months_used": record.months_used,
            "is_partial_season": record.is_partial_season,
        })
    uncertainty_by_year = pd.DataFrame(uncertainty_rows)
    uncertainty_clim = (
        uncertainty_by_year.groupby(["region", "season"], as_index=False)
        .agg(
            pmb_uncertainty=(
                "pmb_uncertainty",
                lambda values: (
                    np.sqrt(np.sum(values.dropna() ** 2)) / len(values.dropna())
                    if len(values.dropna()) else np.nan
                ),
            ),
            n=("season_year", "nunique"),
        )
    )
    return seasonal_clim, uncertainty_clim, seasonal_by_year


def basin_mean_annual_table(basin_monthly: pd.DataFrame) -> pd.DataFrame:
    """Wide basin table of mean annual totals used by Figures 2, 6, and 7."""
    annual, mean_annual = annual_and_mean_annual(basin_monthly)
    wide = mean_annual.pivot(index="region", columns="product",
                             values="precipitation_mm_year").reset_index()
    wide["basin_id"] = pd.to_numeric(wide["region"].str.extract(r"^(\d+)")[0])
    return wide.sort_values("basin_id").reset_index(drop=True)


#%%
# =============================================================================
# SECTION 6. MANUSCRIPT-ANALOGOUS FIGURES 2, 4, 5, 6, AND 7
# =============================================================================

def plot_figure2_basin_maps(basin_annual: pd.DataFrame, basin_mask: xr.DataArray,
                            output_path: str | Path | None = None):
    """Figure 2 analogue: polar maps painted with basin mean annual values."""
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm

    products = [column for column in basin_annual.columns
                if column not in {"region", "basin_id"}]
    ncols = 3
    nrows = int(np.ceil(len(products) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(13, 4.4 * nrows),
                                subplot_kw={"projection": ccrs.SouthPolarStereo()},
                                constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    bounds = np.array([0, 25, 50, 100, 150, 200, 300, 400, 600, 1000, 1600])
    norm = BoundaryNorm(bounds, plt.get_cmap("turbo").N, extend="max")
    mesh = None
    for index, (axis, product) in enumerate(zip(axes, products)):
        lookup = basin_annual.set_index("basin_id")[product]
        painted = xr.full_like(basin_mask, np.nan, dtype=float)
        for basin_id, value in lookup.items():
            painted = xr.where(basin_mask == basin_id, value, painted)
        mesh = axis.pcolormesh(painted.lon, painted.lat, painted,
                               transform=ccrs.PlateCarree(), cmap="turbo", norm=norm)
        axis.coastlines(linewidth=0.5)
        axis.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
        axis.set_title(f"({chr(97 + index)}) {product}\nMean: {np.nanmean(lookup):.0f}")
    for axis in axes[len(products):]:
        axis.set_visible(False)
    figure.colorbar(mesh, ax=axes[:len(products)].tolist(), shrink=0.8,
                    label="Mean annual precipitation [mm yr$^{-1}$]")
    if output_path is not None:
        figure.savefig(Path(output_path), dpi=300, bbox_inches="tight")
    return figure, axes


def plot_figure4_seasonal(regional_monthly: pd.DataFrame,
                          pmb_uncertainty_monthly: pd.DataFrame | None = None,
                          output_path: str | Path | None = None):
    """Figure 4 adapter using the established R1 seasonal plotting function.

    ``pmb_uncertainty_monthly`` is retained as the public argument name for
    backward compatibility, but must contain the already propagated seasonal
    uncertainty table produced by the established uncertainty helpers.
    """
    from program_utile_13Apr2026 import (
        plot_seasonal_climatology_with_pmb_uncertainty,
        product_styles_corr,
    )

    seasonal = regional_monthly.rename(
        columns={"precipitation_mm_season": "precipitation"}
    )
    product_order = tuple(seasonal["product"].drop_duplicates())
    figure, axes = plot_seasonal_climatology_with_pmb_uncertainty(
        seasonal,
        pmb_unc_df=pmb_uncertainty_monthly,
        region_order=("Antarctica", "West Antarctica", "East Antarctica"),
        product_order=product_order,
        product_styles=product_styles_corr,
        figsize=(10, 8),
        ylabel="mm/season",
        y_nbins=4,
        legend_ncol=3,
        unc_label=r"$P_{\mathrm{MB}}$ uncertainty ($\pm 1\sigma$)",
    )
    if output_path is not None:
        figure.savefig(Path(output_path), dpi=300, bbox_inches="tight")
    return figure, axes, seasonal


def plot_figure5_interannual(regional_monthly: pd.DataFrame,
                             pmb_uncertainty_monthly: pd.DataFrame | None = None,
                             output_path: str | Path | None = None):
    """Figure 5 analogue; explicitly marked as a placeholder for one year."""
    annual, _ = annual_and_mean_annual(regional_monthly)
    # Accept the annual uncertainty table returned by the established
    # compute_annual_uncertainty_from_monthly_unc_df helper.
    unc_annual = pmb_uncertainty_monthly
    regions = regional_monthly["region"].drop_duplicates().tolist()
    years = sorted(annual.year.unique())
    figure, axes = plt.subplots(len(regions), 1, figsize=(11, 8.5), sharex=True,
                                constrained_layout=True)
    for axis, region in zip(np.atleast_1d(axes), regions):
        sub = annual[annual.region == region]
        if unc_annual is not None:
            center = sub[sub["product"] == r"$P_{\mathrm{MB}}$"].set_index("year")
            spread = unc_annual[unc_annual.region == region].set_index("year")
            common = center.index.intersection(spread.index)
            sigma = spread.loc[common, "pmb_uncertainty"]
            axis.fill_between(common, center.loc[common, "precipitation_mm_year"] - sigma,
                              center.loc[common, "precipitation_mm_year"] + sigma, color="0.85")
        for product, frame in sub.groupby("product", sort=False):
            axis.plot(frame.year, frame.precipitation_mm_year, marker="o",
                      label=product, **PRODUCT_STYLES.get(product, {}))
        axis.set_title(region)
        axis.set_ylabel("mm yr$^{-1}$")
        axis.grid(alpha=0.25)
        if len(years) < 2:
            axis.text(0.5, 0.08, "PLACEHOLDER: interannual variability requires ≥2 complete years",
                      transform=axis.transAxes, ha="center", fontsize=9,
                      bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.6"})
    axes[0].legend(ncol=3, frameon=False, fontsize=8)
    axes[-1].set_xticks(years)
    if output_path is not None:
        figure.savefig(Path(output_path), dpi=300, bbox_inches="tight")
    return figure, axes, annual


def plot_figure6_basin_scatter(basin_annual: pd.DataFrame,
                               output_path: str | Path | None = None,
                               reference: str = r"$P_{\mathrm{MB}}$"):
    """Figure 6 analogue: basin mean annual values against PMB."""
    products = [c for c in basin_annual.columns if c not in {"region", "basin_id", reference}]
    figure, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    axes = axes.ravel()
    for index, (axis, product) in enumerate(zip(axes, products)):
        paired = basin_annual[["basin_id", reference, product]].dropna()
        x, y = paired[reference].to_numpy(float), paired[product].to_numpy(float)
        positive = (x > 0) & (y > 0)
        x, y = x[positive], y[positive]
        ids = paired.loc[positive, "basin_id"]
        axis.scatter(x, y, c=ids, cmap="turbo", edgecolor="k", s=48)
        limits = [max(1, min(x.min(), y.min()) * 0.7), max(x.max(), y.max()) * 1.3]
        axis.plot(limits, limits, "k--", linewidth=1)
        axis.set(xscale="log", yscale="log", xlim=limits, ylim=limits,
                 xlabel=r"$P_{MB}$ [mm yr$^{-1}$]", ylabel=f"{product} [mm yr$^{{-1}}$]")
        correlation = np.corrcoef(x, y)[0, 1] if len(x) >= 2 else np.nan
        bias = 100 * (np.mean(y) - np.mean(x)) / np.mean(x)
        axis.text(0.04, 0.95, f"CC = {correlation:.2f}\nBias = {bias:+.1f}%",
                  transform=axis.transAxes, va="top")
        axis.set_title(f"({chr(97 + index)}) {product}")
        axis.grid(alpha=0.2, which="both")
    for axis in axes[len(products):]:
        axis.set_visible(False)
    if output_path is not None:
        figure.savefig(Path(output_path), dpi=300, bbox_inches="tight")
    return figure, axes


def plot_figure7_basin_spread(basin_annual: pd.DataFrame, wais_basins: list[int],
                              eais_basins: list[int], output_path: str | Path | None = None,
                              reference: str = r"$P_{\mathrm{MB}}$"):
    """Figure 7 analogue with six-product normalized spread."""
    products = [c for c in basin_annual.columns if c not in {"region", "basin_id"}]
    values = basin_annual[products]
    basin_annual = basin_annual.copy()
    basin_annual["normalized_spread_pct"] = 100 * (values.max(axis=1) - values.min(axis=1)) / values.mean(axis=1)
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    markers = ["s", "D", "o", "^", "P"]
    for axis, (title, basin_ids) in zip(axes, [("WAIS", wais_basins), ("EAIS", eais_basins)]):
        sub = basin_annual[basin_annual.basin_id.isin(basin_ids)].set_index("basin_id").reindex(basin_ids)
        x = np.arange(len(sub))
        axis.bar(x, sub[reference], color="0.82", edgecolor="0.3", label=reference)
        for marker, product in zip(markers, [p for p in products if p != reference]):
            axis.scatter(x, sub[product], marker=marker, s=42,
                         color=PRODUCT_STYLES.get(product, {}).get("color"), label=product)
        top = sub[products].max(axis=1)
        for xpos, value, spread in zip(x, top, sub.normalized_spread_pct):
            axis.text(xpos, value * 1.08, f"{spread:.0f}%", ha="center", fontweight="bold", fontsize=8)
        axis.set_title(title)
        axis.set_xticks(x, sub.index.astype(int))
        axis.set_ylabel("mm yr$^{-1}$")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("IMBIE basin ID")
    axes[-1].legend(ncol=3, frameon=False, fontsize=8)
    if output_path is not None:
        figure.savefig(Path(output_path), dpi=300, bbox_inches="tight")
    return figure, axes, basin_annual

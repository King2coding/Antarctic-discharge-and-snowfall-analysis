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


PRODUCT_STYLES = {
    r"$P_{\mathrm{MB}}$": {"color": "black", "linewidth": 2.8},
    "ERA5": {"color": "tab:blue", "linewidth": 2.2},
    "GPCP V3.3": {"color": "tab:orange", "linewidth": 2.2},
    "GPM PMW V08": {"color": "tab:green", "linewidth": 2.2},
    "UA-HIPA q75": {"color": "magenta", "linewidth": 2.2},
    "AIRS": {"color": "tab:red", "linewidth": 2.2},
}


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
) -> tuple[dict[str, xr.DataArray], pd.DatetimeIndex]:
    """Return all six products on their strict common complete-calendar-year months."""

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
        require_complete_calendar_years=True,
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

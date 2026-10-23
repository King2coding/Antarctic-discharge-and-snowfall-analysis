#%%
"""Reusable monthly-processing utilities for UA-HIPA q75 and AIRS.

The UA-HIPA estimator follows the platform-aware GPM/GPROF convention used in
``program_utile_13Apr2026.py``:

1. average valid orbital retrievals within each platform and calendar month;
2. convert the platform monthly-mean rate from mm h-1 to mm month-1; and
3. average the available platform monthly fields with equal platform weight.

The output retains orbit and platform counts so changes in sampling and
constellation composition remain visible. AIRS daily precipitation (mm day-1)
is summed to calendar-month accumulation, with daily coverage diagnostics.
"""

#%%
# =============================================================================
# SECTION 1. IMPORTS AND CONSTANTS
# =============================================================================

from __future__ import annotations

import calendar
import glob
import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr


UAHIPA_FILENAME_RE = re.compile(
    r"^clavrx_NSS\.GHRR\.(?P<platform>[^.]+)\."
    r"D(?P<yyddd>\d{5})\.S(?P<hhmm>\d{4})\."
)
AIRS_FILENAME_RE = re.compile(r"3A_AIRSV6_IR_HDD\.(?P<date>\d{8})\.nc$")


@dataclass(frozen=True)
class UAHIPARecord:
    """One parsed UA-HIPA orbital file."""

    path: str
    platform: str
    timestamp: pd.Timestamp


@dataclass(frozen=True)
class AIRSRecord:
    """One parsed AIRS daily file."""

    path: str
    timestamp: pd.Timestamp


#%%
# =============================================================================
# SECTION 2. FILE INVENTORIES
# =============================================================================

def parse_uahipa_filename(path: str | os.PathLike) -> UAHIPARecord:
    """Parse platform and orbit-start UTC time from one UA-HIPA filename."""

    match = UAHIPA_FILENAME_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"Unrecognized UA-HIPA filename: {Path(path).name}")

    yyddd = match.group("yyddd")
    hhmm = match.group("hhmm")
    year = 2000 + int(yyddd[:2])
    day_of_year = int(yyddd[2:])
    hour = int(hhmm[:2])
    minute = int(hhmm[2:])

    timestamp = (
        pd.Timestamp(year=year, month=1, day=1)
        + pd.Timedelta(days=day_of_year - 1, hours=hour, minutes=minute)
    )
    return UAHIPARecord(str(path), match.group("platform"), timestamp)


def discover_uahipa_files(
    root: str | os.PathLike,
    years: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Discover every parseable UA-HIPA file, including future year folders."""

    requested_years = None if years is None else {int(year) for year in years}
    rows = []
    for path in sorted(glob.glob(os.path.join(str(root), "*", "*.nc"))):
        try:
            record = parse_uahipa_filename(path)
        except ValueError as exc:
            warnings.warn(str(exc))
            continue
        if requested_years is not None and record.timestamp.year not in requested_years:
            continue
        rows.append(
            {
                "path": record.path,
                "platform": record.platform,
                "timestamp": record.timestamp,
                "year": record.timestamp.year,
                "month": record.timestamp.month,
                "month_start": record.timestamp.to_period("M").to_timestamp(),
            }
        )

    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise FileNotFoundError(f"No parseable UA-HIPA NetCDF files found below {root}")
    return inventory.sort_values(["timestamp", "platform", "path"]).reset_index(drop=True)


def parse_airs_filename(path: str | os.PathLike) -> AIRSRecord:
    """Parse the nominal date from one converted AIRS daily filename."""

    match = AIRS_FILENAME_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"Unrecognized AIRS filename: {Path(path).name}")
    return AIRSRecord(str(path), pd.to_datetime(match.group("date"), format="%Y%m%d"))


def discover_airs_files(
    root: str | os.PathLike,
    years: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Discover converted AIRS daily files and their calendar months."""

    requested_years = None if years is None else {int(year) for year in years}
    rows = []
    for path in sorted(glob.glob(os.path.join(str(root), "*.nc"))):
        try:
            record = parse_airs_filename(path)
        except ValueError as exc:
            warnings.warn(str(exc))
            continue
        if requested_years is not None and record.timestamp.year not in requested_years:
            continue
        rows.append(
            {
                "path": record.path,
                "timestamp": record.timestamp,
                "year": record.timestamp.year,
                "month": record.timestamp.month,
                "month_start": record.timestamp.to_period("M").to_timestamp(),
            }
        )

    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise FileNotFoundError(f"No parseable AIRS NetCDF files found in {root}")
    return inventory.sort_values(["timestamp", "path"]).reset_index(drop=True)


#%%
# =============================================================================
# SECTION 3. STREAMING ACCUMULATORS
# =============================================================================

def _clean_field(field: xr.DataArray, maximum_valid: float | None = None) -> xr.DataArray:
    field = field.astype("float64").where(np.isfinite(field))
    field = field.where(field >= 0)
    if maximum_valid is not None:
        field = field.where(field <= maximum_valid)
    return field


def _streaming_mean(
    file_paths: Iterable[str],
    reader,
) -> tuple[xr.DataArray, xr.DataArray, int, int]:
    """Return a pixelwise mean/count while holding only one input field at a time."""

    total = None
    count = None
    template = None
    used = 0
    failed = 0

    for path in file_paths:
        try:
            field = reader(path)
        except Exception as exc:  # retain a complete processing log and continue
            warnings.warn(f"Skipping {path}: {exc}")
            failed += 1
            continue

        if template is None:
            template = field
        else:
            field = field.reindex_like(template, method=None)

        valid = field.notnull()
        field_zeroed = field.fillna(0.0)
        total = field_zeroed if total is None else total + field_zeroed
        count = valid.astype("int32") if count is None else count + valid.astype("int32")
        used += 1

    if total is None or count is None or template is None:
        raise RuntimeError("No readable fields were available for this group")

    mean = (total / count.where(count > 0)).astype("float32")
    return mean, count.astype("int32"), used, failed


#%%
# =============================================================================
# SECTION 4. UA-HIPA q75 MONTHLY PROCESSING
# =============================================================================

def read_uahipa_q75(
    path: str | os.PathLike,
    group: str = "SH",
    variable: str = "retrieved_precip_q75",
    maximum_valid_rate_mm_h: float | None = None,
) -> xr.DataArray:
    """Read and decode one gridded UA-HIPA q75 orbital rate field."""

    with xr.open_dataset(path, group=group, cache=False) as dataset:
        if variable not in dataset:
            raise KeyError(f"{variable!r} not found; variables={list(dataset.data_vars)}")
        field = dataset[variable].load()

    if set(field.dims) != {"y", "x"}:
        raise ValueError(f"Expected UA-HIPA dimensions y/x, got {field.dims}")
    field = field.transpose("y", "x")
    field = _clean_field(field, maximum_valid=maximum_valid_rate_mm_h)
    field.attrs.update(
        {
            "long_name": "UA-HIPA q75 orbital surface precipitation rate",
            "units": "mm h-1",
            "source_variable": variable,
        }
    )
    return field


def build_uahipa_platform_month(
    platform_inventory: pd.DataFrame,
    maximum_valid_rate_mm_h: float | None = None,
) -> xr.Dataset:
    """Average orbital q75 fields for one platform-month."""

    if platform_inventory.empty:
        raise ValueError("platform_inventory is empty")
    platforms = platform_inventory["platform"].unique()
    months = platform_inventory["month_start"].unique()
    if len(platforms) != 1 or len(months) != 1:
        raise ValueError("Input must contain exactly one platform and one month")

    reader = lambda path: read_uahipa_q75(  # noqa: E731
        path,
        maximum_valid_rate_mm_h=maximum_valid_rate_mm_h,
    )
    mean_rate, valid_orbit_count, used, failed = _streaming_mean(
        platform_inventory["path"].tolist(), reader
    )
    month_start = pd.Timestamp(months[0])
    days_in_month = month_start.days_in_month
    platform = str(platforms[0])

    mean_rate = mean_rate.expand_dims(time=[month_start])
    valid_orbit_count = valid_orbit_count.expand_dims(time=[month_start])
    accumulation = mean_rate * np.float32(24.0 * days_in_month)

    dataset = xr.Dataset(
        {
            "uahipa_q75_rate_mm_h": mean_rate,
            "uahipa_q75_mm_month": accumulation.astype("float32"),
            "valid_orbit_count": valid_orbit_count,
        }
    )
    dataset = dataset.expand_dims(platform=[platform])
    dataset.attrs.update(
        {
            "platform": platform,
            "month": month_start.strftime("%Y-%m"),
            "n_input_files": int(len(platform_inventory)),
            "n_orbits_used": int(used),
            "n_files_failed": int(failed),
            "rate_to_month_conversion": "monthly mean sampled rate * 24 * calendar days",
        }
    )
    return dataset


def build_uahipa_month(
    month_inventory: pd.DataFrame,
    minimum_valid_platforms: int = 1,
    maximum_valid_rate_mm_h: float | None = None,
) -> xr.Dataset:
    """Build one equal-platform-weight UA-HIPA q75 monthly estimate."""

    if month_inventory.empty:
        raise ValueError("month_inventory is empty")

    platform_datasets = []
    for platform, group in month_inventory.groupby("platform", sort=True):
        print(f"  UA-HIPA {group['month_start'].iloc[0]:%Y-%m}: {platform}, {len(group)} files")
        platform_datasets.append(
            build_uahipa_platform_month(
                group,
                maximum_valid_rate_mm_h=maximum_valid_rate_mm_h,
            )
        )

    combined = xr.concat(
        platform_datasets,
        dim="platform",
        join="exact",
        compat="override",
        coords="minimal",
        combine_attrs="drop_conflicts",
    )
    rates = combined["uahipa_q75_rate_mm_h"]
    totals = combined["uahipa_q75_mm_month"]
    valid_platform_count = rates.notnull().sum("platform").astype("int16")

    equal_platform_rate = rates.mean("platform", skipna=True)
    equal_platform_total = totals.mean("platform", skipna=True)
    keep = valid_platform_count >= int(minimum_valid_platforms)

    output = xr.Dataset(
        {
            "uahipa_q75_rate_mm_h": equal_platform_rate.where(keep).astype("float32"),
            "uahipa_q75_mm_month": equal_platform_total.where(keep).astype("float32"),
            "valid_platform_count": valid_platform_count,
            "platform_q75_rate_mm_h": rates.astype("float32"),
            "platform_valid_orbit_count": combined["valid_orbit_count"].astype("int32"),
        }
    )
    output.attrs.update(
        {
            "title": "Monthly UA-HIPA q75 precipitation with equal platform weighting",
            "aggregation": (
                "pixelwise mean of orbital q75 rates within platform-month, followed by "
                "equal-weight mean across available platforms"
            ),
            "minimum_valid_platforms_per_pixel": int(minimum_valid_platforms),
            "platforms": ",".join(str(value) for value in combined.platform.values),
            "important_sampling_note": (
                "Multiplication by full calendar hours treats sampled orbital rates as "
                "representative of the month; orbit and platform counts are retained for QC."
            ),
        }
    )
    return output


def build_all_uahipa_months(
    inventory: pd.DataFrame,
    minimum_valid_platforms: int = 1,
    maximum_valid_rate_mm_h: float | None = None,
) -> xr.Dataset:
    """Process every discovered UA-HIPA month, with no hard-coded year range."""

    monthly = []
    for month_start, group in inventory.groupby("month_start", sort=True):
        print(f"\nBuilding UA-HIPA q75 month {month_start:%Y-%m}")
        monthly.append(
            build_uahipa_month(
                group,
                minimum_valid_platforms=minimum_valid_platforms,
                maximum_valid_rate_mm_h=maximum_valid_rate_mm_h,
            )
        )
    output = xr.concat(monthly, dim="time", join="outer", combine_attrs="drop_conflicts")
    return output.sortby("time")


#%%
# =============================================================================
# SECTION 5. AIRS DAILY-TO-MONTHLY PROCESSING
# =============================================================================

def read_airs_daily(
    path: str | os.PathLike,
    variable: str = "precipitation",
    northern_latitude: float = -60.0,
) -> xr.DataArray:
    """Read one AIRS daily precipitation field south of a latitude boundary."""

    with xr.open_dataset(path, cache=False) as dataset:
        if variable not in dataset:
            raise KeyError(f"{variable!r} not found; variables={list(dataset.data_vars)}")
        field = dataset[variable]
        if "time" in field.dims:
            field = field.isel(time=0, drop=True)
        field = field.load()

    if not {"lat", "lon"}.issubset(field.dims):
        raise ValueError(f"Expected AIRS dimensions lat/lon, got {field.dims}")
    field = field.transpose("lat", "lon")
    if field.lat.values[0] > field.lat.values[-1]:
        field = field.sel(lat=slice(northern_latitude, -90.0))
    else:
        field = field.sel(lat=slice(-90.0, northern_latitude))
    return _clean_field(field)


def build_airs_month(
    month_inventory: pd.DataFrame,
    minimum_daily_coverage_fraction: float = 0.90,
    northern_latitude: float = -60.0,
) -> xr.Dataset:
    """Sum AIRS daily mm/day fields into one mm/month field."""

    if month_inventory.empty:
        raise ValueError("month_inventory is empty")
    months = month_inventory["month_start"].unique()
    if len(months) != 1:
        raise ValueError("Input must contain exactly one AIRS month")

    month_start = pd.Timestamp(months[0])
    expected_days = calendar.monthrange(month_start.year, month_start.month)[1]
    required_days = int(np.ceil(expected_days * minimum_daily_coverage_fraction))

    total = None
    count = None
    template = None
    used_dates = []
    failed = 0
    for row in month_inventory.itertuples(index=False):
        try:
            field = read_airs_daily(row.path, northern_latitude=northern_latitude)
        except Exception as exc:
            warnings.warn(f"Skipping {row.path}: {exc}")
            failed += 1
            continue
        if template is None:
            template = field
        else:
            field = field.reindex_like(template, method=None)
        valid = field.notnull()
        total = field.fillna(0.0) if total is None else total + field.fillna(0.0)
        count = valid.astype("int16") if count is None else count + valid.astype("int16")
        used_dates.append(pd.Timestamp(row.timestamp))

    if total is None or count is None:
        raise RuntimeError(f"No readable AIRS files for {month_start:%Y-%m}")

    accumulation = total.where(count >= required_days).astype("float32")
    output = xr.Dataset(
        {
            "airs_precip_mm_month": accumulation.expand_dims(time=[month_start]),
            "airs_valid_day_count": count.expand_dims(time=[month_start]),
        }
    )
    output.attrs.update(
        {
            "title": "Monthly AIRS precipitation accumulation",
            "aggregation": "sum of valid daily precipitation fields in mm day-1",
            "expected_calendar_days": int(expected_days),
            "n_unique_daily_files_used": int(len(set(used_dates))),
            "n_files_failed": int(failed),
            "minimum_daily_coverage_fraction": float(minimum_daily_coverage_fraction),
            "minimum_valid_days_per_pixel": int(required_days),
        }
    )
    return output


def build_all_airs_months(
    inventory: pd.DataFrame,
    minimum_daily_coverage_fraction: float = 0.90,
    northern_latitude: float = -60.0,
) -> xr.Dataset:
    """Process every discovered AIRS calendar month."""

    monthly = []
    for month_start, group in inventory.groupby("month_start", sort=True):
        print(f"\nBuilding AIRS month {month_start:%Y-%m}")
        monthly.append(
            build_airs_month(
                group,
                minimum_daily_coverage_fraction=minimum_daily_coverage_fraction,
                northern_latitude=northern_latitude,
            )
        )
    return xr.concat(monthly, dim="time", join="exact", combine_attrs="drop_conflicts").sortby("time")


#%%
# =============================================================================
# SECTION 6. STRICT COMMON-PERIOD UTILITIES
# =============================================================================

def valid_months_from_field(field: xr.DataArray, time_name: str = "time") -> pd.DatetimeIndex:
    """Return months containing at least one finite spatial value."""

    spatial_dims = [dim for dim in field.dims if dim != time_name]
    valid = field.notnull().any(dim=spatial_dims)
    return pd.DatetimeIndex(pd.to_datetime(field[time_name].values[valid.values])).to_period("M").to_timestamp()


def strict_common_months(
    product_fields: Mapping[str, xr.DataArray],
    require_complete_calendar_years: bool = True,
) -> pd.DatetimeIndex:
    """Intersect valid months across all products without nearest-time matching."""

    if not product_fields:
        raise ValueError("product_fields is empty")
    common = None
    for name, field in product_fields.items():
        months = valid_months_from_field(field)
        print(f"{name}: {len(months)} valid months, {months.min():%Y-%m} to {months.max():%Y-%m}")
        common = months if common is None else common.intersection(months)

    common = pd.DatetimeIndex(sorted(common.unique()))
    if require_complete_calendar_years:
        counts = pd.Series(common.year).value_counts()
        complete_years = sorted(int(year) for year, count in counts.items() if count == 12)
        common = common[common.year.astype(int).isin(complete_years)]
    if len(common) == 0:
        raise ValueError("No strict common months remain across all requested products")
    return common


def subset_products_to_common_months(
    product_fields: Mapping[str, xr.DataArray],
    require_complete_calendar_years: bool = True,
) -> tuple[dict[str, xr.DataArray], pd.DatetimeIndex]:
    """Find and strictly reindex every product to the same coincident months."""

    common = strict_common_months(
        product_fields,
        require_complete_calendar_years=require_complete_calendar_years,
    )
    subset = {name: field.reindex(time=common) for name, field in product_fields.items()}
    return subset, common


#%%
# =============================================================================
# SECTION 7. NETCDF WRITING
# =============================================================================

def netcdf_encoding(dataset: xr.Dataset) -> dict:
    """Compression and dtypes suitable for the monthly output datasets."""

    encoding = {}
    for name, variable in dataset.data_vars.items():
        is_integer = np.issubdtype(variable.dtype, np.integer)
        encoding[name] = {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": str(variable.dtype),
        }
        if not is_integer:
            encoding[name]["_FillValue"] = np.float32(np.nan)
    return encoding


def write_monthly_dataset(dataset: xr.Dataset, output_path: str | os.PathLike) -> Path:
    """Write a monthly dataset without overwriting an existing result."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    dataset.to_netcdf(output, encoding=netcdf_encoding(dataset))
    return output

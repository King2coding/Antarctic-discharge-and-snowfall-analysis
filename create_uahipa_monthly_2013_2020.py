#%%
"""
Create monthly UA-HIPA precipitation fields for Antarctica.

Input
-----
Orbital-level UA-HIPA / HIRS-AVHRR fusion files organized by year:

    /scratch/omidzandi/AVHRR_retrieved_from_HPC_collocated/YYYY/*.nc

Each file contains a Southern Hemisphere group opened as:

    xr.open_dataset(file, group="SH")

Expected variable:
    precipitation

Assumed units:
    precipitation is instantaneous/rate intensity in mm/hour.

Output
------
Monthly UA-HIPA fields on native 0.25° WGS grid:

    uahipa_precip_monthly_YYYYMM.nc

Variables:
    uahipa_precip_mm_per_h_mean
    uahipa_precip_mm_day
    uahipa_precip_mm_month
    n_orbits_used

Main convention
---------------
To match the comparative-analysis script:

    GPCP mm/month = GPCP mm/day * days_in_month
    ERA5 mm/month = ERA5 m/day * 1000 * days_in_month

Therefore:

    UA-HIPA mm/day   = monthly mean orbital precipitation [mm/h] * 24
    UA-HIPA mm/month = UA-HIPA mm/day * days_in_month

No spatial regridding is done here.
"""

#%%
# =============================================================================
# IMPORTS
# =============================================================================

import os
import re
import gc
import glob
import warnings
from datetime import date
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import xarray as xr

#%%
# =============================================================================
# PATHS AND SETTINGS
# =============================================================================

uahipa_root = r"/scratch/omidzandi/AVHRR_retrieved_from_HPC_collocated"

out_dir = r"/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/uahipa_monthly"
os.makedirs(out_dir, exist_ok=True)

YEAR_START = 2013
YEAR_END = 2020
YEARS = np.arange(YEAR_START, YEAR_END + 1)

GROUP_NAME = "SH"
PRECIP_VAR = "precipitation"

# Assumed UA-HIPA precipitation unit
UAHIPA_RATE_UNIT = "mm h-1"

# Parallel workers
N_WORKERS = max(1, min(16, cpu_count() - 1))

cde_run_dte = str(date.today().strftime("%Y%m%d"))

# Optional: clip physically invalid values
CLIP_NEGATIVE_TO_ZERO = True

# Optional: treat huge values as missing if needed
MAX_VALID_RATE_MM_H = None
# Example:
# MAX_VALID_RATE_MM_H = 100.0

#%%
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_orbit_datetime_from_filename(path):
    """
    Extract approximate orbit start datetime from filename.

    Example filename:
        clavrx_NSS.GHRR.M1.D13140.S0635.E0724.B0347575.MM....nc

    D13140 means:
        year = 2013
        day-of-year = 140

    S0635 means start time 06:35 UTC.

    Returns
    -------
    pd.Timestamp or NaT
    """

    base = os.path.basename(path)

    m = re.search(r"\.D(?P<yyddd>\d{5})\.S(?P<hhmm>\d{4})", base)

    if m is None:
        return pd.NaT

    yyddd = m.group("yyddd")
    hhmm = m.group("hhmm")

    yy = int(yyddd[:2])
    doy = int(yyddd[2:])

    # Files here are 2013 onward. If needed, this can be generalized.
    year = 2000 + yy

    hh = int(hhmm[:2])
    minute = int(hhmm[2:])

    try:
        dt = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        dt = dt + pd.Timedelta(hours=hh, minutes=minute)
    except Exception:
        return pd.NaT

    return dt


def clean_precip_da(da):
    """
    Clean UA-HIPA precipitation DataArray.
    """

    da = da.astype("float32")

    # Replace non-finite with NaN
    da = da.where(np.isfinite(da))

    # Optional: remove negative retrievals
    if CLIP_NEGATIVE_TO_ZERO:
        da = da.where(da >= 0)

    # Optional: remove unrealistic high rates
    if MAX_VALID_RATE_MM_H is not None:
        da = da.where(da <= MAX_VALID_RATE_MM_H)

    return da


def read_single_uahipa_orbit(path, group_name=GROUP_NAME, precip_var=PRECIP_VAR):
    """
    Read one UA-HIPA orbital file and return precipitation DataArray.

    Returns
    -------
    xr.DataArray with dims (y, x), or None if unreadable.
    """

    try:
        ds = xr.open_dataset(path, group=group_name)

        if precip_var not in ds:
            warnings.warn(f"{precip_var} not found in {path}")
            ds.close()
            return None

        da = ds[precip_var].copy()

        # Standardize coordinate names if needed
        # Expected dims are y, x already.
        if "lat" in da.dims and "lon" in da.dims:
            da = da.rename({"lat": "y", "lon": "x"})

        if "y" not in da.dims or "x" not in da.dims:
            warnings.warn(f"Unexpected dims {da.dims} in {path}")
            ds.close()
            return None

        da = clean_precip_da(da)

        # Add source filename as attribute only; do not add as coordinate.
        da.attrs["source_file"] = path
        da.attrs["units"] = UAHIPA_RATE_UNIT

        ds.close()

        return da

    except Exception as e:
        warnings.warn(f"Failed reading {path}: {e}")
        return None


def monthly_mean_from_files(file_list):
    """
    Compute monthly mean precipitation intensity [mm/h] from all orbital files.

    This does not regrid. It assumes all files share the same native UA-HIPA grid.
    """

    valid_sum = None
    valid_count = None
    template = None
    n_orbits_used = 0
    n_files_failed = 0

    for fp in file_list:

        da = read_single_uahipa_orbit(fp)

        if da is None:
            n_files_failed += 1
            continue

        if template is None:
            template = da

        # Align to the first valid file grid, without interpolation.
        # If grids differ, this will align by exact x/y labels and may introduce NaNs.
        da = da.reindex_like(template)

        finite = xr.where(np.isfinite(da), 1.0, 0.0).astype("float32")
        data = da.fillna(0.0)

        if valid_sum is None:
            valid_sum = data
            valid_count = finite
        else:
            valid_sum = valid_sum + data
            valid_count = valid_count + finite

        n_orbits_used += 1

        # avoid accumulating too much memory
        del da, data, finite
        gc.collect()

    if valid_sum is None or valid_count is None or template is None:
        return None, 0, n_files_failed

    monthly_mean_mm_h = valid_sum / valid_count.where(valid_count > 0)

    monthly_mean_mm_h.name = "uahipa_precip_mm_per_h_mean"

    monthly_mean_mm_h.attrs.update({
        "long_name": "UA-HIPA monthly mean precipitation intensity",
        "units": "mm h-1",
        "description": (
            "Mean of valid orbital UA-HIPA precipitation intensities within the month. "
            "No spatial regridding performed."
        ),
        "n_orbits_used": int(n_orbits_used),
        "n_files_failed": int(n_files_failed),
    })

    return monthly_mean_mm_h, n_orbits_used, n_files_failed


def process_one_month(year, month, files_for_month):
    """
    Process one month and save NetCDF output.
    """

    if len(files_for_month) == 0:
        print(f"[{year}-{month:02d}] No files found.")
        return None

    print(f"[{year}-{month:02d}] Processing {len(files_for_month)} files...")

    monthly_mean_mm_h, n_orbits_used, n_files_failed = monthly_mean_from_files(files_for_month)

    if monthly_mean_mm_h is None:
        print(f"[{year}-{month:02d}] No valid orbit data.")
        return None

    month_time = pd.Timestamp(year=year, month=month, day=1)
    days_in_month = month_time.days_in_month

    # Convert to mm/day and mm/month for consistency with GPCP/ERA5 workflow
    uahipa_mm_day = monthly_mean_mm_h * 24.0
    uahipa_mm_day.name = "uahipa_precip_mm_day"

    uahipa_mm_month = uahipa_mm_day * float(days_in_month)
    uahipa_mm_month.name = "uahipa_precip_mm_month"

    # Count valid orbital contributions per pixel
    # Recompute count separately for storage is expensive; instead store orbit count scalar.
    # If you later want per-pixel valid-count, we can add it.
    n_orbits_da = xr.full_like(monthly_mean_mm_h, float(n_orbits_used))
    n_orbits_da.name = "n_orbits_used"

    ds_out = xr.Dataset(
        data_vars={
            "uahipa_precip_mm_per_h_mean": monthly_mean_mm_h,
            "uahipa_precip_mm_day": uahipa_mm_day,
            "uahipa_precip_mm_month": uahipa_mm_month,
            "n_orbits_used": n_orbits_da,
        }
    )

    ds_out = ds_out.expand_dims(time=[month_time])

    ds_out.attrs.update({
        "title": "Monthly UA-HIPA precipitation for Antarctic comparison",
        "source": "UA-HIPA orbital precipitation retrievals",
        "group_used": GROUP_NAME,
        "precipitation_variable": PRECIP_VAR,
        "native_grid": "UA-HIPA WGS grid, no regridding",
        "conversion": (
            "uahipa_precip_mm_day = monthly_mean(mm h-1) * 24; "
            "uahipa_precip_mm_month = uahipa_precip_mm_day * days_in_month"
        ),
        "days_in_month": int(days_in_month),
        "n_orbits_used": int(n_orbits_used),
        "n_files_input": int(len(files_for_month)),
        "n_files_failed": int(n_files_failed),
        "created": cde_run_dte,
    })

    # Clean attrs that can cause writing conflicts
    for v in ds_out.data_vars:
        for key in ["_FillValue", "scale_factor", "add_offset"]:
            ds_out[v].attrs.pop(key, None)

    out_file = os.path.join(
        out_dir,
        f"uahipa_precip_monthly_{year}{month:02d}_{cde_run_dte}.nc"
    )

    encoding = {
        v: {
            "zlib": True,
            "complevel": 4,
            "_FillValue": np.float32(np.nan),
            "dtype": "float32",
        }
        for v in ds_out.data_vars
    }

    ds_out.to_netcdf(out_file, encoding=encoding)

    print(f"[{year}-{month:02d}] Saved: {out_file}")

    ds_out.close()
    gc.collect()

    return out_file


def build_file_inventory():
    """
    Build dataframe of all UA-HIPA orbital files and parsed datetimes.
    """

    rows = []

    for yy in YEARS:
        year_dir = os.path.join(uahipa_root, str(yy))

        files = sorted(glob.glob(os.path.join(year_dir, "*.nc")))

        for fp in files:
            dt = parse_orbit_datetime_from_filename(fp)

            if pd.isna(dt):
                continue

            rows.append({
                "file": fp,
                "datetime": dt,
                "year": int(dt.year),
                "month": int(dt.month),
            })

    inv = pd.DataFrame(rows)

    if inv.empty:
        raise RuntimeError("No UA-HIPA files found or no filenames could be parsed.")

    # Keep only requested years/months
    inv = inv[
        (inv["year"] >= YEAR_START) &
        (inv["year"] <= YEAR_END)
    ].copy()

    inv = inv.sort_values("datetime").reset_index(drop=True)

    return inv


#%%
# =============================================================================
# BUILD MONTHLY UA-HIPA FILES
# =============================================================================

inventory = build_file_inventory()

print("UA-HIPA file inventory:")
print(inventory.head())
print(inventory.tail())
print("Total files:", len(inventory))

print("\nFiles per year/month:")
print(
    inventory
    .groupby(["year", "month"])
    .size()
    .reset_index(name="n_files")
)

# Prepare monthly tasks
tasks = []

for yy in YEARS:
    for mm in range(1, 13):
        sub = inventory[
            (inventory["year"] == yy) &
            (inventory["month"] == mm)
        ]

        files_for_month = sub["file"].tolist()

        tasks.append((yy, mm, files_for_month))


# Run monthly processing in parallel
# Each worker processes one month; inside each month, files are streamed sequentially.
# This avoids loading all orbital files at once.

if N_WORKERS == 1:
    out_files = [
        process_one_month(yy, mm, files)
        for yy, mm, files in tasks
    ]
else:
    with Pool(processes=N_WORKERS) as pool:
        out_files = pool.starmap(process_one_month, tasks)

out_files = [f for f in out_files if f is not None]

print("\nFinished monthly UA-HIPA processing.")
print("Number of monthly files saved:", len(out_files))

# Save inventory/log
inventory_log = os.path.join(
    out_dir,
    f"uahipa_monthly_file_inventory_{YEAR_START}_{YEAR_END}_{cde_run_dte}.csv"
)

inventory.to_csv(inventory_log, index=False)

print("Saved inventory:")
print(inventory_log)

#%%
# =============================================================================
# OPTIONAL: COMBINE MONTHLY FILES INTO ONE MULTI-TIME DATASET
# =============================================================================

if len(out_files) > 0:

    combined_out = os.path.join(
        out_dir,
        f"uahipa_precip_monthly_{YEAR_START}_{YEAR_END}_{cde_run_dte}.nc"
    )

    ds_combined = xr.open_mfdataset(
        sorted(out_files),
        combine="by_coords",
        engine="netcdf4",
        chunks={"time": 12, "y": 180, "x": 1440},
    )

    ds_combined.to_netcdf(
        combined_out,
        encoding={
            v: {
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.float32(np.nan),
                "dtype": "float32",
            }
            for v in ds_combined.data_vars
        }
    )

    ds_combined.close()

    print("\nSaved combined UA-HIPA monthly dataset:")
    print(combined_out)

gc.collect()


#%% Daignostic Check
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ua_file = (
    "/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/"
    "data/uahipa_monthly/uahipa_precip_monthly_2013_2020_20260508.nc"
)

ds = xr.open_dataset(ua_file)

print(ds)
print("Time range:", ds.time.values[0], "to", ds.time.values[-1])
print("Number of months:", ds.sizes["time"])

# ------------------------------------------------------------
# Completeness check
# ------------------------------------------------------------
months = pd.to_datetime(ds.time.values).to_period("M").astype(str)
expected = pd.period_range("2013-01", "2019-12", freq="M").astype(str)
missing = sorted(set(expected) - set(months))

print("Missing months:", missing)

# ------------------------------------------------------------
# First-month sample/orbit count
# ------------------------------------------------------------
ds["n_orbits_used"].isel(time=0).plot()
plt.title("UA-HIPA number of orbits used, first month")
plt.show()

# ------------------------------------------------------------
# First-month precipitation in mm/day
# ------------------------------------------------------------
ds["uahipa_precip_mm_day"].isel(time=0).plot(vmin=0, vmax=5)
plt.title("UA-HIPA precipitation, first month [mm/day]")
plt.show()

# ------------------------------------------------------------
# First-month precipitation in mm/month
# ------------------------------------------------------------
ds["uahipa_precip_mm_month"].isel(time=0).plot(vmin=0, vmax=100)
plt.title("UA-HIPA precipitation, first month [mm/month]")
plt.show()

# ------------------------------------------------------------
# Quick numerical diagnostics
# ------------------------------------------------------------
for var in ["uahipa_precip_mm_per_h_mean", "uahipa_precip_mm_day", "uahipa_precip_mm_month", "n_orbits_used"]:
    da = ds[var]
    print(
        var,
        "min:", float(da.min(skipna=True)),
        "mean:", float(da.mean(skipna=True)),
        "max:", float(da.max(skipna=True)),
    )
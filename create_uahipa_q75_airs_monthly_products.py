#%%
"""Interactive driver for dynamic UA-HIPA q75 and AIRS monthly products.

Run numbered cells manually. The default configuration discovers every year
currently present; no analysis-year range is hard-coded. Set ``YEARS`` to a
list such as ``[2018]`` for a targeted test run.
"""

#%%
# =============================================================================
# SECTION 1. IMPORTS
# =============================================================================

from datetime import date
from pathlib import Path

from uahipa_airs_monthly_utils import (
    build_all_airs_months,
    build_all_uahipa_months,
    discover_airs_files,
    discover_uahipa_files,
    write_monthly_dataset,
)


#%%
# =============================================================================
# SECTION 2. PATHS AND USER SETTINGS
# =============================================================================

UAHIPA_ROOT = Path(
    "/scratch/omidzandi/AVHRR_retrieved_from_HPC/"
    "new_ERA5_based/gridded_WGS_no_L2"
)
AIRS_ROOT = Path(
    "/ra1/pubdat/AVHRR_CloudSat_proj/TOVSAIRS/"
    "airs_from_Eric_converted_to_nc_files_all_years"
)
OUTPUT_ROOT = Path(
    "/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/"
    "data/uahipa_airs_monthly"
)

# None discovers all available years. Example test setting: YEARS = [2018]
YEARS = [2018]

# Parallel execution is intentionally not hidden inside this driver. Each file
# is streamed sequentially so memory use and progress are easy to follow.
MINIMUM_VALID_UAHIPA_PLATFORMS = 1
MAXIMUM_VALID_UAHIPA_RATE_MM_H = None
MINIMUM_AIRS_DAILY_COVERAGE_FRACTION = 0.90

RUN_UAHIPA = True
RUN_AIRS = True

RUN_DATE = date.today().strftime("%Y%m%d")


#%%
# =============================================================================
# SECTION 3. INVENTORY UA-HIPA AND AIRS INPUTS
# =============================================================================

uahipa_inventory = discover_uahipa_files(UAHIPA_ROOT, years=YEARS)
airs_inventory = discover_airs_files(AIRS_ROOT, years=YEARS)

print("\nUA-HIPA files by year, month, and platform:")
print(uahipa_inventory.groupby(["year", "month", "platform"]).size().to_string())

print("\nAIRS daily files by year and month:")
print(airs_inventory.groupby(["year", "month"]).size().to_string())

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
uahipa_inventory.to_csv(OUTPUT_ROOT / f"uahipa_orbit_inventory_{RUN_DATE}.csv", index=False)
airs_inventory.to_csv(OUTPUT_ROOT / f"airs_daily_inventory_{RUN_DATE}.csv", index=False)


#%%
# =============================================================================
# SECTION 4. BUILD EQUAL-PLATFORM-WEIGHT UA-HIPA q75 MONTHLY PRODUCT
# =============================================================================

if RUN_UAHIPA:
    uahipa_monthly = build_all_uahipa_months(
        uahipa_inventory,
        minimum_valid_platforms=MINIMUM_VALID_UAHIPA_PLATFORMS,
        maximum_valid_rate_mm_h=MAXIMUM_VALID_UAHIPA_RATE_MM_H,
    )
    uahipa_start = str(uahipa_monthly.time.dt.year.min().item())
    uahipa_end = str(uahipa_monthly.time.dt.year.max().item())
    uahipa_output = OUTPUT_ROOT / (
        f"uahipa_q75_equal_platform_monthly_{uahipa_start}_{uahipa_end}_{RUN_DATE}.nc"
    )
    write_monthly_dataset(uahipa_monthly, uahipa_output)
    print("Saved:", uahipa_output)


#%%
# =============================================================================
# SECTION 5. BUILD AIRS MONTHLY ACCUMULATION
# =============================================================================

if RUN_AIRS:
    airs_monthly = build_all_airs_months(
        airs_inventory,
        minimum_daily_coverage_fraction=MINIMUM_AIRS_DAILY_COVERAGE_FRACTION,
        northern_latitude=-60.0,
    )
    airs_start = str(airs_monthly.time.dt.year.min().item())
    airs_end = str(airs_monthly.time.dt.year.max().item())
    airs_output = OUTPUT_ROOT / f"airs_monthly_{airs_start}_{airs_end}_{RUN_DATE}.nc"
    write_monthly_dataset(airs_monthly, airs_output)
    print("Saved:", airs_output)


#%%
# =============================================================================
# SECTION 6. INTERACTIVE QUALITY-CONTROL CHECKS
# =============================================================================

if RUN_UAHIPA:
    print(uahipa_monthly)
    print("\nUA-HIPA platforms:", uahipa_monthly.platform.values.tolist())
    print("UA-HIPA valid-platform count by month:")
    print(
        uahipa_monthly["valid_platform_count"]
        .max(("y", "x"))
        .to_series()
    )

if RUN_AIRS:
    print(airs_monthly)
    print("\nAIRS minimum valid-day count by month:")
    print(airs_monthly["airs_valid_day_count"].min(("lat", "lon")).to_series())

# UA-HIPA q75 and AIRS comparison workflow

This focused workflow extends the Antarctic precipitation-mass-budget analysis
without writing into the figure or table directories used for the submitted
JGR-Atmospheres manuscript.

## Scientific conventions

- **UA-HIPA retrieval:** `retrieved_precip_q75` from the `SH` NetCDF group.
- **Within-platform aggregation:** pixelwise mean of valid orbital rates for
  each platform and calendar month.
- **Across-platform aggregation:** equal-weight mean of the available platform
  monthly fields, following the platform-aware GPM/GPROF treatment in
  `program_utile_13Apr2026.py`.
- **UA-HIPA units:** monthly mean sampled rate in mm h-1 multiplied by 24 and
  the number of calendar days. This is a sampling-based monthly estimator, not
  a continuously observed accumulation. Orbit counts, platform fields, and
  valid-platform counts are retained for sensitivity and coverage checks.
- **AIRS units:** daily fields are supplied in mm day-1 and summed by calendar
  month. A pixel must contain at least 90% of the expected daily fields by
  default.
- **Temporal comparison:** exact month-start timestamps are intersected across
  PMB, ERA5, GPCP v3.3, GPM PMW V08, UA-HIPA q75, and AIRS. Only complete
  calendar years are retained. With current UA-HIPA availability this is
  January-December 2018; future complete UA-HIPA years enter automatically.

## Files

1. `uahipa_airs_monthly_utils.py` contains reusable inventory, aggregation,
   coverage, strict-common-period, and NetCDF-writing functions.
2. `create_uahipa_q75_airs_monthly_products.py` is the numbered, cell-based
   preprocessing driver. Run it interactively.
3. `comparative_uahipa_airs_common_period_utils.py` contains regional series,
   paired metrics, and plotting functions.
4. `Comparative_assessment_PMB_UAHIPA_q75_AIRS_common_period.py` is a standalone
   analogous comparison driver. It directly loads PMB, ERA5, GPCP v3.3,
   GPM/GPROF V08, UA-HIPA q75, AIRS, and the basin data; it does not depend on
   another comparison script having been run first.

## Recommended execution order

1. In `create_uahipa_q75_airs_monthly_products.py`, set `YEARS = [2018]` for an
   initial controlled run. Inspect the inventories before running aggregation.
2. Run the UA-HIPA q75 cell and inspect valid orbit/platform counts.
3. Run the AIRS cell and inspect valid-day counts.
4. Open `Comparative_assessment_PMB_UAHIPA_q75_AIRS_common_period.py` in a clean
   kernel and run Sections 1-18 sequentially. The file recreates the established
   0.1-degree WGS84 basin grid and prepares every comparison product itself.
5. In Section 12, confirm that the strict six-product period contains exactly
   the 12 months from January through December 2018 for the current inputs.

When additional UA-HIPA year folders arrive, use `YEARS = None` to discover all
available years or specify a reviewed list. The comparison selects only years
for which every requested product has all 12 coincident months.

## Output locations

- Monthly products and inventories:
  `/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/uahipa_airs_monthly`
- New comparison plots and tables:
  `/home/kkumah/Projects/Antarctic_discharge_work/uahipa_airs_comparison`

The new output directories are separate from the existing manuscript outputs.

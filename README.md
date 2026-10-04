# Antarctic discharge and snowfall analysis

Research code for estimating and evaluating Antarctic snowfall and precipitation
through an observation-constrained mass-budget framework. The workflows combine
ice-sheet discharge and mass-change information with gridded precipitation
products, then evaluate correction factors and long-term consistency at ice-sheet
and drainage-basin scales.

## Repository status

This is a code-only research archive. The scripts preserve the analysis history
used during an active manuscript project; dated and descriptively suffixed files
are retained intentionally. Source code has not been refactored into a package,
and the manuscript, research data, generated figures, and outputs are not included.

## Main workflow areas

- preparation of monthly Antarctic precipitation products by drainage basin;
- calculation of precipitation mass-budget estimates in millimetres and gigatonnes;
- GRACE/GRACE-FO gap filling and discharge-based mass-budget analysis;
- evaluation and correction of satellite precipitation products;
- seasonal, annual, and long-term correction-factor diagnostics;
- SSMIS F17 operational-period and sensor-transition assessment; and
- basin-scale plots, trend diagnostics, and manuscript figures.

The files named `program_utile_13Apr2026.py` and
`PMB_correction_E2_AIS_seasonal_and_single_CF.py` are central analysis utilities
and workflows. Their dated `before_*` variants document intermediate analysis
states and should not be assumed to be interchangeable.

## Data and configuration

The scripts expect external NetCDF and other research datasets, commonly through
absolute paths on the original rain-server environment. These paths are retained
as part of the analysis record. See [`DATA.md`](DATA.md) for the dataset categories
and local-file policy.

## Software environment

The workflows use the scientific and geospatial Python ecosystem, including
packages such as `numpy`, `pandas`, `xarray`, `scipy`, `matplotlib`, `seaborn`,
`cartopy`, `geopandas`, and related NetCDF/geospatial libraries. Exact dependency
versions were not consistently recorded across the analysis history. Reproduction
should therefore begin with the imports and configuration in the selected script.

## Manuscript and citation

The associated manuscript working files are maintained outside this repository
and are not redistributed here. The repository-level citation metadata in
[`CITATION.cff`](CITATION.cff) identifies the code archive without asserting a
publication record that has not been independently verified.

## License

No software license has yet been assigned. Contact the author before reuse or
redistribution.

## Contact

Kwabena Kingsley Kumah — [GitHub profile](https://github.com/King2coding)

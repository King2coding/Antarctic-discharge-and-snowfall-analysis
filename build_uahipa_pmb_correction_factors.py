# =============================================================================
# UA-HIPA PMB-BASED SEASONAL CORRECTION FACTORS
# =============================================================================
#%%
import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from pyproj import CRS
from program_utils import *
from Extra_util_functions import *
from program_utile_13Apr2026 import *
#%%
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

PMB_NAME = r"$P_{\mathrm{MB}}$"
UAHIPA_NAME = "UA-HIPA"

WAIS_IDS = [10, 11, 12, 13, 14, 15, 16, 17]
EAIS_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 18, 19]

REGION_BASINS = {
    "West Antarctica": WAIS_IDS,
    "East Antarctica": EAIS_IDS,
}

REGION_CODE = {
    "West Antarctica": 1,
    "East Antarctica": 2,
}

CRS_WGS84 = "EPSG:4326"
CRS_SH_STEREO = "+proj=stere +lat_0=-90 +lat_ts=-71 +x_0=0 +y_0=0 +lon_0=0 +datum=WGS84"


def compute_uahipa_seasonal_correction_factors(
    region_seasonal_clim_df,
    ref_product=PMB_NAME,
    target_product=UAHIPA_NAME,
    value_col="precipitation",
    region_col="region",
    product_col="product",
    season_col="season",
):
    """
    Compute PMB-based seasonal correction factors for UA-HIPA.

    CF = PMB / UA-HIPA

    Input dataframe must contain:
        region, product, season, precipitation

    Returns
    -------
    cf_df : pd.DataFrame
        Columns:
        region, season, pmb, uahipa, correction_factor
    """

    df = region_seasonal_clim_df.copy()

    ref = (
        df[df[product_col] == ref_product]
        [[region_col, season_col, value_col]]
        .rename(columns={value_col: "pmb"})
    )

    target = (
        df[df[product_col] == target_product]
        [[region_col, season_col, value_col]]
        .rename(columns={value_col: "uahipa"})
    )

    cf = target.merge(ref, on=[region_col, season_col], how="left")

    cf["correction_factor"] = np.where(
        (cf["uahipa"].notna()) & (cf["uahipa"] > 0),
        cf["pmb"] / cf["uahipa"],
        np.nan,
    )

    cf["season"] = pd.Categorical(
        cf["season"],
        categories=SEASON_ORDER,
        ordered=True,
    )

    cf = (
        cf.sort_values([region_col, season_col])
        .reset_index(drop=True)
    )

    return cf
#--------------------------------------------------------------------------------
def open_uahipa_sh_template(uahipa_sh_file):
    """
    Open one representative UA-HIPA SH gridded file.
    Expected shape:
        y: 180
        x: 1440
    Coordinates:
        y = -45 to -89.75
        x = -180 to 179.75
    """
    ds = xr.open_dataset(uahipa_sh_file, group='SH')

    if "precipitation" in ds:
        template = ds["precipitation"].isel(time=0) if "time" in ds["precipitation"].dims else ds["precipitation"]
    else:
        # fallback to first 2D data variable
        first_var = list(ds.data_vars)[0]
        template = ds[first_var]
        if "time" in template.dims:
            template = template.isel(time=0)

    template = template.squeeze(drop=True)

    if set(template.dims) != {"y", "x"}:
        raise ValueError(f"Expected template dims ('y', 'x'), got {template.dims}")

    template = template.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    template = template.rio.write_crs(CRS_WGS84, inplace=False)

    return template

#--------------------------------------------------------------------------------
def prepare_basin_mask_on_uahipa_grid(
    basin_mask_source,
    uahipa_template,
):
    """
    Reproject or match IMBIE basin IDs to the UA-HIPA SH lat-lon grid.

    basin_mask_source should be an xarray DataArray with valid CRS.
    Use nearest-neighbor because basin IDs are categorical.
    """

    basin = basin_mask_source.squeeze(drop=True)

    if not basin.rio.crs:
        raise ValueError("Basin mask has no CRS. Please write CRS before using this function.")

    basin = basin.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    basin_on_uahipa = basin.rio.reproject_match(
        uahipa_template,
        resampling=Resampling.nearest,
    )

    basin_on_uahipa = basin_on_uahipa.squeeze(drop=True)

    if "spatial_ref" in basin_on_uahipa.coords:
        basin_on_uahipa = basin_on_uahipa.drop_vars("spatial_ref")

    return basin_on_uahipa

#--------------------------------------------------------------------------------
def build_region_mask_from_basin_ids(basin_on_uahipa):
    """
    Build region mask:
        0 = outside Antarctica / ocean / no correction
        1 = WAIS
        2 = EAIS
    """

    region_mask = xr.full_like(basin_on_uahipa, 0, dtype=np.int16)

    region_mask = xr.where(basin_on_uahipa.isin(WAIS_IDS), 1, region_mask)
    region_mask = xr.where(basin_on_uahipa.isin(EAIS_IDS), 2, region_mask)

    region_mask = region_mask.rename("region_mask")
    region_mask.attrs.update({
        "long_name": "Antarctic region mask for PMB-based UA-HIPA correction",
        "flag_values": np.array([0, 1, 2], dtype=np.int16),
        "flag_meanings": "no_correction WAIS EAIS",
    })

    return region_mask
#--------------------------------------------------------------------------------

def make_cf_field_for_season(
    cf_df,
    region_mask,
    season,
    region_col="region",
    season_col="season",
    cf_col="correction_factor",
):
    """
    Create one 2D correction-factor field for a given season.
    """

    field = xr.full_like(region_mask.astype(float), np.nan, dtype=np.float32)

    for region_name, code in REGION_CODE.items():
        sub = cf_df[
            (cf_df[region_col] == region_name) &
            (cf_df[season_col].astype(str) == season)
        ]

        if sub.empty:
            cf_value = np.nan
        else:
            cf_value = float(sub[cf_col].iloc[0])

        if np.isfinite(cf_value):
            field = xr.where(region_mask == code, cf_value, field)

    field = field.rename(f"cf_{season}")
    field.attrs.update({
        "long_name": f"PMB-based UA-HIPA seasonal correction factor for {season}",
        "description": "Multiplicative factor defined as PMB / UA-HIPA",
        "units": "1",
        "season": season,
    })

    return field.astype("float32")

#--------------------------------------------------------------------------------
def build_uahipa_cf_dataset(cf_df, basin_on_uahipa, uahipa_template):
    """
    Build final lat-lon SH correction-factor dataset.
    """

    region_mask = build_region_mask_from_basin_ids(basin_on_uahipa)

    data_vars = {
        f"cf_{season}": make_cf_field_for_season(
            cf_df=cf_df,
            region_mask=region_mask,
            season=season,
        )
        for season in SEASON_ORDER
    }

    data_vars["region_mask"] = region_mask

    ds_cf = xr.Dataset(data_vars)

    ds_cf = ds_cf.assign_coords({
        "y": uahipa_template["y"],
        "x": uahipa_template["x"],
    })

    ds_cf.attrs.update({
        "title": "PMB-based seasonal correction factors for UA-HIPA Antarctic precipitation",
        "correction_definition": "CF = P_MB / UA-HIPA",
        "application": "UA-HIPA_corrected = UA-HIPA_original * CF_season",
        "regions": "WAIS and EAIS based on IMBIE basin IDs",
        "region_mask_values": "0=no correction/ocean, 1=WAIS, 2=EAIS",
        "source_reference": "PMB seasonal climatology",
        "target_product": "UA-HIPA",
        "grid": "Southern Hemisphere UA-HIPA 0.25 degree lat-lon grid",
    })

    ds_cf = ds_cf.rio.write_crs(CRS_WGS84, inplace=False)

    return ds_cf

#--------------------------------------------------------------------------------
def clean_encoding_attrs(ds):
    """
    Remove NetCDF encoding keys from variable attrs before saving.
    These keys should be controlled through the encoding dict, not attrs.
    """
    ds = ds.copy()

    bad_attr_keys = [
        "_FillValue",
        "missing_value",
        "scale_factor",
        "add_offset",
    ]

    for v in ds.variables:
        for key in bad_attr_keys:
            ds[v].attrs.pop(key, None)

    return ds

#--------------------------------------------------------------------------------
def save_cf_latlon_nc(ds_cf, out_nc):
    """
    Save correction-factor dataset as NetCDF.
    """

    ds_cf = clean_encoding_attrs(ds_cf)

    encoding = {}

    for v in ds_cf.data_vars:
        if v.startswith("cf_"):
            encoding[v] = {
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
                "zlib": True,
                "complevel": 4,
            }
        elif v == "region_mask":
            encoding[v] = {
                "dtype": "int16",
                "_FillValue": np.int16(0),
                "zlib": True,
                "complevel": 4,
            }

    ds_cf.to_netcdf(out_nc, encoding=encoding)
    print(f"Saved: {out_nc}")
#--------------------------------------------------------------------------------
def save_cf_polar_stereo_nc(
    ds_cf_latlon,
    out_nc,
    target_resolution_m=25000,
):
    """
    Reproject each CF field and region mask to Antarctic polar stereographic.

    Uses nearest-neighbor because the CFs are region constants and the region mask
    is categorical.
    """

    ds_in = ds_cf_latlon.copy()
    ds_in = ds_in.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    ds_in = ds_in.rio.write_crs(CRS_WGS84, inplace=False)

    out_vars = {}

    for var in ds_in.data_vars:
        da = ds_in[var]

        da_stereo = da.rio.reproject(
            CRS_SH_STEREO,
            resolution=target_resolution_m,
            resampling=Resampling.nearest,
        )

        if var.startswith("cf_"):
            da_stereo = da_stereo.astype("float32")
            da_stereo = da_stereo.where(np.isfinite(da_stereo))
            da_stereo = da_stereo.where(da_stereo < 1e20)
            da_stereo = da_stereo.where(da_stereo > -1e20)

        elif var == "region_mask":
            da_stereo = da_stereo.where(np.isfinite(da_stereo), 0)
            da_stereo = da_stereo.where(da_stereo < 1e20, 0)
            da_stereo = da_stereo.fillna(0).astype("int16")

        out_vars[var] = da_stereo

    ds_stereo = xr.Dataset(out_vars)

    ds_stereo.attrs.update(ds_cf_latlon.attrs)
    ds_stereo.attrs.update({
        "grid": "Antarctic polar stereographic",
        "projection": CRS_SH_STEREO,
        "target_resolution_m": target_resolution_m,
    })

    ds_stereo = ds_stereo.rio.write_crs(CRS_SH_STEREO, inplace=False)

    # IMPORTANT: remove conflicting _FillValue attrs before writing
    ds_stereo = clean_encoding_attrs(ds_stereo)

    encoding = {}
    for v in ds_stereo.data_vars:
        if v.startswith("cf_"):
            encoding[v] = {
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
                "zlib": True,
                "complevel": 4,
            }
        elif v == "region_mask":
            encoding[v] = {
                "dtype": "int16",
                "_FillValue": np.int16(0),
                "zlib": True,
                "complevel": 4,
            }

    ds_stereo.to_netcdf(out_nc, encoding=encoding)
    print(f"Saved: {out_nc}")

    return ds_stereo

#%%
# =============================================================================
# MAIN WORKFLOW
# =============================================================================
regional_monthly_cos_df = pd.read_csv('/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/out_dfs/monthly_precip_over_imbie_basins_20260513.csv')
# 1. Seasonal climatology from regional monthly cosine-weighted series
region_seasonal_clim_cos = compute_seasonal_climatology_from_regional_series(
    regional_monthly_cos_df,
    drop_incomplete=True,
)

# 2. Correction factors: PMB / UA-HIPA
uahipa_cf_df = compute_uahipa_seasonal_correction_factors(
    region_seasonal_clim_cos,
    ref_product=r"$P_{\mathrm{MB}}$",
    target_product="UA-HIPA",
)

print(uahipa_cf_df)

# Optional CSV table for Ali/Omid
out_dfs = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/out_dfs'
    
uahipa_cf_df.to_csv(
    os.path.join(out_dfs, "UA_HIPA_PMB_seasonal_correction_factors_table.csv"),
    index=False,
)

# 3. Open representative UA-HIPA SH file
temp_file = 'clavrx_NSS.GHRR.M1.D13140.S0720.E0816.B0347576.SV.hirs_avhrr_fusion.level2_collocated_wgs.nc'
uh_hipa_dir= r"/scratch/omidzandi/AVHRR_retrieved_from_HPC_collocated/2013"
temp_file = os.path.join(uh_hipa_dir, temp_file)
uahipa_template = open_uahipa_sh_template(
    temp_file
)

# 4. Load original IMBIE basin mask
basins_path = r'/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/data/basins'
basins = xr.open_dataarray(os.path.join(basins_path, 'bedmap3_basins_0.1deg.tif'))
basins = basins.where((basins > 1) & np.isfinite(basins))

if not basins.rio.crs:
    basins = basins.rio.write_crs(CRS.from_proj4(CRS_SH_STEREO), inplace=False)

# 5. Regrid basin IDs to UA-HIPA SH lat-lon grid
basin_on_uahipa = prepare_basin_mask_on_uahipa_grid(
    basin_mask_source=basins,
    uahipa_template=uahipa_template,
)

# 6. Build correction-factor dataset on UA-HIPA SH grid
ds_cf_latlon = build_uahipa_cf_dataset(
    cf_df=uahipa_cf_df,
    basin_on_uahipa=basin_on_uahipa,
    uahipa_template=uahipa_template,
)

# 7. Save lat-lon version
save_cf_latlon_nc(
    ds_cf_latlon,
    os.path.join(out_dfs, 
                 "uahipa_pmb_seasonal_correction_factors_SH_latlon_0p25deg.nc"),
)

# 8. Optional: save polar stereographic version
ds_cf_stereo = save_cf_polar_stereo_nc(
    ds_cf_latlon,
    os.path.join(out_dfs, 
                 "uahipa_pmb_seasonal_correction_factors_SH_polarstereo.nc"),
    target_resolution_m=25000,
)
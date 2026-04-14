#%%
from program_utils import *
from Extra_util_functions import *
import matplotlib.patheffects as pe
#%%
# Floating Variables


SATELLITE_CATEGORIES = {
    "ATMS": ["SNPP", "NOAA-20"],
    "DMSP-SSMIS": ["F16", "F17", "F18", "F19"],
    "MHS": [("NOAA", ("NOAA-18", "NOAA-19")),
            ("METOP", ("METOP-A", "METOP-B", "METOP-C"))],
    "GCOM-W1_AMSR2": ["AMSR2"],
}

#---------------------------------------------------------------------------------
product_order_corr = [
    r"$P_{\mathrm{MB}}$",
    "ERA5",
    "GPCP v3.3",
    # "ATMS",
    # "ATMS (corr.)",
    # "MHS",
    # "MHS (corr.)",
    # "DMSP SSMIS",
    # "DMSP SSMIS (corr.)",
    # "AMSR2",
    # "AMSR2 (corr.)",   
    "GPM PMW V07 (corr.)",
    "GPM PMW V07",
]
#---------------------------------------------------------------------------------

product_styles_corr = {
    r"$P_{\mathrm{MB}}$": {"color": "k", "marker": "o", "lw": 2.5},

    "ERA5": {"color": "blue", "marker": "s", "lw": 2.5},
    "GPCP v3.3": {"color": "orange", "marker": "D", "lw": 2.5},

    "ATMS": {"color": "tab:blue", "lw": 1.5},
    "ATMS (corr.)": {"color": "tab:blue", "ls": "--", "lw": 2},

    "MHS": {"color": "lime", "lw": 1.5},
    "MHS (corr.)": {"color": "lime", "ls": "--", "lw": 2},

    "DMSP SSMIS": {"color": "green", "lw": 1.5},
    "DMSP SSMIS (corr.)": {"color": "green", "ls": "--", "lw": 2},

    "AMSR2": {"color": "red", "lw": 1.5},
    "AMSR2 (corr.)": {"color": "red", "ls": "--", "lw": 2},

    "GPM PMW V07": {"color": "cyan", "lw": 3.5},
    "GPM PMW V07 (corr.)": {"color": "cyan", "ls": "--", "lw": 3.5},
}
#---------------------------------------------------------------------------------

corr_targets = [
    # "ATMS",
    # "MHS",
    # "DMSP SSMIS",
    # "AMSR2",
    "GPM PMW V07",
]

product_order = [
    r"$P_{\mathrm{MB}}$",
    "ERA5",
    "GPCP v3.3",
    # "RACMO 2.4p1",
    "ATMS",
    "MHS",
    "DMSP SSMIS",
    "AMSR2",
    "GPM PMW V07",
]
#---------------------------------------------------------------------------------

product_styles = {
    r"$P_{\mathrm{MB}}$": {"color": "k", "marker": "o", "lw": 2.0},
    "ERA5": {"color": "blue", "marker": "s", "lw": 1.8},
    "GPCP v3.3": {"color": "tab:orange", "marker": "D", "lw": 1.8},
    # "RACMO 2.4p1": {"color": "tab:green", "marker": "^", "lw": 1.8},
    "ATMS": {"lw": 1.5},
    "MHS": {"lw": 1.5},
    "DMSP SSMIS": {"lw": 1.5},
    "AMSR2": {"lw": 1.5},
    "GPM PMW V07": {"lw": 1.5},
}

#%%
def relative_bias(ref, test):
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    m = np.isfinite(ref) & np.isfinite(test)
    ref = ref[m]
    test = test[m]
    if len(ref) == 0 or np.nanmean(ref) == 0:
        return np.nan
    return np.nanmean(test - ref) / np.nanmean(ref)

# =============================================================================

# =============================================================================

def p_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 2:
        return np.nan
    return np.corrcoef(x, y)[0, 1]
# =============================================================================

# =============================================================================

def colors_for_basins(basin_ids):
    """
    Color mapping by basin ID using gist_ncar palette like before.
    Assumes basin IDs in 2..19.
    """
    cmap_colors = plt.cm.gist_ncar(np.linspace(0, 1, 19))
    cmap_colors[-1] = np.array([0.60, 0.60, 0.60, 1.0])  # basin 19 gray

    out = []
    for bid in basin_ids:
        idx = int(bid) - 1
        if 0 <= idx < len(cmap_colors):
            out.append(cmap_colors[idx])
        else:
            out.append((0.5, 0.5, 0.5, 1.0))
    return out

def load_basin_grid(basins_path, crs_stereo):
    """
    Load the native 0.1° basin grid and attach CRS if missing.
    """
    basins = xr.open_dataarray(os.path.join(basins_path, 'bedmap3_basins_0.1deg.tif'))

    # Keep only valid IMBIE basin IDs (> 1)
    basins = basins.where((basins > 1) & (basins.notnull()))

    if not basins.rio.crs:
        basins = basins.rio.write_crs(CRS.from_proj4(crs_stereo))

    return basins
# =============================================================================

# =============================================================================

def print_basin_grid_info(basins):
    """
    Print basin grid geometry info.
    """
    basin_transform = basins.rio.transform()
    height, width = basins.data.shape[1:]
    xmin, ymax = basin_transform.c, basin_transform.f
    xres, yres = basin_transform.a, -basin_transform.e
    xmax = xmin + width * xres
    ymin = ymax - height * yres

    print(f"Basin grid: width={width}, height={height}, xres={xres}, yres={yres}")
    print(f"Basin bounds: {(xmin, xmax, ymin, ymax)}")

# =============================================================================

# =============================================================================

def build_target_latlon_template_from_basin_grid(basins):
    """
    Build the common 0.1° lat-lon target grid from the basin raster itself.

    Important:
    - The basin raster is native in polar stereographic.
    - We convert/reproject just the geometry to EPSG:4326 to define the
      comparison grid.
    - This grid becomes the common support for all products.
    """
    basins_2d = basins.squeeze(drop=True)

    basins_2d = basins_2d.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    # Reproject basin IDs to EPSG:4326 just to establish the target geometry
    basin_latlon = basins_2d.rio.reproject("EPSG:4326", resampling=Resampling.nearest)

    # Rename spatial dims to lat/lon for consistency
    if "x" in basin_latlon.dims or "y" in basin_latlon.dims:
        basin_latlon = basin_latlon.rename({"x": "lon", "y": "lat"})

    # Ensure ascending lon / descending lat are handled consistently later
    basin_latlon = basin_latlon.sortby("lon")
    basin_latlon = basin_latlon.sortby("lat", ascending=False)

    # Use one 2D slice as template
    target_template = basin_latlon.copy()
    target_template = target_template.rio.write_crs("EPSG:4326", inplace=False)

    return target_template
# =============================================================================

# =============================================================================


def reproject_basin_ids_to_target_grid(basins, target_template):
    """
    Reproject basin IDs onto the common target lat-lon grid.
    Nearest-neighbor is correct because basin IDs are categorical.
    """
    basins_native = basins.squeeze(drop=True)
    basins_native = basins_native.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    # Write CRS from original basin object if needed
    if not basins_native.rio.crs:
        raise ValueError("Basin grid CRS is missing.")

    basin_mask_latlon = basins_native.rio.reproject_match(
        target_template,
        resampling=Resampling.nearest
    )

    # Rename to lat/lon if needed
    rename_map = {}
    if "x" in basin_mask_latlon.dims:
        rename_map["x"] = "lon"
    if "y" in basin_mask_latlon.dims:
        rename_map["y"] = "lat"
    if rename_map:
        basin_mask_latlon = basin_mask_latlon.rename(rename_map)

    basin_mask_latlon = basin_mask_latlon.sortby("lon")
    basin_mask_latlon = basin_mask_latlon.sortby("lat", ascending=False)

    return basin_mask_latlon
# =============================================================================

# =============================================================================

def make_region_masks_from_basin_mask(basin_mask_latlon, region_basins):
    """
    Build boolean region masks (AIS/WAIS/EAIS) from basin-ID grid on target grid.
    """
    region_masks = {}
    basin_vals = basin_mask_latlon.values

    for region_name, basin_ids in region_basins.items():
        region_masks[region_name] = np.isin(basin_vals, basin_ids)

    return region_masks

# =============================================================================

# =============================================================================

def open_gpcp_monthly(all_gpcp_files):
    """
    Open GPCP monthly data.
    """
    ds = xr.open_mfdataset(
        all_gpcp_files,
        combine="nested",
        concat_dim="time",
        coords="minimal",
        compat="override",
        parallel=True,
        engine="netcdf4",
        chunks={"time": 120, "lat": 180, "lon": 360},
        cache=False
    )

    ds = ds_swaplon(ds)
    return ds
# =============================================================================

# =============================================================================


def open_era5_daily(all_era5_files):
    """
    Open ERA5 daily data using the existing processing helper.
    """
    era5_ds_xr_list = []

    for er5 in all_era5_files:
        if not os.path.exists(er5):
            raise FileNotFoundError(f"ERA5 file not found: {er5}")

        era5_data = process_era5_file(er5)
        era5_ds_xr_list.append(era5_data)

    if not era5_ds_xr_list:
        raise RuntimeError("No ERA5 files were loaded.")

    era5_ds_xr = xr.concat(era5_ds_xr_list, dim="valid_time")
    print("✅ ERA5 loading complete")
    print("-" * 30)

    del era5_ds_xr_list
    gc.collect()

    return era5_ds_xr
# =============================================================================

# =============================================================================

def prepare_gpcp_monthly_on_target(gpcp_ds, target_template):
    """
    Convert GPCP monthly precip field to the common 0.1° target grid
    using nearest neighbor.

    This preserves original grid-cell values while sampling them on the
    finer common analysis grid.
    """
    gpcp_mon = gpcp_ds["sat_gauge_precip"].copy()

    # Subset Antarctica
    gpcp_mon = gpcp_mon.sel(lat=slice(-60, -90))

    # Set spatial metadata for rioxarray
    gpcp_mon = gpcp_mon.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    gpcp_mon = gpcp_mon.rio.write_crs("EPSG:4326", inplace=False)

    # Reproject to common target grid
    gpcp_mon_01 = gpcp_mon.rio.reproject_match(
        target_template,
        resampling=Resampling.nearest
    )

    # Ensure standard dimension names
    rename_map = {}
    if "x" in gpcp_mon_01.dims:
        rename_map["x"] = "lon"
    if "y" in gpcp_mon_01.dims:
        rename_map["y"] = "lat"
    if rename_map:
        gpcp_mon_01 = gpcp_mon_01.rename(rename_map)

    gpcp_mon_01 = gpcp_mon_01.sortby("lon")
    gpcp_mon_01 = gpcp_mon_01.sortby("lat", ascending=False)

    return gpcp_mon_01

# =============================================================================

# =============================================================================

def prepare_era5_monthly_on_target(era5_ds, target_template):
    """
    Convert ERA5 daily precipitation to monthly means/totals on the common
    0.1° target grid using nearest neighbor.

    Assumes process_era5_file() already returns a daily precipitation field
    in a consistent form.
    """
    # Earlier workflow showed valid_time handling; here we normalize to time.
    if "valid_time" in era5_ds.dims:
        era5_da = era5_ds
        time_dim = "valid_time"
    elif "time" in era5_ds.dims:
        era5_da = era5_ds
        time_dim = "time"
    else:
        raise ValueError("ERA5 object has neither 'valid_time' nor 'time' dimension.")

    # Subset Antarctica
    era5_ant = era5_da.sel(latitude=slice(-60, -90))

    # Rename to standard names before reprojection
    rename_map = {}
    if "latitude" in era5_ant.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in era5_ant.dims:
        rename_map["longitude"] = "lon"
    if rename_map:
        era5_ant = era5_ant.rename(rename_map)

    if time_dim == "valid_time":
        era5_ant = era5_ant.rename({"valid_time": "time"})

    # Set CRS and spatial dims
    era5_ant = era5_ant.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    era5_ant = era5_ant.rio.write_crs("EPSG:4326", inplace=False)

    # Reproject daily field to target grid
    era5_ant_01 = era5_ant.rio.reproject_match(
        target_template,
        resampling=Resampling.nearest
    )

    # Standardize dimension names
    rename_map = {}
    if "x" in era5_ant_01.dims:
        rename_map["x"] = "lon"
    if "y" in era5_ant_01.dims:
        rename_map["y"] = "lat"
    if rename_map:
        era5_ant_01 = era5_ant_01.rename(rename_map)

    era5_ant_01 = era5_ant_01.sortby("lon")
    era5_ant_01 = era5_ant_01.sortby("lat", ascending=False)

    # Monthly aggregation
    # Use whatever is appropriate for your ERA5 variable convention:
    # - if daily mean rate -> monthly sum may be needed later
    # - if daily accumulation already in mm/day -> monthly sum is appropriate
    era5_mon_01 = era5_ant_01.resample(time="MS").sum(skipna=True)

    return era5_mon_01

# =============================================================================

# =============================================================================

def prepare_pmb_monthly_on_target(P_mm_mnth, target_template):
    """
    Reproject PMB monthly field from native polar stereographic to the common
    0.1° target lat-lon grid.

    Use Resampling.average because PMB is a continuous precipitation-like field.
    """
    pmb_native = P_mm_mnth.rename({"date": "time"}).copy()

    # Drop auxiliary coords that can interfere with reprojection
    drop_coords = [c for c in ["basin_id", "mapping", "band"] if c in pmb_native.coords]
    pmb_native = pmb_native.drop_vars(drop_coords, errors="ignore")

    # Set spatial metadata
    pmb_native = pmb_native.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    pmb_native = pmb_native.rio.write_crs(P_mm_mnth.mapping.crs_wkt, inplace=False)

    # Reproject to common target
    pmb_latlon = pmb_native.rio.reproject_match(
        target_template,
        resampling=Resampling.average
    )

    # Standardize names
    rename_map = {}
    if "x" in pmb_latlon.dims:
        rename_map["x"] = "lon"
    if "y" in pmb_latlon.dims:
        rename_map["y"] = "lat"
    if rename_map:
        pmb_latlon = pmb_latlon.rename(rename_map)

    pmb_latlon = pmb_latlon.sortby("lon")
    pmb_latlon = pmb_latlon.sortby("lat", ascending=False)

    return pmb_latlon

# =============================================================================

# =============================================================================

def subset_common_period(da, start="2013-01-01", end="2020-12-31"):
    """
    Restrict time series to common analysis period.
    """
    return da.sel(time=slice(start, end))

# =============================================================================

# =============================================================================
def replace_fill_with_nan(da):
    fillv = da.attrs.get("_FillValue", None)
    if fillv is not None:
        da = da.where(da != fillv)
    da = da.where(np.isfinite(da))
    return da

# =============================================================================

# =============================================================================

def make_lat_weight_2d(da_2d, lat_name="lat", lon_name="lon"):
    """
    Build 2D cosine(latitude) weights matching a 2D lat-lon DataArray.
    """
    lat_vals = da_2d[lat_name].values
    lon_vals = da_2d[lon_name].values

    w_lat = np.cos(np.deg2rad(lat_vals))
    w_2d = xr.DataArray(
        np.repeat(w_lat[:, None], len(lon_vals), axis=1),
        coords={lat_name: lat_vals, lon_name: lon_vals},
        dims=(lat_name, lon_name),
    )
    return w_2d

# =============================================================================

# =============================================================================

# =============================================================================
# SECTION 1D. GPM MICROWAVE MONTHLY INPUT PREPARATION
# =============================================================================

def ensure_1d_latlon_coords(da, lat_name="lat", lon_name="lon"):
    """
    Ensure a DataArray has valid 1D latitude/longitude coordinates attached
    to the matching dimensions so rioxarray can compute bounds.
    """
    # Must have dims
    if lat_name not in da.dims or lon_name not in da.dims:
        raise ValueError(f"Expected dims ({lat_name}, {lon_name}), got {da.dims}")

    # If coords missing, try to rebuild from indexes
    if lat_name not in da.coords:
        da = da.assign_coords({lat_name: np.arange(da.sizes[lat_name])})
    if lon_name not in da.coords:
        da = da.assign_coords({lon_name: np.arange(da.sizes[lon_name])})

    lat = da.coords[lat_name]
    lon = da.coords[lon_name]

    # If coordinates are not 1D, fail clearly
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(
            f"{lat_name}/{lon_name} must be 1D for reprojection; "
            f"got lat.ndim={lat.ndim}, lon.ndim={lon.ndim}"
        )

    # Reattach as explicit coords on matching dims
    da = da.assign_coords({
        lat_name: (lat_name, np.asarray(lat.values, dtype=float)),
        lon_name: (lon_name, np.asarray(lon.values, dtype=float)),
    })

    # Sort to keep increasing lon and decreasing lat if needed
    da = da.sortby(lon_name)
    # For Antarctica workflows, descending lat is often fine, but rioxarray can work either way
    # We'll keep natural order if already monotonic; otherwise sort descending if needed.
    lat_vals = da[lat_name].values
    if len(lat_vals) > 1:
        if not (np.all(np.diff(lat_vals) > 0) or np.all(np.diff(lat_vals) < 0)):
            da = da.sortby(lat_name)

    return da

def collect_gpm_family_files(gpm_satellites_path):
    """
    Collect monthly files by microwave family.
    Rename GCOM-W1_AMSR2 -> AMSR2 for convenience.
    """
    satellite_category_files = {}

    for satellite in os.listdir(gpm_satellites_path):
        monthly_path = os.path.join(gpm_satellites_path, satellite, "Monthly")

        if not os.path.isdir(monthly_path):
            continue

        files = glob.glob(
            os.path.join(monthly_path, "**", "*.nc4"),
            recursive=True
        )

        key = "AMSR2" if satellite == "GCOM-W1_AMSR2" else satellite
        satellite_category_files[key] = sorted(files)

    return satellite_category_files


def convert_gpm_rate_to_mm_month(da, time_name="time"):
    """
    Convert monthly mean precipitation rate [mm/hr] to monthly accumulation [mm/month].
    """
    days_in_month = xr.DataArray(
        pd.to_datetime(da[time_name].values).days_in_month,
        dims=[time_name],
        coords={time_name: da[time_name]}
    )
    return da * 24.0 * days_in_month


def clean_gpm_da(da):
    """
    Replace code missing values and non-finite values with NaN.
    """
    # handle common fill values
    da = da.where(np.isfinite(da))
    da = da.where(da != -9999.9)
    da = da.where(da != -9999)
    return da

def subset_antarctica_lat(da, lat_name="lat", north_bound=-55, south_bound=-90):
    """
    Subset Antarctic latitudes while respecting whether latitude is ascending or descending.
    """
    lat_vals = da[lat_name].values
    if len(lat_vals) == 0:
        raise ValueError("Latitude coordinate is empty before Antarctic subsetting.")

    if lat_vals[0] > lat_vals[-1]:
        # descending
        return da.sel({lat_name: slice(north_bound, south_bound)})
    else:
        # ascending
        return da.sel({lat_name: slice(south_bound, north_bound)})

def prepare_one_gpm_family_monthly(
    files,
    sat_name,
    preprocess_func,
    target_template_01deg,
    basin_mask_01deg,
):
    """
    Open one GPM MW family, average duplicate timestamps, convert units,
    reproject to common 0.1° grid, and mask to valid basin domain.
    """
    print(f"Preparing {sat_name} with {len(files)} files")

    ds = xr.open_mfdataset(
        sorted(files),
        preprocess=preprocess_func,
        combine="nested",
        concat_dim="time",
        parallel=True
    )

    da = ds["surfacePrecipitation"].transpose("time", "lat", "lon")

    # average duplicate timestamps within family
    # useful where multiple platforms contribute in same family
    if sat_name != "AMSR2":
        da = da.groupby("time").mean(skipna=True)

    # clean time stamps to month-start
    da = da.assign_coords(
        time=pd.to_datetime(da["time"].values).to_period("M").to_timestamp()
    )

    # clean fill values
    da = clean_gpm_da(da)

    # convert mm/hr -> mm/month
    da = convert_gpm_rate_to_mm_month(da, time_name="time")
    da.name = f"{sat_name}_mm_month"

    # subset period
    da = da.sel(time=slice("2013-01-01", "2020-12-31"))

    # subset Antarctica with latitude-order awareness
    da = subset_antarctica_lat(da, lat_name="lat", north_bound=-55, south_bound=-90)

    if da.sizes["lat"] == 0 or da.sizes["lon"] == 0:
        raise ValueError(
            f"{sat_name}: empty spatial domain after Antarctic subset. "
            f"Check latitude ordering and coordinates."
        )

    # force clean 1D coords for reprojection
    da = ensure_1d_latlon_coords(da, lat_name="lat", lon_name="lon")

    # attach CRS / spatial dims
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    da = da.rio.write_crs("EPSG:4326", inplace=False)

    print(f"{sat_name}: dims={da.dims}, lat={da['lat'].shape}, lon={da['lon'].shape}")

    # remap to common 0.1° grid
    da_01 = da.rio.reproject_match(
        target_template_01deg,
        resampling=Resampling.nearest
    )

    rename_map = {}
    if "x" in da_01.dims:
        rename_map["x"] = "lon"
    if "y" in da_01.dims:
        rename_map["y"] = "lat"
    if rename_map:
        da_01 = da_01.rename(rename_map)

    da_01 = da_01.sortby("lon")
    da_01 = da_01.sortby("lat", ascending=False)

    # clean again after reprojection
    da_01 = clean_gpm_da(da_01)

    # apply basin-analysis mask and final latitude cutoff
    da_01 = da_01.where(basin_mask_01deg.notnull())
    da_01 = da_01.where(da_01["lat"] < -60)

    return da_01


def build_gpm_family_monthly_dict(
    gpm_satellites_path,
    preprocess_func,
    target_template_01deg,
    basin_mask_01deg,
):
    """
    Build monthly 0.1° gridded fields for all GPM MW families.
    """
    satellite_category_files = collect_gpm_family_files(gpm_satellites_path)

    gpm_family_dict = {}

    for sat_name, files in satellite_category_files.items():
        if len(files) == 0:
            continue

        da_01 = prepare_one_gpm_family_monthly(
            files=files,
            sat_name=sat_name,
            preprocess_func=preprocess_func,
            target_template_01deg=target_template_01deg,
            basin_mask_01deg=basin_mask_01deg,
        )

        # rename display name to match your plotting language
        display_name = sat_name
        if sat_name == "DMSP-SSMIS":
            display_name = "DMSP SSMIS"

        gpm_family_dict[display_name] = da_01

    return gpm_family_dict


def build_gpm_pmw_mean(gpm_family_dict, mean_name="GPM PMW V07"):
    """
    Mean across available GPM MW families on the common 0.1° grid.
    """
    valid_arrays = []
    for name, da in gpm_family_dict.items():
        valid_arrays.append(da)

    if len(valid_arrays) == 0:
        raise ValueError("No GPM MW family arrays available to average.")

    stacked = xr.concat(valid_arrays, dim="pmw_family")
    pmw_mean = stacked.mean(dim="pmw_family", skipna=True)
    pmw_mean.name = mean_name
    return pmw_mean
#===============================================================================


def cosine_weighted_mean_masked(da_2d, region_mask, lat_name="lat", lon_name="lon"):
    """
    Cosine-weighted mean of a 2D field over a boolean region mask.

    Parameters
    ----------
    da_2d : xr.DataArray
        2D monthly field on common lat-lon grid.
    region_mask : np.ndarray or xr.DataArray
        Boolean mask with same lat/lon shape as da_2d.
    """
    # Ensure mask is DataArray aligned to da_2d
    if not isinstance(region_mask, xr.DataArray):
        region_mask = xr.DataArray(
            region_mask,
            coords={lat_name: da_2d[lat_name], lon_name: da_2d[lon_name]},
            dims=(lat_name, lon_name),
        )

    # Mask the field
    da_masked = da_2d.where(region_mask)

    # Build cosine weights
    w_2d = make_lat_weight_2d(da_2d, lat_name=lat_name, lon_name=lon_name)

    # Keep weights only where data are valid and inside region
    valid = xr.where(np.isfinite(da_masked), 1.0, np.nan)
    w_valid = w_2d * valid

    num = (da_masked * w_valid).sum(dim=(lat_name, lon_name), skipna=True)
    den = w_valid.sum(dim=(lat_name, lon_name), skipna=True)

    return num / den
# =============================================================================

# =============================================================================

def build_region_monthly_series_cosine(
    product_da,
    region_masks,
    product_name,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
):
    """
    Build monthly cosine-weighted regional mean series for one product.

    Returns tidy dataframe with columns:
    time, region, product, precipitation
    """
    out = []

    for region_name, region_mask in region_masks.items():
        vals = []

        for t in product_da[time_name].values:
            da_2d = product_da.sel({time_name: t})
            reg_mean = cosine_weighted_mean_masked(
                da_2d,
                region_mask=region_mask,
                lat_name=lat_name,
                lon_name=lon_name,
            )

            vals.append(reg_mean.item() if hasattr(reg_mean, "item") else float(reg_mean.values))

        tmp = pd.DataFrame({
            "time": pd.to_datetime(product_da[time_name].values),
            "region": region_name,
            "product": product_name,
            "precipitation": vals,
        })
        out.append(tmp)

    return pd.concat(out, ignore_index=True)

# =============================================================================

# =============================================================================

def build_all_region_monthly_series_cosine(
    product_dict,
    region_masks,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
):
    """
    Build cosine-weighted monthly regional series for all products.

    Parameters
    ----------
    product_dict : dict
        Example:
        {
            r"$P_{\\mathrm{MB}}$": pmb_mon_01,
            "ERA5": era5_mnth_01,
            "GPCP v3.3": gpcp_mon_01,
        }
    """
    all_df = []

    for product_name, da in product_dict.items():
        print(f"Building monthly cosine-weighted series for {product_name} ...")
        df_prod = build_region_monthly_series_cosine(
            product_da=da,
            region_masks=region_masks,
            product_name=product_name,
            lat_name=lat_name,
            lon_name=lon_name,
            time_name=time_name,
        )
        all_df.append(df_prod)

    return pd.concat(all_df, ignore_index=True)

# =============================================================================

# =============================================================================
def compute_monthly_climatology_from_regional_series(df):
    """
    From tidy monthly regional dataframe, compute mean monthly climatology.

    Input columns required:
    time, region, product, precipitation
    """
    out = df.copy()
    out["month"] = pd.to_datetime(out["time"]).dt.month

    clim = (
        out.groupby(["region", "product", "month"], as_index=False)["precipitation"]
        .mean()
        .sort_values(["region", "product", "month"])
    )
    return clim

# =============================================================================

# =============================================================================

def month_to_season(month):
    if month in [12, 1, 2]:
        return "DJF"
    elif month in [3, 4, 5]:
        return "MAM"
    elif month in [6, 7, 8]:
        return "JJA"
    else:
        return "SON"
# =============================================================================

# =============================================================================

def build_season_year(df):
    """
    Assign season year so that Dec belongs to the following DJF year.
    """
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out["year"] = out["time"].dt.year
    out["month"] = out["time"].dt.month
    out["season"] = out["month"].apply(month_to_season)

    out["season_year"] = out["year"]
    out.loc[out["month"] == 12, "season_year"] = out.loc[out["month"] == 12, "year"] + 1

    return out

# =============================================================================

# =============================================================================
def compute_seasonal_totals_from_regional_series(df, drop_incomplete=True):
    """
    Convert monthly regional series to seasonal totals [mm/season].
    """
    out = build_season_year(df)

    seasonal = (
        out.groupby(["region", "product", "season_year", "season"], as_index=False)["precipitation"]
        .sum()
    )

    if drop_incomplete:
        counts = (
            out.groupby(["region", "product", "season_year", "season"], as_index=False)
            .size()
            .rename(columns={"size": "nmonths"})
        )
        seasonal = seasonal.merge(
            counts,
            on=["region", "product", "season_year", "season"],
            how="left"
        )
        seasonal = seasonal[seasonal["nmonths"] == 3].copy()
        seasonal = seasonal.drop(columns="nmonths")

    return seasonal
# =============================================================================

# =============================================================================

def compute_seasonal_climatology_from_regional_series(df, drop_incomplete=True):
    """
    Seasonal climatology [mean mm/season over years].
    """
    seasonal = compute_seasonal_totals_from_regional_series(df, drop_incomplete=drop_incomplete)

    clim = (
        seasonal.groupby(["region", "product", "season"], as_index=False)["precipitation"]
        .mean()
    )

    season_order = {"DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}
    clim["season_order"] = clim["season"].map(season_order)
    clim = clim.sort_values(["region", "product", "season_order"]).drop(columns="season_order")

    return clim
# =============================================================================

# =============================================================================

def compute_annual_totals_from_regional_series(df):
    """
    Annual totals [mm/year] from monthly regional series.
    """
    out = df.copy()
    out["year"] = pd.to_datetime(out["time"]).dt.year

    annual = (
        out.groupby(["region", "product", "year"], as_index=False)["precipitation"]
        .sum()
        .sort_values(["region", "product", "year"])
    )
    return annual

# =============================================================================
# SECTION 6.1. BRIDGE: TIDY REGIONAL MONTHLY DF -> WIDE REGION DATAFRAMES
# =============================================================================

def regional_monthly_tidy_to_region_dict(
    regional_monthly_df,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    time_col="time",
    region_col="region",
    product_col="product",
    value_col="precipitation",
):
    """
    Convert tidy monthly regional dataframe into a dict of wide dataframes:
        {
            "Antarctica": wide_df,
            "West Antarctica": wide_df,
            "East Antarctica": wide_df,
        }

    Each wide_df:
      - index = time
      - columns = products
      - values = precipitation
    """
    out = {}

    for region in region_order:
        sub = regional_monthly_df[regional_monthly_df[region_col] == region].copy()

        wide = (
            sub.pivot(index=time_col, columns=product_col, values=value_col)
            .sort_index()
        )

        wide.index = pd.to_datetime(wide.index)
        out[region] = wide

    return out

# =============================================================================
# SECTION 6.2. CONVENTIONAL SEASONAL SERIES + DESEASONALIZATION
# =============================================================================

def month_to_season(month):
    if month in [12, 1, 2]:
        return "DJF"
    elif month in [3, 4, 5]:
        return "MAM"
    elif month in [6, 7, 8]:
        return "JJA"
    else:
        return "SON"


def _ensure_month_start_index(df):
    out = df.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp(how="start")
    out = out.sort_index()
    return out


def _season_year_and_label_from_index(dt_index):
    idx = pd.to_datetime(dt_index)
    months = idx.month.to_numpy()
    years = idx.year.to_numpy().copy()

    seasons = np.array([month_to_season(m) for m in months], dtype=object)

    # December belongs to following DJF year
    years[months == 12] += 1

    return seasons, years


def _season_rep_timestamp(season_year, season):
    """
    Representative timestamp for each season:
      DJF -> Jan 1
      MAM -> Apr 1
      JJA -> Jul 1
      SON -> Oct 1
    """
    month_map = {"DJF": 1, "MAM": 4, "JJA": 7, "SON": 10}
    return pd.Timestamp(year=int(season_year), month=month_map[season], day=1)


def build_conventional_seasonal_series_from_region_monthly(
    ts_region_monthly,
    seasonal_mode="mean",
    drop_incomplete=True,
):
    """
    Build conventional seasonal time series from monthly regional series.

    Input:
      wide monthly DataFrame with index=time and columns=products
      Units typically mm/month.

    seasonal_mode:
      - "mean": mean of 3 months
      - "sum" : sum of 3 months
    """
    if seasonal_mode not in {"mean", "sum"}:
        raise ValueError("seasonal_mode must be 'mean' or 'sum'")

    ts = _ensure_month_start_index(ts_region_monthly)

    seasons, season_years = _season_year_and_label_from_index(ts.index)

    work = ts.copy()
    work["season"] = seasons
    work["season_year"] = season_years

    pieces = []

    for prod in ts.columns:
        sub = work[[prod, "season", "season_year"]].copy()
        g = sub.groupby(["season_year", "season"])[prod]

        if drop_incomplete:
            counts = g.count()
            valid_keys = counts[counts == 3].index

            if seasonal_mode == "mean":
                agg = g.mean().loc[valid_keys]
            else:
                agg = g.sum().loc[valid_keys]
        else:
            if seasonal_mode == "mean":
                agg = g.mean()
            else:
                agg = g.sum()

        agg = agg.reset_index(name=prod)
        agg["time"] = agg.apply(
            lambda r: _season_rep_timestamp(r["season_year"], r["season"]),
            axis=1
        )
        agg = agg[["time", prod]].set_index("time").sort_index()
        pieces.append(agg)

    seasonal_df = pd.concat(pieces, axis=1).sort_index()
    seasonal_df.index.name = "time"
    return seasonal_df


def deseasonalize_seasonal_series(df_seasonal):
    """
    Deseasonalize a conventional seasonal time series by subtracting
    the climatological mean for each season.
    """
    out = df_seasonal.copy()
    out = _ensure_month_start_index(out)

    month_to_seas = {1: "DJF", 4: "MAM", 7: "JJA", 10: "SON"}
    seas = out.index.month.map(month_to_seas)

    for c in out.columns:
        clim = out[c].groupby(seas).mean()
        out[c] = out[c] - seas.map(clim).values

    return out

def seasonal_scatter_stats_to_wide_table(
    stats_out,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    target_cols=("ERA5", "GPCP v3.3"),
    ref_col=r"$P_{\mathrm{MB}}$",
):
    rows = []

    for targ in target_cols:
        slope_vals = []
        std_vals = []
        cc_vals = []

        for region in region_order:
            st = stats_out.get(region, {}).get(targ, {})
            slope_vals.append(st.get("slope", np.nan))
            std_vals.append(st.get("std", np.nan))
            cc_vals.append(st.get("cc", np.nan))

        rows.append({
            "Comparison": f"{targ} vs {ref_col}",
            "Slope": "/".join([f"{v:.2f}" if np.isfinite(v) else "nan" for v in slope_vals]),
            "Standard deviation": "/".join([f"{v:.2f}" if np.isfinite(v) else "nan" for v in std_vals]),
            "Correlation": "/".join([f"{v:.2f}" if np.isfinite(v) else "nan" for v in cc_vals]),
        })

    return pd.DataFrame(rows)

# =============================================================================
# ANNUAL TOTALS AND 2013–2020 MEAN ANNUAL FIELDS
# =============================================================================

def monthly_to_annual_totals_field(da_monthly, time_name="time"):
    """
    Convert monthly field [mm/month] to annual totals [mm/year].
    Returns DataArray with dims: year, lat, lon
    """
    da = da_monthly.copy()
    years = pd.to_datetime(da[time_name].values).year

    annual = (
        da.assign_coords(year=(time_name, years))
          .groupby("year")
          .sum(dim=time_name, skipna=True)
    )
    return annual


def annual_to_multiyear_mean_field(da_annual, year_start=2013, year_end=2020):
    """
    Mean of annual totals across years.
    Input: year, lat, lon
    Output: lat, lon
    """
    da = da_annual.sel(year=slice(year_start, year_end))
    return da.mean(dim="year", skipna=True)


# =============================================================================
# COSINE-WEIGHTED BASIN MEANS FROM A 2D FIELD
# =============================================================================

def compute_basin_cosine_weighted_means_from_field(
    da_2d,
    basin_mask_2d,
    basin_ids,
    lat_name="lat",
    lon_name="lon",
    basin_name="basin",
    value_name="precipitation",
):
    """
    Compute cosine-weighted mean within each basin from a 2D lat-lon field.

    Parameters
    ----------
    da_2d : xr.DataArray
        2D field on common lat-lon grid, e.g. mean annual precipitation [mm/year]
    basin_mask_2d : xr.DataArray
        Basin ID grid on the same lat-lon grid
    basin_ids : list-like
        Basin IDs to compute
    """
    lat_vals = da_2d[lat_name].values
    lon_vals = da_2d[lon_name].values

    # 2D cosine(lat) weights
    w_lat = np.cos(np.deg2rad(lat_vals))
    w_2d = xr.DataArray(
        np.repeat(w_lat[:, None], len(lon_vals), axis=1),
        coords={lat_name: lat_vals, lon_name: lon_vals},
        dims=(lat_name, lon_name),
    )

    out = []

    for bid in basin_ids:
        mask = (basin_mask_2d == bid)
        da_b = da_2d.where(mask)

        valid = xr.where(np.isfinite(da_b), 1.0, np.nan)
        w_valid = w_2d * valid

        num = (da_b * w_valid).sum(dim=(lat_name, lon_name), skipna=True)
        den = w_valid.sum(dim=(lat_name, lon_name), skipna=True)

        mean_val = (num / den).item() if float(den.values) > 0 else np.nan

        out.append({
            basin_name: int(bid),
            value_name: mean_val,
        })

    return pd.DataFrame(out)
#%% The plots
# =============================================================================
# HELPER FUNCTIONS FOR NICE SYMMETRIC AXES
# =============================================================================

def _round_up_nice(x):
    """
    Round a positive number up to a nice value:
    1, 2, 2.5, 5, 10 × 10^n
    """
    if not np.isfinite(x) or x <= 0:
        return 1.0

    exp = np.floor(np.log10(x))
    frac = x / (10 ** exp)

    if frac <= 1:
        nice = 1
    elif frac <= 2:
        nice = 2
    elif frac <= 2.5:
        nice = 2.5
    elif frac <= 5:
        nice = 5
    else:
        nice = 10

    return nice * (10 ** exp)

# =============================================================================

# =============================================================================

def _get_axis_limits(x, y, pad_frac=0.08, symmetric=True):
    """
    Compute nice panel limits from x and y.
    """
    vals = np.concatenate([np.asarray(x).ravel(), np.asarray(y).ravel()])
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return (-1, 1)

    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    if symmetric:
        vmax_abs = max(abs(vmin), abs(vmax))
        vmax_abs = vmax_abs * (1 + pad_frac)
        vmax_abs = _round_up_nice(vmax_abs)
        return (-vmax_abs, vmax_abs)
    else:
        span = vmax - vmin
        pad = span * pad_frac if span > 0 else 1.0
        lo = _round_up_nice(abs(vmin - pad))
        hi = _round_up_nice(abs(vmax + pad))
        return (-lo if vmin < 0 else 0, hi)
# =============================================================================

# =============================================================================


def _nice_symmetric_ticks(panel_lims, max_ticks=5):
    """
    Choose nice symmetric ticks like:
      [-3,-2,-1,0,1,2,3]
      [-10,-5,0,5,10]
      [-2,0,2]
    depending on range.

    Returns a compact set of ticks for plotting.
    """
    lo, hi = panel_lims
    vmax = max(abs(lo), abs(hi))

    if vmax <= 0:
        return np.array([0])

    # candidate step sizes
    candidates = [0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100]

    step = candidates[-1]
    for c in candidates:
        n = int(np.floor(vmax / c))
        ticks = np.arange(-n * c, n * c + 0.5 * c, c)
        ticks = ticks[np.abs(ticks) <= vmax + 1e-9]
        if len(ticks) <= max_ticks and len(ticks) >= 3:
            step = c
            break

    n = int(np.floor(vmax / step))
    ticks = np.arange(-n * step, n * step + 0.5 * step, step)

    # force inclusion of exact endpoints when clean
    if len(ticks) < 3:
        ticks = np.array([-vmax, 0, vmax])

    return ticks

# =============================================================================

# =============================================================================

def regression_stats(x, y):
    """
    Basic regression stats.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "cc": np.nan,
            "n": len(x),
        }

    slope, intercept = np.polyfit(x, y, 1)
    cc = np.corrcoef(x, y)[0, 1]

    return {
        "slope": slope,
        "intercept": intercept,
        "cc": cc,
        "n": len(x),
    }
# =============================================================================
# PLOT HELPER 1. MONTHLY CLIMATOLOGY
# =============================================================================


def plot_monthly_climatology(
    clim_df,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP v3.3"),
    product_styles=None,
    figsize=(10, 9),
    ylabel="mm/month",
    y_nbins=4,
    legend_ncol=3,
):
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, axes = plt.subplots(len(region_order), 1, figsize=figsize, sharex=True)

    if len(region_order) == 1:
        axes = [axes]

    for ax, region in zip(axes, region_order):
        sub = clim_df[clim_df["region"] == region]

        for prod in product_order:
            ss = sub[sub["product"] == prod].sort_values("month")
            if ss.empty:
                continue

            style = {} if product_styles is None else product_styles.get(prod, {}).copy()
            ax.plot(
                ss["month"],
                ss["precipitation"],
                label=prod,
                **style
            )

        ax.set_title(region, fontweight="bold", fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=y_nbins))

    # shared y label
    fig.supylabel(ylabel, x=0.08, fontweight="bold", fontsize=18)

    # legend outside bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.001),
        ncol=legend_ncol,
        fontsize=15,
        frameon=False
    )

    # axes[-1].set_xlabel("Month", fontweight="bold")
    plt.tight_layout(rect=[0.05, 0.06, 1, 1])
    return fig, axes


# =============================================================================
# PLOT HELPER 2. SEASONAL CLIMATOLOGY
# =============================================================================

def plot_seasonal_climatology(
    clim_df,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP v3.3"),
    product_styles=None,
    figsize=(10, 8),
    ylabel="mm/season",
    y_nbins=4,
    legend_ncol=3,
):
    season_labels = ["DJF", "MAM", "JJA", "SON"]

    fig, axes = plt.subplots(len(region_order), 1, figsize=figsize, sharex=True)

    if len(region_order) == 1:
        axes = [axes]

    for ax, region in zip(axes, region_order):
        sub = clim_df[clim_df["region"] == region]

        for prod in product_order:
            ss = sub[sub["product"] == prod].copy()
            if ss.empty:
                continue

            ss["season"] = pd.Categorical(ss["season"], categories=season_labels, ordered=True)
            ss = ss.sort_values("season")

            style = {} if product_styles is None else product_styles.get(prod, {}).copy()
            ax.plot(
                ss["season"],
                ss["precipitation"],
                label=prod,
                **style
            )

        ax.set_title(region, fontweight="bold", fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=y_nbins))

    fig.supylabel(ylabel, x=0.06, fontweight="bold", fontsize=18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.001),
        ncol=legend_ncol,
        fontsize=15,
        frameon=False
    )

    # axes[-1].set_xlabel("Season", fontweight="bold")
    plt.tight_layout(rect=[0.05, 0.06, 1, 1])
    return fig, axes

# =============================================================================
# PLOT HELPER 3. INTERANNUAL VARIABILITY
# =============================================================================

def plot_interannual_variability(
    annual_df,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    product_order=(r"$P_{\mathrm{MB}}$", "ERA5", "GPCP v3.3"),
    product_styles=None,
    figsize=(10, 9),
    ylabel="mm/year",
    y_nbins=4,
    legend_ncol=3,
):
    fig, axes = plt.subplots(len(region_order), 1, figsize=figsize, sharex=True)

    if len(region_order) == 1:
        axes = [axes]

    for ax, region in zip(axes, region_order):
        sub = annual_df[annual_df["region"] == region]

        for prod in product_order:
            ss = sub[sub["product"] == prod].sort_values("year")
            if ss.empty:
                continue

            style = {} if product_styles is None else product_styles.get(prod, {}).copy()
            ax.plot(
                ss["year"],
                ss["precipitation"],
                label=prod,
                **style
            )

        ax.set_title(region, fontweight="bold", fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=y_nbins))

    fig.supylabel(ylabel, x=0.06, fontweight="bold", fontsize=18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.001),
        ncol=legend_ncol,
        fontsize=15,
        frameon=False
    )

    # axes[-1].set_xlabel("Year", fontweight="bold")
    plt.tight_layout(rect=[0.05, 0.06, 1, 1])
    return fig, axes


# =============================================================================
# SECTION 7. PLOT REGIONAL SEASONAL ANOMALY TIME SERIES (3x1)
# =============================================================================

def plot_seasonal_anomaly_timeseries_regions_3x1(
    ts_seasonal_anom_dict,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    ref_col=r"$P_{\mathrm{MB}}$",
    target_cols=("ERA5", "GPCP v3.3"),
    product_styles=None,
    figsize=(10, 9),
    y_nbins=4,
    ylabel="mm/season",
    legend_ncol=3,
):
    plot_cols = (ref_col,) + tuple(target_cols)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for ax, region in zip(axes, region_order):
        df = ts_seasonal_anom_dict[region].copy()

        for col in plot_cols:
            if col not in df.columns:
                continue

            style = {} if product_styles is None else product_styles.get(col, {}).copy()
            ax.plot(
                df.index,
                df[col],
                label=col,
                **style
            )

        ax.axhline(0, color="gray", lw=1.0, alpha=0.8)
        ax.set_title(region, fontweight="bold", fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=y_nbins))

    fig.supylabel(ylabel, x=0.06, fontweight="bold", fontsize=18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.001),
        ncol=legend_ncol,
        fontsize=15,
        frameon=False
    )

    axes[-1].set_xlabel("Time", fontweight="bold")
    plt.tight_layout(rect=[0.05, 0.06, 1, 1])

    return fig, axes

# =============================================================================
# SEASONAL ANOMALY SCATTER PLOT (OLD STYLE, UPDATED)
# =============================================================================

def plot_seasonal_anomaly_scatter_regions_3xN(
    region_anom_dict,
    region_order=("Antarctica", "West Antarctica", "East Antarctica"),
    ref_col=r"$P_{\mathrm{MB}}$",
    target_cols=("ERA5", "GPCP v3.3"),
    panel_titles=None,
    figsize=(10.8, 11.5),
    share_lims=False,
    lims=None,
    equal_axes=True,
    point_size=60,
    point_alpha=0.90,
    add_regression=True,
    add_one_to_one=True,
    add_zero_lines=True,
    show_stats_text=True,
):
    """
    Scatter plot:
      rows = regions
      cols = products

    lims can be:
      - None
      - tuple/list like (-3, 3) applied to all panels
      - dict keyed by region_name, e.g.
            {
              "Antarctica": (-3, 3),
              "West Antarctica": (-10, 10),
              "East Antarctica": (-3, 3),
            }
    """

    target_cols = list(target_cols)
    region_order = list(region_order)

    if panel_titles is None:
        panel_titles = {c: c for c in target_cols}

    # -------------------------------------------------------------------------
    # global limits if requested
    # -------------------------------------------------------------------------
    global_lims = None
    if share_lims and lims is None:
        vals = []
        for region_name in region_order:
            df = region_anom_dict[region_name]
            use_cols = [ref_col] + [c for c in target_cols if c in df.columns]
            arr = df[use_cols].to_numpy(dtype=float).ravel()
            vals.append(arr)

        vals = np.concatenate(vals)
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            global_lims = (-1, 1)
        else:
            vmax = np.nanmax(np.abs(vals))
            vmax = 1.0 if (not np.isfinite(vmax) or vmax == 0) else vmax
            vmax = _round_up_nice(vmax * 1.08)
            global_lims = (-vmax, vmax)

    nrows = len(region_order)
    ncols = len(target_cols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    stats_out = {}

    for i, region_name in enumerate(region_order):
        df = region_anom_dict[region_name].copy()
        stats_out[region_name] = {}

        for j, targ in enumerate(target_cols):
            ax = axes[i, j]

            if ref_col not in df.columns or targ not in df.columns:
                ax.axis("off")
                continue

            sub = df[[ref_col, targ]].dropna().copy()
            x = sub[ref_col].values.astype(float)
            y = sub[targ].values.astype(float)

            st = regression_stats(x, y)

            # residual std
            if np.isfinite(st["slope"]) and len(x) >= 2:
                yhat = st["intercept"] + st["slope"] * x
                resid_std = np.std(y - yhat, ddof=1) if len(x) > 2 else np.nan
            else:
                resid_std = np.nan

            st["std"] = resid_std
            stats_out[region_name][targ] = st

            # -----------------------------------------------------------------
            # choose limits
            # -----------------------------------------------------------------
            if share_lims:
                panel_lims = global_lims if global_lims is not None else lims
            else:
                if isinstance(lims, dict):
                    panel_lims = lims.get(region_name, None)
                elif isinstance(lims, (tuple, list, np.ndarray)) and len(lims) == 2:
                    panel_lims = tuple(lims)
                else:
                    panel_lims = None

                if panel_lims is None:
                    panel_lims = _get_axis_limits(x, y, pad_frac=0.08, symmetric=True)

            # scatter
            ax.scatter(
                x, y,
                s=point_size,
                alpha=point_alpha,
                color="k",
                edgecolor="b",
                linewidth=0.5,
                zorder=3
            )

            # one-to-one line
            if add_one_to_one:
                ax.plot(panel_lims, panel_lims, color="k", lw=1.0, ls="--", zorder=2)

            # regression line
            if add_regression and np.isfinite(st["slope"]):
                xx = np.linspace(panel_lims[0], panel_lims[1], 100)
                yy = st["intercept"] + st["slope"] * xx
                ax.plot(xx, yy, color="red", lw=1.4, zorder=4)

            # zero lines
            if add_zero_lines:
                ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.8, zorder=1)
                ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.8, zorder=1)

            ax.set_xlim(panel_lims)
            ax.set_ylim(panel_lims)

            ticks = _nice_symmetric_ticks(panel_lims, max_ticks=5)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            if equal_axes:
                ax.set_aspect("equal", adjustable="box")

            ax.grid(True, alpha=0.25)

            # titles
            if i == 0:
                ax.set_title(panel_titles.get(targ, targ), fontsize=16, fontweight="bold", pad=8)

            # y labels
            if j == 0:
                ax.set_ylabel(f"{region_name}\n{targ} anomaly", fontsize=13, fontweight="bold")
            else:
                ax.set_ylabel(f"{targ} anomaly", fontsize=12, fontweight="bold")

            # x labels
            if i == nrows - 1:
                ax.set_xlabel(f"{ref_col} anomaly", fontsize=13, fontweight="bold")

            # stats text
            if show_stats_text:
                ax.text(
                    0.04, 0.96,
                    f"Slope={st['slope']:.2f}\nCC={st['cc']:.2f}",
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=11.5,
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
                )

    plt.tight_layout()
    return fig, axes, stats_out


# =============================================================================
# BASIN MULTI-YEAR MEAN ANNUAL SCATTER
# =============================================================================

def plot_basin_mean_scatter(
    df,
    ref_col=r"$P_{\mathrm{MB}}$",
    target_cols=("ERA5", "GPCP v3.3"),
    basin_col="basin",
    figsize=(10, 4.8),
    point_size=55,
    log_scale=True,
    log_min=5,
):
    ncols = len(target_cols)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)
    axes = axes[0]

    stats_out = {}

    for ax, targ in zip(axes, target_cols):
        sub = df[[basin_col, ref_col, targ]].dropna().copy()

        x = sub[ref_col].values.astype(float)
        y = sub[targ].values.astype(float)
        basin_ids = sub[basin_col].values.astype(int)

        # scatter
        ax.scatter(
            x, y,
            s=point_size,
            color="plum",
            edgecolor="k",
            alpha=0.9,
            zorder=3
        )

        # label basins
        for xi, yi, bid in zip(x, y, basin_ids):
            ax.text(xi, yi, str(bid), fontsize=9, ha="left", va="bottom")

        # regression
        st = regression_stats(x, y)
        stats_out[targ] = st

        if np.isfinite(st["slope"]):
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            yy = st["intercept"] + st["slope"] * xx
            ax.plot(xx, yy, color="red", lw=1.5, zorder=4)

        # one-to-one
        lo = min(np.nanmin(x), np.nanmin(y))
        hi = max(np.nanmax(x), np.nanmax(y))
        ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1.0, zorder=2)

        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(left=log_min)
            ax.set_ylim(bottom=log_min)

        ax.grid(True, alpha=0.25)
        ax.set_title(targ, fontweight="bold")
        ax.set_xlabel(f"{ref_col} [mm/yr]", fontweight="bold")
        ax.set_ylabel(f"{targ} [mm/yr]", fontweight="bold")

        ax.text(
            0.05, 0.95,
            f"CC={st['cc']:.2f}\nSlope={st['slope']:.2f}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

    plt.tight_layout()
    return fig, axes, stats_out

# =============================================================================
# BASIN-RANKED COMPARISON
# =============================================================================

def plot_basin_ranked_comparison(
    df,
    ref_col=r"$P_{\mathrm{MB}}$",
    target_cols=("ERA5", "GPCP v3.3"),
    basin_col="basin",
    figsize=(12, 5.2),
):
    work = df.copy().sort_values(ref_col).reset_index(drop=True)
    x = np.arange(len(work))

    fig, ax = plt.subplots(figsize=figsize)

    # PMB bars
    ax.bar(
        x,
        work[ref_col].values,
        color="lightgray",
        edgecolor="gray",
        width=0.72,
        label=ref_col,
        zorder=1
    )

    marker_map = {
        "ERA5": ("s", "blue"),
        "GPCP v3.3": ("d", "orange"),
    }

    for targ in target_cols:
        marker, color = marker_map.get(targ, ("o", None))
        ax.scatter(
            x,
            work[targ].values,
            marker=marker,
            color=color,
            s=70,
            zorder=3,
            label=targ,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(work[basin_col].astype(int).astype(str))
    ax.set_ylabel("mm/year", fontweight="bold")
    ax.set_xlabel("Basin", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

    plt.tight_layout()
    return fig, ax, work

def plot_pmb_scatter_oldstyle(
    df_mean_yr_acc,
    ref,
    products,
    high_thresh=500.0,
    scale="log",                 # "linear" or "log"
    log_min=5,
    log_ticks=(5, 10, 20, 50, 100, 200, 500, 1000, 2000),
    ncols=3,
    figsize_per_col=4.6,
    figsize_per_row=4.2,
    share_axes=False,
    show_ylabel_only_left=False,
    global_max=2000.0,
):
    """
    Old-style basin scatter:
      - colored by basin ID
      - basin labels for notable points
      - CC + Bias text
      - 1:1 dashed line
      - log axes with explicit ticks
    """

    pretty = {
        r"$P_{\mathrm{MB}}$": r"$P_{\mathrm{MB}}$",
        "GPCP": "GPCP v3.3",
        "GPCP v3.3": "GPCP v3.3",
        "ERA5": "ERA5",
        "GPM PMW V07": "GPM PMW V07",
        "RACMO": "RACMO v2.4p1",
    }

    n = len(products)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per_col, nrows * figsize_per_row),
        sharex=share_axes, sharey=share_axes
    )
    axes = np.array(axes).ravel()

    stats_out = {}

    for k, (ax, prod) in enumerate(zip(axes, products)):
        yname = pretty.get(prod, prod)

        # valid rows and arrays
        sub = df_mean_yr_acc[[ref, prod, "basin"]].dropna().copy()
        sub["basin"] = pd.to_numeric(sub["basin"], errors="coerce")
        sub = sub[np.isfinite(sub["basin"])].copy()
        sub["basin"] = sub["basin"].astype(int)

        x_all = sub[ref].to_numpy(dtype=float)
        y_all = sub[prod].to_numpy(dtype=float)
        b_all = sub["basin"].to_numpy(dtype=int)

        # log requires strictly positive values
        if scale == "log":
            pos = (x_all > 0) & (y_all > 0)
        else:
            pos = np.isfinite(x_all) & np.isfinite(y_all)

        x = x_all[pos]
        y = y_all[pos]
        b = b_all[pos]

        cols = colors_for_basins(b)

        # scatter
        ax.scatter(
            x, y,
            c=cols, s=110, alpha=0.90,
            edgecolor="k", linewidths=0.6, zorder=2
        )

        # annotate selected basins
        diff = np.abs(x - y)
        mask = (diff >= high_thresh) | (((b >= 13) & (b <= 18)) | (x >= 500))

        for xx, yy, bb in zip(x[mask], y[mask], b[mask]):
            ax.annotate(
                f"{int(bb)}",
                xy=(xx, yy), xycoords="data",
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=12, color="black",
                clip_on=False,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                zorder=4
            )

        # limits, scales, 1:1 line
        if scale == "log":
            lims = (log_min, global_max)
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            lims = (0.0, global_max)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_aspect("equal", adjustable="box")

        # stats inside plotting range
        in_box = (x >= lims[0]) & (x <= lims[1]) & (y >= lims[0]) & (y <= lims[1])

        if np.count_nonzero(in_box) >= 2:
            cc = round(p_corr(x[in_box], y[in_box]), 2)
            bias = round(relative_bias(x[in_box], y[in_box]) * 100, 2)
        else:
            cc, bias = np.nan, np.nan

        stats_out[prod] = {"cc": cc, "bias_percent": bias}

        ax.text(
            0.03, 0.97,
            f"CC={cc:.2f}\nBias={bias:.2f} %",
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=14,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

        # labels
        ref_nme = pretty.get(ref, ref)
        ax.set_title(yname, fontsize=16, fontweight="bold", pad=6)
        ax.set_xlabel(f"{ref_nme} [mm/yr]", fontsize=13, fontweight="bold")

        if (not show_ylabel_only_left) or (k % ncols == 0):
            ax.set_ylabel(f"{yname} [mm/yr]", fontsize=13, fontweight="bold")
        else:
            ax.set_ylabel("")

        ax.tick_params(labelsize=12)

        # ticks & grid
        if scale == "log":
            ax.set_xticks(log_ticks)
            ax.set_yticks(log_ticks)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        else:
            step = 500.0
            ax.set_xticks(np.arange(0, global_max + step, step))
            ax.set_yticks(np.arange(0, global_max + step, step))
            ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6)

    for ax in axes[len(products):]:
        ax.axis("off")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.03, hspace=0.2)
    return fig, axes, stats_out


# -----------------------------------------------------------------------------
# Basin-ranked plot with PMB bars + product points + dual spread annotations
# Adapted to the new cosine-weighted basin dataframe
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_basin_spread_points_dual(
    df,
    basin_col="basin",
    ref_col=r"$P_{\mathrm{MB}}$",          # new PMB column name
    prod_cols=None,                        # products to plot as points (must NOT include ref_col)
    prod_labels=None,                      # optional labels: list aligned with prod_cols or dict col->label
    product_styles=None,                   # your product_styles / product_styles_corr
    non_gpm_group=None,                    # spread group 1 (must include ref_col)
    gpm_group=None,                        # spread group 2 (must include ref_col)
    figsize=(13, 5.2),
    log_scale=True,
    ylim=(10, 2000),
    pmb_bar_color="lightgray",
    pmb_edge_color="black",
    annotate_non_gpm_color="black",
    annotate_gpm_color="dimgray",
    annotate_fontsize=10,
    legend_ncol=4,
    place_key=True,
    key_loc=(0.02, 0.98),
    key_fontsize=10,
):
    """
    Basin plot with:
      - P_MB as bars (ref_col)
      - product points for prod_cols using product_styles
      - TWO spread annotations per basin:
          S_nonGPM = spread among non_gpm_group (includes ref_col)
          S_GPM    = spread among gpm_group     (includes ref_col)

    Spread definition:
        S = (max(P_i) - min(P_i)) / mean(P_i) * 100%
    """

    # -------------------------------------------------------------------------
    # defaults / safety
    # -------------------------------------------------------------------------
    if product_styles is None:
        product_styles = {}

    if prod_cols is None:
        prod_cols = [
            "ERA5",
            "GPCP v3.3",
            "GPM PMW V07",          # placeholder for later
            "GPM PMW V07 (corr.)",  # placeholder for later
        ]

    # never plot ref_col as marker
    prod_cols = [c for c in prod_cols if c != ref_col]

    # labels mapping
    if prod_labels is None:
        label_map = {c: c for c in prod_cols}
    elif isinstance(prod_labels, dict):
        label_map = {c: prod_labels.get(c, c) for c in prod_cols}
    else:
        if len(prod_labels) != len(prod_cols):
            raise ValueError("prod_labels must have same length as prod_cols (or be a dict).")
        label_map = {c: lab for c, lab in zip(prod_cols, prod_labels)}

    # spread groups
    if non_gpm_group is None:
        non_gpm_group = [ref_col, "ERA5", "GPCP v3.3"]

    if gpm_group is None:
        # keep placeholder structure now; when GPM arrives it will drop in naturally
        gpm_group = [ref_col, "GPM PMW V07"]

    if ref_col not in non_gpm_group:
        non_gpm_group = [ref_col] + list(non_gpm_group)
    if ref_col not in gpm_group:
        gpm_group = [ref_col] + list(gpm_group)

    # -------------------------------------------------------------------------
    # prep df
    # -------------------------------------------------------------------------
    df_plot = df.copy()
    df_plot[basin_col] = pd.to_numeric(df_plot[basin_col], errors="coerce")
    df_plot = df_plot[np.isfinite(df_plot[basin_col])].copy()
    df_plot[basin_col] = df_plot[basin_col].astype(int)

    df_plot = df_plot.dropna(subset=[ref_col])
    df_plot = df_plot.sort_values(basin_col)

    basins = df_plot[basin_col].values
    x = np.arange(len(basins))

    # one row per basin expected
    df_plot = df_plot.set_index(basin_col).loc[basins]

    # -------------------------------------------------------------------------
    # figure
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # PMB bars
    ax.bar(
        x,
        df_plot[ref_col].values.astype(float),
        color=pmb_bar_color,
        edgecolor=pmb_edge_color,
        linewidth=1.0,
        label=r"$P_{\mathrm{MB}}$",
        zorder=1,
    )

    # -------------------------------------------------------------------------
    # points for products
    # -------------------------------------------------------------------------
    fallback_markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", ">", "<"]
    fallback_sizes   = [8, 8, 8, 8, 8, 8, 8, 9, 8, 8, 8]

    for i, col in enumerate(prod_cols):
        if col not in df_plot.columns:
            continue

        y = df_plot[col].values.astype(float)
        mask = np.isfinite(y)

        st = product_styles.get(col, {})
        color  = st.get("color", None)
        marker = st.get("marker", fallback_markers[i % len(fallback_markers)])
        ms     = st.get("markersize", fallback_sizes[i % len(fallback_sizes)])

        # optional hollow ERA5 styling
        is_hollow = (col == "ERA5")
        mfc = "white" if is_hollow else (color if color is not None else None)

        ax.plot(
            x[mask],
            y[mask],
            linestyle="None",
            marker=marker,
            markersize=ms,
            color=color,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.6,
            label=label_map.get(col, col),
            zorder=4,
        )

    # -------------------------------------------------------------------------
    # y-scale
    # -------------------------------------------------------------------------
    if log_scale:
        bottom, top = ylim
        top = top * 1.35   # extra headroom for stacked annotations
        ax.set_yscale("log")
        ax.set_ylim(bottom, top)

        log_ticks = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
        ax.set_yticks([t for t in log_ticks if bottom <= t <= ylim[1]])
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())

    ax.tick_params(axis="y", labelsize=12)

    # -------------------------------------------------------------------------
    # spread helpers
    # -------------------------------------------------------------------------
    def spread_pct_for_group(df_local, cols):
        arrs = []
        for c in cols:
            if c in df_local.columns:
                arrs.append(df_local[c].values.astype(float))

        if len(arrs) < 2:
            return np.full(len(df_local), np.nan)

        vals  = np.vstack(arrs)
        vmin  = np.nanmin(vals, axis=0)
        vmax  = np.nanmax(vals, axis=0)
        vmean = np.nanmean(vals, axis=0)

        out = np.full_like(vmean, np.nan, dtype=float)
        ok = np.isfinite(vmin) & np.isfinite(vmax) & np.isfinite(vmean) & (vmean != 0)
        out[ok] = (vmax[ok] - vmin[ok]) / vmean[ok] * 100.0
        return out

    spread_non_gpm = spread_pct_for_group(df_plot, non_gpm_group)
    spread_gpm     = spread_pct_for_group(df_plot, gpm_group)

    # top value for annotation placement
    cols_for_top = sorted(set([ref_col] + list(prod_cols) + list(non_gpm_group) + list(gpm_group)))
    top_stack = np.vstack([df_plot[c].values.astype(float) for c in cols_for_top if c in df_plot.columns])
    top_val_all = np.nanmax(top_stack, axis=0)

    # -------------------------------------------------------------------------
    # annotate spreads
    # -------------------------------------------------------------------------
    y_top_axis = ax.get_ylim()[1]
    y_bot_axis = ax.get_ylim()[0]

    for xi, top_val, s_ref, s_gpm in zip(x, top_val_all, spread_non_gpm, spread_gpm):
        if not np.isfinite(top_val) or top_val <= 0:
            continue

        if log_scale:
            y1 = min(top_val * 1.45, y_top_axis * 0.985)  # non-GPM
            y2 = min(top_val * 1.18, y_top_axis * 0.93)   # GPM
        else:
            y1 = top_val + 0.09 * (y_top_axis - y_bot_axis)
            y2 = top_val + 0.03 * (y_top_axis - y_bot_axis)

        if np.isfinite(s_ref):
            ax.text(
                xi, y1, f"{int(round(s_ref))}%",
                ha="center", va="bottom",
                fontsize=annotate_fontsize,
                color=annotate_non_gpm_color,
                zorder=6
            )

        if np.isfinite(s_gpm):
            ax.text(
                xi, y2, f"{int(round(s_gpm))}%",
                ha="center", va="bottom",
                fontsize=annotate_fontsize,
                color=annotate_gpm_color,
                zorder=6
            )

    # -------------------------------------------------------------------------
    # cosmetics
    # -------------------------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(basins, ha="center", fontsize=14)
    ax.set_xlabel("Basin", fontsize=16)
    ax.set_ylabel("[mm/year]", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # small spread key
    if place_key:
        ax.text(
            key_loc[0], key_loc[1],
            "% = spread(P_MB, ERA5, GPCP v3.3)",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=key_fontsize,
            color=annotate_non_gpm_color,
            zorder=10,
        )
        ax.text(
            key_loc[0], key_loc[1] - 0.055,
            "% = spread(P_MB, GPM PMW V07)",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=key_fontsize,
            color=annotate_gpm_color,
            zorder=10,
        )

    # legend
    handles, labels = ax.get_legend_handles_labels()

    seen = set()
    new_handles, new_labels = [], []
    for h, lab in zip(handles, labels):
        if lab in seen:
            continue
        seen.add(lab)
        new_handles.append(h)
        new_labels.append(lab)

    ax.legend(
        new_handles, new_labels,
        fontsize=14,
        ncol=legend_ncol,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    return fig, ax, spread_non_gpm, spread_gpm



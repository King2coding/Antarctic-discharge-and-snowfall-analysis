#%%
def run_uahipa_cf_orbit_sample_diagnostic(
    uahipa_root,
    cf_latlon_nc,
    n_sample=1000,
    years=range(2013, 2020),
    group_name="SH",
    precip_var="precipitation",
    era5_var="ERA5_tp",
    seed=42,
    clip_negative=True,
    figsize=(14, 5),
    save_fig=None,
):
    """
    Quick diagnostic for PMB-based UA-HIPA seasonal correction factors.

    This function randomly samples UA-HIPA orbital files, applies the seasonal
    correction factors in two ways:

        (1) WAIS-only correction
        (2) WAIS + EAIS correction

    and compares zonal means against:

        - original UA-HIPA
        - WAIS-corrected UA-HIPA
        - WAIS+EAIS-corrected UA-HIPA
        - ERA5 layer already stored in the UA-HIPA orbital file

    Parameters
    ----------
    uahipa_root : str
        Root directory containing yearly UA-HIPA orbital files, e.g.
        /scratch/omidzandi/AVHRR_retrieved_from_HPC_collocated

    cf_latlon_nc : str
        NetCDF file containing lat-lon correction factors.
        Expected variables:
            cf_DJF, cf_MAM, cf_JJA, cf_SON, region_mask

        Expected dims:
            y, x

        region_mask convention assumed:
            1 = WAIS
            2 = EAIS
            0 or NaN = outside correction domain

    n_sample : int
        Number of orbital files to sample.

    years : iterable
        Years to sample from.

    group_name : str
        NetCDF group name in UA-HIPA orbital file. Usually "SH".

    precip_var : str
        UA-HIPA precipitation variable name.

    era5_var : str
        ERA5 precipitation variable inside the UA-HIPA file.

    seed : int
        Random seed for reproducible sampling.

    clip_negative : bool
        If True, negative precipitation values are masked.

    figsize : tuple
        Figure size.

    save_fig : str or None
        If provided, save the diagnostic figure to this path.

    Returns
    -------
    fig, axes, diagnostic_dict

    diagnostic_dict contains:
        sampled_files
        lat_mean_final
        lon_mean_final
        n_files_used
        n_files_failed
    """

    import os
    import re
    import glob
    import warnings
    import numpy as np
    import pandas as pd
    import xarray as xr
    import matplotlib.pyplot as plt

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def parse_orbit_datetime_from_filename(path):
        """
        Extract approximate orbit start datetime from UA-HIPA filename.

        Example:
            clavrx_NSS.GHRR.M1.D13140.S0635.E0724...

        D13140 = 2013 day-of-year 140
        S0635  = 06:35 UTC
        """
        base = os.path.basename(path)

        m = re.search(r"\.D(?P<yyddd>\d{5})\.S(?P<hhmm>\d{4})", base)
        if m is None:
            return pd.NaT

        yyddd = m.group("yyddd")
        hhmm = m.group("hhmm")

        yy = int(yyddd[:2])
        doy = int(yyddd[2:])
        year = 2000 + yy

        hh = int(hhmm[:2])
        minute = int(hhmm[2:])

        try:
            dt = (
                pd.Timestamp(year=year, month=1, day=1)
                + pd.Timedelta(days=doy - 1)
                + pd.Timedelta(hours=hh, minutes=minute)
            )
        except Exception:
            return pd.NaT

        return dt

    def month_to_season(month):
        if month in [12, 1, 2]:
            return "DJF"
        elif month in [3, 4, 5]:
            return "MAM"
        elif month in [6, 7, 8]:
            return "JJA"
        else:
            return "SON"

    def clean_precip(da):
        da = da.astype("float32")
        da = da.where(np.isfinite(da))

        if clip_negative:
            da = da.where(da >= 0)

        return da

    def build_file_inventory():
        rows = []

        for yy in years:
            year_dir = os.path.join(uahipa_root, str(yy))
            files = sorted(glob.glob(os.path.join(year_dir, "*.nc")))

            for fp in files:
                dt = parse_orbit_datetime_from_filename(fp)

                if pd.isna(dt):
                    continue

                if dt.year not in list(years):
                    continue

                rows.append({
                    "file": fp,
                    "datetime": dt,
                    "year": int(dt.year),
                    "month": int(dt.month),
                    "season": month_to_season(int(dt.month)),
                })

        inv = pd.DataFrame(rows)

        if inv.empty:
            raise RuntimeError("No valid UA-HIPA orbital files found.")

        inv = inv.sort_values("datetime").reset_index(drop=True)
        return inv

    def read_orbit_file(fp):
        """
        Read UA-HIPA precipitation and ERA5 from one SH group.
        """
        ds = xr.open_dataset(fp, group=group_name)

        if precip_var not in ds:
            ds.close()
            raise KeyError(f"{precip_var} not found in {fp}")

        if era5_var not in ds:
            ds.close()
            raise KeyError(f"{era5_var} not found in {fp}")

        uahipa = ds[precip_var].copy()
        era5 = ds[era5_var].copy()

        # Standardize coordinate names if needed
        if "lat" in uahipa.dims and "lon" in uahipa.dims:
            uahipa = uahipa.rename({"lat": "y", "lon": "x"})
        if "lat" in era5.dims and "lon" in era5.dims:
            era5 = era5.rename({"lat": "y", "lon": "x"})

        if "y" not in uahipa.dims or "x" not in uahipa.dims:
            ds.close()
            raise ValueError(f"Unexpected UA-HIPA dims {uahipa.dims} in {fp}")

        if "y" not in era5.dims or "x" not in era5.dims:
            ds.close()
            raise ValueError(f"Unexpected ERA5 dims {era5.dims} in {fp}")

        uahipa = clean_precip(uahipa)
        era5 = clean_precip(era5)

        ds.close()

        return uahipa, era5

    def get_cf_for_orbit(ds_cf, template_da, season):
        """
        Extract seasonal CF and region mask, then align to the orbital grid.

        Since both CF and UA-HIPA orbital files are lat-lon WGS with y/x dims,
        this uses interp_like/reindex_like depending on coordinate equality.
        """
        cf_name = f"cf_{season}"

        if cf_name not in ds_cf:
            raise KeyError(f"{cf_name} not found in correction factor file.")

        if "region_mask" not in ds_cf:
            raise KeyError("region_mask not found in correction factor file.")

        cf = ds_cf[cf_name]
        region_mask = ds_cf["region_mask"]

        # Standardize dims if needed
        rename_map = {}
        if "lat" in cf.dims:
            rename_map["lat"] = "y"
        if "lon" in cf.dims:
            rename_map["lon"] = "x"

        if rename_map:
            cf = cf.rename(rename_map)
            region_mask = region_mask.rename(rename_map)

        # Align to orbital grid.
        # If coordinates match exactly, reindex_like is enough.
        # If not, interpolate CF and nearest-neighbor region mask.
        same_y = (
            "y" in cf.coords and "y" in template_da.coords
            and cf.sizes["y"] == template_da.sizes["y"]
            and np.allclose(cf["y"].values, template_da["y"].values)
        )
        same_x = (
            "x" in cf.coords and "x" in template_da.coords
            and cf.sizes["x"] == template_da.sizes["x"]
            and np.allclose(cf["x"].values, template_da["x"].values)
        )

        if same_y and same_x:
            cf_match = cf.reindex_like(template_da)
            region_match = region_mask.reindex_like(template_da)
        else:
            cf_match = cf.interp(
                y=template_da["y"],
                x=template_da["x"],
                method="nearest",
            )
            region_match = region_mask.interp(
                y=template_da["y"],
                x=template_da["x"],
                method="nearest",
            )

        cf_match = cf_match.astype("float32")
        region_match = region_match.astype("float32")

        return cf_match, region_match

    def apply_cf_versions(uahipa_orig, cf_match, region_match):
        """
        Create WAIS-only and WAIS+EAIS corrected versions.

        region_mask convention:
            1 = WAIS
            2 = EAIS
        """
        # keep original where correction is not applied
        uahipa_wais = uahipa_orig.copy()
        uahipa_both = uahipa_orig.copy()

        wais_mask = region_match == 1
        both_mask = (region_match == 1) | (region_match == 2)

        uahipa_wais = xr.where(
            wais_mask & np.isfinite(cf_match),
            uahipa_orig * cf_match,
            uahipa_orig,
        )

        uahipa_both = xr.where(
            both_mask & np.isfinite(cf_match),
            uahipa_orig * cf_match,
            uahipa_orig,
        )

        uahipa_wais.name = "uahipa_wais_corr"
        uahipa_both.name = "uahipa_wais_eais_corr"

        return uahipa_wais, uahipa_both

    def zonal_means_from_fields(fields):
        """
        Compute zonal means by latitude and longitude for each field.
        """
        lat_means = {}
        lon_means = {}

        for name, da in fields.items():
            da = da.where(np.isfinite(da))

            lat_means[name] = da.mean(dim="x", skipna=True)
            lon_means[name] = da.mean(dim="y", skipna=True)

        return lat_means, lon_means

    def accumulate_mean_dict(accum_dict, new_dict):
        """
        Append DataArrays into dictionary of lists.
        """
        for name, da in new_dict.items():
            if name not in accum_dict:
                accum_dict[name] = []
            accum_dict[name].append(da)

    def finalize_mean_dict(accum_dict, concat_dim="sample"):
        """
        Convert dictionary of DataArray lists to sample-mean DataArrays.
        """
        out = {}

        for name, da_list in accum_dict.items():
            if len(da_list) == 0:
                continue

            stacked = xr.concat(da_list, dim=concat_dim)
            out[name] = stacked.mean(dim=concat_dim, skipna=True)

        return out

    # ---------------------------------------------------------------------
    # Load correction factors
    # ---------------------------------------------------------------------
    ds_cf = xr.open_dataset(cf_latlon_nc)

    # Remove possible encoding attrs that can interfere later if user saves
    for v in ds_cf.data_vars:
        for key in ["_FillValue", "scale_factor", "add_offset"]:
            ds_cf[v].attrs.pop(key, None)

    # ---------------------------------------------------------------------
    # Build and sample inventory
    # ---------------------------------------------------------------------
    inventory = build_file_inventory()

    if n_sample > len(inventory):
        warnings.warn(
            f"Requested n_sample={n_sample}, but only {len(inventory)} files found. "
            f"Using all available files."
        )
        n_sample = len(inventory)

    sampled = (
        inventory
        .sample(n=n_sample, random_state=seed)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print(f"Total available files: {len(inventory)}")
    print(f"Sampled files: {len(sampled)}")
    print(sampled.groupby(["year", "season"]).size().reset_index(name="n"))

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------
    lat_accum = {}
    lon_accum = {}

    n_files_used = 0
    n_files_failed = 0

    for i, row in sampled.iterrows():
        fp = row["file"]
        season = row["season"]

        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(sampled)}")

        try:
            uahipa_orig, era5 = read_orbit_file(fp)

            cf_match, region_match = get_cf_for_orbit(
                ds_cf=ds_cf,
                template_da=uahipa_orig,
                season=season,
            )

            uahipa_wais_corr, uahipa_both_corr = apply_cf_versions(
                uahipa_orig=uahipa_orig,
                cf_match=cf_match,
                region_match=region_match,
            )

            # ------------------------------------------------------------
            # Limit diagnostic to Antarctic land only
            # region_mask convention:
            #   1 = WAIS
            #   2 = EAIS
            # ------------------------------------------------------------
            land_mask = (region_match == 1) | (region_match == 2)

            fields = {
                "UA-HIPA original": uahipa_orig.where(land_mask),
                "UA-HIPA WAIS-corr.": uahipa_wais_corr.where(land_mask),
                "UA-HIPA WAIS+EAIS-corr.": uahipa_both_corr.where(land_mask),
                "ERA5": era5.where(land_mask),
            }

            lat_means, lon_means = zonal_means_from_fields(fields)

            accumulate_mean_dict(lat_accum, lat_means)
            accumulate_mean_dict(lon_accum, lon_means)

            n_files_used += 1

        except Exception as e:
            n_files_failed += 1
            warnings.warn(f"Failed processing {fp}: {e}")
            continue

    if n_files_used == 0:
        raise RuntimeError("No files were successfully processed.")

    lat_mean_final = finalize_mean_dict(lat_accum, concat_dim="sample")
    lon_mean_final = finalize_mean_dict(lon_accum, concat_dim="sample")

    ds_cf.close()

    # ---------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------
    plot_styles = {
    "UA-HIPA original": {
        "color": "gray",
        "lw": 2.0,
        "ls": "-",
        "label": "UA-HIPA original",
    },
    "UA-HIPA WAIS-corr.": {
        "color": "magenta",
        "lw": 2.2,
        "ls": "-",
        "label": "UA-HIPA WAIS-corr.",
    },
    "UA-HIPA WAIS+EAIS-corr.": {
        "color": "darkorange",
        "lw": 2.4,
        "ls": "--",
        "label": "UA-HIPA WAIS+EAIS-corr.",
    },
    "ERA5": {
        "color": "blue",
        "lw": 2.4,
        "ls": "-",
        "label": "ERA5",
    },
    }

    plot_order = [
        "UA-HIPA original",
        "UA-HIPA WAIS-corr.",
        "UA-HIPA WAIS+EAIS-corr.",
        "ERA5",
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ------------------------------------------------------------------
    # Zonal mean by latitude
    # ------------------------------------------------------------------
    ax = axes[0]

    for name in plot_order:
        if name not in lat_mean_final:
            continue

        da = lat_mean_final[name]
        st = plot_styles.get(name, {})

        ax.plot(
            da.values,
            da["y"].values,
            color=st.get("color", None),
            lw=st.get("lw", 2.0),
            ls=st.get("ls", "-"),
            label=st.get("label", name),
        )

    ax.set_title("Land-only zonal mean by latitude", fontsize=18, fontweight="bold")
    ax.set_xlabel("Precipitation rate", fontweight="bold")
    ax.set_ylabel("Latitude", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Zonal mean by longitude
    # ------------------------------------------------------------------
    ax = axes[1]

    for name in plot_order:
        if name not in lon_mean_final:
            continue

        da = lon_mean_final[name]
        st = plot_styles.get(name, {})

        ax.plot(
            da["x"].values,
            da.values,
            color=st.get("color", None),
            lw=st.get("lw", 2.0),
            ls=st.get("ls", "-"),
            label=st.get("label", name),
        )

    ax.set_title("Land-only zonal mean by longitude", fontsize=18, fontweight="bold")
    ax.set_xlabel("Longitude", fontweight="bold")
    ax.set_ylabel("Precipitation rate", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if save_fig is not None:
        fig.savefig(save_fig, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_fig}")

    diagnostic_dict = {
        "sampled_files": sampled,
        "lat_mean_final": lat_mean_final,
        "lon_mean_final": lon_mean_final,
        "n_files_used": n_files_used,
        "n_files_failed": n_files_failed,
    }

    print(f"Files successfully used: {n_files_used}")
    print(f"Files failed: {n_files_failed}")

    return fig, axes, diagnostic_dict


#%%
fig, axes, diag = run_uahipa_cf_orbit_sample_diagnostic(
    uahipa_root="/scratch/omidzandi/AVHRR_retrieved_from_HPC_collocated",
    cf_latlon_nc="/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/out_dfs/uahipa_pmb_seasonal_correction_factors_SH_latlon_0p25deg.nc",
    n_sample=1000,
    years=range(2013, 2020),
    group_name="SH",
    precip_var="precipitation",
    era5_var="ERA5_tp",
    seed=42,
    save_fig="/ra1/pubdat/AVHRR_CloudSat_proj/Antarctic_discharge_analysis/figures/uahipa_cf_orbit_sample_zonal_diagnostic.png",
)
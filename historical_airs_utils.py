"""Small, importable workers for historical AIRS preprocessing.

The functions in this module deliberately perform only file-level work.  They
are kept outside the interactive driver so ``ProcessPoolExecutor`` workers can
import them reliably when the driver is executed through a VS Code/Jupyter
kernel.
"""

import numpy as np
import xarray as xr


def read_airs_daily_antarctic_field(file_path, latitude_limit=-60.0):
    """Read one AIRS daily file and return only its Antarctic precipitation.

    Parameters
    ----------
    file_path : str
        Path to one converted AIRS daily NetCDF file.
    latitude_limit : float, default -60
        Northern boundary of the retained Southern Hemisphere domain.

    Returns
    -------
    numpy.ndarray
        A two-dimensional ``float32`` array ordered as ``(lat, lon)``.

    Notes
    -----
    The NetCDF file is closed before the array is returned.  Restricting each
    result to latitudes south of 60 degrees reduces inter-process transfer and
    prevents the parent kernel from accumulating full global daily fields.
    """
    with xr.open_dataset(file_path, engine="netcdf4", cache=False) as dataset:
        field = (
            dataset["precipitation"]
            .isel(time=0, drop=True)
            .sel(lat=slice(latitude_limit, -90))
            .values
        )

    # An owned array is required because the backing NetCDF file is now closed.
    return np.asarray(field, dtype=np.float32).copy()


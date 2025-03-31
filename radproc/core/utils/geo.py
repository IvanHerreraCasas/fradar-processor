# core/utils/geo.py

import xarray as xr
import logging

# Although xradar is used via accessor, importing helps clarity
import xradar # noqa

logger = logging.getLogger(__name__)

def georeference_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Georeferences the radar dataset using the xradar accessor.

    Assumes the input dataset has the necessary coordinate information
    (like latitude, longitude, range, azimuth, elevation) for xradar
    to perform the georeferencing.

    Args:
        ds: The input xarray Dataset (typically a single scan).

    Returns:
        A new xarray Dataset with added 'x', 'y' coordinates and potentially
        a specified projection CRS attribute. Returns the original dataset
        if georeferencing fails.
    """
    try:
        # Check if already georeferenced (simple check)
        if 'x' in ds.coords and 'y' in ds.coords and 'crs' in ds.attrs:
            logger.debug("Dataset appears to be already georeferenced. Skipping.")
            return ds

        logger.debug("Applying georeferencing using ds.xradar.georeference()")
        ds_geo = ds.xradar.georeference()
        logger.debug("Georeferencing applied successfully.")
        return ds_geo
    except AttributeError as ae:
         # Handle cases where the xradar accessor might not be available or methods fail
         logger.error(f"Failed to georeference dataset. xradar accessor might be missing or data incomplete: {ae}", exc_info=True)
         # Return the original dataset to allow processing to potentially continue
         return ds
    except Exception as e:
        logger.error(f"An unexpected error occurred during georeferencing: {e}", exc_info=True)
        # Return the original dataset on unexpected errors
        return ds
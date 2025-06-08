# core/utils/geo.py
import numpy as np
import xarray as xr
import logging
import pyproj # Import pyproj
from typing import Tuple, Any, Optional
import xradar

from radproc.core.analysis import logger

# Although xradar is used via accessor, importing helps clarity
try:
    import rioxarray # Used to access CRS info stored by xarray-spatial/rioxarray
except ImportError:
    rioxarray = None # Mark as potentially unavailable


logger = logging.getLogger(__name__)

def get_dataset_crs(ds: xr.Dataset) -> Optional[Any]:
    """
    Attempts to extract the Coordinate Reference System (CRS) from a dataset.

    Prioritizes finding a variable holding CRS info in its attributes (like 'crs_wkt', 'spatial_ref').
    Checks common attributes on the dataset itself and uses accessors (rio, xradar) as fallbacks.

    Args:
        ds: The input xarray Dataset.

    Returns:
        A pyproj.CRS object if successfully parsed, otherwise None or potentially raw attributes/string.
    """
    crs = None
    crs_source_description = "Unknown" # Track where CRS info came from

    # --- Priority 1: Check for CRS variable attributes ---
    # Common names for variables holding CRS info in attributes
    potential_crs_var_names = ['crs_wkt', 'spatial_ref', 'crs']
    for var_name in potential_crs_var_names:
        if var_name in ds.variables: # Check data_vars and coords
            logger.debug(f"Found potential CRS variable: '{var_name}'")
            crs_var = ds[var_name]
            if hasattr(crs_var, 'attrs') and crs_var.attrs:
                crs_attrs = crs_var.attrs
                # A) Try parsing WKT directly from 'crs_wkt' attribute if it exists
                if 'crs_wkt' in crs_attrs:
                    try:
                        crs = pyproj.CRS.from_wkt(crs_attrs['crs_wkt'])
                        crs_source_description = f"WKT from '{var_name}.attrs['crs_wkt']'"
                        logger.debug(f"Successfully created pyproj.CRS from {crs_source_description}")
                        return crs # Success
                    except Exception as e:
                        logger.warning(f"Could not parse WKT from '{var_name}.attrs['crs_wkt']': {e}")

                # B) Try creating from CF convention attributes
                if crs is None: # Only if WKT didn't work or wasn't present
                    try:
                        crs = pyproj.CRS.from_cf(crs_attrs)
                        crs_source_description = f"CF attributes from '{var_name}.attrs'"
                        logger.debug(f"Successfully created pyproj.CRS from {crs_source_description}")
                        return crs # Success
                    except Exception as e:
                        logger.warning(f"Could not create pyproj.CRS from '{var_name}' CF attributes: {e}")

                # C) Fallback: If parsing failed but we found attributes, store them
                if crs is None:
                    crs = crs_attrs # Store the raw attributes as a last resort from this variable
                    crs_source_description = f"Raw attributes from '{var_name}.attrs'"
                    logger.debug(f"Storing {crs_source_description} as CRS fallback.")
                    break # Stop checking other variable names if we found attributes here

            else:
                logger.debug(f"Variable '{var_name}' found but has no attributes.")
        # If we successfully parsed a pyproj.CRS, we would have returned already

def georeference_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Georeferences the radar dataset using the xradar accessor.
    Ensures CRS information is attached and squeezes singleton dimensions
    from resulting 'x' and 'y' coordinates.

    Args:
        ds: The input xarray Dataset (typically a single scan).

    Returns:
        A new xarray Dataset with added 2D 'x', 'y' coordinates and CRS attribute.
        Returns the original dataset if georeferencing fails.
    """
    if xradar is None:
         logger.error("xradar library not found. Cannot perform georeferencing.")
         return ds
    try:
        # Check if already georeferenced (including dimensionality check)
        crs_present = get_dataset_crs(ds) is not None
        coords_present = 'x' in ds.coords and 'y' in ds.coords
        dims_correct = False
        if coords_present:
             dims_correct = ds['x'].ndim == 2 and ds['y'].ndim == 2

        if coords_present and crs_present and dims_correct:
            logger.debug("Dataset appears to be already georeferenced with correct dims. Skipping.")
            return ds
        elif coords_present and crs_present and not dims_correct:
             logger.warning("Dataset georeferenced but x/y dims are not 2D. Attempting re-georeference.")

        logger.debug("Applying georeferencing using ds.xradar.georeference()...")
        # --- Perform Georeferencing ---
        ds_geo = ds.xradar.georeference()
        logger.debug("Initial georeferencing complete.")

        # --- Squeeze Singleton Dims from x and y ---
        modified = False
        if 'x' in ds_geo.coords and ds_geo['x'].ndim > 2:
            try:
                original_x_shape = ds_geo['x'].shape
                # Squeeze all singleton dimensions (like time=1 or elevation=1)
                ds_geo['x'] = ds_geo['x'].squeeze()
                if ds_geo['x'].ndim == 2:
                    #logger.info(f"Squeezed 'x' coordinates from {original_x_shape} to {ds_geo['x'].shape}")
                    modified = True
                #elif ds_geo['x'].ndim == original_x_shape.ndim:
                #     logger.debug(f"'x' coordinate shape {original_x_shape} unchanged after squeeze (no singletons).")
                else:
                     logger.warning(f"Squeezing 'x' resulted in unexpected shape {ds_geo['x'].shape} from {original_x_shape}. Check data.")
                     # Revert? Or proceed with caution? Let's proceed but log warning.
            except Exception as squeeze_err:
                 logger.warning(f"Could not squeeze 'x' coordinate: {squeeze_err}")

        if 'y' in ds_geo.coords and ds_geo['y'].ndim > 2:
             try:
                original_y_shape = ds_geo['y'].shape
                ds_geo['y'] = ds_geo['y'].squeeze()
                if ds_geo['y'].ndim == 2:
                   #logger.info(f"Squeezed 'y' coordinates from {original_y_shape} to {ds_geo['y'].shape}")
                    modified = True
                #elif ds_geo['y'].ndim == original_y_shape.ndim:
                #   #logger.debug(f"'y' coordinate shape {original_y_shape} unchanged after squeeze (no singletons).")
                else:
                   logger.warning(f"Squeezing 'y' resulted in unexpected shape {ds_geo['y'].shape} from {original_y_shape}. Check data.")
             except Exception as squeeze_err:
                 logger.warning(f"Could not squeeze 'y' coordinate: {squeeze_err}")
        # --- End Squeeze ---

        # Verify CRS was added or exists after georeferencing
        if get_dataset_crs(ds_geo) is None:
             logger.warning("Georeferencing applied, but CRS information could not be verified afterwards.")
        elif modified:
             logger.debug("Georeferencing and coordinate squeezing applied successfully.")
        else:
             logger.debug("Georeferencing applied successfully (no squeezing needed).")


        # Final check on dimensions before returning
        if 'x' in ds_geo.coords and ds_geo['x'].ndim != 2:
             logger.error(f"Final 'x' coordinate dimension is {ds_geo['x'].ndim}, expected 2.")
        if 'y' in ds_geo.coords and ds_geo['y'].ndim != 2:
             logger.error(f"Final 'y' coordinate dimension is {ds_geo['y'].ndim}, expected 2.")

        return ds_geo
    except AttributeError as ae:
         logger.error(f"Failed to georeference dataset. xradar accessor might be missing or data incomplete: {ae}", exc_info=True)
         return ds
    except Exception as e:
        logger.error(f"An unexpected error occurred during georeferencing: {e}", exc_info=True)
        return ds

def transform_point_to_dataset_crs(target_lat: float, target_lon: float, ds_crs: Any) -> Optional[Tuple[float, float]]:
    """
    Transforms geographic coordinates (Lat/Lon) to the dataset's projected CRS.

    Args:
        target_lat: Target latitude in decimal degrees.
        target_lon: Target longitude in decimal degrees.
        ds_crs: The dataset's target CRS (preferably a pyproj.CRS object,
                but can attempt to handle strings or dicts).

    Returns:
        A tuple (target_x, target_y) in the dataset's CRS, or None if transformation fails.
    """
    source_crs = "EPSG:4326" # Standard WGS84 Geographic CRS

    try:
        # Ensure target_crs is a pyproj.CRS object
        if not isinstance(ds_crs, pyproj.CRS):
            logger.debug(f"Attempting to create pyproj.CRS from provided ds_crs (type: {type(ds_crs)})")
            try:
                if isinstance(ds_crs, dict):
                    target_crs_obj = pyproj.CRS.from_cf(ds_crs)
                elif isinstance(ds_crs, str):
                    target_crs_obj = pyproj.CRS.from_string(ds_crs) # Handles WKT, Proj strings, EPSG codes
                else:
                    logger.error(f"Unsupported ds_crs type for pyproj conversion: {type(ds_crs)}")
                    return None
            except Exception as crs_parse_err:
                logger.error(f"Failed to parse dataset CRS for transformation: {crs_parse_err}", exc_info=True)
                return None
        else:
            target_crs_obj = ds_crs # It's already a pyproj object

        # Create the transformer
        # always_xy=True expects input (lon, lat) and outputs (x, y) order
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs_obj, always_xy=True)

        # Perform the transformation
        target_x, target_y = transformer.transform(target_lon, target_lat)
        logger.debug(f"Transformed ({target_lon=}, {target_lat=}) from {source_crs} to "
                     f"({target_x=:.2f}, {target_y=:.2f}) in dataset CRS.")
        return target_x, target_y

    except ImportError:
        logger.error("pyproj library is required for coordinate transformations but not found.")
        return None
    except Exception as e:
        logger.error(f"Coordinate transformation failed: {e}", exc_info=True)
        return None

def find_nearest_indices(ds_geo: xr.Dataset, target_lat: float, target_lon: float, max_dist_meters: float = 200.0) -> \
Optional[Tuple[int, int]]:
    """
    Finds the azimuth and range indices in the dataset closest to a target lat/lon.
    Transforms the target coordinates, calculates distances in the projected space,
    and finds the minimum distance indices. Checks if the minimum distance
    is within a specified threshold.
    Args:
        ds_geo: Georeferenced xarray.Dataset.
        target_lat: Target latitude (decimal degrees).
        target_lon: Target longitude (decimal degrees).
        max_dist_meters: Maximum allowable distance (meters) between target point
                         and closest radar bin center. If exceeded, returns None.
    Returns:
        A tuple (azimuth_index, range_index) or None.
    """
    logger.debug(f"Finding nearest indices for Lat={target_lat}, Lon={target_lon}")
    ds_crs = get_dataset_crs(ds_geo)
    if ds_crs is None:
        logger.error(f"Cannot find CRS in georeferenced dataset for point ({target_lat}, {target_lon}).")
        return None
    transformed_coords = transform_point_to_dataset_crs(target_lat, target_lon, ds_crs)
    if transformed_coords is None:
        logger.error("Coordinate transformation failed for index finding.")
        return None
    target_x, target_y = transformed_coords
    if not all(coord in ds_geo.coords and ds_geo[coord].ndim == 2 for coord in ['x', 'y']):
        logger.error("Dataset missing 2D 'x' or 'y' coordinates for index finding.")
        return None
    try:
        x_coords = ds_geo['x'].data
        y_coords = ds_geo['y'].data
        dist_sq = (x_coords - target_x) ** 2 + (y_coords - target_y) ** 2
        dist_sq_computed = dist_sq.compute() if hasattr(dist_sq, 'compute') else dist_sq
        if np.all(np.isnan(dist_sq_computed)):
            logger.warning("All calculated distances are NaN. Cannot find minimum for index.")
            return None
        min_flat_index = np.nanargmin(dist_sq_computed)
        az_idx, rg_idx = np.unravel_index(min_flat_index, dist_sq_computed.shape)
        min_dist = np.sqrt(dist_sq_computed[az_idx, rg_idx])
        if max_dist_meters is not None and min_dist > max_dist_meters:
            logger.warning(f"Nearest bin ({min_dist:.1f}m) for point ({target_lat}, {target_lon}) "
                           f"exceeds max_dist_meters ({max_dist_meters}m). No valid indices.")
            return None
        return int(az_idx), int(rg_idx)
    except ValueError:
        logger.warning("Could not find minimum distance (all distances likely NaN).")
        return None
    except Exception as e:
        logger.error(f"Error finding minimum distance index: {e}", exc_info=True)
        return None

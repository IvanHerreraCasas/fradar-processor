# core/analysis.py

import os
import logging
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timezone

# Import necessary functions from other modules
from .config import get_all_points_config, get_setting
from .utils.geo import get_dataset_crs, georeference_dataset, transform_point_to_dataset_crs
# CSV handler import will be mostly for generate_point_timeseries & calculate_accumulation's export/legacy parts
from .utils.csv_handler import read_timeseries_csv, write_timeseries_csv  # Removed unused CSV functions for this step
# data.py imports will be adjusted as get_filepaths_in_range is deprecated
from .data import read_scan  # extract_scan_key_metadata will be used by processor, not directly here
# Import DB Manager
from .db_manager import (
    get_connection, release_connection, get_or_create_variable_id,
    batch_insert_timeseries_data, update_point_cached_indices_in_db
)

logger = logging.getLogger(__name__)


# --- File-based Index Caching Helpers ---
# REMOVED: _get_index_cache_path, load_cached_indices, save_cached_indices
# This functionality is now handled by storing indices in the radproc_points DB table.

# --- Core Analysis Functions ---

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
    logger.debug(f"Transformed target point to dataset CRS: X={target_x:.2f}, Y={target_y:.2f}")

    if not all(coord in ds_geo.coords and ds_geo[coord].ndim == 2 for coord in ['x', 'y']):
        logger.error("Dataset missing 2D 'x' or 'y' coordinates for index finding.")
        return None

    try:
        x_coords = ds_geo['x'].data
        y_coords = ds_geo['y'].data
        dist_sq = (x_coords - target_x) ** 2 + (y_coords - target_y) ** 2
        dist_sq_computed = dist_sq.compute() if hasattr(dist_sq, 'compute') else dist_sq
    except Exception as e:
        logger.error(f"Error calculating distances for index finding: {e}", exc_info=True)
        return None

    try:
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
        logger.debug(f"Minimum distance ({min_dist:.2f}m) found at indices: Az={az_idx}, Ran={rg_idx}.")
        return int(az_idx), int(rg_idx)
    except ValueError:  # nanargmin raises ValueError if all inputs are NaN
        logger.warning("Could not find minimum distance (all distances likely NaN).")
        return None
    except Exception as e:
        logger.error(f"Error finding minimum distance index: {e}", exc_info=True)
        return None


def extract_point_value(ds: xr.Dataset, variable: str, az_idx: int, rg_idx: int) -> float:
    """
    Extracts data value for a variable at given azimuth/range indices. Assumes single time step.
    Returns np.nan on failure.
    """
    try:
        if variable not in ds.data_vars:
            logger.warning(f"Variable '{variable}' not found in dataset for value extraction.")
            return np.nan
        # Assumes ds_geo has already been prepared with a single time and elevation dimension
        data_point = ds[variable].isel(time=0, elevation=0, azimuth=az_idx, range=rg_idx)
        value = data_point.compute().item() if hasattr(data_point.data, 'compute') else data_point.item()

        if isinstance(value, (int, float, np.number)) and not np.isnan(value):
            return float(value)
        logger.debug(f"Invalid or NaN value for '{variable}' at az={az_idx}, rg={rg_idx}: {value}")
        return np.nan
    except IndexError:
        logger.error(f"Indices az={az_idx}, rg={rg_idx} out of bounds for var '{variable}'.")
        return np.nan
    except Exception as e:
        logger.error(f"Failed to extract value for '{variable}' at az={az_idx}, rg={rg_idx}: {e}", exc_info=True)
        return np.nan


# --- Orchestrator Functions ---

def update_timeseries_for_scan(ds_geo: xr.Dataset,
                               scan_precise_timestamp: datetime,
                               scan_elevation: float):
    """
    Updates timeseries data in PostgreSQL for relevant points using data from a single scan.
    Extracts all 'timeseries_default_variables' for points whose 'target_elevation'
    matches the scan's elevation. The scan_precise_timestamp and scan_elevation are
    passed from processor.py after being extracted via extract_scan_key_metadata.

    Args:
        ds_geo: The georeferenced xarray.Dataset for the current scan (single time step).
        scan_precise_timestamp: The precise, timezone-aware UTC timestamp of the scan.
        scan_elevation: The elevation of the scan.
    """
    logger.info(
        f"Starting DB timeseries update for scan at {scan_precise_timestamp.isoformat()}, Elev: {scan_elevation:.2f}°")

    # 1. Get Configuration
    all_points_db_config: List[Dict[str, Any]] = get_all_points_config()  # Reads from DB via config.py
    default_variables_to_extract: List[str] = get_setting('app.timeseries_default_variables', [])

    if not all_points_db_config:
        logger.info("No points defined in the database. Skipping timeseries update.")
        return
    if not default_variables_to_extract:
        logger.info("No 'app.timeseries_default_variables' configured. Skipping timeseries update.")
        return

    conn = None
    try:
        conn = get_connection()
        new_data_for_batch_insert: List[Dict[str, Any]] = []
        points_contributing_data_count = 0
        elevation_tolerance = 0.1  # degrees

        # 2. Loop Through Points defined in DB
        for point_config in all_points_db_config:
            point_id = point_config.get('point_id')
            point_name = point_config.get('point_name')
            target_lat = point_config.get('latitude')
            target_lon = point_config.get('longitude')
            point_target_elevation = point_config.get('target_elevation')

            if not all([point_id is not None, point_name, target_lat is not None,
                        target_lon is not None, point_target_elevation is not None]):
                logger.warning(f"Skipping point due to incomplete DB configuration: {point_config}")
                continue

            # Check if scan elevation matches this point's target elevation
            if abs(scan_elevation - point_target_elevation) > elevation_tolerance:
                logger.debug(
                    f"Point '{point_name}': Scan elev {scan_elevation:.2f} != Target elev {point_target_elevation:.2f}. Skipping.")
                continue

            logger.debug(
                f"Point '{point_name}' (ID: {point_id}) matches scan elevation. Processing default variables...")

            # Get or Calculate Radar Grid Indices for this point
            az_idx = point_config.get('cached_azimuth_index')
            rg_idx = point_config.get('cached_range_index')
            current_indices: Optional[Tuple[int, int]] = None

            if az_idx is not None and rg_idx is not None:  # Check if None, not just falsy (0 is valid)
                current_indices = (int(az_idx), int(rg_idx))
                logger.debug(
                    f"Using cached DB indices for '{point_name}': Az={current_indices[0]}, Rg={current_indices[1]}")
            else:
                logger.debug(f"No cached DB indices for '{point_name}'. Calculating...")
                calculated_indices = find_nearest_indices(ds_geo, target_lat, target_lon)
                if calculated_indices:
                    current_indices = calculated_indices
                    if not update_point_cached_indices_in_db(conn, point_id, current_indices[0], current_indices[1]):
                        logger.warning(
                            f"Failed to save updated indices to DB for point '{point_name}' (ID: {point_id}).")
                    else:
                        logger.info(
                            f"Calculated and saved indices for '{point_name}' (ID: {point_id}): {current_indices}")
                else:
                    logger.warning(
                        f"Could not find valid radar grid indices for point '{point_name}'. Skipping variables for this point in this scan.")
                    continue

            if not current_indices:
                continue

            # Extract all default variables for this point using the determined indices
            point_had_data_added = False
            for var_name_to_extract in default_variables_to_extract:
                variable_id = get_or_create_variable_id(conn, var_name_to_extract)
                if variable_id is None:
                    logger.error(
                        f"Could not get/create DB ID for var '{var_name_to_extract}' for point '{point_name}'. Skipping this variable.")
                    continue

                value = extract_point_value(ds_geo, var_name_to_extract, current_indices[0], current_indices[1])
                if not np.isnan(value):
                    new_data_for_batch_insert.append({
                        'timestamp': scan_precise_timestamp,  # Passed as argument
                        'point_id': point_id,
                        'variable_id': variable_id,
                        'value': value
                    })
                    point_had_data_added = True
                    logger.debug(
                        f"Point '{point_name}', Var '{var_name_to_extract}': Value={value:.2f} prepared for insert.")
                else:
                    logger.debug(f"Point '{point_name}', Var '{var_name_to_extract}': Extracted NaN or invalid value.")

            if point_had_data_added:
                points_contributing_data_count += 1

        # 3. Batch Insert Data into PostgreSQL
        if new_data_for_batch_insert:
            logger.info(
                f"Attempting to batch insert {len(new_data_for_batch_insert)} data points for {points_contributing_data_count} point(s)...")
            if batch_insert_timeseries_data(conn, new_data_for_batch_insert):
                logger.info(f"Successfully inserted {len(new_data_for_batch_insert)} data points.")
            else:
                logger.error("Batch insert failed for timeseries data.")
        else:
            logger.info("No new timeseries data to insert from this scan.")

    except ConnectionError as ce:
        logger.error(f"Database connection error during timeseries update: {ce}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during database timeseries update: {e}", exc_info=True)
    finally:
        if conn:
            release_connection(conn)

    logger.info(f"DB timeseries update finished for scan. Points contributing data: {points_contributing_data_count}")


# --- generate_point_timeseries and calculate_accumulation placeholders ---
def generate_point_timeseries(
        point_names: List[str],
        start_dt: datetime,
        end_dt: datetime,
        specific_variables: Optional[List[str]] = None
) -> bool:
    logger.warning("generate_point_timeseries: DB-aware implementation is PENDING. Current call is a placeholder.")
    if not point_names: return False
    logger.info(f"[Placeholder] generate_point_timeseries for points '{', '.join(point_names)}' "
                f"from {start_dt.isoformat()} to {end_dt.isoformat()}. "
                f"Vars: {specific_variables or get_setting('app.timeseries_default_variables', [])}")
    return True


def calculate_accumulation(
        point_name: str,
        start_dt: datetime,
        end_dt: datetime,
        interval: str,
        rate_variable: str,
        output_file_path: str
) -> bool:
    logger.warning("calculate_accumulation: DB-aware implementation is PENDING. Current call is a placeholder.")
    logger.info(f"[Placeholder] calculate_accumulation for point '{point_name}' to '{output_file_path}'.")
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as f:
            f.write("timestamp,value\n")  # Dummy CSV header
        return True
    except Exception as e:
        logger.error(f"Placeholder calculate_accumulation failed to write dummy file: {e}")
        return False
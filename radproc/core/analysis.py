# core/analysis.py

import os
import logging
from typing import Optional, Tuple, Dict, List, Any, Set
from collections import defaultdict
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timezone

# Import necessary functions from other modules
from .config import get_all_points_config, get_setting, get_point_config  # get_point_config is DB-aware
from .utils.geo import get_dataset_crs, georeference_dataset, transform_point_to_dataset_crs
# CSV handler only for calculate_accumulation's output, and potentially migration (not used here)
from .utils.csv_handler import write_timeseries_csv
from .data import read_scan  # extract_scan_key_metadata is used by processor
# Import DB Manager
from .db_manager import (
    get_connection, release_connection, get_or_create_variable_id,
    batch_insert_timeseries_data, update_point_cached_indices_in_db,
    get_existing_timestamps_for_multiple_points,  # Key for this function
    query_scan_log_for_timeseries_processing  # Key for this function
)

logger = logging.getLogger(__name__)


# --- Core Analysis Functions (find_nearest_indices, extract_point_value) ---
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
        f"Starting DB timeseries update for current scan at {scan_precise_timestamp.isoformat()}, Elev: {scan_elevation:.2f}°")
    all_points_db_config: List[Dict[str, Any]] = get_all_points_config()
    default_variables_to_extract: List[str] = get_setting('app.timeseries_default_variables', [])
    if not all_points_db_config or not default_variables_to_extract:
        logger.info("No points defined or no default variables configured. Skipping timeseries update.")
        return
    conn = None
    try:
        conn = get_connection()
        new_data_for_batch_insert: List[Dict[str, Any]] = []
        points_contributing_data_count = 0
        elevation_tolerance = 0.1
        for point_config in all_points_db_config:
            point_id = point_config.get('point_id');
            point_name = point_config.get('point_name')
            target_lat = point_config.get('latitude');
            target_lon = point_config.get('longitude')
            point_target_elevation = point_config.get('target_elevation')
            if not all([point_id is not None, point_name, target_lat is not None, target_lon is not None,
                        point_target_elevation is not None]):
                logger.warning(f"Skipping point due to incomplete DB config: {point_config}");
                continue
            if abs(scan_elevation - point_target_elevation) > elevation_tolerance: continue
            az_idx = point_config.get('cached_azimuth_index');
            rg_idx = point_config.get('cached_range_index')
            current_indices: Optional[Tuple[int, int]] = (int(az_idx),
                                                          int(rg_idx)) if az_idx is not None and rg_idx is not None else None
            if current_indices is None:
                calculated_indices = find_nearest_indices(ds_geo, target_lat, target_lon)
                if calculated_indices:
                    current_indices = calculated_indices
                    update_point_cached_indices_in_db(conn, point_id, current_indices[0], current_indices[1])
                else:
                    logger.warning(f"Could not find indices for point '{point_name}'."); continue
            if not current_indices: continue
            point_had_data_added = False
            for var_name_to_extract in default_variables_to_extract:
                variable_id = get_or_create_variable_id(conn, var_name_to_extract)
                if variable_id is None: continue
                value = extract_point_value(ds_geo, var_name_to_extract, current_indices[0], current_indices[1])
                if not np.isnan(value):
                    new_data_for_batch_insert.append(
                        {'timestamp': scan_precise_timestamp, 'point_id': point_id, 'variable_id': variable_id,
                         'value': value})
                    point_had_data_added = True
            if point_had_data_added: points_contributing_data_count += 1
        if new_data_for_batch_insert:
            if batch_insert_timeseries_data(conn, new_data_for_batch_insert):
                logger.info(f"Inserted {len(new_data_for_batch_insert)} data points.")
            else:
                logger.error("Batch insert failed.")
        else:
            logger.info("No new timeseries data to insert.")
    except ConnectionError as ce:
        logger.error(f"DB connection error: {ce}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        if conn: release_connection(conn)
    logger.info(f"DB timeseries update finished. Points contributing: {points_contributing_data_count}")


def generate_point_timeseries(
        point_names: List[str],
        start_dt: datetime,  # Expected to be timezone-aware UTC
        end_dt: datetime,  # Expected to be timezone-aware UTC
        specific_variables: Optional[List[str]] = None
) -> bool:
    """
    Generates/updates timeseries data in PostgreSQL for a list of points.
    Uses radproc_scan_log to find relevant scans and processes only data not
    already present in timeseries_data.
    """
    if not point_names:
        logger.warning("generate_point_timeseries called with no point names.")
        return True  # Nothing to do, considered success

    logger.info(f"Starting historical timeseries generation for points: {', '.join(point_names)}")
    logger.info(f"Time range: {start_dt.isoformat()} to {end_dt.isoformat()}")

    # 1. Determine Variables to Process
    variables_to_process: List[str]
    if specific_variables and len(specific_variables) > 0:
        variables_to_process = specific_variables
        logger.info(f"Using specified variables: {variables_to_process}")
    else:
        variables_to_process = get_setting('app.timeseries_default_variables', [])
        if not variables_to_process:
            logger.error("No specific_variables provided and 'app.timeseries_default_variables' "
                         "is not configured or empty. Cannot proceed.")
            return False
        logger.info(f"Using app default variables: {variables_to_process}")

    conn = None
    global_new_data_to_insert: List[Dict[str, Any]] = []
    total_scans_opened_and_processed = 0

    try:
        conn = get_connection()

        # 2. Resolve Variable Names to IDs
        variable_name_to_id_map: Dict[str, Optional[int]] = {}
        for var_name in variables_to_process:
            variable_name_to_id_map[var_name] = get_or_create_variable_id(conn, var_name)
            if variable_name_to_id_map[var_name] is None:
                logger.error(f"Failed to resolve ID for variable '{var_name}'. It will be skipped.")

        active_variables_map = {name: id for name, id in variable_name_to_id_map.items() if id is not None}
        if not active_variables_map:
            logger.error("No valid variable IDs could be resolved. Cannot proceed.")
            return False
        logger.info(f"Processing with resolved variable IDs: {active_variables_map}")

        # 3. Get Point Configurations and Group by Target Elevation
        all_db_points_map = {p['point_name']: p for p in get_all_points_config()}  # Fetch all points once

        points_by_elevation: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        point_details_for_processing: Dict[int, Dict[str, Any]] = {}  # Store point_id -> full_config

        for name in point_names:
            point_config = all_db_points_map.get(name)
            if not point_config or point_config.get('target_elevation') is None or point_config.get('point_id') is None:
                logger.warning(f"Point '{name}' misconfigured or not found in DB. Skipping.")
                continue
            points_by_elevation[float(point_config['target_elevation'])].append(point_config)
            point_details_for_processing[point_config['point_id']] = point_config

        if not points_by_elevation:
            logger.warning("No valid points found for processing after configuration check.")
            return True  # Successfully did nothing

        # 4. Main Processing Loop: Per Elevation Group
        for target_elevation, points_in_group in points_by_elevation.items():
            logger.info(f"Processing elevation group: {target_elevation:.2f}° "
                        f"(Points: {[p['point_name'] for p in points_in_group]})")

            # 4a. Get candidate scans from radproc_scan_log for this elevation and time range
            candidate_scans_from_log: List[Tuple[str, datetime, int]] = \
                query_scan_log_for_timeseries_processing(conn, target_elevation, start_dt, end_dt)

            if not candidate_scans_from_log:
                logger.info(f"No scans found in radproc_scan_log for elev {target_elevation:.2f}° in range.")
                continue
            logger.info(
                f"Found {len(candidate_scans_from_log)} candidate scans in log for elev {target_elevation:.2f}°.")

            # 4b. Fetch all existing timestamps from timeseries_data for all points/variables in this group
            point_var_pairs_for_query: List[Tuple[int, int]] = []
            for p_conf in points_in_group:
                for var_id in active_variables_map.values():
                    point_var_pairs_for_query.append((p_conf['point_id'], var_id))

            existing_db_timestamps_map: Dict[Tuple[int, int], Set[datetime]] = \
                get_existing_timestamps_for_multiple_points(conn, point_var_pairs_for_query, start_dt, end_dt)

            # 4c. Inner Loop: Iterate through candidate scan files (from log)
            opened_scans_cache: Dict[str, xr.Dataset] = {}  # Cache opened ds_geo objects per scan_filepath

            for scan_filepath, precise_scan_dt, _scan_log_id in candidate_scans_from_log:
                scan_needs_opening = False
                # Check if this scan's data is needed for *any* point/variable in the current group
                for point_config in points_in_group:
                    point_id = point_config['point_id']
                    for var_id in active_variables_map.values():
                        if precise_scan_dt not in existing_db_timestamps_map.get((point_id, var_id), set()):
                            scan_needs_opening = True
                            break
                    if scan_needs_opening: break

                if not scan_needs_opening:
                    logger.debug(f"Scan {os.path.basename(scan_filepath)} (Time: {precise_scan_dt.isoformat()}) "
                                 "data already exists for all relevant point/variable combinations. Skipping full read.")
                    continue

                # Open scan if needed (and not already opened in this iteration)
                ds_geo = None
                if scan_filepath in opened_scans_cache:
                    ds_geo = opened_scans_cache[scan_filepath]
                else:
                    logger.info(f"Opening and processing scan: {scan_filepath}")
                    ds_raw = read_scan(scan_filepath, variables=list(active_variables_map.keys()))
                    if ds_raw is None: continue
                    total_scans_opened_and_processed += 1

                    # Verify elevation from data
                    scan_elev_from_data = float(ds_raw['elevation'].item())
                    if abs(scan_elev_from_data - target_elevation) > 0.1:  # elevation_tolerance
                        logger.warning(f"Scan {scan_filepath} actual elev {scan_elev_from_data:.2f} "
                                       f"mismatches group target {target_elevation:.2f}. Skipping scan.")
                        ds_raw.close()
                        continue

                    ds_geo = georeference_dataset(ds_raw)
                    ds_raw.close()  # Close raw after georef
                    if not ('x' in ds_geo.coords and 'y' in ds_geo.coords):
                        logger.warning(f"Failed to georeference {scan_filepath}. Skipping.")
                        ds_geo.close()
                        continue
                    opened_scans_cache[scan_filepath] = ds_geo

                # Process each point in the current elevation group against this opened ds_geo
                for point_config in points_in_group:
                    point_id = point_config['point_id']
                    point_name = point_config['point_name']

                    current_indices: Optional[Tuple[int, int]]
                    cached_az_idx = point_config.get('cached_azimuth_index')
                    cached_rg_idx = point_config.get('cached_range_index')

                    if cached_az_idx is not None and cached_rg_idx is not None:
                        current_indices = (int(cached_az_idx), int(cached_rg_idx))
                    else:
                        calculated_indices = find_nearest_indices(ds_geo, point_config['latitude'],
                                                                  point_config['longitude'])
                        if calculated_indices:
                            current_indices = calculated_indices
                            update_point_cached_indices_in_db(conn, point_id, current_indices[0], current_indices[1])
                            # Update in-memory point_config for next use with this ds_geo if multiple vars
                            point_details_for_processing[point_id]['cached_azimuth_index'] = current_indices[0]
                            point_details_for_processing[point_id]['cached_range_index'] = current_indices[1]
                        else:
                            logger.warning(f"Could not find indices for point '{point_name}' in scan {scan_filepath}.")
                            continue  # Skip this point for this scan

                    if not current_indices: continue

                    for var_name, var_id in active_variables_map.items():
                        # Check again if this specific (point, var, timestamp) is missing
                        if precise_scan_dt not in existing_db_timestamps_map.get((point_id, var_id), set()):
                            value = extract_point_value(ds_geo, var_name, current_indices[0], current_indices[1])
                            if not np.isnan(value):
                                global_new_data_to_insert.append({
                                    'timestamp': precise_scan_dt, 'point_id': point_id,
                                    'variable_id': var_id, 'value': value
                                })
                                # Add to in-memory set to avoid duplicates *within this run*
                                existing_db_timestamps_map.setdefault((point_id, var_id), set()).add(precise_scan_dt)

                # Close dataset if it was opened and is not needed for other elevations
                # (It won't be, as we iterate per elevation group)
                if ds_geo and scan_filepath in opened_scans_cache:
                    opened_scans_cache.pop(scan_filepath).close()

        # 5. Batch Insert All New Data
        if global_new_data_to_insert:
            logger.info(f"Attempting to batch insert {len(global_new_data_to_insert)} new data points.")
            if not batch_insert_timeseries_data(conn, global_new_data_to_insert):
                logger.error("Batch insert failed for historical timeseries data.")
                return False  # Indicate failure
        else:
            logger.info("No new data points to insert after processing all scans.")

        logger.info(
            f"Historical timeseries generation finished. Scans opened: {total_scans_opened_and_processed}. New data points added: {len(global_new_data_to_insert)}.")
        return True

    except ConnectionError as ce:
        logger.error(f"DB connection error in historical timeseries: {ce}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error in historical timeseries: {e}", exc_info=True)
        return False
    finally:
        if conn:
            release_connection(conn)


def calculate_accumulation(
        point_name: str,
        start_dt: datetime,
        end_dt: datetime,
        interval: str,
        rate_variable: str,  # This is the name of the variable, e.g., "RATE"
        output_file_path: str
) -> bool:
    logger.warning("calculate_accumulation: DB-aware implementation is PENDING. Current call is a placeholder.")
    logger.info(f"[Placeholder] calculate_accumulation for point '{point_name}' to '{output_file_path}'.")
    # This will be refactored next.
    try:
        # Create a dummy output file for now
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as f:
            f.write("timestamp,value\n")  # Dummy CSV header
        return True
    except Exception as e:
        logger.error(f"Placeholder calculate_accumulation failed to write dummy file: {e}")
        return False
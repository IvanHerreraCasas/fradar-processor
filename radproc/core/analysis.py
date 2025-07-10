# core/analysis.py

import logging
from typing import Optional, Tuple, Dict, List, Any, Set
from collections import defaultdict
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timezone

# Import necessary functions from other modules
from .config import get_all_points_config, get_setting, get_point_config
from .utils.geo import georeference_dataset, find_nearest_indices
from .utils.csv_handler import write_timeseries_csv  # Used for outputting accumulation result
from .data import read_ppi_scan, extract_point_value, \
    read_volume_from_cfradial  # extract_scan_key_metadata is used by processor
# Import DB Manager
from .db_manager import (
    get_connection, release_connection, get_or_create_variable_id,
    batch_insert_timeseries_data, update_point_cached_indices_in_db,
    get_existing_timestamps_for_multiple_points,
    query_scan_log_for_timeseries_processing,
    query_timeseries_data_for_point, get_processed_volume_paths, update_point_height
)

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


    # Define a fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        """Fallback for tqdm if not installed. Simply returns the iterable."""
        # You can add a print statement here if you want to indicate progress without tqdm
        # print(f"Processing: {kwargs.get('desc', 'items')}...")
        return iterable

logger = logging.getLogger(__name__)


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
                    logger.warning(f"Could not find indices for point '{point_name}'.");
                    continue
            if not current_indices: continue
            point_had_data_added = False
            for var_name_to_extract in default_variables_to_extract:
                variable_id = get_or_create_variable_id(conn, var_name_to_extract)
                if variable_id is None: continue
                value, height = extract_point_value(ds_geo, var_name_to_extract, current_indices[0], current_indices[1])
                if not np.isnan(value):
                    new_data_for_batch_insert.append(
                        {'timestamp': scan_precise_timestamp, 'point_id': point_id, 'variable_id': variable_id,
                         'value': value,
                         'source_version': 'raw'})
                    point_had_data_added = True

                    if point_config.get('height') is None and not np.isnan(height):
                        update_point_height(conn, point_id, height)
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


def _generate_corrected_point_timeseries(
        point_names: List[str],
        start_dt: datetime,
        end_dt: datetime,
        version: str,
        specific_variables: Optional[List[str]],
        interactive_mode: bool
) -> bool:
    """
    Generates timeseries data from corrected volumetric files using an efficient
    sweep-centric approach.
    """
    logger.info(f"Starting corrected timeseries generation for version '{version}'.")
    conn = get_connection()
    if not conn: return False

    try:
        # --- Configuration Setup ---
        variables_to_process = specific_variables or get_setting('app.timeseries_default_variables', [])
        if not variables_to_process:
            logger.error("No variables specified or configured for timeseries processing.")
            return False

        variable_map = {name: get_or_create_variable_id(conn, name) for name in variables_to_process}
        all_points_map = {p['point_name']: p for p in get_all_points_config()}
        points_to_process = [all_points_map[name] for name in point_names if name in all_points_map]

        # --- Group points by target elevation ---
        points_by_elevation = {}
        for point in points_to_process:
            target_elevation = point.get('target_elevation')
            if target_elevation is not None:
                elevation_key = round(target_elevation, 1)
                if elevation_key not in points_by_elevation:
                    points_by_elevation[elevation_key] = []
                points_by_elevation[elevation_key].append(point)

        if not points_by_elevation:
            logger.info("No points with a defined 'target_elevation' to process.")
            return True

        # --- Database Query for Existing Data ---
        point_var_pairs_for_query = [
            (p['point_id'], v_id) for p in points_to_process for v_id in variable_map.values() if v_id
        ]
        existing_db_ts_map = get_existing_timestamps_for_multiple_points(
            conn, point_var_pairs_for_query, start_dt, end_dt, version_filter=version
        )

        # --- File and Data Processing Loop ---
        corrected_volumes = get_processed_volume_paths(conn, start_dt, end_dt, version)
        if not corrected_volumes:
            logger.info(f"No corrected volumes found for version '{version}' in the given time range.")
            return True

        global_new_data_to_insert = []
        volume_iterator = tqdm(corrected_volumes, desc=f"Corrected Volumes ({version})", unit="vol",
                               disable=not interactive_mode)

        for vol_path, vol_timestamp in volume_iterator:

            # Check if this volume's data is already in the DB for all points/variables
            is_needed = any(
                vol_timestamp not in existing_db_ts_map.get((p['point_id'], v_id), set())
                for p in points_to_process
                for v_id in variable_map.values() if v_id
            )
            if not is_needed:
                logger.debug(
                    f"Skipping volume {vol_path} at {vol_timestamp.isoformat()} as all required data already exists.")
                continue

            dtree = read_volume_from_cfradial(vol_path)
            if not dtree: continue

            # --- Loop through sweeps ---
            for sweep_name in dtree.children:
                sweep_ds = dtree[sweep_name].ds
                if 'elevation' not in sweep_ds.coords:
                    if sweep_name.startswith('sweep'):
                        logger.warning(f"Elevation not in sweep coords for sweep: {sweep_name}")
                    continue

                current_elevation = round(float(sweep_ds.elevation.values[0]), 1)
                matching_points = points_by_elevation.get(current_elevation)
                if not matching_points:
                    continue

                sweep_ds_geo = georeference_dataset(sweep_ds)
                if not ('x' in sweep_ds_geo.coords and 'y' in sweep_ds_geo.coords):
                    logger.warning(f"Georeferencing failed for sweep {current_elevation} in {vol_path}. Skipping.")
                    if sweep_ds_geo and sweep_ds_geo is not sweep_ds: sweep_ds_geo.close()
                    continue

                for point_config in matching_points:
                    point_id = point_config['point_id']

                    az_idx, rg_idx = point_config.get('cached_azimuth_index'), point_config.get('cached_range_index')
                    if az_idx is None or rg_idx is None:
                        indices = find_nearest_indices(sweep_ds_geo, point_config['latitude'],
                                                       point_config['longitude'])
                        if indices:
                            az_idx, rg_idx = indices
                            update_point_cached_indices_in_db(conn, point_id, az_idx, rg_idx)

                    if az_idx is not None and rg_idx is not None:
                        for var_name, var_id in variable_map.items():
                            if not var_id: continue

                            # Granular check for existing data points
                            if vol_timestamp in existing_db_ts_map.get((point_id, var_id), set()):
                                continue

                            value, height = extract_point_value(sweep_ds_geo, var_name, az_idx, rg_idx)
                            if not np.isnan(value):
                                global_new_data_to_insert.append({
                                    'timestamp': vol_timestamp,
                                    'point_id': point_id,
                                    'variable_id': var_id,
                                    'value': value,
                                    'source_version': version
                                })

                                if point_config.get('height') is None and not np.isnan(height):
                                    update_point_height(conn, point_id, height)
                            else:
                                logger.debug(f"NaN value for {var_name} at point {point_config['point_name']}")

                if sweep_ds_geo and sweep_ds_geo is not sweep_ds:
                    sweep_ds_geo.close()

            dtree.close()

        # --- Batch Insert (Unchanged) ---
        if global_new_data_to_insert:
            logger.info(f"Performing batch insert with {len(global_new_data_to_insert)} new corrected data points...")
            if not batch_insert_timeseries_data(conn, global_new_data_to_insert):
                logger.error("Final batch insert failed for corrected data.")
                return False

        return True
    finally:
        release_connection(conn)


def _generate_raw_point_timeseries(
        point_names: List[str],
        start_dt: datetime,
        end_dt: datetime,
        specific_variables: Optional[List[str]] = None,
        interactive_mode: bool = False,
) -> bool:
    """Generates timeseries data from raw scan files."""
    if not point_names:
        logger.warning("generate_point_timeseries called with no point names.")
        return True

    logger.info("Starting raw historical timeseries generation.")
    logger.info(f"Time range: {start_dt.isoformat()} to {end_dt.isoformat()}")

    variables_to_process: List[str]
    if specific_variables and len(specific_variables) > 0:
        variables_to_process = specific_variables
        logger.info(f"Processing specific variables: {', '.join(variables_to_process)}")
    else:
        variables_to_process = get_setting('app.timeseries_default_variables', [])
        if not variables_to_process:
            logger.error(
                "No specific_variables provided and no 'app.timeseries_default_variables' configured. Cannot proceed.")
            return False
        logger.info(f"Processing default variables: {', '.join(variables_to_process)}")

    conn = None
    global_new_data_to_insert: List[Dict[str, Any]] = []
    total_scans_opened_and_processed = 0
    overall_success = True

    try:
        conn = get_connection()
        variable_name_to_id_map: Dict[str, Optional[int]] = {
            v_name: get_or_create_variable_id(conn, v_name) for v_name in variables_to_process
        }
        active_variables_map = {name: id_val for name, id_val in variable_name_to_id_map.items() if id_val is not None}

        if not active_variables_map:
            logger.error("No valid variable IDs found for the specified/default variables. Cannot proceed.")
            return False

        all_db_points_map = {p_conf['point_name']: p_conf for p_conf in get_all_points_config()}

        points_by_elevation: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        point_details_for_processing: Dict[int, Dict[str, Any]] = {}

        for name in point_names:
            point_config_from_db = all_db_points_map.get(name)
            if not point_config_from_db or point_config_from_db.get(
                    'target_elevation') is None or point_config_from_db.get('point_id') is None:
                logger.warning(f"Point '{name}' not found in DB or is misconfigured. Skipping.")
                continue

            point_id = point_config_from_db['point_id']
            target_elevation = float(point_config_from_db['target_elevation'])
            points_by_elevation[target_elevation].append(point_config_from_db)
            point_details_for_processing[point_id] = point_config_from_db

        if not points_by_elevation:
            logger.warning("No valid points found for processing after checking configuration.")
            return True

        elevation_tolerance = get_setting('app.elevation_tolerance_for_points', 0.1)

        # +++ Outer loop progress bar (for elevation groups) +++
        elevation_iterator = tqdm(points_by_elevation.items(), desc="Processing Elevation Groups", unit="group",
                                  disable=not (TQDM_AVAILABLE and interactive_mode))
        for target_elevation, points_in_group in elevation_iterator:
            if TQDM_AVAILABLE and interactive_mode:  # Update description for the outer loop
                elevation_iterator.set_description(f"Elev Group {target_elevation:.2f}°")

            logger.info(f"Processing for target elevation: {target_elevation:.2f}° (+/- {elevation_tolerance}°)")

            candidate_scans: List[Tuple[str, datetime, int]] = query_scan_log_for_timeseries_processing(
                conn,
                target_elevation,
                start_dt,
                end_dt,
                elevation_tolerance=elevation_tolerance
            )

            if not candidate_scans:
                logger.info(
                    f"No scans found in log for elevation group {target_elevation:.2f}° in the given time range.")
                continue

            logger.info(f"Found {len(candidate_scans)} candidate scans for elevation group {target_elevation:.2f}°.")

            point_var_pairs_for_query: List[Tuple[int, int]] = [
                (p['point_id'], v_id) for p in points_in_group for v_id in active_variables_map.values()
            ]

            existing_db_ts_map: Dict[Tuple[int, int], Set[datetime]] = get_existing_timestamps_for_multiple_points(
                conn, point_var_pairs_for_query, start_dt, end_dt
            )

            # +++ Inner loop progress bar (for scans within an elevation group) +++
            scan_iterator = tqdm(candidate_scans, desc=f"Scans at {target_elevation:.2f}°", unit="scan", leave=False,
                                 disable=not (TQDM_AVAILABLE and interactive_mode))
            for scan_filepath, precise_scan_dt, _scan_log_id in scan_iterator:
                scan_actually_needed_for_a_point = False
                for p_conf_check in points_in_group:
                    p_id_check = p_conf_check['point_id']
                    for var_id_check in active_variables_map.values():
                        if precise_scan_dt not in existing_db_ts_map.get((p_id_check, var_id_check), set()):
                            scan_actually_needed_for_a_point = True
                            break
                    if scan_actually_needed_for_a_point:
                        break

                if not scan_actually_needed_for_a_point:
                    logger.debug(
                        f"Scan {scan_filepath} at {precise_scan_dt.isoformat()} not needed. Skipping file read.")
                    continue

                logger.debug(f"Processing scan file: {scan_filepath} for timestamp {precise_scan_dt.isoformat()}")
                ds_raw = None
                ds_geo = None
                try:
                    ds_raw = read_ppi_scan(scan_filepath, list(active_variables_map.keys()))
                    if ds_raw is None:
                        logger.warning(f"Failed to read scan: {scan_filepath}. Skipping.")
                        overall_success = False
                        continue

                    total_scans_opened_and_processed += 1

                    scan_elev_from_data = float(ds_raw['elevation'].item())
                    if abs(scan_elev_from_data - target_elevation) > elevation_tolerance:
                        logger.warning(
                            f"Scan {scan_filepath} (Elev: {scan_elev_from_data:.2f}) outside tolerance for group {target_elevation:.2f}. Skipping point extraction.")
                        ds_raw.close()
                        continue

                    ds_geo = georeference_dataset(ds_raw)
                    if not ('x' in ds_geo.coords and 'y' in ds_geo.coords):
                        logger.warning(f"Georeferencing failed for {scan_filepath}. Skipping point extraction.")
                        overall_success = False
                        ds_raw.close()
                        if ds_geo and ds_geo is not ds_raw: ds_geo.close()
                        continue

                    for p_conf in points_in_group:
                        p_id = p_conf['point_id']
                        p_name = p_conf['point_name']
                        az_idx_cached = p_conf.get('cached_azimuth_index')
                        rg_idx_cached = p_conf.get('cached_range_index')
                        current_indices: Optional[Tuple[int, int]] = None

                        if az_idx_cached is not None and rg_idx_cached is not None:
                            current_indices = (int(az_idx_cached), int(rg_idx_cached))
                        else:
                            logger.debug(f"No cached indices for point '{p_name}'. Calculating...")
                            calculated_indices = find_nearest_indices(ds_geo, p_conf['latitude'], p_conf['longitude'])
                            if calculated_indices:
                                current_indices = calculated_indices
                                update_point_cached_indices_in_db(conn, p_id, current_indices[0], current_indices[1])
                                p_conf['cached_azimuth_index'] = current_indices[0]
                                p_conf['cached_range_index'] = current_indices[1]
                            else:
                                logger.warning(
                                    f"Could not find indices for point '{p_name}' in scan {scan_filepath}. Skipping.")
                                continue

                        if not current_indices: continue

                        for var_name, var_id in active_variables_map.items():
                            if precise_scan_dt not in existing_db_ts_map.get((p_id, var_id), set()):
                                value, height = extract_point_value(ds_geo, var_name, current_indices[0], current_indices[1])
                                if not np.isnan(value):
                                    global_new_data_to_insert.append({
                                        'timestamp': precise_scan_dt,
                                        'point_id': p_id,
                                        'variable_id': var_id,
                                        'value': value,
                                        'source_version': 'raw'
                                    })
                                    existing_db_ts_map.setdefault((p_id, var_id), set()).add(precise_scan_dt)
                                    if p_conf.get('height') is None and not np.isnan(height):
                                        update_point_height(conn, p_id, height)
                                else:
                                    logger.debug(
                                        f"NaN value for {var_name} at point {p_name} from {scan_filepath}, not adding.")
                except Exception as e_scan:
                    logger.error(f"Error processing scan file {scan_filepath}: {e_scan}", exc_info=True)
                    overall_success = False
                finally:
                    if ds_geo is not None:
                        try:
                            ds_geo.close()
                        except Exception:
                            pass
                    if ds_raw is not None and ds_raw is not ds_geo:
                        try:
                            ds_raw.close()
                        except Exception:
                            pass
            # +++ End inner loop for scans +++
        # +++ End outer loop for elevation groups +++

        if global_new_data_to_insert:
            logger.info(f"Performing final batch insert with {len(global_new_data_to_insert)} new data points...")
            if not batch_insert_timeseries_data(conn, global_new_data_to_insert):
                logger.error("Final batch insert failed.")
                overall_success = False

        logger.info(
            f"Historical timeseries generation finished. Total scans opened: {total_scans_opened_and_processed}. Success: {overall_success}")
        return overall_success

    except ConnectionError as ce:
        logger.error(f"Database connection error during timeseries generation: {ce}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during historical timeseries generation: {e}", exc_info=True)
        return False
    finally:
        if conn:
            release_connection(conn)


def generate_timeseries(
        point_names: List[str],
        start_dt: datetime,
        end_dt: datetime,
        source: str = 'raw',
        version: Optional[str] = None,
        specific_variables: Optional[List[str]] = None,
        interactive_mode: bool = False
) -> bool:
    """
    Orchestrates timeseries generation from either raw or corrected data sources.

    Args:
        point_names: List of point names to process.
        start_dt: The start of the time range.
        end_dt: The end of the time range.
        source: The data source, either 'raw' or 'corrected'.
        version: The correction version to use (required if source is 'corrected').
        specific_variables: A list of specific variables to process.
        interactive_mode: Enables/disables tqdm progress bars.

    Returns:
        True if the process completed successfully, False otherwise.
    """
    logger.info(f"--- Starting Timeseries Generation from '{source.upper()}' source ---")

    if source == 'corrected':
        if not version:
            logger.error("A --version must be specified when using --source corrected.")
            return False
        return _generate_corrected_point_timeseries(point_names, start_dt, end_dt, version, specific_variables,
                                                    interactive_mode)
    else:  # source == 'raw'
        return _generate_raw_point_timeseries(point_names, start_dt, end_dt, specific_variables, interactive_mode)


def calculate_accumulation(
        point_name: str,
        start_dt: datetime,
        end_dt: datetime,
        interval: str,
        rate_variable: str,
        output_file_path: str,
        source: str = 'raw',
        version: Optional[str] = None
) -> bool:
    """
    Calculates accumulated precipitation for a point and time range.
    Ensures source rate data is in PostgreSQL, then queries it,
    performs calculations, and saves results to a CSV file.
    """
    logger.info(f"Calculating accumulation for point '{point_name}' from source '{source}'.")
    if source == 'corrected' and not version:
        logger.error("A --version must be specified when using --source corrected for accumulation.")
        return False

    conn = None
    try:
        # Step 1: Ensure the source data exists by calling the main orchestrator
        # This will now correctly generate raw or corrected timeseries data as needed.
        logger.info(f"Ensuring source rate data ('{rate_variable}') for point '{point_name}' is in DB...")
        if not generate_timeseries(
                point_names=[point_name], start_dt=start_dt, end_dt=end_dt,
                specific_variables=[rate_variable], source=source, version=version
        ):
            logger.error(f"Failed to ensure source timeseries data. Cannot calculate accumulation.")
            return False

        conn = get_connection()

        # Step 2: Get point_id and variable_id (no change here)
        point_config = get_point_config(point_name)
        if not point_config: logger.error(f"Point '{point_name}' not found."); return False
        point_id = point_config['point_id']
        variable_id = get_or_create_variable_id(conn, rate_variable)
        if not variable_id: logger.error(f"Could not resolve ID for variable '{rate_variable}'."); return False

        # Step 3: Query the data, now with versioning
        logger.info(f"Fetching '{rate_variable}' data for point '{point_name}' (source: {source}) from DB...")
        df_rate = query_timeseries_data_for_point(conn, point_id, variable_id, start_dt, end_dt, rate_variable,
                                                  source_version=version if source == 'corrected' else 'raw')

        if df_rate.empty:
            logger.warning("No rate data found for the specified parameters. Cannot calculate accumulation.")
            return True  # Not an error, just no data.

        logger.info(f"Successfully fetched {len(df_rate)} rate data points from database.")

        # 4. Perform Accumulation Calculation (using pandas, as before)
        df_rate.set_index('timestamp', inplace=True)
        df_rate.sort_index(inplace=True)

        time_diff = df_rate.index.to_series().diff()
        duration_h = time_diff.dt.total_seconds() / 3600.0

        rate_numeric = pd.to_numeric(df_rate[rate_variable], errors='coerce')
        incremental_precip_mm = rate_numeric * duration_h
        incremental_precip_mm = incremental_precip_mm.fillna(0)  # First diff is NaN, treat as 0 precip

        try:
            accumulated_series = incremental_precip_mm.resample(interval, label='right', closed='right').sum()
        except ValueError as e:  # Handles bad interval strings for resample
            logger.error(f"Invalid resampling interval '{interval}': {e}")
            return False

        accum_col_name = f'precip_acc_{interval.replace("-", "_").replace(":", "_")}_mm'  # Ensure filename-friendly
        output_df = accumulated_series.reset_index(name=accum_col_name)

        logger.info(f"Accumulation calculation complete. Generated {len(output_df)} intervals.")

        # 5. Prepare Metadata and Write Output CSV
        metadata = {
            "Point Name": point_name,
            "Target Latitude": point_config.get('latitude', 'N/A'),
            "Target Longitude": point_config.get('longitude', 'N/A'),
            "Target Elevation (deg)": point_config.get('target_elevation', 'N/A'),
            "Height (m)": f"{point_config.get('height', 'N/A'):.2f}",
            "Source Rate Variable": rate_variable,
            "Accumulation Interval": interval,
            "Analysis Start Time (UTC)": start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "Analysis End Time (UTC)": end_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "Data Source": source,
            "Generated Timestamp (UTC)": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        }

        write_timeseries_csv(output_file_path, output_df, metadata)  # This handles dir creation & overwrite
        logger.info(f"Accumulation results successfully saved to: {output_file_path}")
        return True

    except Exception as e:
        logger.error(f"Unexpected error during accumulation: {e}", exc_info=True)
        return False
    finally:
        if conn: release_connection(conn)
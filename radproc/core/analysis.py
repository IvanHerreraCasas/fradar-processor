# core/analysis.py

import os
import json
import logging
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta, timezone

# Import necessary functions from other modules
from .config import get_setting, get_point_config, get_config # Added get_config
from .utils.geo import get_dataset_crs, georeference_dataset, transform_point_to_dataset_crs
# Import CSV handler
from .utils.csv_handler import read_timeseries_csv, append_to_timeseries_csv, write_metadata_header_if_needed, write_timeseries_csv
# Import data finder and reader
from .data import get_filepaths_in_range, read_scan
# Import georeferencing from utils (used in historical)

logger = logging.getLogger(__name__)

# --- Index Caching Helpers ---
# ( _get_index_cache_path, load_cached_indices, save_cached_indices - already added )
def _get_index_cache_path(point_name: str, timeseries_dir: str) -> str:
    """Constructs the file path for a point's index cache."""
    return os.path.join(timeseries_dir, f".{point_name}.index.json") # Hidden file

def load_cached_indices(point_name: str, timeseries_dir: str) -> Optional[Tuple[int, int]]:
    """Loads cached azimuth and range indices for a given point."""
    cache_path = _get_index_cache_path(point_name, timeseries_dir)
    if not os.path.exists(cache_path):
        logger.debug(f"Index cache not found for point '{point_name}' at {cache_path}")
        return None
    try:
        with open(cache_path, 'r') as f: data = json.load(f)
        if isinstance(data, dict) and 'azimuth_index' in data and 'range_index' in data:
            az_idx = int(data['azimuth_index'])
            rg_idx = int(data['range_index'])
            logger.info(f"Loaded cached indices for point '{point_name}': az={az_idx}, rg={rg_idx}")
            return az_idx, rg_idx
        else:
            logger.warning(f"Invalid data structure in index cache file: {cache_path}")
            return None
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"Failed to load or parse index cache file {cache_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading index cache {cache_path}: {e}", exc_info=True)
        return None

def save_cached_indices(point_name: str, timeseries_dir: str, az_idx: int, rg_idx: int):
    """Saves calculated azimuth and range indices to a cache file."""
    cache_path = _get_index_cache_path(point_name, timeseries_dir)
    data = {'azimuth_index': az_idx, 'range_index': rg_idx}
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f: json.dump(data, f, indent=2)
        logger.info(f"Saved indices cache for point '{point_name}' to {cache_path}")
    except Exception as e:
        logger.error(f"Failed to save index cache file {cache_path}: {e}", exc_info=True)

# --- Core Analysis Functions ---

def find_nearest_indices(ds_geo: xr.Dataset, target_lat: float, target_lon: float, max_dist_meters: float = 200.0) -> Optional[Tuple[int, int]]:
    """
    Finds the azimuth and range indices in the dataset closest to a target lat/lon.

    Transforms the target coordinates, calculates distances in the projected space,
    and finds the minimum distance indices. Optionally checks if the minimum distance
    is within a specified threshold.

    Args:
        ds_geo: Georeferenced xarray.Dataset (must contain 2D 'x', 'y' coordinates
                and have accessible CRS info).
        target_lat: Target latitude (decimal degrees).
        target_lon: Target longitude (decimal degrees).
        max_dist_meters: Maximum allowable distance (in meters) between the target point
                         and the center of the closest radar bin. If exceeded, returns None.
                         Set to None or float('inf') to disable check.

    Returns:
        A tuple (azimuth_index, range_index) if found within threshold, otherwise None.
    """
    logger.debug(f"Finding nearest indices for Lat={target_lat}, Lon={target_lon}")

    # 1. Get Dataset CRS
    ds_crs = get_dataset_crs(ds_geo)
    if ds_crs is None:
        logger.error(f"Cannot find CRS in georeferenced dataset for point ({target_lat}, {target_lon}). Skipping index search.")
        return None # Cannot proceed without CRS

    # 2. Transform Target Point to Dataset CRS
    transformed_coords = transform_point_to_dataset_crs(target_lat, target_lon, ds_crs)
    if transformed_coords is None:
        logger.error("Cannot find nearest indices: Coordinate transformation failed.")
        return None
    target_x, target_y = transformed_coords
    logger.debug(f"Transformed target point to dataset CRS: X={target_x:.2f}, Y={target_y:.2f}")

    # 3. Check if necessary coordinates exist in dataset
    if 'x' not in ds_geo.coords or 'y' not in ds_geo.coords or \
       ds_geo['x'].ndim != 2 or ds_geo['y'].ndim != 2:
        logger.error("Cannot find nearest indices: Dataset missing 2D 'x' or 'y' coordinates.")
        logger.info("Dataset coordinates: x.shape={}, y.shape={}".format(ds_geo['x'].shape, ds_geo['y'].shape))
        return None

    # 4. Calculate Squared Distances
    try:
        # Ensure x and y are computed if they are Dask arrays
        # Use .data to work with underlying numpy/dask array directly for efficiency if possible
        x_coords = ds_geo['x'].data
        y_coords = ds_geo['y'].data
        az_coords = ds_geo['azimuth'] # Usually not Dask, but access directly
        rg_coords = ds_geo['range']
        # Broadcasting should handle the subtraction
        dist_sq = (x_coords - target_x)**2 + (y_coords - target_y)**2

        # If dist_sq is a Dask array, compute it
        if hasattr(dist_sq, 'compute'):
            logger.debug("Computing distances (Dask array detected)...")
            dist_sq_computed = dist_sq.compute()
        else:
            dist_sq_computed = dist_sq # Assuming it's already a NumPy array

    except Exception as e:
        logger.error(f"Error calculating distances: {e}", exc_info=True)
        return None

    # 5. Find Minimum Distance Index
    try:
        # Check if all distances are NaN (e.g., if coords were bad)
        if np.all(np.isnan(dist_sq_computed)):
             logger.warning("All calculated distances are NaN. Cannot find minimum.")
             return None

        # Find the flat index of the minimum non-NaN value
        min_flat_index = np.nanargmin(dist_sq_computed)
        # Convert the flat index back to 2D indices (azimuth, range)
        az_idx, rg_idx = np.unravel_index(min_flat_index, dist_sq_computed.shape)

        # --- MODIFIED/ADDED LOGGING ---
        try:
            # Retrieve the actual azimuth and range values using the found indices
            azimuth_value = az_coords.isel({az_coords.dims[0]: az_idx}).item() # Use actual dim name if not 'azimuth'
            range_value = rg_coords.isel({rg_coords.dims[0]: rg_idx}).item()   # Use actual dim name if not 'range'

            # Calculate the actual minimum distance
            min_dist = np.sqrt(dist_sq_computed[az_idx, rg_idx])

            logger.debug(
                f"Minimum distance ({min_dist:.2f} m) found at indices: Az={az_idx}, Ran={rg_idx}. "
                f"Corresponding coordinates: Azimuth={azimuth_value:.2f} deg, Range={range_value:.1f} m"
            )
            # Optional: Add warning if min_dist is large?
            # avg_range_step = np.mean(np.diff(rg_coords.values)) if len(rg_coords) > 1 else 1000
            # if min_dist > avg_range_step * 0.75: # Example threshold
            #     logger.warning(f"Nearest bin center is relatively far ({min_dist:.1f}m) from the target point.")

        except Exception as lookup_err:
            # Log if looking up the values fails, but still return indices if found
            logger.warning(f"Found indices (Az={az_idx}, Ran={rg_idx}), but failed to look up coordinate values: {lookup_err}")

        # Return the indices (needed for caching and data extraction)
        return int(az_idx), int(rg_idx) # Cast to standard int just in case

    except ValueError as ve:
        # nanargmin raises ValueError if all inputs are NaN
        logger.warning(f"Could not find minimum distance (likely all distances were NaN): {ve}")
        return None
    except Exception as e:
        logger.error(f"Error finding minimum distance index: {e}", exc_info=True)
        return None


def extract_point_value(ds: xr.Dataset, variable: str, az_idx: int, rg_idx: int) -> float:
    """
    Extracts the data value for a specific variable at given azimuth/range indices.

    Assumes the input dataset has only one time step.

    Args:
        ds: Input xarray.Dataset (single time step).
        variable: Name of the variable to extract.
        az_idx: Azimuth index.
        rg_idx: Range index.

    Returns:
        The extracted data value as a float, or np.nan if extraction fails
        or the value is invalid.
    """
    try:
        if variable not in ds.data_vars:
            logger.warning(f"Variable '{variable}' not found in dataset for value extraction.")
            return np.nan

        # Select the data point using integer indexing
        data_point = ds[variable].isel(time=0, azimuth=az_idx, range=rg_idx)

        # If data is Dask-based, compute it to get the actual value
        if hasattr(data_point.data, 'compute'):
            value = data_point.compute().item()
        else:
            value = data_point.item() # Assumes NumPy backend

        # Check for valid numeric type and handle potential NaNs from source
        if isinstance(value, (int, float, np.number)) and not np.isnan(value):
            return float(value)
        else:
            logger.debug(f"Invalid or NaN value found for '{variable}' at az={az_idx}, rg={rg_idx}. Value: {value}")
            return np.nan

    except IndexError:
        logger.error(f"Indices az={az_idx}, rg={rg_idx} are out of bounds for variable '{variable}'.")
        return np.nan
    except Exception as e:
        logger.error(f"Failed to extract value for '{variable}' at az={az_idx}, rg={rg_idx}: {e}", exc_info=True)
        return np.nan


# --- Orchestrator Functions ---

def update_timeseries_for_scan(ds_geo: xr.Dataset):
    """
    Updates timeseries CSV files for all defined points using data from a single scan.

    Called by the processor during the 'run' command workflow.

    Args:
        ds_geo: The georeferenced xarray.Dataset for the current scan (single time step).
    """
    logger.info("Starting automatic timeseries update for current scan...")

    # 1. Get Configuration
    all_points_config = get_setting('points_config.points', [])
    timeseries_dir = get_setting('app.timeseries_dir')

    if not isinstance(all_points_config, list) or not all_points_config:
        logger.debug("No points defined in 'points.yaml'. Skipping timeseries update.")
        return
    if not timeseries_dir:
        logger.error("Automatic timeseries update failed: 'app.timeseries_dir' is not configured.")
        return

    # 2. Extract Scan Info
    try:
        # Ensure timestamp is timezone-aware UTC
        scan_ts = pd.to_datetime(ds_geo['time'].values.item()).tz_localize('UTC')
        # Get dataset CRS (needed for finding indices if cache miss)
        ds_crs = get_dataset_crs(ds_geo)
        if ds_crs is None:
            logger.error("Cannot update timeseries: Dataset CRS could not be determined for index finding.")
            return
    except Exception as e:
        logger.error(f"Failed to extract time/CRS from dataset for timeseries update: {e}", exc_info=True)
        return

    logger.debug(f"Processing timeseries update for scan time: {scan_ts}")

    # 3. Loop Through Points
    rows_to_append: Dict[str, Dict] = {} # Key: csv_path, Value: row data dict {'timestamp':..., 'VAR':...}
    points_processed_count = 0
    points_failed_count = 0

    for point_config in all_points_config:
        if not isinstance(point_config, dict): continue # Skip invalid entries

        point_name = point_config.get('name')
        target_lat = point_config.get('latitude')
        target_lon = point_config.get('longitude')
        target_elevation = point_config.get('elevation')
        variable = point_config.get('variable') # Use the default variable for this point

        if not point_name or target_lat is None or target_lon is None or not variable:
            logger.warning(f"Skipping invalid point definition in points.yaml: {point_config}")
            continue
        
        scan_elevation = ds_geo['elevation'].item() # Get elevation from dataset
        if scan_elevation != target_elevation:
            logger.warning(f"Scan elevation ({scan_elevation}) does not match target elevation ({target_elevation}). Skipping point '{point_name}'.")
            points_failed_count += 1
            continue

        logger.debug(f"Processing point '{point_name}' for variable '{variable}'...")

        # Determine paths
        try:
            # Ensure timeseries_dir exists (create if necessary)
            # Do this once before the loop? Or here per point is safer? Per point is fine.
            os.makedirs(timeseries_dir, exist_ok=True)
            csv_path = os.path.join(timeseries_dir, f"{point_name}_{variable}.csv")
            index_cache_path = _get_index_cache_path(point_name, timeseries_dir) # Uses helper
        except OSError as e:
             logger.error(f"Cannot create/access timeseries directory '{timeseries_dir}' for point '{point_name}'. Error: {e}")
             points_failed_count += 1
             continue # Skip this point

        # Find or Cache Indices
        cached_indices = load_cached_indices(point_name, timeseries_dir)
        if cached_indices is None:
            indices = find_nearest_indices(ds_geo, target_lat, target_lon) # Max dist check inside
            if indices is not None:
                cached_indices = indices
                az_idx, rg_idx = indices
                save_cached_indices(point_name, timeseries_dir, az_idx, rg_idx)
                logger.info(f"Determined and cached indices for '{point_name}': {cached_indices}")
                units = ds_geo[variable].attrs.get('units', 'unknown')
                
                # --- Gather Metadata & Write Header ---
                metadata_for_header = {
                        "Point Name": point_name,
                        "Target Latitude": f"{target_lat:.6f}",
                        "Target Longitude": f"{target_lon:.6f}",
                        "Target Elevation (deg)": f"{target_elevation:.1f}",
                        "Variable": f"{variable} ({units})",
                        "Radar Latitude": f"{ds_geo['latitude'].item():.6f}",
                        "Radar Longitude": f"{ds_geo['longitude'].item():.6f}",
                        "Radar Altitude (m)": f"{ds_geo['altitude'].item():.1f}",
                        "Nearest Bin Azimuth (deg)": f"{ds_geo['azimuth'].isel(azimuth=az_idx).item():.2f}",
                        "Nearest Bin Range (m)": f"{ds_geo['range'].isel(range=rg_idx).item():.1f}",
                        "Nearest Bin Azimuth Index": az_idx,
                        "Nearest Bin Range Index": rg_idx,
                }
                write_metadata_header_if_needed(csv_path, metadata_for_header)
                # ------------------------------------
            else:
                logger.warning(f"Could not find valid indices for point '{point_name}'. Skipping timeseries update for this scan.")
                # Don't mark as total failure, maybe point is temporarily out of range?
                # Or maybe log failure count? Let's just log warning for now.
                continue # Skip this point for this scan

        # Extract Value
        if cached_indices:
            az_idx, rg_idx = cached_indices
            value = extract_point_value(ds_geo, variable, az_idx, rg_idx)

            # Prepare row data (use np.nan if extraction failed)
            row_data = {'timestamp': scan_ts, variable: value}
            rows_to_append[csv_path] = row_data
            points_processed_count += 1
            logger.debug(f"Prepared data for point '{point_name}': {value=}")
        # (If cached_indices is still None after trying, we already continued)


    # 4. Append Data to CSVs
    if rows_to_append:
        logger.info(f"Appending data for {len(rows_to_append)} points to CSV files...")
        append_success_count = 0
        append_fail_count = 0
        for csv_path, row_data in rows_to_append.items():
            try:
                new_data_df = pd.DataFrame([row_data])
                # Ensure timestamp column type is correct before append function formats it
                new_data_df['timestamp'] = pd.to_datetime(new_data_df['timestamp'], utc=True)
                append_to_timeseries_csv(csv_path, new_data_df)
                append_success_count += 1
            except Exception as e:
                logger.error(f"Failed to append data to {csv_path}: {e}", exc_info=True)
                append_fail_count += 1

        logger.info(f"CSV append finished. Success: {append_success_count}, Failed: {append_fail_count}")
    else:
        logger.info("No new timeseries data to append for this scan.")

    logger.info(f"Automatic timeseries update finished for scan {scan_ts}. Points processed: {points_processed_count}, Points failed/skipped: {points_failed_count}")

def generate_point_timeseries(
    point_name: str,
    start_dt: datetime,
    end_dt: datetime,
    variable_override: Optional[str] = None
) -> Optional[str]:
    """
    Generates or updates a timeseries CSV for a specific point and elevation,
    processing only scans within the requested time range whose data is not
    already present in the CSV.

    Args:
        point_name: The name of the point (defined in points.yaml).
        start_dt: The start datetime for the processing range (timezone-aware UTC).
        end_dt: The end datetime for the processing range (timezone-aware UTC).
        variable_override: Optional variable name to extract, overriding the point's default.
    """
    logger.info(f"Starting/Updating timeseries generation for point '{point_name}'")
    logger.info(f"Requested range: {start_dt} to {end_dt}")

    # 1. Get Configuration
    point_config = get_point_config(point_name)
    if not point_config: return None

    target_lat = point_config['latitude']
    target_lon = point_config['longitude']
    target_elevation = point_config['elevation']
    default_variable = point_config['variable']
    variable_to_extract = variable_override if variable_override else default_variable
    elevation_tolerance = 0.1

    timeseries_dir = get_setting('app.timeseries_dir')
    processed_scan_dir = get_setting('app.output_dir')
    if not timeseries_dir or not processed_scan_dir:
        logger.error("Timeseries generation failed: 'timeseries_dir' or 'output_dir' not configured.")
        return None
    os.makedirs(timeseries_dir, exist_ok=True)

    csv_path = os.path.join(timeseries_dir, f"{point_name}_{variable_to_extract}.csv")
    logger.info(f"Target CSV: {csv_path}")
    logger.info(f"Target Elevation: {target_elevation:.1f} deg")

    # 2. Read Existing Timestamps
    logger.debug("Reading existing timestamps from CSV...")
    existing_df = read_timeseries_csv(csv_path) # Handles file not found, ensures UTC timestamps
    existing_timestamps = set()
    if not existing_df.empty and 'timestamp' in existing_df.columns:
        # Convert to pd.Timestamp for reliable comparison if not already
        existing_timestamps = set(pd.to_datetime(existing_df['timestamp'], utc=True))
        logger.info(f"Found {len(existing_timestamps)} existing timestamps in {csv_path}.")
    else:
        logger.info(f"No existing timestamps found in {csv_path}.")

    # 3. Find Candidate Files in User Range
    logger.info(f"Searching for processed scan files between {start_dt} and {end_dt}...")
    filepaths_tuples = get_filepaths_in_range(processed_scan_dir, start_dt, end_dt)
    if not filepaths_tuples:
        logger.info("No scan files found in the specified range and directory.")
        # Still might need to write header if CSV is new/empty
        # Proceed to header writing check
    else:
         logger.info(f"Found {len(filepaths_tuples)} candidate scan files in range.")

    # 4. Load/Calculate Indices & Prepare Metadata
    cached_indices = load_cached_indices(point_name, timeseries_dir)
    metadata_for_header: Optional[Dict[str, Any]] = None
    first_processed_ds_geo: Optional[xr.Dataset] = None # Hold reference to gather metadata

    # 5. Process Files & Extract Data for *NEW* Timestamps
    new_rows = []
    processed_scan_count = 0
    added_data_count = 0
    error_count = 0

    for filepath, file_dt_nominal in filepaths_tuples: # file_dt_nominal from filename parsing
        ds = None
        ds_geo = None
        try:
            processed_scan_count += 1
            logger.debug(f"Checking file [{processed_scan_count}/{len(filepaths_tuples)}]: {os.path.basename(filepath)}")

            # --- Get Actual Timestamp from File & Check if Exists ---
            # We need to read the file briefly to get the *precise* timestamp
            # associated with the data, filename might be slightly off.
            # Optimization: Could parse filename first for a quick check, but reading is safer.
            # Let's parse filename first for efficiency. file_dt_nominal is already UTC-aware.
            if file_dt_nominal in existing_timestamps:
                logger.debug(f"Timestamp {file_dt_nominal} already exists in CSV. Skipping.")
                continue

            # --- Process if Timestamp is New ---
            # Read scan
            ds = read_scan(filepath, variables=[variable_to_extract])
            if ds is None: raise ValueError("Failed to read scan")

            # Get precise timestamp from dataset and ensure UTC
            scan_ts_precise = pd.to_datetime(ds['time'].values.item()).tz_localize('UTC')

            # Double-check precise timestamp against existing set
            if scan_ts_precise in existing_timestamps:
                 logger.debug(f"Precise timestamp {scan_ts_precise} already exists in CSV (filename nominal: {file_dt_nominal}). Skipping.")
                 ds.close(); ds=None; continue

            # Filter by Elevation
            scan_elevation = ds['elevation'].item()
            if scan_elevation != target_elevation:
                logger.debug(f"Skipping scan: Elevation {scan_elevation:.1f} != Target {target_elevation:.1f}")
                ds.close(); ds = None; continue

            # Georeference
            ds_geo = georeference_dataset(ds)
            if 'x' not in ds_geo.coords: raise ValueError("Failed to georeference scan")

            # Determine indices & Metadata ONCE
            if cached_indices is None:
                logger.debug(f"Attempting to determine indices/metadata for '{point_name}'...")
                indices = find_nearest_indices(ds_geo, target_lat, target_lon)
                if indices is not None:
                    cached_indices = indices
                    save_cached_indices(point_name, timeseries_dir, indices[0], indices[1])
                    logger.info(f"Determined and cached indices for '{point_name}': {cached_indices}")
                else:
                    # If indices cannot be determined, stop processing
                    logger.error(f"Could not determine valid indices for point '{point_name}'. Aborting timeseries update.")
                    if ds: ds.close()
                    return None # Signal failure

            # Extract value if indices are known
            if cached_indices:
                az_idx, rg_idx = cached_indices
                value = extract_point_value(ds_geo, variable_to_extract, az_idx, rg_idx)
                # Use the precise timestamp from the data
                new_rows.append({'timestamp': scan_ts_precise, variable_to_extract: value})
                added_data_count += 1
                # Add to existing timestamps set immediately to handle duplicates within this run
                existing_timestamps.add(scan_ts_precise)
                
                units = ds_geo[variable_to_extract].attrs.get('units', 'unknown')
                
                if metadata_for_header is None:
                    # --- Gather Metadata for Header ---
                    metadata_for_header = {
                        "Point Name": point_name,
                        "Target Latitude": f"{target_lat:.6f}",
                        "Target Longitude": f"{target_lon:.6f}",
                        "Target Elevation (deg)": f"{target_elevation:.1f}",
                        "Variable": f"{variable_to_extract} ({units})",
                        "Radar Latitude": f"{ds_geo['latitude'].item():.6f}",
                        "Radar Longitude": f"{ds_geo['longitude'].item():.6f}",
                        "Radar Altitude (m)": f"{ds_geo['altitude'].item():.1f}",
                        "Nearest Bin Azimuth (deg)": f"{ds_geo['azimuth'].isel(azimuth=az_idx).item():.2f}",
                        "Nearest Bin Range (m)": f"{ds_geo['range'].isel(range=rg_idx).item():.1f}",
                        "Nearest Bin Azimuth Index": az_idx,
                        "Nearest Bin Range Index": rg_idx,
                    }
            else:
                 logger.error("Indices became unavailable unexpectedly.")
                 error_count += 1

            ds.close(); ds = None # Close dataset

        except Exception as e:
            error_count += 1
            logger.error(f"Error processing file {filepath} for timeseries: {e}", exc_info=True)
        finally:
            if ds is not None: ds.close()


    # 6. Write Metadata Header (if we determined it)
    if metadata_for_header:
         logger.debug(f"Ensuring metadata header is written to {csv_path}...")
         try:
              write_metadata_header_if_needed(csv_path, metadata_for_header)
         except Exception as header_err:
              logger.error(f"Failed to write metadata header: {header_err}", exc_info=True)
              # Decide if this is fatal? Probably not, data can still be appended.

    # 7. Append New Data to CSV
    if new_rows:
        logger.info(f"Appending {len(new_rows)} new data points to {csv_path}...")
        try:
            new_data_df = pd.DataFrame(new_rows)
            new_data_df['timestamp'] = pd.to_datetime(new_data_df['timestamp'], utc=True)
            # Sort new data before appending (append function doesn't guarantee order)
            new_data_df = new_data_df.sort_values(by='timestamp')
            append_to_timeseries_csv(csv_path, new_data_df)
            logger.info("Successfully appended new data.")
        except Exception as e:
            logger.error(f"Failed to append data to {csv_path}: {e}", exc_info=True)
            return None # Appending failed, signal error
    else:
        logger.info("No new data points to append for the specified range.")

    logger.info(f"Timeseries generation/update finished for point '{point_name}'.")
    logger.info(f"Files Scanned: {len(filepaths_tuples)}, Scans Processed: {processed_scan_count}, New Data Points Added: {added_data_count}, Errors: {error_count}")
    return csv_path # Signal success
    
def calculate_accumulation(
    point_name: str,
    start_dt: datetime,
    end_dt: datetime,
    interval: str,
    rate_variable: str,
    output_file_path: str
) -> bool:
    """
    Calculates accumulated precipitation, ensuring source rate data is updated first.

    Checks the source rate CSV, generates any missing data within the requested
    [start_dt, end_dt] range by calling generate_point_timeseries, then performs
    the accumulation calculation and saves the result.

    Args:
        point_name: The name of the point.
        start_dt: The start datetime for analysis (timezone-aware UTC).
        end_dt: The end datetime for analysis (timezone-aware UTC).
        interval: Pandas frequency string for accumulation.
        rate_variable: The name of the rate variable column.
        output_file_path: The full path for the output accumulation CSV file.

    Returns:
        True if successful, False otherwise.
    """
    logger.info(f"Calculating accumulation for point '{point_name}', interval '{interval}'")
    logger.info(f"Requested analysis range: {start_dt} to {end_dt}")
    logger.info(f"Using rate variable: '{rate_variable}'")

    # --- a. Load Config & Validate ---
    timeseries_dir = get_setting('app.timeseries_dir')
    processed_scan_dir = get_setting('app.output_dir') # Needed for generate_point_timeseries
    if not timeseries_dir:
        logger.error("Accumulation failed: 'app.timeseries_dir' is not configured.")
        return False
    if not processed_scan_dir:
         # generate_point_timeseries needs this, log error here too
        logger.error("Accumulation potentially incomplete: 'app.output_dir' (processed scan directory) is not configured for updating source data.")
        # Decide whether to proceed without updating source? Let's proceed but warn heavily.
        # return False # Stricter: Stop if source cannot be updated.

    point_config = get_point_config(point_name)
    if not point_config: return False # Error logged by getter

    # Construct source path
    source_csv_path = os.path.join(timeseries_dir, f"{point_name}_{rate_variable}.csv")

    # --- d. Read Source Data ---
    logger.info(f"Reading source data from: {source_csv_path}")
    df_rate = read_timeseries_csv(source_csv_path) # Read current state

    if df_rate.empty:
        logger.warning(f"Source rate file '{source_csv_path}' is empty. Cannot calculate accumulation.")
        # Optionally create empty output file with header?
        # metadata = {... gather metadata ...}
        # write_timeseries_csv(output_file_path, pd.DataFrame(columns=['timestamp', f'precip_acc_{...}']), metadata)
        return True

    if rate_variable not in df_rate.columns:
        logger.error(f"Rate variable column '{rate_variable}' not found in source file: {source_csv_path}")
        return False


    # --- e. Filter Data by Requested Accumulation Time Range ---
    # Use the original start_dt and end_dt provided by the user for accumulation
    logger.info(f"Filtering rate data for accumulation range: {start_dt} to {end_dt}")
    df_filtered = df_rate[(df_rate['timestamp'] >= start_dt) & (df_rate['timestamp'] <= end_dt)].copy()

    if df_filtered.empty:
        logger.info(f"No rate data found within the specified accumulation range ({start_dt} to {end_dt}).")
        return True

    logger.info(f"Using {len(df_filtered)} rate data points for accumulation calculation.")

    # --- f. Prepare for Calculation ---
    df_filtered.set_index('timestamp', inplace=True)
    df_filtered.sort_index(inplace=True)

    # --- g. Calculate Incremental Precipitation ---
    time_diff = df_filtered.index.to_series().diff()
    duration_h = time_diff.dt.total_seconds() / 3600.0
    rate_numeric = pd.to_numeric(df_filtered[rate_variable], errors='coerce')
    incremental_precip_mm = rate_numeric * duration_h
    incremental_precip_mm = incremental_precip_mm.fillna(0)

    # --- h. Resample and Sum ---
    logger.info(f"Resampling incremental precipitation to interval: '{interval}'")
    try:
        accumulated_series = incremental_precip_mm.resample(interval, label='right', closed='right').sum()
    except ValueError as e:
        logger.error(f"Invalid resampling interval '{interval}': {e}")
        return False

    accum_col_name = f'precip_acc_{interval}_mm'.replace('-', '_')
    output_df = accumulated_series.reset_index(name=accum_col_name)

    logger.info(f"Accumulated precipitation columns: {output_df.columns.tolist()}")
    logger.info(f"Accumulation calculation complete. Generated {len(output_df)} intervals.")

    # --- i. Prepare Metadata Header ---
    metadata = {
        "Point Name": point_name,
        "Target Latitude": point_config.get('latitude', 'N/A'),
        "Target Longitude": point_config.get('longitude', 'N/A'),
        "Target Elevation (deg)": point_config.get('elevation', 'N/A'),
        "Source Rate Variable": rate_variable,
        "Accumulation Interval": interval,
        "Analysis Start Time (UTC)": start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "Analysis End Time (UTC)": end_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "Source Rate File": os.path.basename(source_csv_path),
        "Generated Timestamp (UTC)": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    # --- j. Write Output CSV ---
    try:
        # Use the dedicated writer function which handles metadata and overwrites
        write_timeseries_csv(output_file_path, output_df, metadata)
        return True # Signal success
    except Exception as write_err:
        logger.error(f"Failed to write accumulation output file {output_file_path}: {write_err}", exc_info=True)
        return False # Signal failure
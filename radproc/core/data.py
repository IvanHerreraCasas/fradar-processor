# core/data.py

import xarray as xr
import numpy as np
import os
import logging
from datetime import datetime
from typing import List, Optional, Tuple

from .utils.helpers import parse_datetime_from_filename, parse_date_from_dirname
from .config import get_setting  # Import config access function

# Setup logger for this module
logger = logging.getLogger(__name__)


def _preprocess_scan(ds: xr.Dataset, azimuth_step: float) -> xr.Dataset:
    """
    Internal function to preprocess a single radar scan Dataset.
    - Reindexes azimuth to a regular grid.
    - Sets time dimension to the floored minute.
    - Sets elevation dimension using 'sweep_fixed_angle'.

    Args:
        ds: The input xarray Dataset from xr.open_dataset.
        azimuth_step: The desired regular azimuth step in degrees.

    Returns:
        The preprocessed xarray Dataset.
    """
    # Ensure 'sweep_fixed_angle' exists for elevation dimension
    if 'sweep_fixed_angle' not in ds.coords and 'sweep_fixed_angle' not in ds.data_vars:
        logger.error("Variable 'sweep_fixed_angle' not found in dataset. Cannot set elevation dimension.")
        raise ValueError("Missing 'sweep_fixed_angle' in input dataset")  # Or handle more gracefully

    # 1. Reindex Azimuth
    azimuth_angles = np.arange(0, 360, azimuth_step)
    try:
        # Check if azimuth dimension exists and is multi-dimensional before reindexing
        if 'azimuth' in ds.dims and ds['azimuth'].ndim > 0:
            ds = ds.reindex(azimuth=azimuth_angles, method="nearest", tolerance=azimuth_step)  # Added tolerance
        else:
            logger.warning("Azimuth dimension not found or is scalar, skipping reindexing.")
    except ValueError as e:
        logger.warning(f"Could not reindex azimuth: {e}. Skipping azimuth reindexing.")

    # 2. Standardize Time Dimension
    if 'time' in ds.coords or 'time' in ds.data_vars:
        try:
            scan_time = ds['time'].values
            first_time_val = np.atleast_1d(scan_time)[0]
            if np.isnat(first_time_val):
                time_minute_floor = np.datetime64(datetime.now().replace(second=0, microsecond=0))
            elif isinstance(first_time_val, np.datetime64):
                time_minute_floor = first_time_val.astype('datetime64[m]')
            else:
                dt_obj = datetime.utcfromtimestamp(first_time_val.item())
                time_minute_floor = np.datetime64(dt_obj.replace(second=0, microsecond=0))

            if 'time' in ds.coords: ds = ds.drop_vars("time", errors='ignore')
            if 'time' in ds.data_vars: ds = ds.drop_vars("time", errors='ignore')

            ds = ds.expand_dims({"time": [time_minute_floor]})
            ds = ds.set_coords("time")
        except Exception as e:
            logger.error(f"Error processing time dimension: {e}. Time dimension might be incorrect.")
            time_minute_floor = np.datetime64(datetime.now().replace(second=0, microsecond=0))
            ds = ds.expand_dims({"time": [time_minute_floor]})
            ds = ds.set_coords("time")
    else:
        logger.warning("Time coordinate/variable not found. Adding current time as fallback.")
        time_minute_floor = np.datetime64(datetime.now().replace(second=0, microsecond=0))
        ds = ds.expand_dims({"time": [time_minute_floor]})
        ds = ds.set_coords("time")

    # 3. Standardize Elevation Dimension
    try:
        elevation_val = np.atleast_1d(ds['sweep_fixed_angle'].values)[0]
        if 'elevation' in ds.coords: ds = ds.drop_vars("elevation", errors='ignore')
        if 'elevation' in ds.data_vars: ds = ds.drop_vars("elevation", errors='ignore')
        ds = ds.expand_dims({"elevation": [elevation_val]})
        ds = ds.set_coords("elevation")
    except Exception as e:
        logger.error(f"Error processing elevation dimension: {e}. Elevation dimension might be incorrect.")
        ds = ds.expand_dims({"elevation": [0.0]})
        ds = ds.set_coords("elevation")

    return ds


def read_scan(filepath: str, variables: Optional[List[str]] = None) -> Optional[xr.Dataset]:
    """
    Reads a single raw radar scan file (e.g., .scnx.gz), preprocesses it,
    and returns it as an xarray.Dataset.

    Args:
        filepath: Path to the radar scan file.
        variables: Optional list of variables to select. If None, attempts to load all.

    Returns:
        A preprocessed xarray.Dataset or None on error.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return None

    azimuth_step = get_setting('radar.azimuth_step', default=0.5)
    ds_raw = None
    try:
        ds_raw = xr.open_dataset(filepath, engine="furuno")
        ds_processed = _preprocess_scan(ds_raw, azimuth_step)

        if variables:
            essential_coords = ['time', 'elevation', 'range', 'azimuth', 'latitude', 'longitude', 'x', 'y']
            vars_to_keep = list(set(variables + [coord for coord in essential_coords if coord in ds_processed]))
            available_vars = [v for v in vars_to_keep if v in ds_processed.variables]

            if not any(v in available_vars for v in variables if v in ds_processed.data_vars):
                logger.error(f"None of the requested data variables {variables} are available in {filepath}")
                ds_processed.close()
                return None

            ds_final = ds_processed[available_vars]
        else:
            ds_final = ds_processed

        logger.info(f"Successfully read and preprocessed scan: {filepath}")
        return ds_final

    except Exception as e:
        logger.error(f"Failed to read or preprocess file {filepath}: {e}", exc_info=True)
        return None
    finally:
        if ds_raw is not None:
            try:
                ds_raw.close()
            except Exception as close_err:
                logger.warning(f"Error closing raw dataset for {filepath}: {close_err}")


def get_scan_elevation(scan_filepath: str) -> Optional[float]:
    """
    Reads a scan file briefly to extract its elevation.

    Args:
        scan_filepath: Path to the radar scan file.

    Returns:
        The elevation angle as a float, or None if extraction fails.
    """
    ds = None
    try:
        # Read minimal data - 'read_scan' handles preprocessing including setting 'elevation'
        ds = read_scan(scan_filepath, variables=None)  # Read metadata & coords
        if ds is not None and 'elevation' in ds.coords:
            elevation = float(ds['elevation'].item())
            logger.debug(f"Extracted elevation {elevation:.2f} from {scan_filepath}")
            return elevation
        elif ds is not None:
            logger.warning(f"Elevation coordinate not found after reading scan: {scan_filepath}")
            return None
        else:
            # read_scan failed, error already logged by it
            return None
    except Exception as e:
        logger.error(f"Failed to read or get elevation from scan {scan_filepath}: {e}", exc_info=True)
        return None
    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass


def get_filepaths_in_range(
        base_dir: str,
        start_dt: datetime,
        end_dt: datetime,
        elevation_filter: Optional[float] = None
) -> List[Tuple[str, datetime]]:
    """
    Finds radar scan file paths within a specified base directory (containing
    {ElevationCode}/{YYYYMMDD} subfolders) that fall within the given datetime range
    and optionally match the elevation filter.

    Args:
        base_dir: The root directory for processed scans.
        start_dt: The start datetime for the search range (inclusive).
        end_dt: The end datetime for the search range (inclusive).
        elevation_filter: Optional specific elevation angle to search for.

    Returns:
        A sorted list of tuples, where each tuple contains (filepath, file_datetime).
        Returns an empty list if base_dir doesn't exist or no files are found.
    """
    if not os.path.isdir(base_dir):
        logger.warning(f"Base directory for searching files not found: {base_dir}")
        return []

    start_date = start_dt.date()
    end_date = end_dt.date()
    matching_files: List[Tuple[str, datetime]] = []
    elevation_codes_to_scan: List[str] = []

    logger.info(f"Searching for files in {base_dir} between {start_dt} and {end_dt}")

    if elevation_filter is not None:
        logger.info(f"Filtering for elevation: {elevation_filter:.2f}°")
        elevation_codes_to_scan.append(f"{int(round(elevation_filter * 100)):03d}")
    else:
        logger.info("Scanning all elevations.")
        try:
            elevation_codes_to_scan = [
                d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
            ]
        except Exception as e:
            logger.error(f"Error listing elevation directories in {base_dir}: {e}")
            return []
        if not elevation_codes_to_scan:
            logger.warning(
                f"No potential elevation code directories found in {base_dir}. Checking for old 'YYYYMMDD' structure...")
            # --- Fallback to Old Structure (Optional & Temporary) ---
            # You might remove this fallback after migration is complete.
            try:
                for dir_name in os.listdir(base_dir):
                    dir_path = os.path.join(base_dir, dir_name)
                    current_dir_date = parse_date_from_dirname(dir_name)
                    if os.path.isdir(dir_path) and current_dir_date:
                        if start_date <= current_dir_date <= end_date:
                            logger.debug(f"Scanning OLD structure directory: {dir_path}")
                            for filename in os.listdir(dir_path):
                                if filename.endswith(".scnx.gz"):
                                    filepath = os.path.join(dir_path, filename)
                                    file_dt = parse_datetime_from_filename(filename)
                                    if file_dt and start_dt <= file_dt <= end_dt:
                                        # If filtering by elevation, we must check it here
                                        if elevation_filter is not None:
                                            scan_elev = get_scan_elevation(filepath)
                                            if scan_elev is not None and abs(scan_elev - elevation_filter) < 0.1:
                                                matching_files.append((filepath, file_dt))
                                        else:  # No elevation filter, add it
                                            matching_files.append((filepath, file_dt))
                if matching_files:
                    logger.warning("Found files in OLD YYYYMMDD structure. Consider running 'radproc reorg-scans'.")
                    matching_files.sort(key=lambda x: x[1])
                    return matching_files
                else:
                    return []  # Nothing found in old or new structure
            except Exception as e:
                logger.error(f"Error scanning for old structure: {e}")
                return []
            # --- End Fallback ---

    logger.debug(f"Will scan elevation codes: {elevation_codes_to_scan}")

    try:
        for elev_code in elevation_codes_to_scan:
            elev_dir_path = os.path.join(base_dir, elev_code)
            if not os.path.isdir(elev_dir_path):
                continue

            for date_dir_name in os.listdir(elev_dir_path):
                date_dir_path = os.path.join(elev_dir_path, date_dir_name)
                if os.path.isdir(date_dir_path):
                    current_dir_date = parse_date_from_dirname(date_dir_name)
                    if current_dir_date and start_date <= current_dir_date <= end_date:
                        logger.debug(f"Scanning directory: {date_dir_path}")
                        for filename in os.listdir(date_dir_path):
                            if filename.endswith(".scnx.gz"):
                                filepath = os.path.join(date_dir_path, filename)
                                file_dt = parse_datetime_from_filename(filename)
                                if file_dt and start_dt <= file_dt <= end_dt:
                                    matching_files.append((filepath, file_dt))
    except Exception as e:
        logger.error(f"Error scanning directories in {base_dir}: {e}", exc_info=True)
        return []

    matching_files.sort(key=lambda x: x[1])
    logger.info(f"Found {len(matching_files)} files in range.")
    return matching_files
# core/data.py

import xarray as xr
import numpy as np
import os
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Callable

from core.utils.helpers import parse_datetime_from_filename, parse_date_from_dirname
from core.config import get_setting # Import config access function

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
         raise ValueError("Missing 'sweep_fixed_angle' in input dataset") # Or handle more gracefully

    # 1. Reindex Azimuth
    azimuth_angles = np.arange(0, 360, azimuth_step)
    try:
        # Check if azimuth dimension exists and is multi-dimensional before reindexing
        if 'azimuth' in ds.dims and ds['azimuth'].ndim > 0:
             ds = ds.reindex(azimuth=azimuth_angles, method="nearest", tolerance=azimuth_step) # Added tolerance
        else:
            logger.warning("Azimuth dimension not found or is scalar, skipping reindexing.")
            # If azimuth is scalar or missing, we might need to handle this case differently
            # depending on the expected input data structure. For now, we proceed.

    except ValueError as e:
        logger.warning(f"Could not reindex azimuth: {e}. Skipping azimuth reindexing.")


    # 2. Standardize Time Dimension
    # Check if time dimension/coordinate exists
    if 'time' in ds.coords or 'time' in ds.data_vars:
        try:
            scan_time = ds['time'].values # Get the time value(s)
            # Handle scalar vs array time - take the first if array
            first_time_val = np.atleast_1d(scan_time)[0]
            # Ensure it's a numpy datetime64 before flooring
            if np.isnat(first_time_val):
                 logger.warning("Time value is NaT (Not a Time). Using current time as fallback.")
                 time_minute_floor = np.datetime64(datetime.now().replace(second=0, microsecond=0))
            elif isinstance(first_time_val, np.datetime64):
                time_minute_floor = first_time_val.astype('datetime64[m]') # Floor to the minute
            else:
                 # Attempt conversion if it's not datetime64 (e.g., cftime)
                 try:
                     dt_obj = datetime.utcfromtimestamp(first_time_val.item()) # Example conversion
                     time_minute_floor = np.datetime64(dt_obj.replace(second=0, microsecond=0))
                 except Exception as conv_err:
                     logger.warning(f"Could not convert time value {first_time_val} to datetime64: {conv_err}. Using current time.")
                     time_minute_floor = np.datetime64(datetime.now().replace(second=0, microsecond=0))

            # Remove original time coord/var if it exists, avoiding errors if it was just a scalar value used
            if 'time' in ds.coords: ds = ds.drop_vars("time", errors='ignore')
            if 'time' in ds.data_vars: ds = ds.drop_vars("time", errors='ignore')

            # Add new time dimension
            ds = ds.expand_dims({"time": [time_minute_floor]})
            ds = ds.set_coords("time") # Ensure time is a coordinate
        except Exception as e:
            logger.error(f"Error processing time dimension: {e}. Time dimension might be incorrect.")
            # Fallback: add current time? Or raise error? Let's try adding current time.
            time_minute_floor = np.datetime64(datetime.now().replace(second=0, microsecond=0))
            ds = ds.expand_dims({"time": [time_minute_floor]})
            ds = ds.set_coords("time")

    else:
        logger.warning("Time coordinate/variable not found. Adding current time as fallback.")
        # Fallback if time doesn't exist at all
        time_minute_floor = np.datetime64(datetime.now().replace(second=0, microsecond=0))
        ds = ds.expand_dims({"time": [time_minute_floor]})
        ds = ds.set_coords("time")


    # 3. Standardize Elevation Dimension
    try:
        # Extract elevation value - handle potential scalar/array
        elevation_val = np.atleast_1d(ds['sweep_fixed_angle'].values)[0]

        # Remove original elevation coord/var if it exists
        if 'elevation' in ds.coords: ds = ds.drop_vars("elevation", errors='ignore')
        if 'elevation' in ds.data_vars: ds = ds.drop_vars("elevation", errors='ignore')

        # Add new elevation dimension
        ds = ds.expand_dims({"elevation": [elevation_val]})
        ds = ds.set_coords("elevation") # Ensure elevation is a coordinate
    except Exception as e:
         logger.error(f"Error processing elevation dimension using 'sweep_fixed_angle': {e}. Elevation dimension might be incorrect.")
         # Decide on fallback? Maybe add a default elevation like 0.0?
         ds = ds.expand_dims({"elevation": [0.0]}) # Example fallback
         ds = ds.set_coords("elevation")


    return ds


def read_scan(filepath: str, variables: Optional[List[str]] = None) -> Optional[xr.Dataset]:
    """
    Reads a single raw radar scan file (e.g., .scnx.gz), preprocesses it,
    and returns it as an xarray.Dataset.

    Args:
        filepath: Path to the radar scan file.
        variables: Optional list of variables to select. If None, attempts to load all.
                   'sweep_fixed_angle' is always needed internally for preprocessing.

    Returns:
        A preprocessed xarray.Dataset containing the requested variables for the scan,
        or None if an error occurs during reading or preprocessing.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return None

    # Get azimuth step from config for preprocessing
    azimuth_step = get_setting('radar.azimuth_step', default=0.5)

    ds_raw = None  # Initialize raw dataset variable
    try:
        # Open the dataset using the furuno engine.
        # Remove 'preprocess' argument as it's not accepted by the backend.
        # Keep 'compat' if it doesn't cause an error, otherwise remove it too.
        # Let's assume 'compat' might be handled by xarray wrapper or accepted by backend.
        ds_raw = xr.open_dataset(
            filepath,
            engine="furuno",
        )

        # --- Apply preprocessing MANUALLY after loading ---
        ds_processed = _preprocess_scan(ds_raw, azimuth_step)
        # --- End Manual Preprocessing ---

        # After preprocessing (dims are set), select the final requested variables
        if variables:
            # Ensure essential coordinates are kept (adjust as needed based on your engine/data)
            essential_coords = ['time', 'elevation', 'range', 'azimuth', 'latitude', 'longitude', 'x', 'y']
            # Add sweep_fixed_angle if it's a coordinate/variable needed downstream
            # and not just for preprocessing elevation dimension
            # essential_coords.append('sweep_fixed_angle')

            vars_to_keep = list(set(variables + [coord for coord in essential_coords if coord in ds_processed]))

            try:
                # Check if vars_to_keep actually exist in the processed dataset
                missing_in_processed = set(vars_to_keep) - set(ds_processed.variables)
                if missing_in_processed:
                     logger.warning(f"Variables {missing_in_processed} not found in dataset *after* preprocessing. Available: {list(ds_processed.variables)}")
                     # Adjust vars_to_keep to only include available ones
                     vars_to_keep = [v for v in vars_to_keep if v in ds_processed.variables]


                ds_final = ds_processed[vars_to_keep] # Create the final dataset subset

            except KeyError as e:
                # This catch might be less likely now with the check above, but keep for safety
                missing_vars = set(vars_to_keep) - set(ds_processed.variables)
                logger.warning(f"Requested/essential variables missing after loading/preprocessing: {missing_vars}. Error: {e}")
                available_vars_to_keep = [v for v in vars_to_keep if v in ds_processed.variables]
                ds_final = ds_processed[available_vars_to_keep] # Return subset of what's available

                if not any(v in ds_final.data_vars for v in variables if v in ds_final.data_vars): # Check if any requested *data* var remains
                     logger.error(f"None of the requested data variables {variables} are available in {filepath}")
                     ds_final.close() # Close the subset
                     ds_processed.close() # Close the intermediate processed dataset
                     # ds_raw was already closed by _preprocess_scan implicit copy or is closed in finally
                     return None
        else:
            # If no specific variables requested, return the whole preprocessed dataset
            ds_final = ds_processed

        logger.info(f"Successfully read and preprocessed scan: {filepath}")
        return ds_final # Return the final dataset (subset or full)

    except FileNotFoundError:
        logger.error(f"File disappeared before processing: {filepath}")
        return None
    except ValueError as ve: # Catch specific errors from _preprocess_scan
        logger.error(f"Preprocessing error for {filepath}: {ve}")
        return None
    except TypeError as te: # Catch potential TypeError from open_dataset (e.g., if 'compat' is also bad)
         logger.error(f"TypeError during dataset opening for {filepath}: {te}. Check engine arguments.", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"Failed to read or preprocess file {filepath}: {e}", exc_info=True) # Log traceback
        return None
    finally:
        # Ensure the raw dataset opened by xr.open_dataset is closed
        # Note: _preprocess_scan might return a modified copy, leaving ds_raw open.
        # ds_processed and ds_final might also need closing if errors occur before return.
        # However, returning the dataset object means the caller is responsible for closing it.
        # We mainly need to ensure ds_raw is closed if an error happens *after* it's opened
        # but *before* ds_final is successfully returned.
        if ds_raw is not None:
            try:
                # Check if ds_final exists and IS ds_raw (meaning no variables selected AND _preprocess didn't make a copy)
                # This is complex. Simplest robust approach: always close ds_raw here if it was opened.
                # The returned object ds_final (which might be ds_processed or a subset) is the caller's responsibility.
                ds_raw.close()
                # logger.debug(f"Closed raw dataset for {filepath}") # Optional debug logging
            except Exception as close_err:
                logger.warning(f"Error closing raw dataset for {filepath}: {close_err}", exc_info=True)


def get_filepaths_in_range(base_dir: str, start_dt: datetime, end_dt: datetime) -> List[Tuple[str, datetime]]:
    """
    Finds radar scan file paths within a specified base directory (containing YYYYMMDD subfolders)
    that fall within the given datetime range.

    Args:
        base_dir: The root directory containing date-structured subfolders (e.g., .../processed_data/).
        start_dt: The start datetime for the search range (inclusive).
        end_dt: The end datetime for the search range (inclusive).

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

    logger.info(f"Searching for files in {base_dir} between {start_dt} and {end_dt}")

    try:
        # Iterate through items in the base directory
        for dir_name in os.listdir(base_dir):
            dir_path = os.path.join(base_dir, dir_name)

            # Check if it's a directory and parse its date
            if os.path.isdir(dir_path):
                current_dir_date = parse_date_from_dirname(dir_name)

                # Skip directory if its date is outside the desired range
                if current_dir_date is None or current_dir_date < start_date or current_dir_date > end_date:
                    continue

                # If the directory date is within the range, scan its files
                logger.debug(f"Scanning directory: {dir_path}")
                for filename in os.listdir(dir_path):
                    # Assuming files are like *.scnx.gz - adjust if needed
                    if filename.endswith(".scnx.gz"):
                        filepath = os.path.join(dir_path, filename)
                        file_dt = parse_datetime_from_filename(filename)

                        # If datetime parsed correctly and is within the full start/end range
                        if file_dt and start_dt <= file_dt <= end_dt:
                            matching_files.append((filepath, file_dt))

    except Exception as e:
        logger.error(f"Error scanning directories in {base_dir}: {e}", exc_info=True)
        return [] # Return empty list on error during scan

    # Sort files chronologically before returning
    matching_files.sort(key=lambda x: x[1])

    logger.info(f"Found {len(matching_files)} files in range.")
    return matching_files

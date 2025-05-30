# core/data.py

import xarray as xr
import numpy as np
import os
import logging
import warnings
from datetime import datetime, timezone
import pandas as pd
from typing import List, Optional, Tuple

from .utils.helpers import parse_datetime_from_filename  # Keep for nominal_ts
from .config import get_setting
from .utils.helpers import parse_scan_sequence_number

logger = logging.getLogger(__name__)


def _preprocess_scan(ds: xr.Dataset, azimuth_step: float) -> xr.Dataset:
    """
    Internal function to preprocess a single radar scan Dataset.
    - Determines a single representative time for the scan.
    - Reindexes azimuth to a regular grid.
    - Sets a single time dimension/coordinate (floored to the minute).
    - Sets elevation dimension using 'sweep_fixed_angle'.

    Args:
        ds: The input xarray Dataset from xr.open_dataset (will be copied).
        azimuth_step: The desired regular azimuth step in degrees.

    Returns:
        The preprocessed xarray Dataset.
    Raises:
        ValueError: If a valid representative time cannot be determined.
    """
    # --- Step 0: Determine the single representative timestamp for the entire scan ---
    # This should be done BEFORE azimuth reindexing, from the original time values.
    # We'll use the first valid (non-NaT) time value from the original dataset.

    py_dt_representative = None
    source_file_path = ds.encoding.get("source", None)
    filename_for_logs = os.path.basename(source_file_path) if source_file_path else "unknown_file"

    if 'time' in ds.coords and ds['time'].size > 0:
        # Find the first non-NaT time in the original time array
        original_time_values = np.atleast_1d(ds['time'].values)
        valid_time_indices = np.where(np.isnat(original_time_values) == False)[0]

        if valid_time_indices.size > 0:
            first_valid_time_val = original_time_values[valid_time_indices[0]]
            pd_ts = pd.Timestamp(first_valid_time_val)
            with warnings.catch_warnings():  # Suppress nanosecond warning
                warnings.filterwarnings("ignore", message="Discarding nonzero nanoseconds in conversion.",
                                        category=UserWarning)
                py_dt_representative = pd_ts.to_pydatetime()

            if py_dt_representative.tzinfo is None:
                py_dt_representative = py_dt_representative.replace(tzinfo=timezone.utc)
            else:
                py_dt_representative = py_dt_representative.astimezone(timezone.utc)
            logger.debug(
                f"Determined representative scan time for {filename_for_logs} from original data: {py_dt_representative.isoformat()}")
        else:
            logger.warning(f"All internal 'time' values are NaT for {filename_for_logs}.")
    else:
        logger.warning(f"'time' coordinate not found or empty in original data for {filename_for_logs}.")

    # Fallback to filename time if representative time couldn't be found from data
    if py_dt_representative is None and source_file_path:
        nominal_dt_from_filename = parse_datetime_from_filename(filename_for_logs)
        if nominal_dt_from_filename:
            logger.info(
                f"Using nominal timestamp from filename as representative time for {filename_for_logs}: {nominal_dt_from_filename.isoformat()}")
            py_dt_representative = nominal_dt_from_filename

    if py_dt_representative is None:
        logger.error(f"Cannot determine a valid representative time for scan {filename_for_logs}. Cannot preprocess.")
        raise ValueError(f"Cannot determine a valid representative time for {filename_for_logs}")

    # This is the single, floored UTC time that will represent the whole scan
    representative_time_minute_floor_utc = py_dt_representative.replace(second=0, microsecond=0)

    with warnings.catch_warnings():  # Suppress np.datetime64 timezone warning
        warnings.filterwarnings("ignore", message="no explicit representation of timezones available for np.datetime64",
                                category=UserWarning)
        np_representative_time_minute_floor = np.datetime64(representative_time_minute_floor_utc)

    # --- Actual Dataset Modification Starts Here (use a copy) ---
    ds_processed = ds.copy()

    # --- Step 1: Reindex Azimuth ---
    if 'sweep_fixed_angle' not in ds_processed.coords and 'sweep_fixed_angle' not in ds_processed.data_vars:
        logger.error(f"Variable 'sweep_fixed_angle' not found in {filename_for_logs}. Cannot set elevation.")
        raise ValueError("Missing 'sweep_fixed_angle'")

    azimuth_angles = np.arange(0, 360, azimuth_step)
    if 'azimuth' in ds_processed.dims and ds_processed['azimuth'].ndim > 0:
        try:
            # Reindex. Note that ds_processed['time'] might get NaTs here if it was a dependent coord.
            ds_processed = ds_processed.reindex(azimuth=azimuth_angles, method="nearest", tolerance=azimuth_step)
        except ValueError as e:
            logger.warning(f"Could not reindex azimuth for {filename_for_logs}: {e}. Skipping reindex step.")
    else:
        logger.warning(f"Azimuth dimension not found or is scalar in {filename_for_logs}, skipping reindexing.")

    # --- Step 2: Standardize Time Dimension ---
    # Remove the old 'time' coordinate (which might now have NaTs or be multi-valued)
    # and replace it with the single representative time.
    if 'time' in ds_processed.coords:
        ds_processed = ds_processed.drop_vars("time", errors='ignore')
    if 'time' in ds_processed.data_vars:  # Should not happen, but defensive
        ds_processed = ds_processed.drop_vars("time", errors='ignore')

    ds_processed = ds_processed.expand_dims({"time": [np_representative_time_minute_floor]})
    ds_processed = ds_processed.set_coords("time")

    # --- Step 3: Standardize Elevation Dimension ---
    try:
        elevation_val = np.atleast_1d(ds_processed['sweep_fixed_angle'].values)[0]
        if 'elevation' in ds_processed.coords: ds_processed = ds_processed.drop_vars("elevation", errors='ignore')
        if 'elevation' in ds_processed.data_vars: ds_processed = ds_processed.drop_vars("elevation", errors='ignore')
        ds_processed = ds_processed.expand_dims({"elevation": [float(elevation_val)]})
        ds_processed = ds_processed.set_coords("elevation")
    except Exception as e:
        logger.error(f"Error processing elevation for {filename_for_logs}: {e}. Using 0.0 fallback.")
        ds_processed = ds_processed.expand_dims({"elevation": [0.0]}).set_coords("elevation")

    return ds_processed

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
        ds_processed = _preprocess_scan(ds_raw.copy(),
                                        azimuth_step)  # Pass a copy to avoid modifying ds_raw if it's used later in finally
        if variables:
            essential_coords = ['time', 'elevation', 'range', 'azimuth', 'latitude', 'longitude', 'x', 'y']
            vars_to_keep = list(set(variables + [coord for coord in essential_coords if coord in ds_processed]))
            available_vars = [v for v in vars_to_keep if v in ds_processed]
            if not any(v in available_vars for v in variables if v in ds_processed.data_vars):  # Check actual data vars
                logger.error(f"None of requested data variables {variables} available in {filepath} after processing.")
                ds_processed.close()
                return None
            ds_final = ds_processed[available_vars]
        else:
            ds_final = ds_processed
        logger.info(f"Successfully read and preprocessed scan: {filepath}")
        return ds_final
    except Exception as e:
        logger.error(f"Failed to read/preprocess {filepath}: {e}", exc_info=True)
        if ds_processed is not None and ds_final is not ds_processed: ds_processed.close()  # type: ignore
        return None
    finally:
        if ds_raw is not None: ds_raw.close()


def extract_scan_key_metadata(scan_filepath: str) -> Optional[Tuple[datetime, float, int]]:
    """
    Reads a scan file to extract its precise internal timestamp (UTC),
    elevation, and scan sequence number (_N from filename).

    Returns:
        Tuple (precise_timestamp_utc, elevation, sequence_number) or None if extraction fails.
    """
    ds_raw = None
    try:
        filename = os.path.basename(scan_filepath)
        sequence_number = parse_scan_sequence_number(filename)
        if sequence_number is None:
            logger.error(f"Sequence number is required for {filename}. Cannot extract metadata.")
            return None

        ds_raw = xr.open_dataset(scan_filepath, engine="furuno")

        actual_scan_time_from_data: Optional[datetime] = None
        if 'time' in ds_raw.coords and ds_raw['time'].size > 0:
            original_time_values = np.atleast_1d(ds_raw['time'].values)
            valid_time_indices = np.where(np.isnat(original_time_values) == False)[0]
            if valid_time_indices.size > 0:
                first_valid_time_val = original_time_values[valid_time_indices[0]]
                pd_ts = pd.Timestamp(first_valid_time_val)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Discarding nonzero nanoseconds in conversion.",
                                            category=UserWarning)
                    py_dt_from_raw = pd_ts.to_pydatetime()

                actual_scan_time_from_data = py_dt_from_raw.replace(
                    tzinfo=timezone.utc) if py_dt_from_raw.tzinfo is None \
                    else py_dt_from_raw.astimezone(timezone.utc)
            else:
                logger.warning(f"All internal 'time' values in {scan_filepath} are NaT.")
        else:
            logger.warning(f"'time' coordinate not found or empty in raw scan data for {scan_filepath}.")

        if actual_scan_time_from_data is None:  # If still None, try filename as last resort for precise_timestamp
            logger.warning(
                f"Could not determine precise time from internal data of {scan_filepath}. Attempting to use filename time.")
            actual_scan_time_from_data = parse_datetime_from_filename(filename)  # This is already UTC aware
            if actual_scan_time_from_data:
                logger.info(
                    f"Used filename time as precise_timestamp for {scan_filepath}: {actual_scan_time_from_data.isoformat()}")
            else:
                logger.error(
                    f"Internal time is invalid AND could not parse time from filename for {scan_filepath}. Cannot index accurately.")
                return None

        elevation: Optional[float] = None
        if 'sweep_fixed_angle' in ds_raw.coords and ds_raw['sweep_fixed_angle'].size > 0:
            elevation = float(np.atleast_1d(ds_raw['sweep_fixed_angle'].values)[0])
        elif 'elevation' in ds_raw.coords and ds_raw['elevation'].size > 0:
            elevation = float(np.atleast_1d(ds_raw['elevation'].values)[0])

        if elevation is None:
            logger.error(f"Elevation information not found in {scan_filepath}.")
            return None

        logger.debug(
            f"Extracted from {scan_filepath}: PreciseTS={actual_scan_time_from_data.isoformat()}, Elev={elevation:.2f}, SeqNum={sequence_number}")
        return actual_scan_time_from_data, elevation, sequence_number

    except Exception as e:
        logger.error(f"Failed to open or extract key metadata from scan {scan_filepath}: {e}", exc_info=True)
        return None
    finally:
        if ds_raw is not None:
            ds_raw.close()

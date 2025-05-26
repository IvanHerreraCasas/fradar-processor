# core/data.py

import xarray as xr
import numpy as np
import os
import logging
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
        logger.error("Variable 'sweep_fixed_angle' not found. Cannot set elevation dimension.")
        raise ValueError("Missing 'sweep_fixed_angle'")

    # 1. Reindex Azimuth
    azimuth_angles = np.arange(0, 360, azimuth_step)
    if 'azimuth' in ds.dims and ds['azimuth'].ndim > 0:
        try:
            ds = ds.reindex(azimuth=azimuth_angles, method="nearest", tolerance=azimuth_step)
        except ValueError as e:
            logger.warning(f"Could not reindex azimuth: {e}. Skipping.")
    else:
        logger.warning("Azimuth dimension not found or is scalar, skipping reindexing.")

    # 2. Standardize Time Dimension
    if 'time' in ds.coords or 'time' in ds.data_vars:
        try:
            scan_time_val = np.atleast_1d(ds['time'].values)[0]
            if np.isnat(scan_time_val):
                py_dt = datetime.now(timezone.utc)
            else:
                py_dt = pd.to_datetime(scan_time_val).to_pydatetime()

            time_minute_floor_utc = (py_dt.replace(tzinfo=timezone.utc) if py_dt.tzinfo is None \
                                         else py_dt.astimezone(timezone.utc)).replace(second=0, microsecond=0)

            np_time_minute_floor = np.datetime64(time_minute_floor_utc)

            if 'time' in ds.coords: ds = ds.drop_vars("time", errors='ignore')
            if 'time' in ds.data_vars: ds = ds.drop_vars("time", errors='ignore')
            ds = ds.expand_dims({"time": [np_time_minute_floor]})
            ds = ds.set_coords("time")
        except Exception as e:
            logger.error(f"Error processing time dimension: {e}. Using current time fallback.")
            np_time_fallback = np.datetime64(datetime.now(timezone.utc).replace(second=0, microsecond=0))
            ds = ds.expand_dims({"time": [np_time_fallback]}).set_coords("time")
    else:
        logger.warning("Time coordinate not found. Adding current time fallback.")
        np_time_fallback = np.datetime64(datetime.now(timezone.utc).replace(second=0, microsecond=0))
        ds = ds.expand_dims({"time": [np_time_fallback]}).set_coords("time")

    # 3. Standardize Elevation Dimension
    try:
        elevation_val = np.atleast_1d(ds['sweep_fixed_angle'].values)[0]
        if 'elevation' in ds.coords: ds = ds.drop_vars("elevation", errors='ignore')
        if 'elevation' in ds.data_vars: ds = ds.drop_vars("elevation", errors='ignore')
        ds = ds.expand_dims({"elevation": [float(elevation_val)]})  # Ensure float
        ds = ds.set_coords("elevation")
    except Exception as e:
        logger.error(f"Error processing elevation: {e}. Using 0.0 fallback.")
        ds = ds.expand_dims({"elevation": [0.0]}).set_coords("elevation")
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
    ds = None
    try:
        filename = os.path.basename(scan_filepath)
        sequence_number = parse_scan_sequence_number(filename)
        if sequence_number is None:
            logger.warning(
                f"Could not parse sequence number from filename: {filename}. Cannot include in scan log accurately.")
            # Decide if this is fatal. For now, let's allow proceeding without seq num if we must.
            # Or return None here to enforce sequence number presence:
            # return None
            # For now, we'll make it required for robust volume grouping.
            if sequence_number is None:  # Re-check, explicit for clarity
                logger.error(f"Sequence number is required. Could not parse from {filename}.")
                return None

        # Use read_scan with variables=None to get coordinates including time and elevation.
        ds = read_scan(scan_filepath, variables=None)
        if ds is not None and 'time' in ds.coords and 'elevation' in ds.coords:
            time_val_np64 = ds['time'].values.item()
            py_dt = pd.to_datetime(time_val_np64).to_pydatetime()
            precise_ts_utc = py_dt.replace(tzinfo=timezone.utc) if py_dt.tzinfo is None \
                else py_dt.astimezone(timezone.utc)

            elevation = float(ds['elevation'].item())

            logger.debug(
                f"Extracted from {scan_filepath}: PreciseTS={precise_ts_utc.isoformat()}, Elev={elevation:.2f}, SeqNum={sequence_number}")
            return precise_ts_utc, elevation, sequence_number
        elif ds is not None:
            logger.warning(f"Precise time or elevation coordinate not found after reading scan: {scan_filepath}")
            return None
        else:  # read_scan failed
            return None
    except Exception as e:
        logger.error(f"Failed to read or get precise data from scan {scan_filepath}: {e}", exc_info=True)
        return None
    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass

# DEPRECATE get_filepaths_in_range:
# def get_filepaths_in_range(...)
# This function's role is now primarily handled by querying radproc_scan_log
# via db_manager.query_scan_log_for_timeseries_processing()
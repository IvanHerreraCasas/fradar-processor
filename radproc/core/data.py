# radproc/core/data.py

import xarray as xr
import xradar as xd
import numpy as np
import os
import logging
import warnings
from datetime import datetime, timezone
import pandas as pd
from typing import List, Optional, Tuple

from xarray.core import datatree

from .utils.helpers import parse_datetime_from_filename  # Keep for nominal_ts
from .config import get_setting
from .utils.helpers import parse_scan_sequence_number

logger = logging.getLogger(__name__)

def reindex_ppi(ds: xr.Dataset, azimuth_step: float = 0.5) -> xr.Dataset:
    """
    Reindexes a PPI scan to a regular azimuth grid, handling the 0/360 degree wrap-around.

    Args:
        ds: The input xarray Dataset containing the PPI scan. It must have an
            'azimuth' coordinate.
        azimuth_step: The desired angular resolution in degrees for the new grid.

    Returns:
        A new xarray Dataset reindexed to the regular grid.
    """
    # 1. Define the ideal, uniform target grid
    target_azimuths = np.arange(0, 360, azimuth_step)

    # Ensure the original azimuth coordinate is sorted
    ds = ds.sortby('azimuth')
    original_azimuths = ds.azimuth

    # 2. Create the "wrapped" data to handle the seam
    end_of_scan = ds.where(original_azimuths > (360 - 10))
    end_of_scan['azimuth'] = end_of_scan['azimuth'] - 360

    start_of_scan = ds.where(original_azimuths < 10)
    start_of_scan['azimuth'] = start_of_scan['azimuth'] + 360

    # 3. Combine original and wrapped data
    combined_ds = xr.concat([ds, end_of_scan, start_of_scan], dim='azimuth')

    # 4. Sort the combined data to create a monotonic index
    combined_ds = combined_ds.sortby('azimuth')

    # 5. Perform the reindexing on the now-sorted combined dataset
    new_ds = combined_ds.reindex(
        azimuth=target_azimuths,
        method="nearest",
        tolerance=azimuth_step
    )

    return new_ds

def preprocess_scan(ds: xr.Dataset, azimuth_step: float, for_volume: bool = False) -> Optional[xr.Dataset]:
    """
    Internal function to preprocess a single radar scan Dataset.

    If for_volume=False:
        - Reindexes azimuth to a regular grid.
        - Sets time dimension to the floored minute of the first time value.
        - Sets elevation dimension using the first 'sweep_fixed_angle' value.
    If for_volume=True:
        - Only reindexes azimuth, preserving per-azimuth time and elevation.
    """
    ds_processed = ds.copy()

    # --- Step 1: Reindex Azimuth (Common to both modes) ---
    try:
        if 'azimuth' in ds_processed.dims:
            ds_processed = reindex_ppi(ds_processed, azimuth_step)
    except Exception as e:
        logger.error(f"Could not reindex azimuth: {e}. Skipping this sweep.", exc_info=True)
        return None

    # --- Step 2: Apply different logic based on the context ---
    if not for_volume:
        # --- SINGLE-SCAN MODE: Revert to the old, robust simplification logic ---
        logger.debug("Preprocessing for single-scan mode (collapsing coordinates).")

        # 2a. Standardize Time Dimension
        try:
            time_minute_floor = ds_processed.time.dt.floor("1min").min().values

            if 'time' in ds_processed.coords: ds_processed = ds_processed.drop_vars("time", errors='ignore')
            ds_processed = ds_processed.expand_dims({"time": [time_minute_floor]})
            ds_processed = ds_processed.set_coords("time")
        except Exception as e:
            logger.error(f"Error processing time dimension in single-scan mode: {e}.")
            return None

        logger.error("test")

        # 2b. Standardize Elevation Dimension
        try:
            elevation_val = np.atleast_2d(ds_processed['sweep_fixed_angle'].values)[0,0]
            if 'elevation' in ds_processed.coords: ds_processed = ds_processed.drop_vars("elevation", errors='ignore')
            ds_processed = ds_processed.expand_dims({"elevation": [float(elevation_val)]})
            ds_processed = ds_processed.set_coords("elevation")
        except Exception as e:
            logger.error(f"Error processing elevation dimension: {e}.")
            ds_processed = ds_processed.expand_dims({"elevation": [0.0]}).set_coords("elevation")

    else:
        # --- VOLUME MODE: Preserve details ---
        logger.debug("Preprocessing for volume mode (preserving coordinates).")
        # For volume creation, we need the detailed per-azimuth time,
        # so we do nothing to collapse it. We also leave elevation alone.
        pass

    return ds_processed


def read_ppi_scan(filepath: str, variables: Optional[List[str]] = None, for_volume: bool = False) -> Optional[
    xr.Dataset]:
    """
    Reads a single raw Furuno PPI scan file (.scnx.gz).

    Args:
        filepath: The full path to the raw scan file.
        variables: A list of specific variables to load.
        for_volume: If True, skips preprocessing steps that are incompatible
                    with multi-sweep volume creation.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found for reading: {filepath}")
        return None

    azimuth_step = get_setting('radar.azimuth_step', default=0.5)

    try:
        # Load the raw sweep data
        ds = xr.open_dataset(filepath, engine='furuno', group='sweep_0', decode_times=True)
        # Pass the for_volume flag to the preprocessor
        ds_processed = preprocess_scan(ds, azimuth_step=azimuth_step, for_volume=for_volume)

        logger.info(f"Successfully read PPI scan '{os.path.basename(filepath)}'")
        return ds_processed
    except Exception as e:
        logger.error(f"Failed to read file {filepath} with engine 'furuno': {e}", exc_info=True)
        return None

def read_volume_from_cfradial(filepath: str) -> Optional[datatree.DataTree]:
    """
    Reads a multi-sweep CfRadial2 volume file into a DataTree object.

    Args:
        filepath: The full path to the CfRadial2 (.nc) file.

    Returns:
        A datatree.DataTree object where each child node represents a sweep.
    """
    if not os.path.exists(filepath):
        logger.error(f"CfRadial volume file not found: {filepath}")
        return None
    if not filepath.endswith('.nc'):
        logger.error(f"Unsupported file type for read_volume_from_cfradial: {filepath}")
        return None

    try:
        # Use xradar's dedicated datatree opener for CfRadial2 files
        tree = xd.io.open_cfradial1_datatree(filepath)
        logger.info(f"Successfully read volume '{os.path.basename(filepath)}' into datatree.")
        return tree
    except Exception as e:
        logger.error(f"Failed to read CfRadial2 file {filepath}: {e}")
        return None

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

def extract_point_value(ds: xr.Dataset, variable: str, az_idx: int, rg_idx: int) -> Tuple[float, float]:
    """
    Extracts data value and height for a variable at given azimuth/range indices. Assumes single time step.
    Returns (np.nan, np.nan) on failure.
    """
    try:
        if variable not in ds.data_vars:
            logger.warning(f"Variable '{variable}' not found in dataset for value extraction.")
            return np.nan, np.nan

        data_point = ds[variable].isel(azimuth=az_idx, range=rg_idx)
        value = data_point.compute().item() if hasattr(data_point.data, 'compute') else data_point.item()

        height = np.nan
        if 'z' in ds.coords:
            height = ds.isel(azimuth=az_idx, range=rg_idx)['z'].item()

        if isinstance(value, (int, float, np.number)) and not np.isnan(value):
            return float(value), float(height)

        return np.nan, np.nan
    except IndexError:
        logger.error(f"Indices az={az_idx}, rg={rg_idx} out of bounds for var '{variable}'.")
        return np.nan, np.nan
    except Exception as e:
        logger.error(f"Failed to extract value for '{variable}' at az={az_idx}, rg={rg_idx}: {e}", exc_info=True)
        return np.nan, np.nan
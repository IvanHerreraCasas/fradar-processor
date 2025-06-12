# radproc/core/volume_processor.py

import os
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import pyart
import xarray as xr
from xarray import DataTree

import pandas as pd
import numpy as np

from .retrievals import apply_corrections
from .data import read_ppi_scan
from .db_manager import get_connection, release_connection, add_processed_volume_log, get_scan_paths_for_volume
from .config import get_setting
from .utils.helpers import parse_scan_sequence_number

logger = logging.getLogger(__name__)


def create_volume_from_files(
    scans_with_elevation: List[Tuple[str, float]],  version: str) -> Optional[pyart.core.Radar]:
    """
    Takes a list of raw scan file paths, processes them into a single volume,
    applies corrections, applies subsetting based on the config, and returns a final Py-ART Radar object.

    This function is self-contained and does not interact with the database,
    making it suitable for testing in a notebook.

    Args:
        scan_filepaths: An ordered list of file paths for the scans in the volume.
        version: The processing version to apply (e.g., 'v1_0').

    Returns:
        A corrected Py-ART Radar object, or None if processing fails.
    """
    logger.info(f"Creating volume with subsetting for version '{version}'.")
    config = get_setting(f'corrections.{version}')
    subset_config = config.get('subsetting', {})

    # 1. Filter Scans by Elevation
    elevations_to_keep = subset_config.get('elevations_to_keep')
    if elevations_to_keep:
        # Create a set for efficient lookup
        keep_elevs = set(elevations_to_keep)
        # Filter the list, allowing for a small tolerance
        scan_paths = [
            path for path, elev in scans_with_elevation
            if any(abs(elev - keep_elev) < 0.1 for keep_elev in keep_elevs)
        ]
        logger.info(f"Filtered to {len(scan_paths)} sweeps based on 'elevations_to_keep' config.")
    else:
        # If not specified, keep all sweeps
        scan_paths = [path for path, elev in scans_with_elevation]

    if not scan_paths:
        logger.warning("No scans remain after elevation filtering.")
        return None

    # 2. Determine Variables to Load
    variables_to_keep = subset_config.get('variables_to_keep', [])
    # Make sure we always load fields required for corrections (e.g., KDP, ZDR)
    # This is a placeholder for more intelligent logic that would parse the
    # full correction config to determine dependencies.
    required_for_corrections = ['KDP', 'ZDR', 'DBZH']
    variables_to_load = list(set(variables_to_keep + required_for_corrections))
    logger.info(f"Attempting to load variables: {variables_to_load}")

    # 1. Read each sweep using our "volume-aware" reader
    sweep_datasets = []
    for path in scan_paths:
        ds = read_ppi_scan(path, for_volume=True)
        if ds:
            # Transform sweep_number to a scalar variable
            sweep_num = parse_scan_sequence_number(os.path.basename(path))
            if sweep_num is not None:
                ds['sweep_number'] = xr.DataArray(sweep_num)
            sweep_datasets.append(ds)

    if not sweep_datasets:
        logger.error("No scans could be successfully read for this volume.")
        return None

    # 3. Manually construct the complete DataTree object
    first_sweep = sweep_datasets[0]
    last_sweep = sweep_datasets[-1]

    root_attrs = {
        "Conventions": "Cf/Radial", "history": f"Created by RadProc {version}",
        "time_coverage_start": f"{pd.to_datetime(first_sweep.time.min().values).isoformat()}Z",
        "time_coverage_end": f"{pd.to_datetime(last_sweep.time.max().values).isoformat()}Z",
    }
    root_ds = xr.Dataset(
        coords={"latitude": first_sweep.latitude, "longitude": first_sweep.longitude, "altitude": first_sweep.altitude},
        attrs=root_attrs)

    radar_params_ds = xr.Dataset(coords=root_ds.coords)
    georef_ds = xr.Dataset(coords=root_ds.coords)
    calib_vars = {key: first_sweep.attrs[key] for key in ['tx_power_h', 'antenna_gain_v'] if key in first_sweep.attrs}
    radar_calib_ds = xr.Dataset(calib_vars)

    tree_dict = {
        "/": root_ds, "/radar_parameters": radar_params_ds,
        "/georeferencing_correction": georef_ds, "/radar_calibration": radar_calib_ds,
    }
    for i, sweep_ds in enumerate(sweep_datasets):
        tree_dict[f"/sweep_{i}"] = sweep_ds

    volume_tree = DataTree.from_dict(tree_dict)

    # 4. Convert the datatree to a Py-ART Radar object
    logger.info("Converting datatree to Py-ART Radar object...")
    combined_radar = volume_tree.pyart.to_radar()

    # 5. Apply advanced corrections
    logger.info(f"Applying corrections for version '{version}'...")
    corrected_volume = apply_corrections(combined_radar, version=version)

    # 6. Add missing optional attributes for writer compatibility.
    # Scalar attributes, set to None if they don't exist
    optional_scalar_attrs = [
        'scan_rate', 'antenna_transition', 'altitude_agl', 'target_scan_rate'
    ]
    for attr in optional_scalar_attrs:
        if not hasattr(corrected_volume, attr):
            setattr(corrected_volume, attr, None)

    # Per-sweep attributes, set to None if they don't exist
    optional_sweep_attrs = ['rays_are_indexed', 'ray_angle_res']
    for attr in optional_sweep_attrs:
        if not hasattr(corrected_volume, attr):
            setattr(corrected_volume, attr, None)

    # Moving platform attributes (for fixed platforms, these are None)
    moving_platform_attrs = [
        'rotation', 'tilt', 'roll', 'drift', 'heading', 'pitch', 'georefs_applied'
    ]
    for attr in moving_platform_attrs:
        if not hasattr(corrected_volume, attr):
            setattr(corrected_volume, attr, None)

    # Dictionary attributes, set to an empty dict if they don't exist
    optional_dict_attrs = ['instrument_parameters', 'radar_calibration']
    for attr in optional_dict_attrs:
        if not hasattr(corrected_volume, attr) or getattr(corrected_volume, attr) is None:
            # If the attribute is missing, create it from the DataTree if possible,
            # otherwise create an empty dictionary.
            if attr == 'radar_calibration' and 'radar_calibration' in volume_tree:
                calib_dict = {}
                calib_ds = volume_tree['radar_calibration'].ds
                for var_name in calib_ds.data_vars:
                    calib_dict[var_name] = {
                        'data': np.array([calib_ds[var_name].item()])
                    }
                setattr(corrected_volume, attr, calib_dict)
            else:
                setattr(corrected_volume, attr, {})  # Set to empty dict as a fallback

    # 7. Final Subsetting of Fields
    if variables_to_keep:
        final_fields_to_keep = set(variables_to_keep)
        # Get a list of all current field names to iterate over
        current_field_names = list(corrected_volume.fields.keys())
        for field_name in current_field_names:
            if field_name not in final_fields_to_keep:
                corrected_volume.fields.pop(field_name)
                logger.debug(f"Removed field not in 'variables_to_keep': {field_name}")

    # --- Step 5: FINAL OPTIMIZATION - Apply Quantization ---

    quantization_config =  config.get('quantization', {})
    if quantization_config:
        logger.info("Applying quantization to fields for file size reduction...")
        for field_name, n_digits in quantization_config.items():
            if field_name in corrected_volume.fields:
                # This special key tells the netCDF4 writer to quantize the data.
                corrected_volume.fields[field_name]['_Least_significant_digit'] = n_digits
                logger.debug(f"Set _Least_significant_digit={n_digits} for field '{field_name}'")

    logger.info("Radar object prepared and corrected with all optional attributes. Ready for writing.")

    return corrected_volume


def process_volume(volume_identifier: datetime, version: str = 'v1_0') -> bool:
    """
    Orchestrates the processing of a complete volume of radar scans. It fetches
    file paths from the database, calls the core processing function, saves
    the resulting file, and logs it back to the database.
    """
    logger.info(f"--- Starting Volume Processing for ID: {volume_identifier.isoformat()} ---")

    conn = get_connection()
    try:
        # 1. Get file paths from the database
        scan_paths = get_scan_paths_for_volume(conn, volume_identifier)
        if not scan_paths:
            logger.error(f"No scan paths found for volume_identifier {volume_identifier.isoformat()}.")
            return False

        # 2. Call the core, self-contained function to get the corrected radar object
        corrected_radar = create_volume_from_files(scan_paths, version)

        if not corrected_radar:
            logger.error(f"Core volume creation failed for volume ID {volume_identifier.isoformat()}.")
            return False

        # 3. Save the corrected volume as a single CfRadial2 file
        cfradial_dir = get_setting('app.cfradial_dir')
        if not cfradial_dir:
            logger.error("'app.cfradial_dir' not configured.")
            return False

        date_str = volume_identifier.strftime("%Y%m%d")
        cfradial_version_dir = os.path.join(cfradial_dir, version, date_str)
        os.makedirs(cfradial_version_dir, exist_ok=True)

        time_str = volume_identifier.strftime("%Y%m%d_%H%M%S")
        cfradial_filename = f"volume_{time_str}.nc"
        cfradial_filepath = os.path.join(cfradial_version_dir, cfradial_filename)

        logger.info(f"Saving corrected volume to {cfradial_filepath} as CfRadial2...")
        pyart.io.write_cfradial(cfradial_filepath, corrected_radar)
        logger.info("Corrected volume saved successfully.")

        # 4. Log the new volume file to the database
        add_processed_volume_log(conn, volume_identifier, cfradial_filepath, version)

    except Exception as e:
        logger.error(f"An unexpected error occurred during volume processing for ID {volume_identifier.isoformat()}: {e}", exc_info=True)
        return False
    finally:
        release_connection(conn)

    logger.info(f"--- Finished Volume Processing for ID: {volume_identifier.isoformat()} ---")
    return True
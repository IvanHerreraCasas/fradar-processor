# radproc/core/volume_processor.py

import os
import logging
from datetime import datetime
from typing import List, Optional

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


def create_volume_from_files(scan_filepaths: List[str], version: str) -> Optional[pyart.core.Radar]:
    """
    Takes a list of raw scan file paths, processes them into a single volume,
    applies corrections, and returns a final Py-ART Radar object.

    This function is self-contained and does not interact with the database,
    making it suitable for testing in a notebook.

    Args:
        scan_filepaths: An ordered list of file paths for the scans in the volume.
        version: The processing version to apply (e.g., 'v1_0').

    Returns:
        A corrected Py-ART Radar object, or None if processing fails.
    """
    logger.info(f"Creating volume from {len(scan_filepaths)} files for version '{version}'.")

    # 1. Read each sweep using our "volume-aware" reader
    sweep_datasets = []
    for path in scan_filepaths:
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

    # 2. Manually construct the complete DataTree object
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

    # 3. Convert the datatree to a Py-ART Radar object
    logger.info("Converting datatree to Py-ART Radar object...")
    combined_radar = volume_tree.pyart.to_radar()

    # 4. Apply advanced corrections
    logger.info(f"Applying corrections for version '{version}'...")
    corrected_volume = apply_corrections(combined_radar, version=version)

    # 5. Add missing optional attributes for writer compatibility.
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
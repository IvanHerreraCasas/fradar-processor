# radproc/core/volume_processor.py

import os
import logging
from datetime import datetime
from typing import List

import pyart
from xarray.core import datatree

from . import preprocessing
from .data import read_ppi_scan
from .db_manager import get_connection, release_connection, add_processed_volume_log, get_scan_paths_for_volume
from .config import get_setting

logger = logging.getLogger(__name__)


def process_volume(volume_identifier: datetime, version: str = 'v1_0') -> bool:
    """
    Processes a complete volume of radar scans.

    This is the core function of the batch processing workflow. It fetches all
    raw PPI scan files for a given volume, combines them into a single Py-ART
    Radar object, applies advanced corrections, saves the result as a
    CfRadial2 file, and logs it to the database.

    Args:
        volume_identifier: The timestamp identifier for the volume to process.
        version: The correction parameter version to apply (e.g., 'v1_0').

    Returns:
        True if processing was successful, False otherwise.
    """
    logger.info(f"--- Starting Volume Processing for ID: {volume_identifier.isoformat()} ---")

    conn = get_connection()
    try:
        # 1. Get all raw file paths for this volume
        scan_paths = get_scan_paths_for_volume(conn, volume_identifier)
        if not scan_paths:
            logger.error(f"No scan paths found for volume_identifier {volume_identifier.isoformat()}.")
            return False

        # 2. Read each PPI into a Dataset and build a dictionary of sweeps
        sweep_datasets = {}
        for i, path in enumerate(scan_paths):
            ds_sweep = read_ppi_scan(path)  # Use our specific PPI reader
            if ds_sweep:
                sweep_datasets[f"sweep_{i}"] = ds_sweep
            else:
                logger.warning(f"Failed to read scan {path}. It will be excluded from the volume.")

        if not sweep_datasets:
            logger.error("No scans could be successfully read for this volume.")
            return False

        # 3. Create a DataTree from the dictionary of sweep Datasets
        logger.info(f"Building datatree from {len(sweep_datasets)} sweeps...")
        volume_tree = datatree.DataTree.from_dict(sweep_datasets)

        # 4. Use the high-level accessor to convert the datatree to a Py-ART Radar object
        logger.info("Converting datatree to Py-ART Radar object...")
        combined_radar = volume_tree.pyart.to_radar()

        # 5. Apply advanced corrections (this part remains the same)
        corrected_volume = preprocessing.apply_corrections(combined_radar, version=version)

        # 6. Save the corrected volume as a single CfRadial2 file
        cfradial_dir = get_setting('app.cfradial_dir')
        if not cfradial_dir:
            logger.error("'app.cfradial_dir' not configured. Cannot save corrected volume.")
            return False

        date_str = volume_identifier.strftime("%Y%m%d")
        cfradial_version_dir = os.path.join(cfradial_dir, version, date_str)
        os.makedirs(cfradial_version_dir, exist_ok=True)

        # Create a filename based on the volume identifier timestamp
        time_str = volume_identifier.strftime("%Y%m%d_%H%M%S")
        cfradial_filename = f"volume_{time_str}.nc"
        cfradial_filepath = os.path.join(cfradial_version_dir, cfradial_filename)

        logger.info(f"Saving corrected volume to {cfradial_filepath}")
        pyart.io.write_cfradial(cfradial_filepath, corrected_volume)
        logger.info("Corrected volume saved successfully.")

        # 7. Log the new volume file to the database
        add_processed_volume_log(conn, volume_identifier, cfradial_filepath, version)

    except Exception as e:
        logger.error(f"An unexpected error occurred during volume processing for ID {volume_identifier.isoformat()}: {e}", exc_info=True)
        return False
    finally:
        release_connection(conn)

    logger.info(f"--- Finished Volume Processing for ID: {volume_identifier.isoformat()} ---")
    return True
# core/processor.py

import os
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

# --- Core Imports ---
from core.config import get_setting, get_config
from core.data import read_scan # Keep get_filepaths_in_range for historical
from core.utils.geo import georeference_dataset
from core.utils.helpers import move_processed_file # Keep parse_datetime_from_filename indirectly
from core.visualization.style import get_plot_style
from core.visualization.plotter import create_ppi_image
from core.utils.upload_queue import add_scan_to_queue
# --- End Core Imports ---

# Setup logger for this module
logger = logging.getLogger(__name__)


def process_new_scan(filepath: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Processes a single raw radar scan file based on the configured mode.
    'ftp_only': Queues scan for FTP, skips local move/plotting. Deletes local on SUCCESSFUL QUEUED UPLOAD via worker.
    'standard': Performs local plotting/saving, moves local file, optionally queues moved file for FTP.
    'disabled': Performs local plotting/saving, moves local file. No FTP.

    Args:
        filepath: The absolute path to the raw radar scan file (.scnx.gz).
        config: The application configuration dictionary (optional, uses get_config() if None). 

    Returns:
        True if local plot generation was successful (in standard/ftp_only modes)
             OR if queuing for FTP was successful (in ftp_only mode),
        False otherwise (e.g., read failure, plot save failure, critical queue add failure).
    """
    if config is None: config = get_config()
    logger.info(f"Starting processing for scan: {filepath}")

    # --- Get Relevant Configuration ---
    # FTP settings
    scan_upload_mode = get_setting('ftp.scan_upload_mode', 'disabled')
    upload_in_standard_mode = get_setting('ftp.upload_in_standard_mode', False)
    ftp_servers = get_setting('ftp.servers', [])

    # App settings
    variables_to_process: Dict[str, str] = get_setting('app.variables_to_process', {})
    images_dir: Optional[str] = get_setting('app.images_dir')
    output_dir: Optional[str] = get_setting('app.output_dir') # Local move destination
    watermark_path: Optional[str] = get_setting('app.watermark_path')
    move_local_in_standard = get_setting('app.move_file_after_processing', True) # Check if local move desired in standard

    # Style settings
    watermark_zoom: float = get_setting('styles.defaults.watermark_zoom', 0.05)

    # --- Basic Validations ---
    if not os.path.exists(filepath):
         logger.error(f"Input file does not exist: {filepath}")
         return False
    if not variables_to_process:
        logger.warning("No 'variables_to_process' defined. Skipping plot generation.")
        # Continue if only FTP queuing is needed, but might need adjustments
    if not images_dir and scan_upload_mode != 'disabled': # Plotting needed unless fully disabled
        logger.error("Configuration missing 'app.images_dir'. Cannot save plots.")
        # Can we proceed in ftp_only mode without plots? Plan says plots happen. So, error out.
        return False
    if scan_upload_mode == 'standard' and move_local_in_standard and not output_dir:
        logger.warning("Scan mode is 'standard' with local move enabled, but 'app.output_dir' is not set. Disabling local move.")
        move_local_in_standard = False
    if scan_upload_mode != 'disabled' and not ftp_servers:
        logger.warning(f"FTP mode '{scan_upload_mode}' enabled but no servers configured. Disabling FTP actions.")
        scan_upload_mode = 'disabled'
    if scan_upload_mode != 'disabled' and not variables_to_process and scan_upload_mode != 'ftp_only':
         # Plotting needed unless strictly ftp_only (where we skip it)
         logger.error("Plotting variables ('app.variables_to_process') needed but not configured.")
         return False
    if scan_upload_mode != 'disabled' and not images_dir and scan_upload_mode != 'ftp_only':
         logger.error("Image directory ('app.images_dir') needed but not configured.")
         return False

    # --- Mode 1: FTP Upload Scan ONLY (No Plotting, No Local Move) ---
    if scan_upload_mode == 'ftp_only':
        logger.info(f"Mode 'ftp_only': Queuing scan file for FTP upload: {filepath}")
        # Minimal read JUST for metadata might be an optimization, but queueing needs the path anyway.
        # For simplicity, we just queue the path directly. The worker needs the file later.
        queued_at_least_one = False
        queueing_succeeded = True # Assume success unless adding fails
        for server_config in ftp_servers:
            if add_scan_to_queue(filepath, server_config):
                queued_at_least_one = True
            else:
                # Error logged by add_scan_to_queue
                queueing_succeeded = False # Mark critical failure if ADDING fails

        if not queued_at_least_one and ftp_servers: # Check if we intended to queue but failed for all
            logger.error(f"Mode 'ftp_only': Failed to queue scan file for any FTP server: {filepath}")
            return False # Failed to perform the primary action

        logger.info(f"Mode 'ftp_only': Successfully queued scan file. Local file deletion handled by worker upon successful upload.")
        # Return True indicating the task was successfully handed off to the queue.
        return queueing_succeeded

    # --- Main Processing Logic ---
    ds = None
    local_plots_succeeded = False # Track success of local plot saving
    queueing_succeeded = True # Assume true unless adding to queue fails critically

    try:
        # --- Steps required for ALL modes that involve plotting ---
        if scan_upload_mode != 'disabled': # Or adjust if plotting should be skipped in ftp_only
            # 1. Read Scan
            logger.debug(f"Reading scan data for variables: {list(variables_to_process.keys())}")
            vars_to_read = list(variables_to_process.keys()) if variables_to_process else None
            if not vars_to_read:
                logger.warning("No variables configured to read for plotting/processing.")
                # Need to decide if we can proceed. If mode='disabled' or standard without upload, failure.
                # If standard WITH upload, maybe we just upload scan? Requires rethinking flow.
                # For now, assume reading/plotting is the primary goal if not ftp_only.
                return False
            ds = read_scan(filepath, variables=list(variables_to_process.keys()))
            if ds is None: return False # Cannot proceed

            # 2. Georeference
            logger.debug("Georeferencing dataset...")
            ds_geo = georeference_dataset(ds)
            if 'x' not in ds_geo.coords or 'y' not in ds_geo.coords:
                 logger.warning("Dataset potentially missing georeference coordinates ('x','y'). Plotting might fail.")

            # 3. Extract Metadata (for plot filenames/paths)
            try:
                dt_np = np.datetime64(ds_geo['time'].values.item())
                dt = dt_np.astype(datetime)
                elevation = float(ds_geo['elevation'].values.item())
                date_str = dt.strftime("%Y%m%d")
                datetime_file_str = dt.strftime("%Y%m%d_%H%M")
                elevation_code = f"{int(elevation * 100):03d}"
            except Exception as meta_err:
                logger.error(f"Failed to extract metadata: {meta_err}", exc_info=True)
                return False # Critical for filenames

            # 4. Generate and Save Plots Locally
            if variables_to_process and images_dir:
                 logger.debug(f"Generating plots for {list(variables_to_process.keys())}...")
                 for variable in variables_to_process.keys():
                     if variable not in ds_geo.data_vars:
                          logger.warning(f"Variable '{variable}' not in dataset for plotting. Skipping.")
                          continue
                     plot_style = get_plot_style(variable)
                     if plot_style is None: continue # Error logged by get_plot_style

                     image_bytes = create_ppi_image(ds_geo, variable, plot_style, watermark_path, watermark_zoom)
                     if image_bytes:
                         image_sub_dir = os.path.join(images_dir, variable, date_str, elevation_code)
                         os.makedirs(image_sub_dir, exist_ok=True)
                         image_filename = f"{variable}_{elevation_code}_{datetime_file_str}.png"
                         image_filepath = os.path.join(image_sub_dir, image_filename)
                         try:
                             with open(image_filepath, 'wb') as f: f.write(image_bytes)
                             logger.info(f"Saved plot: {image_filepath}")
                             local_plots_succeeded = True # Mark success if at least one saves
                         except IOError as e:
                             logger.error(f"Failed to save plot '{image_filepath}': {e}", exc_info=True)
                     else:
                         logger.warning(f"Failed to generate image bytes for variable '{variable}'.")
            else:
                 logger.debug("Skipping plot generation (no variables or images_dir configured).")
                 local_plots_succeeded = True # No plots to fail on, counts as success for this step

        # --- Handle Scan File Based on Mode ---
        moved_filepath = None 
        move_attempted = False
        move_succeeded = True # Default to true if move not attempted

        if scan_upload_mode == 'standard':
            logger.info("Mode 'standard': Handling local move and potential FTP queueing.")
            if move_local_in_standard and output_dir:
                move_attempted = True
                logger.debug(f"Attempting to move scan file: {filepath} -> {output_dir}")
                try:
                    moved_filepath = move_processed_file(filepath, output_dir)
                    if moved_filepath: logger.info(f"Moved scan file to: {moved_filepath}")
                    else: move_succeeded = False; logger.error(f"Failed to move scan file locally: {filepath}")
                except Exception as move_err:
                    move_succeeded = False; logger.error(f"Error moving source file {filepath}: {move_err}", exc_info=True)
            else:
                 logger.debug(f"Local move is disabled or output_dir not set. Using original path for FTP: {filepath}")
                 moved_filepath = filepath # Use original path if not moved

            # Queue for FTP if enabled AND local plotting succeeded AND (move succeeded OR wasn't attempted)
            if local_plots_succeeded and move_succeeded and upload_in_standard_mode and moved_filepath:
                 logger.info(f"Queueing processed scan file for FTP upload: {moved_filepath}")
                 queueing_succeeded = True
                 for server_config in ftp_servers:
                      if not add_scan_to_queue(moved_filepath, server_config):
                          queueing_succeeded = False # Log critical add failure
                 # Informational logging about queue outcome, doesn't affect overall success bool
                 if not queueing_succeeded: logger.error("Failed to add scan file to FTP queue for one or more servers.")
            # Return success based on local plotting and moving (if attempted)
            return local_plots_succeeded and move_succeeded

        elif scan_upload_mode == 'disabled':
            logger.info("Mode 'disabled': Moving scan file locally if configured.")
            if move_local_in_standard and output_dir:
                move_attempted = True
                logger.debug(f"Attempting to move scan file: {filepath} -> {output_dir}")
                try:
                    moved_filepath = move_processed_file(filepath, output_dir)
                    if moved_filepath: logger.info(f"Moved scan file to: {moved_filepath}")
                    else: move_succeeded = False; logger.error(f"Failed to move scan file locally: {filepath}")
                except Exception as move_err:
                    move_succeeded = False; logger.error(f"Error moving source file {filepath}: {move_err}", exc_info=True)
            else:
                logger.debug("Local move is disabled or output_dir not set.")
            # Return success based on local plotting and moving (if attempted)
            return local_plots_succeeded and move_succeeded

        else: # Should not be reached if validation is correct
             logger.error(f"Internal error: Invalid scan_upload_mode '{scan_upload_mode}' reached main processing.")
             return False

    except Exception as e:
        logger.error(f"Unhandled error during standard/disabled processing of {filepath}: {e}", exc_info=True)
        return False
    finally:
        # Ensure dataset is closed if it was opened
        if ds is not None:
            try: ds.close()
            except Exception: pass

def generate_historical_plots(start_dt: datetime, end_dt: datetime, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Finds processed radar scan files within a date range and regenerates plots for them.

    Args:
        start_dt: The start datetime for the reprocessing range.
        end_dt: The end datetime for the reprocessing range.
        config: The application configuration dictionary (optional, uses get_config() if None).
    """
    if config is None:
        config = get_config()

    logger.info(f"Starting historical plot generation from {start_dt} to {end_dt}.")

    # Directory where processed .scnx.gz files are stored (used for searching)
    processed_data_dir = config.get('app', {}).get('output_dir')

    if not processed_data_dir:
        logger.error("Configuration missing 'app.output_dir'. Cannot find historical files.")
        return

    # --- 1. Find Files in Range ---
    # Note: get_filepaths_in_range expects the base dir where YYYYMMDD folders are
    filepaths_tuples = get_filepaths_in_range(processed_data_dir, start_dt, end_dt)

    if not filepaths_tuples:
        logger.info("No historical scan files found in the specified range.")
        return

    logger.info(f"Found {len(filepaths_tuples)} files to reprocess.")

    # --- 2. Loop and Process Each File ---
    processed_count = 0
    failed_count = 0
    # Consider adding tqdm here if running interactively for long periods
    # from tqdm import tqdm
    # for filepath, _ in tqdm(filepaths_tuples, desc="Reprocessing Scans"):
    for filepath, file_dt in filepaths_tuples:
         logger.info(f"Reprocessing historical file: {filepath} (Time: {file_dt})")
         # We pass the config down, process_new_scan should NOT move the file again
         # We need a way to tell process_new_scan not to move the file during reprocessing.
         # Option 1: Modify process_new_scan to accept a `move_file=False` argument.
         # Option 2: Temporarily modify the config dictionary passed down. (Less clean)
         # Let's assume Option 1 is preferred for cleaner design (but requires modifying process_new_scan slightly).
         # For now, we rely on the `move_file_after_processing` flag in the main config,
         # or we comment out the move call in process_new_scan if it's problematic.
         # A better approach: Add a flag to config or argument to disable move during reprocessing.

         # Let's simulate disabling move by modifying a copy of the config
         temp_config = config.copy()
         if 'app' not in temp_config: temp_config['app'] = {}
         temp_config['app']['move_file_after_processing'] = False # Disable moving for reprocessing

         try:
             # Pass the modified config to prevent moving
             success = process_new_scan(filepath, config=temp_config)
             if success:
                 processed_count += 1
             else:
                 failed_count += 1
                 logger.warning(f"Failed to reprocess: {filepath}")
         except Exception as e:
             failed_count += 1
             logger.error(f"Critical error during reprocessing of {filepath}: {e}", exc_info=True)

    logger.info(f"Historical plot generation finished. Processed: {processed_count}, Failed: {failed_count}")

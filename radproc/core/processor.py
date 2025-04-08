# core/processor.py

import os
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

# --- Core Imports ---
# Use get_config in addition to get_setting
from core.config import get_setting, get_config
from core.data import read_scan, get_filepaths_in_range # Keep get_filepaths_in_range for historical
from core.utils.geo import georeference_dataset
from core.utils.helpers import move_processed_file
from core.visualization.style import get_plot_style
from core.visualization.plotter import create_ppi_image
from core.utils.upload_queue import add_scan_to_queue, add_image_to_queue
# +++ Import the new timeseries update function +++
from core.analysis import update_timeseries_for_scan
# +++++++++++++++++++++++++++++++++++++++++++++++++

# Setup logger for this module
logger = logging.getLogger(__name__)


def process_new_scan(filepath: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Processes a single raw radar scan file based on configured modes for scans and images.
    Also triggers automatic timeseries updates if enabled.

    Args:
        filepath: The absolute path to the raw radar scan file (.scnx.gz).
        config: The application configuration dictionary (optional, uses get_config() if None).

    Returns:
        True if essential steps succeeded (read, plot saving if needed, queuing if needed), False otherwise.
    """
    # --- Configuration Loading and Initial Setup ---
    if config is None: config = get_config()
    logger.info(f"Starting processing for scan: {filepath}")

    scan_upload_mode = get_setting('ftp.scan_upload_mode', 'disabled')
    upload_scans_in_standard_mode = get_setting('ftp.upload_scans_in_standard_mode', False)
    image_upload_mode = get_setting('ftp.image_upload_mode', 'disabled')
    upload_images_in_standard_mode = get_setting('ftp.upload_images_in_standard_mode', False)
    ftp_servers = get_setting('ftp.servers', [])
    variables_to_process: Dict[str, str] = get_setting('app.variables_to_process', {})
    images_dir: Optional[str] = get_setting('app.images_dir')
    output_dir: Optional[str] = get_setting('app.output_dir')
    watermark_path: Optional[str] = get_setting('app.watermark_path')
    move_local_in_standard = get_setting('app.move_file_after_processing', True)
    watermark_zoom: float = get_setting('styles.defaults.watermark_zoom', 0.05)
    enable_timeseries = get_setting('app.enable_timeseries_updates', False)

    # --- Determine if Plotting is Needed ---
    # Plotting is required if images need saving locally OR if images need uploading
    plotting_required = (variables_to_process and images_dir) or (image_upload_mode != 'disabled' and variables_to_process)
    # If plotting is required but images_dir is missing, it's an error
    if plotting_required and not images_dir:
        logger.error("Configuration missing 'app.images_dir', which is required for plot generation/upload.")
        return False

    # --- Basic Validations ---
    if not os.path.exists(filepath):
         logger.error(f"Input file does not exist: {filepath}")
         return False
    if not variables_to_process and plotting_required:
        logger.error("Plotting required by config, but no 'variables_to_process' defined.")
        return False
    # If any upload mode is enabled, servers must be configured
    if (scan_upload_mode != 'disabled' or image_upload_mode != 'disabled') and not ftp_servers:
        logger.warning(f"FTP mode enabled for scans ('{scan_upload_mode}') or images ('{image_upload_mode}') but no servers configured. Disabling FTP.")
        scan_upload_mode = 'disabled'
        image_upload_mode = 'disabled'
    # Validate output_dir if moving scan in standard mode
    if scan_upload_mode == 'standard' and move_local_in_standard and not output_dir:
        logger.warning("Scan mode is 'standard' with local move enabled, but 'app.output_dir' is not set. Disabling local move.")
        move_local_in_standard = False


    # --- Mode 1: Scan Upload Only ('ftp_only' for scans, 'disabled' for images) ---
    # This mode still requires plotting if image_upload_mode is NOT disabled
    if scan_upload_mode == 'ftp_only' and image_upload_mode == 'disabled':
        logger.info(f"Mode 'ftp_only' (scan), 'disabled' (image): Queuing scan file {filepath}")
        queued_at_least_one_scan = False
        scan_queueing_succeeded = True
        for server_config in ftp_servers:
            if add_scan_to_queue(filepath, server_config):
                queued_at_least_one_scan = True
            else:
                scan_queueing_succeeded = False # Critical add failure

        if not queued_at_least_one_scan and ftp_servers:
            logger.error(f"Mode 'ftp_only' (scan): Failed to queue scan file for any FTP server: {filepath}")
            return False
        # Skip local move, skip plotting. Success depends on queuing the scan.
        logger.info(f"Mode 'ftp_only' (scan): Successfully queued scan file.")
        return scan_queueing_succeeded


    # --- Main Processing Logic (Reading, Plotting, Moving, Queuing) ---
    ds = None
    ds_geo = None # Keep track of the georeferenced dataset
    local_plots_succeeded = False # Assume false unless plots generated and saved
    overall_success = True      # Assume true, set to false on critical failures

    try:
        # --- Reading and Georeferencing (Required for plotting, timeseries, or standard mode) ---
        # Determine if we absolutely need to read the data
        needs_data_read = plotting_required or enable_timeseries or scan_upload_mode != 'ftp_only'
        if needs_data_read:
            logger.debug(f"Reading scan data for variables: {list(variables_to_process.keys())}")
            # Determine variables needed (for plotting and potentially timeseries defaults)
            vars_needed = set(variables_to_process.keys()) if variables_to_process else set()
            if enable_timeseries:
                 # Add default variables from points config if needed for index finding/extraction
                 all_points = get_setting('points_config.points', [])
                 for p in all_points:
                     if isinstance(p, dict) and p.get('variable'):
                         vars_needed.add(p['variable'])

            if not vars_needed:
                # If plotting/timeseries enabled but no variables identified, log warning
                 if plotting_required or enable_timeseries:
                     logger.warning("No variables identified for plotting or timeseries extraction.")
                 # Can still proceed if only moving/uploading scan file in standard mode
            vars_to_read = list(variables_to_process.keys()) if variables_to_process else None
            if not vars_to_read and plotting_required: # Should have been caught earlier, but double-check
                 logger.error("Cannot proceed: Plotting required but no variables configured.")
                 return False

            ds = read_scan(filepath, variables=vars_to_read)
            if ds is None:
                logger.error(f"Failed to read scan file: {filepath}")
                return False # Critical failure

            logger.debug("Georeferencing dataset...")
            ds_geo = georeference_dataset(ds)
            # Basic check for successful georeferencing
            if 'x' not in ds_geo.coords or 'y' not in ds_geo.coords:
                 logger.warning("Georeferencing may have failed (missing x/y coords). Plotting/Timeseries might fail.")

            # +++ Automatic Timeseries Update +++
            # Perform this *after* georeferencing but *before* potential file move
            if enable_timeseries and ds_geo is not None:
                logger.info("Triggering automatic timeseries update...")
                try:
                    update_timeseries_for_scan(ds_geo)
                except Exception as ts_update_err:
                    # Log error but DO NOT let it stop the main processing workflow
                    logger.error(f"Error during automatic timeseries update: {ts_update_err}", exc_info=True)
                logger.info("Automatic timeseries update attempt finished.")
            elif enable_timeseries:
                 logger.warning("Skipping timeseries update because georeferenced dataset is invalid.")
            # ++++++++++++++++++++++++++++++++++++

            

            # Generate and Save Plots Locally (if required)
            saved_image_paths: Dict[str, str] = {} # Store paths of saved images {variable: path}
            if plotting_required and variables_to_process and images_dir and ds_geo is not None:
                 logger.debug(f"Generating plots for {list(variables_to_process.keys())}...")# Extract Metadata
                 try:
                     dt_np = np.datetime64(ds_geo['time'].values.item())
                     dt = dt_np.astype(datetime)
                     elevation = float(ds_geo['elevation'].values.item())
                     date_str = dt.strftime("%Y%m%d")
                     datetime_file_str = dt.strftime("%Y%m%d_%H%M")
                     elevation_code = f"{int(elevation * 100):03d}"
                 except Exception as meta_err:
                     logger.error(f"Failed to extract metadata: {meta_err}", exc_info=True)
                     return False
                 
                 num_plots_succeeded = 0
                 for variable in variables_to_process.keys():
                     if variable not in ds_geo.data_vars: continue
                     plot_style = get_plot_style(variable)
                     if plot_style is None: continue

                     image_bytes = create_ppi_image(ds_geo, variable, plot_style, watermark_path, watermark_zoom)
                     if image_bytes:
                         image_sub_dir = os.path.join(images_dir, variable, date_str, elevation_code)
                         os.makedirs(image_sub_dir, exist_ok=True)
                         image_filename = f"{variable}_{elevation_code}_{datetime_file_str}.png"
                         image_filepath = os.path.join(image_sub_dir, image_filename)
                         try:
                             with open(image_filepath, 'wb') as f: f.write(image_bytes)
                             logger.info(f"Saved plot: {image_filepath}")
                             saved_image_paths[variable] = image_filepath # Store path
                             num_plots_succeeded += 1
                         except IOError as e:
                             logger.error(f"Failed to save plot '{image_filepath}': {e}")
                             overall_success = False # Saving plot is critical if plotting required
                     else:
                         logger.warning(f"Failed to generate image bytes for variable '{variable}'.")
                         overall_success = False # Generating plot is critical if required

                 # Check if at least one plot succeeded if plotting was the goal
                 local_plots_succeeded = num_plots_succeeded > 0
                 if plotting_required and not local_plots_succeeded:
                     logger.error("Plotting was required, but failed to generate/save any plots.")
                     overall_success = False # Set failure flag

            elif not plotting_required:
                 logger.debug("Skipping plot generation as not required by config.")
                 local_plots_succeeded = True # Not required = success for this step

        # --- Handle Scan File ---
        moved_filepath = filepath # Default to original path
        move_succeeded = True

        # Only move if NOT in 'ftp_only' mode for scans
        if scan_upload_mode != 'ftp_only':
            if move_local_in_standard and output_dir:
                logger.debug(f"Attempting to move scan file: {filepath} -> {output_dir}")
                try:
                    moved_path_result = move_processed_file(filepath, output_dir)
                    if moved_path_result:
                        moved_filepath = moved_path_result
                        logger.info(f"Moved scan file to: {moved_filepath}")
                    else:
                        move_succeeded = False
                        overall_success = False # Failed to move when expected
                        logger.error(f"Failed to move scan file locally: {filepath}")
                except Exception as move_err:
                    move_succeeded = False
                    overall_success = False
                    logger.error(f"Error moving source file {filepath}: {move_err}", exc_info=True)
            else:
                logger.debug("Local move is disabled or output_dir not set (in standard/disabled mode).")
        else:
            logger.debug(f"Scan file local move skipped due to scan_upload_mode='ftp_only'.")


        # --- Queueing Logic ---
        if not overall_success: # Don't queue if critical steps failed
             logger.warning("Skipping FTP queueing due to previous errors.")
             return False

        scan_queued = False
        images_queued = False

        # Queue Scan File?
        if scan_upload_mode == 'ftp_only':
             # Special case handled earlier if image_mode was disabled.
             # If image_mode is 'only' or 'also', we process plots first, then queue scan here.
             logger.info(f"Mode 'ftp_only' (scan): Queuing scan file {filepath} after potential plotting.")
             scan_queueing_succeeded = True
             for server_config in ftp_servers:
                  if not add_scan_to_queue(filepath, server_config):
                      scan_queueing_succeeded = False # Track critical add failure
             if not scan_queueing_succeeded: overall_success = False
             scan_queued = True

        elif scan_upload_mode == 'standard' and upload_scans_in_standard_mode and move_succeeded:
             logger.info(f"Mode 'standard' (scan): Queuing scan file {moved_filepath}")
             scan_queueing_succeeded = True
             for server_config in ftp_servers:
                  if not add_scan_to_queue(moved_filepath, server_config):
                      scan_queueing_succeeded = False
             if not scan_queueing_succeeded: overall_success = False # Consider if this is critical failure
             scan_queued = True

        # Queue Image Files?
        if local_plots_succeeded and saved_image_paths: # Check if plots were actually saved
            should_queue_images = False
            if image_upload_mode == 'only':
                should_queue_images = True
            elif image_upload_mode == 'also':
                # Queue if scan is 'ftp_only' OR (scan is 'standard' AND upload_images_in_standard is true)
                if scan_upload_mode == 'ftp_only' or \
                   (scan_upload_mode == 'standard' and upload_images_in_standard_mode):
                    should_queue_images = True

            if should_queue_images:
                logger.info(f"Queueing {len(saved_image_paths)} generated image(s)...")
                image_queueing_succeeded = True
                for variable, img_path in saved_image_paths.items():
                    for server_config in ftp_servers:
                        if not add_image_to_queue(img_path, server_config):
                             image_queueing_succeeded = False # Track critical add failure
                if not image_queueing_succeeded: overall_success = False
                images_queued = True

        # --- Final Logging and Return ---
        if scan_queued: logger.info("Scan file(s) queued for upload.")
        if images_queued: logger.info("Image file(s) queued for upload.")
        if not scan_queued and not images_queued and (scan_upload_mode != 'disabled' or image_upload_mode != 'disabled'):
            logger.debug("No files were queued for upload based on current modes.")

        logger.info(f"Processing finished for {filepath}. Overall success: {overall_success}")
        return overall_success

    except Exception as e:
        logger.error(f"Unhandled error during processing of {filepath}: {e}", exc_info=True)
        return False
    finally:
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

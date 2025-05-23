# core/processor.py

import os
import logging
import numpy as np
import pandas as pd  # Added for pd.to_datetime
from datetime import datetime, timezone  # Ensured timezone is imported
from typing import Dict, Any, Optional
import shutil

# --- Core Imports ---
from .config import get_setting, get_config  # Corrected import
from .data import read_scan, get_filepaths_in_range
from .utils.geo import georeference_dataset
from .utils.helpers import move_processed_file  # This is the function we modified
from .visualization.style import get_plot_style
from .visualization.plotter import create_ppi_image
from .utils.upload_queue import add_scan_to_queue, add_image_to_queue
from .analysis import update_timeseries_for_scan

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
        True if essential steps succeeded, False otherwise.
    """
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
    realtime_dir = get_setting('app.realtime_image_dir')

    plotting_required = (variables_to_process and images_dir) or \
                        (image_upload_mode != 'disabled' and variables_to_process)
    if plotting_required and not images_dir:
        logger.error("Config missing 'app.images_dir', required for plot generation/upload.")
        return False

    if not os.path.exists(filepath):
        logger.error(f"Input file does not exist: {filepath}")
        return False
    if not variables_to_process and plotting_required:
        logger.error("Plotting required, but no 'variables_to_process' defined.")
        return False
    if (scan_upload_mode != 'disabled' or image_upload_mode != 'disabled') and not ftp_servers:
        logger.warning("FTP mode enabled but no servers configured. Disabling FTP.")
        scan_upload_mode = 'disabled'
        image_upload_mode = 'disabled'
    if scan_upload_mode == 'standard' and move_local_in_standard and not output_dir:
        logger.warning("'output_dir' not set for standard scan mode with local move. Disabling move.")
        move_local_in_standard = False

    if scan_upload_mode == 'ftp_only' and image_upload_mode == 'disabled':
        logger.info(f"Mode 'ftp_only' (scan), 'disabled' (image): Queuing scan file {filepath}")
        queued_at_least_one_scan = False
        scan_queueing_succeeded = True
        for server_config in ftp_servers:
            if add_scan_to_queue(filepath, server_config):
                queued_at_least_one_scan = True
            else:
                scan_queueing_succeeded = False
        if not queued_at_least_one_scan and ftp_servers:
            logger.error(f"Mode 'ftp_only' (scan): Failed to queue for any FTP server: {filepath}")
            return False
        logger.info("Mode 'ftp_only' (scan): Successfully queued scan file.")
        return scan_queueing_succeeded

    ds = None
    ds_geo = None
    local_plots_succeeded = False
    overall_success = True
    saved_image_paths: Dict[str, str] = {}
    scan_elevation_val: Optional[float] = None
    scan_datetime_val: Optional[datetime] = None

    try:
        needs_data_read = plotting_required or enable_timeseries or scan_upload_mode != 'ftp_only'
        if needs_data_read:
            vars_to_read = list(variables_to_process.keys()) if variables_to_process else None
            if not vars_to_read and plotting_required:
                logger.error("Cannot proceed: Plotting required but no variables configured.")
                return False

            ds = read_scan(filepath, variables=vars_to_read)
            if ds is None:
                logger.error(f"Failed to read scan file: {filepath}")
                return False

            logger.debug("Georeferencing dataset...")
            ds_geo = georeference_dataset(ds)
            if 'x' not in ds_geo.coords or 'y' not in ds_geo.coords:
                logger.warning("Georeferencing may have failed (missing x/y coords).")

            # Extract elevation and datetime for moving and other uses
            if ds_geo is not None and 'elevation' in ds_geo.coords and 'time' in ds_geo.coords:
                try:
                    scan_elevation_val = float(ds_geo['elevation'].item())
                    dt_val_from_ds = pd.to_datetime(ds_geo['time'].item())
                    if dt_val_from_ds.tzinfo is None:
                        scan_datetime_val = dt_val_from_ds.tz_localize(timezone.utc)
                    else:
                        scan_datetime_val = dt_val_from_ds.tz_convert(timezone.utc)
                    logger.debug(f"Extracted from ds_geo: elev={scan_elevation_val}, dt={scan_datetime_val}")
                except Exception as e:
                    logger.error(f"Failed to extract elevation/datetime from ds_geo: {e}")
                    overall_success = False  # Critical for moving file correctly

            if enable_timeseries and ds_geo is not None and overall_success:
                logger.info("Triggering automatic timeseries update...")
                try:
                    update_timeseries_for_scan(ds_geo)
                except Exception as ts_update_err:
                    logger.error(f"Error during automatic timeseries update: {ts_update_err}", exc_info=True)
                logger.info("Automatic timeseries update attempt finished.")
            elif enable_timeseries and not ds_geo:
                logger.warning("Skipping timeseries update: georeferenced dataset is invalid.")

            if plotting_required and variables_to_process and images_dir and ds_geo is not None and overall_success:
                logger.debug(f"Generating plots for {list(variables_to_process.keys())}...")
                # Reuse scan_datetime_val and scan_elevation_val if available
                # For plotting, dt_utc and elevation for title/filename comes from ds_geo directly
                try:
                    dt_np_plot = np.datetime64(ds_geo['time'].values.item())  # For plot title/filename
                    dt_plot = dt_np_plot.astype(datetime)
                    elevation_plot = float(ds_geo['elevation'].values.item())  # For plot title/filename
                    date_str_plot = dt_plot.strftime("%Y%m%d")
                    datetime_file_str_plot = dt_plot.strftime("%Y%m%d_%H%M")
                    elevation_code_plot = f"{int(round(elevation_plot * 100)):03d}"
                except Exception as meta_err:
                    logger.error(f"Failed to extract metadata for plotting: {meta_err}", exc_info=True)
                    return False  # Critical for plotting

                num_plots_succeeded = 0
                for variable in variables_to_process.keys():
                    if variable not in ds_geo.data_vars: continue
                    plot_style = get_plot_style(variable)
                    if plot_style is None: continue

                    image_bytes = create_ppi_image(ds_geo, variable, plot_style, watermark_path, watermark_zoom)
                    if image_bytes:
                        image_sub_dir = os.path.join(images_dir, variable, date_str_plot, elevation_code_plot)
                        os.makedirs(image_sub_dir, exist_ok=True)
                        image_filename = f"{variable}_{elevation_code_plot}_{datetime_file_str_plot}.png"
                        image_filepath = os.path.join(image_sub_dir, image_filename)
                        try:
                            with open(image_filepath, 'wb') as f:
                                f.write(image_bytes)
                            logger.info(f"Saved plot: {image_filepath}")
                            saved_image_paths[variable] = image_filepath
                            num_plots_succeeded += 1

                            if realtime_dir:
                                realtime_filename = f"realtime_{variable}_{elevation_code_plot}.png"
                                realtime_filepath = os.path.join(realtime_dir, realtime_filename)
                                realtime_filepath_tmp = f"{realtime_filepath}.{os.getpid()}.tmp"
                                try:
                                    os.makedirs(realtime_dir, exist_ok=True)
                                    shutil.copyfile(image_filepath, realtime_filepath_tmp)
                                    os.replace(realtime_filepath_tmp, realtime_filepath)
                                    logger.debug(f"Updated realtime image: {realtime_filepath}")
                                except OSError as copy_err:
                                    logger.error(
                                        f"Failed to copy/replace realtime image {realtime_filepath}: {copy_err}")
                                    if os.path.exists(realtime_filepath_tmp):
                                        try:
                                            os.remove(realtime_filepath_tmp)
                                        except OSError:
                                            pass
                        except IOError as e:
                            logger.error(f"Failed to save plot '{image_filepath}': {e}")
                            overall_success = False
                    else:
                        logger.warning(f"Failed to generate image bytes for variable '{variable}'.")
                        overall_success = False

                local_plots_succeeded = num_plots_succeeded > 0
                if plotting_required and not local_plots_succeeded:
                    logger.error("Plotting required, but failed to generate/save any plots.")
                    overall_success = False
            elif not plotting_required:
                logger.debug("Skipping plot generation as not required by config.")
                local_plots_succeeded = True

        # --- Handle Scan File ---
        moved_filepath = filepath
        move_succeeded = True

        if not overall_success:  # If prior critical steps like reading or getting elev/dt failed.
            logger.warning("Skipping scan file move due to previous critical errors.")
            move_succeeded = False
        elif scan_upload_mode != 'ftp_only':
            if move_local_in_standard and output_dir:
                if scan_elevation_val is not None and scan_datetime_val is not None:
                    logger.debug(f"Attempting to move scan file: {filepath} to {output_dir} "
                                 f"(Elev: {scan_elevation_val:.2f}, Time: {scan_datetime_val.isoformat()})")
                    try:
                        moved_path_result = move_processed_file(
                            filepath, output_dir, scan_elevation_val, scan_datetime_val  # NEW CALL
                        )
                        if moved_path_result:
                            moved_filepath = moved_path_result
                            logger.info(f"Moved scan file to: {moved_filepath}")
                        else:
                            move_succeeded = False
                            overall_success = False
                            logger.error(f"Failed to move scan file locally: {filepath}")
                    except Exception as move_err:
                        move_succeeded = False
                        overall_success = False
                        logger.error(f"Error moving source file {filepath}: {move_err}", exc_info=True)
                else:
                    move_succeeded = False
                    overall_success = False
                    logger.error(f"Cannot move scan file: elevation or datetime not available from dataset {filepath}.")
            else:
                logger.debug(
                    "Local move is disabled or output_dir not set (in standard/disabled mode). File remains at original path.")
        else:
            logger.debug("Scan file local move skipped due to scan_upload_mode='ftp_only'.")

        # --- Queueing Logic ---
        if not overall_success:
            logger.warning("Skipping FTP queueing due to previous errors.")
            return False  # Ensure we don't proceed if overall_success became false

        scan_queued = False
        images_queued = False

        if scan_upload_mode == 'ftp_only':
            logger.info(f"Mode 'ftp_only' (scan): Queuing scan file {filepath} after potential plotting.")
            current_scan_queueing_succeeded = True
            for server_config in ftp_servers:
                if not add_scan_to_queue(filepath, server_config):  # Use original filepath
                    current_scan_queueing_succeeded = False
            if not current_scan_queueing_succeeded: overall_success = False
            scan_queued = True
        elif scan_upload_mode == 'standard' and upload_scans_in_standard_mode and move_succeeded:
            logger.info(f"Mode 'standard' (scan): Queuing MOVED scan file {moved_filepath}")
            current_scan_queueing_succeeded = True
            for server_config in ftp_servers:
                if not add_scan_to_queue(moved_filepath, server_config):
                    current_scan_queueing_succeeded = False
            if not current_scan_queueing_succeeded: overall_success = False
            scan_queued = True
        elif scan_upload_mode == 'standard' and upload_scans_in_standard_mode and not move_succeeded:
            logger.warning(
                f"Scan mode is 'standard' and upload enabled, but file move failed or was skipped. Scan at {filepath} will not be queued.")

        if local_plots_succeeded and saved_image_paths:
            should_queue_images = False
            if image_upload_mode == 'only':
                should_queue_images = True
            elif image_upload_mode == 'also':
                if scan_upload_mode == 'ftp_only' or \
                        (scan_upload_mode == 'standard' and upload_images_in_standard_mode):
                    should_queue_images = True

            if should_queue_images:
                logger.info(f"Queueing {len(saved_image_paths)} generated image(s)...")
                current_image_queueing_succeeded = True
                for variable, img_path in saved_image_paths.items():
                    for server_config in ftp_servers:
                        if not add_image_to_queue(img_path, server_config):
                            current_image_queueing_succeeded = False
                if not current_image_queueing_succeeded: overall_success = False
                images_queued = True

        if scan_queued: logger.info("Scan file(s) queued for upload.")
        if images_queued: logger.info("Image file(s) queued for upload.")
        if not scan_queued and not images_queued and (
                scan_upload_mode != 'disabled' or image_upload_mode != 'disabled'):
            logger.debug("No files were queued for upload based on current modes and outcomes.")

        logger.info(f"Processing finished for {filepath}. Overall success: {overall_success}")
        return overall_success

    except Exception as e:
        logger.error(f"Unhandled error during processing of {filepath}: {e}", exc_info=True)
        return False
    finally:
        if ds_geo is not None:  # Close ds_geo if it was created
            try:
                ds_geo.close()
            except Exception:
                pass
        if ds is not None:  # ds should be closed if ds_geo was not created or if ds_geo is different
            try:
                ds.close()
            except Exception:
                pass


# `generate_historical_plots` remains unchanged as it calls `process_new_scan`
# and the logic for not re-moving files is handled by passing a modified config
# or ideally by adding a `move_file=False` parameter to `process_new_scan` in future.
def generate_historical_plots(start_dt: datetime, end_dt: datetime, config: Optional[Dict[str, Any]] = None) -> None:
    if config is None:
        config = get_config()
    logger.info(f"Starting historical plot generation from {start_dt} to {end_dt}.")
    processed_data_dir = config.get('app', {}).get('output_dir')
    if not processed_data_dir:
        logger.error("Config missing 'app.output_dir'. Cannot find historical files.")
        return

    # Use the NEW get_filepaths_in_range that supports ElevationCode/YYYYMMDD
    # For reprocess, we want all elevations, so elevation_filter is None.
    filepaths_tuples = get_filepaths_in_range(processed_data_dir, start_dt, end_dt, elevation_filter=None)

    if not filepaths_tuples:
        logger.info("No historical scan files found in the specified range.")
        return
    logger.info(f"Found {len(filepaths_tuples)} files to reprocess.")

    processed_count = 0
    failed_count = 0
    for filepath, file_dt in filepaths_tuples:
        logger.info(f"Reprocessing historical file: {filepath} (Time: {file_dt})")
        temp_config = config.copy()  # Create a shallow copy
        if 'app' not in temp_config: temp_config['app'] = {}
        temp_config['app']['move_file_after_processing'] = False  # Disable moving

        try:
            success = process_new_scan(filepath, config=temp_config)
            if success:
                processed_count += 1
            else:
                failed_count += 1
                logger.warning(f"Failed to reprocess: {filepath}")
        except Exception as e:
            failed_count += 1
            logger.error(f"Critical error reprocessing {filepath}: {e}", exc_info=True)
    logger.info(f"Historical plot generation finished. Processed: {processed_count}, Failed: {failed_count}")
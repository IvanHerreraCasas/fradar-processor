# core/processor.py

import os
import logging
import pandas as pd
from datetime import datetime, timezone  # Ensure timezone is imported
from typing import Dict, Any, Optional
import shutil

# --- Core Imports ---
from .config import get_setting, get_config
from .data import read_scan  # Removed get_filepaths_in_range, extract_scan_key_metadata
from .utils.geo import georeference_dataset
from .utils.helpers import move_processed_file, parse_datetime_from_filename, \
    parse_scan_sequence_number  # Added new helpers
from .visualization.style import get_plot_style
from .visualization.plotter import create_ppi_image
from .utils.upload_queue import add_scan_to_queue, add_image_to_queue
from .analysis import update_timeseries_for_scan  # generate_point_timeseries (for historical) not called directly here
# Import DB Manager for logging to scan_log
from .db_manager import get_connection, release_connection, add_scan_to_log

logger = logging.getLogger(__name__)


def process_new_scan(filepath: str, config_override: Optional[Dict[str, Any]] = None) -> bool:
    """
    Processes a single raw radar scan file.
    Workflow:
    1. Parse filename for sequence number & nominal timestamp.
    2. Read scan data (opens file once).
    3. Extract precise timestamp & elevation from dataset.
    4. Georeference.
    5. Update timeseries data in DB (if enabled).
    6. Generate and save plots (if enabled).
    7. Move processed scan file to its new structured path.
    8. Log scan details to radproc_scan_log in DB.
    9. Queue file for FTP upload (if enabled).
    """
    app_config = get_config() if config_override is None else config_override
    logger.info(f"Starting processing for scan: {filepath}")

    # Load relevant settings
    scan_upload_mode = get_setting('ftp.scan_upload_mode', 'disabled')
    # ... (load other settings as before: image_upload_mode, ftp_servers, etc.)
    variables_to_process: Dict[str, str] = get_setting('app.variables_to_process', {})
    images_dir: Optional[str] = get_setting('app.images_dir')
    output_dir: Optional[str] = get_setting('app.output_dir')  # Base dir for processed scans
    watermark_path: Optional[str] = get_setting('app.watermark_path')
    move_local_in_standard = get_setting('app.move_file_after_processing', True)  # This flag controls local move
    watermark_zoom: float = get_setting('styles.defaults.watermark_zoom', 0.05)
    enable_timeseries_realtime = get_setting('app.enable_timeseries_updates', False)  # Real-time specific flag
    realtime_dir_plots = get_setting('app.realtime_image_dir')  # For realtime plots

    # Initial validations (file existence, output_dir if moving, etc.)
    if not os.path.exists(filepath):
        logger.error(f"Input file does not exist: {filepath}");
        return False
    if not output_dir:  # output_dir is now essential for logging to scan_log with final path
        logger.error("'app.output_dir' must be configured.");
        return False

    original_filename = os.path.basename(filepath)

    # 1. Parse sequence number and nominal timestamp from filename
    sequence_number = parse_scan_sequence_number(original_filename)
    if sequence_number is None:
        logger.error(
            f"Could not parse sequence number from {original_filename}. Cannot process this scan for scan_log or volume grouping.")
        return False  # Critical for scan_log and volume grouping

    nominal_timestamp = parse_datetime_from_filename(original_filename)  # For reference in scan_log

    # --- Early exit for 'ftp_only' mode (if no plotting/timeseries needed before FTP) ---
    # This mode might need re-evaluation. If we FTP original files before any processing,
    # it's simple. But if we need precise_ts for FTP path (as per earlier FTP change),
    # we need to read the file. For now, assuming ftp_only queues the original path.
    # The `add_scan_to_queue` in ftp_client was changed to determine its own path.
    ftp_servers = get_setting('ftp.servers', [])
    if scan_upload_mode == 'ftp_only' and not get_setting(
            'ftp.upload_images_in_standard_mode') and not enable_timeseries_realtime and not variables_to_process:
        logger.info(f"Mode 'ftp_only' for scan, no other processing. Queuing: {filepath}")
        # ... (FTP queuing logic as before, using original filepath) ...
        queued_ok = all(add_scan_to_queue(filepath, s_conf) for s_conf in ftp_servers) if ftp_servers else True
        return queued_ok

    ds = None
    ds_geo = None
    precise_scan_timestamp: Optional[datetime] = None
    scan_elevation: Optional[float] = None
    overall_success = True  # Assume success, set to False on critical failures

    try:
        # 2. Read scan data (opens file once)
        # Determine variables needed for reading (plots + timeseries defaults)
        vars_for_read = set(variables_to_process.keys()) if variables_to_process else set()
        if enable_timeseries_realtime:
            vars_for_read.update(get_setting('app.timeseries_default_variables', []))

        logger.debug(f"Reading scan {filepath} for variables: {vars_for_read or 'all (coords only)'}")
        ds = read_scan(filepath, variables=list(vars_for_read) if vars_for_read else None)

        if ds is None:
            logger.error(f"Failed to read scan file: {filepath}");
            return False

        # 3. Extract precise timestamp & elevation from dataset
        try:
            time_val_np64 = ds['time'].values.item()
            py_dt = pd.to_datetime(time_val_np64).to_pydatetime()
            precise_scan_timestamp = py_dt.replace(tzinfo=timezone.utc) if py_dt.tzinfo is None \
                else py_dt.astimezone(timezone.utc)
            scan_elevation = float(ds['elevation'].item())
            logger.info(
                f"Scan properties: PreciseTS={precise_scan_timestamp.isoformat()}, Elev={scan_elevation:.2f}, Seq={sequence_number}")
        except Exception as e:
            logger.error(f"Failed to extract precise timestamp or elevation from dataset: {e}", exc_info=True)
            overall_success = False;
            raise  # Re-raise to stop processing if these are missing

        # 4. Georeference
        ds_geo = georeference_dataset(ds)
        if not ('x' in ds_geo.coords and 'y' in ds_geo.coords):
            logger.error(f"Georeferencing failed for {filepath}. Missing x/y coords.")
            overall_success = False;
            raise ValueError("Georeferencing failed")

        # 5. Update timeseries data in DB (if enabled)
        if enable_timeseries_realtime and overall_success:
            logger.info(f"Updating timeseries in DB for scan at {precise_scan_timestamp.isoformat()}...")
            # Pass ds_geo, and the now confirmed precise_scan_timestamp and scan_elevation
            update_timeseries_for_scan(ds_geo, precise_scan_timestamp, scan_elevation)
            logger.info("Timeseries DB update call finished.")

        # 6. Generate and save plots (if enabled)
        saved_image_paths: Dict[str, str] = {}
        if variables_to_process and images_dir and overall_success:
            # ... (plotting logic remains largely the same, using ds_geo, scan_elevation, precise_scan_timestamp for filenames/titles)
            # Ensure plot filenames use precise_scan_timestamp and scan_elevation for consistency
            dt_plot = precise_scan_timestamp  # Use precise for consistency
            elevation_plot = scan_elevation
            date_str_plot = dt_plot.strftime("%Y%m%d")
            datetime_file_str_plot = dt_plot.strftime("%Y%m%d_%H%M")  # Minute precision for plot filename
            elevation_code_plot = f"{int(round(elevation_plot * 100)):03d}"

            num_plots_succeeded = 0
            for variable_name in variables_to_process.keys():
                if variable_name not in ds_geo.data_vars: logger.warning(
                    f"Plot var {variable_name} not in ds_geo."); continue
                plot_style = get_plot_style(variable_name)
                if not plot_style: logger.warning(f"No plot style for {variable_name}."); continue

                image_bytes = create_ppi_image(ds_geo, variable_name, plot_style, watermark_path, watermark_zoom)
                if image_bytes:
                    # ... (image saving logic as before, using derived plot dt/elev/var for path) ...
                    image_sub_dir = os.path.join(images_dir, variable_name, date_str_plot, elevation_code_plot)
                    os.makedirs(image_sub_dir, exist_ok=True)
                    image_filename = f"{variable_name}_{elevation_code_plot}_{datetime_file_str_plot}.png"
                    image_filepath = os.path.join(image_sub_dir, image_filename)
                    try:
                        with open(image_filepath, 'wb') as f:
                            f.write(image_bytes)
                        logger.info(f"Saved plot: {image_filepath}")
                        saved_image_paths[variable_name] = image_filepath;
                        num_plots_succeeded += 1
                        if realtime_dir_plots:  # Copy to realtime plot dir
                            rt_fname = f"realtime_{variable_name}_{elevation_code_plot}.png"
                            # ... (realtime copy logic as before) ...
                            rt_fpath = os.path.join(realtime_dir_plots, rt_fname)
                            rt_fpath_tmp = f"{rt_fpath}.{os.getpid()}.tmp"
                            try:
                                os.makedirs(realtime_dir_plots, exist_ok=True)
                                shutil.copyfile(image_filepath, rt_fpath_tmp)
                                os.replace(rt_fpath_tmp, rt_fpath)
                                logger.debug(f"Updated realtime plot: {rt_fpath}")
                            except OSError as rt_e:
                                logger.error(f"Failed realtime plot update {rt_fpath}: {rt_e}")
                    except IOError as e:
                        logger.error(f"Failed to save plot {image_filepath}: {e}"); overall_success = False
                else:
                    logger.warning(f"Failed to generate image bytes for {variable_name}."); overall_success = False
            if not num_plots_succeeded and variables_to_process: overall_success = False

        # --- Steps after all data extraction and plotting ---
        if not overall_success:
            logger.error(f"Critical error during processing of {filepath}. Aborting before move and DB logging.")
            return False  # Return early before move/log if something went wrong

        # 7. Move processed scan file
        moved_filepath = filepath  # Path of the scan file if not moved
        if move_local_in_standard:
            if not output_dir:  # Should have been caught earlier, but re-check
                logger.error("Cannot move file: output_dir is not set, but move_local_in_standard is true.")
                overall_success = False
            elif scan_elevation is None or precise_scan_timestamp is None:  # Should not happen if we re-raised earlier
                logger.error("Cannot move file: scan elevation or precise timestamp is missing.")
                overall_success = False
            else:
                try:
                    logger.info(f"Moving processed file {filepath}...")
                    # Pass original filepath, base output_dir, and the extracted scan_elevation & precise_scan_timestamp
                    moved_path_result = move_processed_file(filepath, output_dir, scan_elevation,
                                                            precise_scan_timestamp)
                    if moved_path_result:
                        moved_filepath = moved_path_result  # Update to the new path
                        logger.info(f"Successfully moved scan to: {moved_filepath}")
                    else:  # move_processed_file returned None, implying a non-critical error it handled
                        logger.warning(
                            f"Scan file move from {filepath} might have had an issue (returned None). File may remain at original location.")
                        # overall_success = False # Decide if this is fatal
                except Exception as move_err:  # Catch errors from move_processed_file if it raises
                    logger.error(f"Failed to move scan file {filepath}: {move_err}", exc_info=True)
                    overall_success = False
        else:
            logger.info(f"Local move disabled. Processed file remains at: {filepath}")

        if not overall_success:
            logger.error(f"Error after data processing or during move for {filepath}. Aborting before DB logging.")
            return False

        # 8. Log scan details to radproc_scan_log in DB
        conn_log = None
        try:
            conn_log = get_connection()
            logger.info(f"Logging scan to radproc_scan_log: {moved_filepath}")
            # volume_identifier is initially NULL, to be filled by group-volumes job
            add_scan_to_log(conn_log, moved_filepath, precise_scan_timestamp, scan_elevation,
                            sequence_number, nominal_timestamp, volume_identifier=None)
        except Exception as log_err:
            logger.error(f"Failed to log scan {moved_filepath} to database: {log_err}", exc_info=True)
            # This might be considered non-fatal for the file processing itself, but bad for catalog.
            # overall_success = False # Optional: make this a fatal error
        finally:
            if conn_log: release_connection(conn_log)

        # 9. Queue file for FTP upload (if enabled)
        # Uses `moved_filepath` if successfully moved, otherwise original `filepath`
        # The ftp_client's upload_scan_file now determines its own remote path using get_scan_elevation
        upload_path_for_ftp = moved_filepath if move_local_in_standard and moved_filepath != filepath else filepath

        upload_scans_in_standard_mode = get_setting('ftp.upload_scans_in_standard_mode', False)
        upload_images_in_standard_mode = get_setting('ftp.upload_images_in_standard_mode', False)

        if scan_upload_mode == 'standard' and upload_scans_in_standard_mode:
            logger.info(f"Queuing scan for FTP (standard mode): {upload_path_for_ftp}")
            for server_config in ftp_servers: add_scan_to_queue(upload_path_for_ftp, server_config)
        elif scan_upload_mode == 'ftp_only':  # This case might be hit if other processing was also enabled
            logger.info(f"Queuing scan for FTP (ftp_only mode after other processing): {filepath}")
            for server_config in ftp_servers: add_scan_to_queue(filepath, server_config)

        if saved_image_paths and get_setting('ftp.image_upload_mode', 'disabled') != 'disabled':
            should_queue_images = False
            if get_setting('ftp.image_upload_mode') == 'only':
                should_queue_images = True
            elif get_setting('ftp.image_upload_mode') == 'also' and \
                    (scan_upload_mode == 'ftp_only' or (
                            scan_upload_mode == 'standard' and upload_images_in_standard_mode)):
                should_queue_images = True

            if should_queue_images:
                logger.info(f"Queuing {len(saved_image_paths)} plot images for FTP...")
                for img_path in saved_image_paths.values():
                    for server_config in ftp_servers: add_image_to_queue(img_path, server_config)

        logger.info(f"Processing finished for {original_filename}. Overall success: {overall_success}")
        return overall_success

    except Exception as e:
        logger.error(f"Unhandled error during processing of {filepath}: {e}", exc_info=True)
        return False
    finally:
        # Ensure datasets are closed
        if ds_geo is not None:
            try:
                ds_geo.close()
            except Exception:
                pass
        if ds is not None and ds is not ds_geo:  # If ds_geo is a different object or ds_geo failed
            try:
                ds.close()
            except Exception:
                pass

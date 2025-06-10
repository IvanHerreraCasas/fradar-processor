# core/processor.py

import os
import logging
import pandas as pd
import  xarray as xr
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import shutil

# --- Core Imports ---
from .config import get_setting, get_config
from .data import read_ppi_scan
from .utils.geo import georeference_dataset
from .utils.helpers import move_processed_file, parse_datetime_from_filename, parse_scan_sequence_number
from .visualization.style import get_plot_style
from .visualization.plotter import create_ppi_image
from .utils.upload_queue import add_scan_to_queue, add_image_to_queue
from .analysis import update_timeseries_for_scan
from .db_manager import get_connection, release_connection, add_scan_to_log


logger = logging.getLogger(__name__)


def _read_and_georeference_scan(filepath: str) -> Optional[Tuple[xr.Dataset, datetime, float, int]]:
    """
    Reads, extracts metadata from, and georeferences a single raw scan file.

    Args:
        filepath: The full path to the raw scan file.

    Returns:
        A tuple containing the georeferenced xarray Dataset, the precise UTC
        timestamp, the scan elevation, and the sequence number. Returns None on failure.
    """
    logger.info(f"Reading and georeferencing: {os.path.basename(filepath)}")
    sequence_number = parse_scan_sequence_number(os.path.basename(filepath))
    if sequence_number is None:
        logger.error(f"Could not parse sequence number from {os.path.basename(filepath)}.")
        return None

    # Determine which variables to read based on config
    variables_to_process: Dict[str, str] = get_setting('app.variables_to_process', {})
    enable_timeseries_realtime = get_setting('app.enable_timeseries_updates', False)
    vars_for_read = set(variables_to_process.keys()) if variables_to_process else set()
    if enable_timeseries_realtime:
        vars_for_read.update(get_setting('app.timeseries_default_variables', []))

    ds = read_ppi_scan(filepath, variables=list(vars_for_read) if vars_for_read else None)
    if ds is None:
        logger.error(f"Failed to read scan file: {filepath}")
        return None

    try:
        time_val_np64 = ds['time'].values.item()
        py_dt = pd.to_datetime(time_val_np64).to_pydatetime()
        precise_scan_timestamp = py_dt.replace(tzinfo=timezone.utc) if py_dt.tzinfo is None else py_dt.astimezone(timezone.utc)
        scan_elevation = float(ds['elevation'].item())
        logger.info(f"Scan properties: PreciseTS={precise_scan_timestamp.isoformat()}, Elev={scan_elevation:.2f}, Seq={sequence_number}")

        ds_geo = georeference_dataset(ds)
        if not ('x' in ds_geo.coords and 'y' in ds_geo.coords):
            logger.error(f"Georeferencing failed for {filepath}. Missing x/y coords.")
            ds.close()
            return None

        # ds is now part of ds_geo, so we only need to manage ds_geo
        return ds_geo, precise_scan_timestamp, scan_elevation, sequence_number
    except Exception as e:
        logger.error(f"Error extracting metadata or georeferencing {filepath}: {e}", exc_info=True)
        if ds: ds.close()
        return None


def _run_raw_data_pipelines(ds_geo: xr.Dataset, precise_scan_timestamp: datetime, scan_elevation: float) -> Dict[str, str]:
    """
    Runs processing pipelines that use the raw (but georeferenced) data,
    such as plotting and real-time timeseries updates.

    Args:
        ds_geo: The georeferenced xarray Dataset.
        precise_scan_timestamp: The precise UTC timestamp of the scan.
        scan_elevation: The elevation of the scan.

    Returns:
        A dictionary of paths to any saved plot images.
    """
    saved_image_paths = {}

    # Real-time timeseries update pipeline
    if get_setting('app.enable_timeseries_updates', False):
        logger.info("Running real-time timeseries update pipeline...")
        update_timeseries_for_scan(ds_geo, precise_scan_timestamp, scan_elevation)

    # Plotting pipeline
    variables_to_process = get_setting('app.variables_to_process', {})
    if variables_to_process and get_setting('app.images_dir'):
        logger.info("Running plotting pipeline...")
        images_dir = get_setting('app.images_dir')
        watermark_path = get_setting('app.watermark_path')
        watermark_zoom = get_setting('styles.defaults.watermark_zoom', 0.05)
        realtime_dir_plots = get_setting('app.realtime_image_dir')

        dt_plot = precise_scan_timestamp
        elevation_plot = scan_elevation
        date_str_plot = dt_plot.strftime("%Y%m%d")
        datetime_file_str_plot = dt_plot.strftime("%Y%m%d_%H%M")
        elevation_code_plot = f"{int(round(elevation_plot * 100)):03d}"

        for var_name in variables_to_process.keys():
            if var_name not in ds_geo.data_vars:
                logger.warning(f"Plot variable '{var_name}' not found in dataset. Skipping plot.")
                continue

            plot_style = get_plot_style(var_name)
            if not plot_style:
                logger.warning(f"No plot style defined for '{var_name}'. Skipping plot.")
                continue

            image_bytes = create_ppi_image(ds_geo, var_name, plot_style, watermark_path, watermark_zoom)
            if image_bytes:
                image_sub_dir = os.path.join(images_dir, var_name, date_str_plot, elevation_code_plot)
                os.makedirs(image_sub_dir, exist_ok=True)
                image_filename = f"{var_name}_{elevation_code_plot}_{datetime_file_str_plot}.png"
                image_filepath = os.path.join(image_sub_dir, image_filename)
                try:
                    with open(image_filepath, 'wb') as f: f.write(image_bytes)
                    logger.info(f"Saved plot: {image_filepath}")
                    saved_image_paths[var_name] = image_filepath

                    if realtime_dir_plots:
                        rt_fname = f"realtime_{var_name}_{elevation_code_plot}.png"
                        rt_fpath = os.path.join(realtime_dir_plots, rt_fname)
                        rt_fpath_tmp = f"{rt_fpath}.{os.getpid()}.tmp"
                        shutil.copyfile(image_filepath, rt_fpath_tmp)
                        os.replace(rt_fpath_tmp, rt_fpath)
                except IOError as e:
                    logger.error(f"Failed to save plot {image_filepath}: {e}")
            else:
                logger.warning(f"Failed to generate image bytes for {var_name}.")
    return saved_image_paths


def _handle_file_movement_and_logging(filepath: str, elevation: float, timestamp: datetime, seq_num: int) -> Tuple[Optional[str], Optional[int]]:
    """
    Moves the raw scan file to its final destination and logs it to the database.

    Args:
        filepath: The current path of the raw scan file.
        elevation: The scan's elevation angle.
        timestamp: The scan's precise UTC timestamp.
        seq_num: The scan's sequence number.

    Returns:
        A tuple containing the new path of the moved file and its scan_log_id.
    """
    output_dir = get_setting('app.output_dir')
    moved_filepath = filepath

    if get_setting('app.move_file_after_processing', True):
        logger.info(f"Moving processed file {filepath}...")
        moved_path_result = move_processed_file(filepath, output_dir, elevation, timestamp)
        if moved_path_result:
            moved_filepath = moved_path_result
        else:
            logger.warning(f"Scan file move from {filepath} might have had an issue.")
    else:
        logger.info(f"Local file move disabled. Processed file remains at: {filepath}")

    conn = get_connection()
    scan_log_id = None
    try:
        nominal_ts = parse_datetime_from_filename(os.path.basename(filepath))
        scan_log_id = add_scan_to_log(conn, moved_filepath, timestamp, elevation, seq_num, nominal_ts)
        if scan_log_id is None:
            logger.error(f"Failed to log raw scan {moved_filepath} to database or retrieve its ID.")
    finally:
        release_connection(conn)

    return moved_filepath, scan_log_id


def _queue_files_for_upload(scan_filepath: str, saved_image_paths: Dict[str, str]):
    """
    Queues the primary scan file and any generated images for FTP upload
    based on the application configuration.

    Args:
        scan_filepath: The final path of the processed scan file.
        saved_image_paths: A dictionary of paths to saved plot images.
    """
    logger.info("Checking FTP queueing requirements...")
    ftp_servers = get_setting('ftp.servers', [])
    if not ftp_servers:
        logger.info("No FTP servers configured. Skipping queueing.")
        return

    scan_upload_mode = get_setting('ftp.scan_upload_mode', 'disabled')
    upload_scans_in_standard_mode = get_setting('ftp.upload_scans_in_standard_mode', False)
    upload_images_in_standard_mode = get_setting('ftp.upload_images_in_standard_mode', False)

    if scan_upload_mode == 'standard' and upload_scans_in_standard_mode:
        logger.info(f"Queuing scan for FTP (standard mode): {scan_filepath}")
        for server_config in ftp_servers: add_scan_to_queue(scan_filepath, server_config)

    if saved_image_paths and get_setting('ftp.image_upload_mode', 'disabled') != 'disabled':
        should_queue_images = False
        image_upload_mode = get_setting('ftp.image_upload_mode')
        if image_upload_mode == 'only' or (image_upload_mode == 'also' and upload_images_in_standard_mode):
            should_queue_images = True

        if should_queue_images:
            logger.info(f"Queuing {len(saved_image_paths)} plot images for FTP...")
            for img_path in saved_image_paths.values():
                for server_config in ftp_servers: add_image_to_queue(img_path, server_config)


def process_new_scan(filepath: str, config_override: Optional[Dict[str, Any]] = None) -> bool:
    """
    Processes a single raw radar scan file by executing a pipeline of
    reading, correcting, saving, and logging.

    Args:
        filepath: The full path to the incoming raw scan file.
        config_override: Optional dictionary to override global config.

    Returns:
        True if the entire process completes successfully, False otherwise.
    """
    app_config = get_config() if config_override is None else config_override
    logger.info(f"=== Starting New Scan Processing: {os.path.basename(filepath)} ===")

    if not os.path.exists(filepath):
        logger.error(f"Input file does not exist: {filepath}")
        return False

    ds_geo = None
    overall_success = True

    try:
        # --- Stage 1: Read and Prepare Data ---
        read_result = _read_and_georeference_scan(filepath)
        if read_result is None:
            return False  # Critical failure, cannot proceed.
        ds_geo, precise_ts, elevation, seq_num = read_result

        # --- Stage 2: Process Raw Data Products (Plots, etc.) ---
        saved_image_paths = _run_raw_data_pipelines(ds_geo, precise_ts, elevation)

        # --- Stage 3: Handle Raw File Logistics ---
        moved_filepath, scan_log_id = _handle_file_movement_and_logging(filepath, elevation, precise_ts, seq_num)
        if not moved_filepath or not scan_log_id:
            logger.error("Failed to move raw file or log it to the database. Aborting further processing.")
            return False # This is a critical step for linking corrected data.

        # --- Stage 4: Queue files for upload ---
        _queue_files_for_upload(moved_filepath, saved_image_paths)

    except Exception as e:
        logger.error(f"Unhandled exception in process_new_scan for {filepath}: {e}", exc_info=True)
        overall_success = False
    finally:
        if ds_geo:
            try:
                ds_geo.close()
            except Exception:
                pass
        logger.info(f"=== Finished Processing: {os.path.basename(filepath)}. Success: {overall_success} ===")

    return overall_success
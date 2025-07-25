#!/usr/bin/env python3
from typing import Optional, List
import matplotlib
try:
    matplotlib.use("Agg")
    print("Matplotlib backend set to 'Agg'.")
except Exception as e:
    print(f"Warning: Could not set Matplotlib backend to 'Agg': {e}")

import argparse
import os
import sys
import logging
import pandas as pd
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta
from pathlib import Path
import psycopg2

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# --- Import Core Functions ---
from ..core.config import load_config, get_setting, get_all_points_config
from ..core.file_monitor import start_monitoring
from ..core.analysis import generate_timeseries, calculate_accumulation
from ..core.visualization.animator import create_animation
from ..core.utils.upload_queue import start_worker, stop_worker
from ..core.data import extract_scan_key_metadata
from ..core.utils.helpers import parse_datetime_from_filename, move_processed_file
from ..core.db_manager import get_connection, release_connection, add_scan_to_log, \
    get_ungrouped_scans_for_volume_assignment, update_volume_identifier_for_scans, find_latest_scan_for_sequence, \
    get_unprocessed_volume_identifiers
from ..core import plotting_manager
from ..core.volume_processor import process_volume

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
# ---------------------------------

# --- Logging Setup (remains the same) ---
def _setup_logger(
    log_file: str,
    log_level_file_str: str = 'DEBUG',
    log_level_console_str: str = 'INFO',
    log_max_bytes: int = 5*1024*1024,
    log_backup_count: int = 5
    ):
        """Sets up file and console logging using configuration values."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        log_level_file = getattr(logging, log_level_file_str.upper(), logging.DEBUG)
        log_level_console = getattr(logging, log_level_console_str.upper(), logging.INFO)
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                print(f"Created log directory: {log_dir}")
            file_handler = RotatingFileHandler(log_file, maxBytes=log_max_bytes, backupCount=log_backup_count, encoding='utf-8')
            file_handler.setLevel(log_level_file)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            print(f"File logging enabled: Level={logging.getLevelName(log_level_file)}, File={log_file}")
        except PermissionError: print(f"ERROR: Permission denied to write log file: {log_file}", file=sys.stderr)
        except Exception as e: print(f"ERROR: Failed to set up file logger ({log_file}): {e}", file=sys.stderr)
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level_console)
            console_formatter = logging.Formatter('%(levelname)s: [%(name)s] %(message)s')
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
            print(f"Console logging enabled: Level={logging.getLevelName(log_level_console)}")
        except Exception as e: print(f"ERROR: Failed to set up console logger: {e}", file=sys.stderr)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        root_logger.info(f"Logging initialized.")
        return root_logger

# --- TQDM Fallback ---
def tqdm_fallback_iterator(iterable, *args, **kwargs):
    """A simple generator fallback if tqdm is not installed."""
    return iter(iterable)

# --- CLI Command Functions ---

def cli_run(args: argparse.Namespace):
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Monitor Mode ===")
    worker_started = False
    try:
        logger.info("Starting FTP queue worker thread...")
        start_worker()
        worker_started = True
        logger.info("Starting file monitor...")
        start_monitoring()
        logger.info("Monitor Mode process finished.")
    except KeyboardInterrupt:
         logger.info("Monitor Mode interrupted by user.")
    except Exception as e:
        logger.exception("!!! Monitor Mode Failed Unexpectedly !!!")
        sys.exit(1)
    finally:
         if worker_started:
             logger.info("Signalling FTP queue worker to stop...")
             stop_worker()
             logger.info("Exiting Monitor Mode.")
         else:
             logger.info("Exiting Monitor Mode (worker not started or failed).")

def cli_timeseries(args: argparse.Namespace):
    """Handler for the 'timeseries' command."""
    logger = logging.getLogger(__name__)
    logger.info(f"=== Starting Historical Timeseries Generation ===")

    # Logic to determine which points to process (same as before)
    points_to_process_names: List[str]
    if args.points:
        points_to_process_names = args.points
    else:
        all_points_configs = get_all_points_config()
        if not all_points_configs:
            logger.warning("No points found in the database. Nothing to process.")
            return
        points_to_process_names = [p['point_name'] for p in all_points_configs if 'point_name' in p]

    if not points_to_process_names:
        logger.warning("No points determined for processing. Exiting.")
        return

    dt_format = "%Y%m%d_%H%M"
    try:
        start_dt_utc = datetime.strptime(args.start, dt_format).replace(tzinfo=timezone.utc)
        end_dt_utc = datetime.strptime(args.end, dt_format).replace(tzinfo=timezone.utc)
    except ValueError:
        logger.error(f"Invalid date format. Please use YYYYMMDD_HHMM.")
        sys.exit(1)

    # Call the new orchestrator function
    generate_timeseries(
        point_names=points_to_process_names,
        start_dt=start_dt_utc,
        end_dt=end_dt_utc,
        specific_variables=[args.variable] if args.variable else None,
        source=args.source, # To be added when analysis functions are complete
        version=args.version,
        interactive_mode=True,
    )

def cli_accumulate(args: argparse.Namespace):
    """Handler for the 'accumulate' command."""
    logger = logging.getLogger(__name__)
    logger.info(f"=== Starting Precipitation Accumulation ===")
    logger.info(f"Point: '{args.point_name}', Range: {args.start} -> {args.end}, Interval: {args.interval}, Variable: {args.variable or 'RATE'}")
    try: pd.Timedelta(args.interval)
    except ValueError: logger.error(f"Invalid interval: '{args.interval}'."); sys.exit(1)
    dt_format = "%Y%m%d_%H%M"
    try:
        start_dt_utc = datetime.strptime(args.start, dt_format).replace(tzinfo=timezone.utc)
        end_dt_utc = datetime.strptime(args.end, dt_format).replace(tzinfo=timezone.utc)
    except ValueError: logger.error(f"Invalid date format."); sys.exit(1)
    if start_dt_utc >= end_dt_utc: logger.error("Start must be before end."); sys.exit(1)
    output_file = args.output_file
    if not output_file:
        timeseries_dir = get_setting('app.timeseries_dir')
        if not timeseries_dir: logger.error("'app.timeseries_dir' must be set."); sys.exit(1)
        rate_variable = args.variable or 'RATE'
        default_filename = f"{args.point_name}_{rate_variable}_{args.interval}_acc.csv"
        output_file = os.path.join(timeseries_dir, default_filename)
        logger.info(f"Using default output: {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        success = calculate_accumulation(
            point_name=args.point_name,
            start_dt=start_dt_utc,
            end_dt=end_dt_utc,
            interval=args.interval,
            rate_variable=(args.variable or 'RATE'),
            output_file_path=output_file,
            source=args.source,
            version=args.version
        )
        if success: logger.info(f"Accumulation finished. Output: {output_file}")
        else: logger.error("Accumulation failed."); sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred during accumulation:")
        sys.exit(1)

def cli_animate(args: argparse.Namespace):
    # ... (existing code remains the same) ...
    logger = logging.getLogger(__name__)
    logger.info(f"=== Starting Animation Generation ===")
    dt_format = "%Y%m%d_%H%M"
    try:
        start_dt_utc = datetime.strptime(args.start, dt_format).replace(tzinfo=timezone.utc)
        end_dt_utc = datetime.strptime(args.end, dt_format).replace(tzinfo=timezone.utc)
        elevation_float = float(args.elevation)
    except ValueError: logger.error(f"Invalid date or elevation format."); sys.exit(1)
    if start_dt_utc >= end_dt_utc: logger.error("Start must be before end."); sys.exit(1)
    try:
        success = create_animation(
            variable=args.variable,
            elevation=elevation_float,
            start_dt=start_dt_utc,
            end_dt=end_dt_utc,
            output_filename=args.output_file,
            plot_extent=args.extent,
            include_watermark=(not args.no_watermark),
            fps=args.fps
        )
        if success: logger.info(f"Animation finished. Output: {args.output_file}")
        else: logger.error("Animation failed."); sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred during animation:")
        sys.exit(1)

def cli_reorg_scans(args: argparse.Namespace):
    """
    Handler for the 'reorg-scans' command.
    Reorganizes processed scan files from YYYYMMDD/ structure to
    ElevationCode/YYYYMMDD/ structure.
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Scan File Reorganization ===")

    output_dir = args.output_dir if args.output_dir else get_setting('app.output_dir')
    if not output_dir:
        logger.error("Output directory not specified and not found in config ('app.output_dir'). Cannot proceed.")
        sys.exit(1)
    if not os.path.isdir(output_dir):
        logger.error(f"Specified output directory does not exist: {output_dir}")
        sys.exit(1)

    logger.info(f"Scanning directory: {output_dir}")
    if args.dry_run:
        logger.info("DRY RUN active: No files will be moved or directories deleted.")
    if args.delete_empty:
        logger.info("DELETE EMPTY active: Empty YYYYMMDD source directories will be deleted after successful moves.")

    moved_count = 0
    skipped_count = 0
    error_count = 0
    deleted_dir_count = 0

    old_date_dirs_to_process = []
    for item_name in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item_name)
        if os.path.isdir(item_path) and item_name.isdigit() and len(item_name) == 8: # Simple check for YYYYMMDD
            old_date_dirs_to_process.append(item_path)

    if not old_date_dirs_to_process:
        logger.info("No old-structure YYYYMMDD directories found to process.")
        logger.info("--- Reorganization Summary ---")
        logger.info("No files processed.")
        return

    dir_iterator_func = tqdm if TQDM_AVAILABLE else tqdm_fallback_iterator
    dir_iterator = dir_iterator_func(old_date_dirs_to_process, desc="Date Dirs", unit="dir", leave=False)

    for date_dir_path in dir_iterator:
        if TQDM_AVAILABLE:
            dir_iterator.set_description(f"Processing Dir: {os.path.basename(date_dir_path)}")
        else:
            logger.info(f"Processing Date Directory: {os.path.basename(date_dir_path)}")

        scnx_files_in_dir = [f for f in os.listdir(date_dir_path) if f.endswith(".scnx.gz")]

        if not scnx_files_in_dir:
            logger.info(f"No .scnx.gz files found in {date_dir_path}.")
            if args.delete_empty:
                try:
                    if not os.listdir(date_dir_path):
                        log_msg = f"DRY RUN: Would delete empty directory: {date_dir_path}" if args.dry_run else f"Deleting empty directory: {date_dir_path}"
                        logger.info(log_msg)
                        if not args.dry_run:
                            os.rmdir(date_dir_path)
                            deleted_dir_count += 1
                    else:
                        logger.info(f"Directory {date_dir_path} is not empty (contains other files). Will not delete.")
                except OSError as e:
                    logger.error(f"Failed to delete directory {date_dir_path}: {e}")
            continue


        files_successfully_processed_from_dir = 0

        file_iterator_func = tqdm if TQDM_AVAILABLE else tqdm_fallback_iterator
        file_iterator = file_iterator_func(scnx_files_in_dir, desc=f"Files in {os.path.basename(date_dir_path)}",
                                           unit="file", leave=False)

        for filename in file_iterator:
            source_filepath = os.path.join(date_dir_path, filename)

            # --- MODIFIED SECTION TO USE extract_scan_key_metadata ---
            metadata = extract_scan_key_metadata(source_filepath)
            elevation: Optional[float] = None
            if metadata:
                _precise_ts, elevation, _sequence_num = metadata # We primarily need elevation here
            # ---------------------------------------------------------

            scan_datetime = parse_datetime_from_filename(filename) # Nominal datetime from filename

            if elevation is None: # Check if elevation was successfully extracted
                logger.warning(f"Could not get elevation for {source_filepath} using extract_scan_key_metadata. Skipping.")
                skipped_count += 1
                continue
            if scan_datetime is None: # Nominal datetime is still useful for directory structure if precise isn't used there
                logger.warning(f"Could not parse nominal datetime from filename {filename}. Skipping.")
                skipped_count += 1
                continue

            # The move_processed_file function uses the provided elevation and scan_datetime
            # to construct the new path.
            elevation_code = f"{int(round(elevation * 100)):03d}"
            date_str = scan_datetime.strftime("%Y%m%d") # Use nominal datetime for path structure
            new_target_dir_check = os.path.join(output_dir, elevation_code, date_str)
            new_target_filepath_check = os.path.join(new_target_dir_check, filename)

            if os.path.exists(new_target_filepath_check):
                logger.warning(
                    f"Target file {new_target_filepath_check} already exists. Skipping move for {source_filepath}.")
                skipped_count += 1
                files_successfully_processed_from_dir += 1
                continue

            log_msg_dry_run = f"DRY RUN: Would move '{source_filepath}' to new structure (Elev: {elevation:.2f}, Nominal DT: {scan_datetime.isoformat()})"
            log_msg_action = f"Attempting to move '{source_filepath}'..."
            logger.info(log_msg_dry_run if args.dry_run else log_msg_action)

            if not args.dry_run:
                try:
                    # move_processed_file takes the elevation and the datetime for target path construction
                    moved_path = move_processed_file(source_filepath, output_dir, elevation, scan_datetime)
                    if moved_path:
                        logger.info(f"Successfully moved to: {moved_path}")
                        moved_count += 1
                        files_successfully_processed_from_dir += 1
                    else:
                        logger.error(f"Move failed for {source_filepath} (move_processed_file returned None).")
                        error_count += 1
                except Exception as e:
                    logger.error(f"Error moving file {source_filepath}: {e}", exc_info=True)
                    error_count += 1
            else:
                moved_count += 1
                files_successfully_processed_from_dir += 1
        if args.delete_empty and files_successfully_processed_from_dir == len(scnx_files_in_dir):
            try:
                if not os.listdir(date_dir_path):
                    log_msg = f"DRY RUN: Would delete successfully processed and now empty YYYYMMDD directory: {date_dir_path}" if args.dry_run else f"Deleting successfully processed and now empty YYYYMMDD directory: {date_dir_path}"
                    logger.info(log_msg)
                    if not args.dry_run:
                        os.rmdir(date_dir_path)
                        deleted_dir_count += 1
                else:
                    logger.info(f"Directory {date_dir_path} processed but still contains other files. Will not delete.")
            except OSError as e:
                logger.error(f"Failed to delete source directory {date_dir_path}: {e}")


    logger.info("--- Reorganization Summary ---")
    logger.info(f"Files processed for moving (actual moves or dry run actions): {moved_count}")
    logger.info(f"Files skipped (target existed, or error getting elev/dt): {skipped_count}")
    logger.info(f"Errors during move operations: {error_count}")
    if args.delete_empty:
        logger.info(f"Old YYYYMMDD directories deleted: {deleted_dir_count}")
    logger.info("Scan file reorganization finished.")

def cli_index_scans(args: argparse.Namespace):
    """Handler for the 'index-scans' command."""
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Scan Indexing ===")

    output_dir = args.output_dir if args.output_dir else get_setting('app.output_dir')
    if not output_dir:
        logger.error("Output directory not specified and not found in config ('app.output_dir'). Cannot proceed.")
        sys.exit(1)
    if not os.path.isdir(output_dir):
        logger.error(f"Specified output directory does not exist: {output_dir}")
        sys.exit(1)

    logger.info(f"Scanning directory: {output_dir}")
    if args.dry_run:
        logger.info("DRY RUN active: No changes will be made to the database.")

    found_count = 0
    indexed_count = 0
    skipped_count = 0
    error_count = 0
    conn = None

    try:
        conn = get_connection()
        logger.info("Database connection established.")

        files_to_process = []
        logger.info("Collecting scan files...")
        # Walk through ElevationCode/YYYYMMDD/ structure
        for root, dirs, files in os.walk(output_dir):
            for filename in files:
                if filename.endswith(".scnx.gz"):
                    files_to_process.append(os.path.join(root, filename))

        if not files_to_process:
            logger.info("No .scnx.gz files found to index.")
            return

        logger.info(f"Found {len(files_to_process)} scan files to process.")

        # Determine iterator (tqdm or fallback)
        iterator_func = tqdm if TQDM_AVAILABLE else tqdm_fallback_iterator
        file_iterator = iterator_func(files_to_process, desc="Indexing Scans", unit="file")

        for filepath in file_iterator:
            found_count += 1
            logger.debug(f"Processing: {filepath}")

            metadata = extract_scan_key_metadata(filepath)
            if metadata is None:
                logger.warning(f"Could not extract metadata for {filepath}. Skipping.")
                skipped_count += 1
                continue

            precise_ts, elevation, sequence_num = metadata
            nominal_ts = parse_datetime_from_filename(os.path.basename(filepath))

            if args.dry_run:
                logger.info(f"DRY RUN: Would add to log: {filepath} | TS: {precise_ts} | Elev: {elevation} | Seq: {sequence_num}")
                indexed_count += 1
            else:
                try:
                    scan_id = add_scan_to_log(conn, filepath, precise_ts, elevation, sequence_num, nominal_ts, volume_identifier=None)
                    if scan_id is not None:
                        logger.debug(f"Indexed {filepath} with ID {scan_id}.")
                        indexed_count += 1
                    else:
                        # This happens if it already exists (ON CONFLICT DO NOTHING)
                        logger.debug(f"Scan {filepath} likely already indexed. Skipping DB insert.")
                        skipped_count += 1 # Count as skipped if already exists
                except Exception as db_err:
                    logger.error(f"Database error indexing {filepath}: {db_err}", exc_info=True)
                    error_count += 1

    except Exception as e:
        logger.exception(f"An unexpected error occurred during scan indexing:")
        error_count += 1 # Count a general error
    finally:
        if conn:
            release_connection(conn)
            logger.info("Database connection released.")

    logger.info("--- Indexing Summary ---")
    logger.info(f"Files Found:    {found_count}")
    logger.info(f"Files Indexed:  {indexed_count}")
    logger.info(f"Files Skipped:  {skipped_count} (Metadata error or already indexed)")
    logger.info(f"Errors:         {error_count}")
    logger.info("Scan indexing finished.")

def cli_group_volumes(args: argparse.Namespace):
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Volume Grouping (Scan-by-Scan Sequential Linking Logic) ===")
    if args.dry_run:
        logger.info("DRY RUN active: No changes will be made to the database.")

    conn = None
    scans_volume_id_assigned_count = 0
    error_count = 0 # For general errors, not for handled unique conflicts by db_manager
    # skipped_due_to_conflict_count is implicitly handled by db_manager logging and successful_updates count

    # Configuration for grouping logic
    # Max time allowed BETWEEN two sequential scans in the same volume (e.g., _000 -> _001)
    MAX_INTER_SCAN_GAP = timedelta(minutes=get_setting('app.max_inter_scan_gap_minutes', 1))
    # Max window to search BACKWARDS for a predecessor scan. Should be comfortably larger than MAX_INTER_SCAN_GAP.
    PREDECESSOR_SEARCH_WINDOW_SECONDS = int(
        (MAX_INTER_SCAN_GAP.total_seconds()) * 2.5)

    lookback_hours = args.lookback_hours
    limit = args.limit

    logger.info(f"Parameters: Lookback={lookback_hours}h, Limit={limit}, MaxInterScanGap={MAX_INTER_SCAN_GAP}")

    try:
        conn = get_connection()
        logger.info("Database connection established.")

        ungrouped_scans = get_ungrouped_scans_for_volume_assignment(conn, lookback_hours, limit)

        # Sort by precise_timestamp first, then sequence number for deterministic processing.
        # This helps ensure that if _000 and _001 arrive nearly simultaneously, _000 is processed first,
        # allowing _001 to find its grouped predecessor.
        ungrouped_scans.sort(key=lambda x: (x['precise_timestamp'], x['scan_sequence_number']))

        if not ungrouped_scans:
            logger.info("No ungrouped scans found to process.")
            return

        logger.info(f"Fetched {len(ungrouped_scans)} ungrouped scans for potential volume assignment.")
        iterator_func = tqdm if TQDM_AVAILABLE else tqdm_fallback_iterator

        for current_scan in iterator_func(ungrouped_scans, desc="Processing Scans for Volume ID", unit="scan"):
            scan_log_id = current_scan['scan_log_id']
            seq_num = current_scan['scan_sequence_number']
            current_ts = current_scan['precise_timestamp']
            # nominal_filename_timestamp is fetched by get_ungrouped_scans_for_volume_assignment
            current_nominal_ts = current_scan.get('nominal_filename_timestamp')

            target_volume_id_to_assign = None

            if seq_num == 0:
                # This is a base scan. It defines its own volume ID using its nominal timestamp, falling back to precise.
                target_volume_id_to_assign = current_nominal_ts if current_nominal_ts is not None else current_ts
                logger.debug(
                    f"Scan {scan_log_id} (Seq 0) defines new potential volume ID: {target_volume_id_to_assign.isoformat() if target_volume_id_to_assign else 'N/A'}")
            else:
                # Not a base scan (seq_num > 0). Try to find its predecessor S_{n-1}.
                predecessor_seq_num = seq_num - 1

                # Search for the predecessor S_{n-1} in the DB
                # We look for the latest S_{n-1} that occurred just before current_scan's timestamp
                predecessor_scan_data = find_latest_scan_for_sequence(
                    conn,
                    predecessor_seq_num,
                    current_ts,
                    PREDECESSOR_SEARCH_WINDOW_SECONDS
                )
                if predecessor_scan_data:
                    pred_ts = predecessor_scan_data['precise_timestamp']
                    pred_vol_id = predecessor_scan_data.get('volume_identifier')
                    if pred_vol_id is not None:
                        time_diff_from_pred = current_ts - pred_ts
                        if timedelta(seconds=0) < time_diff_from_pred <= MAX_INTER_SCAN_GAP:
                            target_volume_id_to_assign = pred_vol_id
                            logger.debug(
                                f"Scan {scan_log_id} (Seq {seq_num}) will attempt to join volume {target_volume_id_to_assign.isoformat()} "
                                f"from predecessor {predecessor_scan_data['scan_log_id']} (Seq {predecessor_seq_num}). Gap: {time_diff_from_pred}")
                        else:
                            logger.debug(
                                f"Scan {scan_log_id} (Seq {seq_num}): Predecessor {predecessor_scan_data['scan_log_id']} (Seq {predecessor_seq_num}) "
                                f"is grouped to {pred_vol_id.isoformat()}, but time gap {time_diff_from_pred} > {MAX_INTER_SCAN_GAP}. Cannot join.")
                    else:
                        logger.debug(
                            f"Scan {scan_log_id} (Seq {seq_num}): Predecessor {predecessor_scan_data['scan_log_id']} (Seq {predecessor_seq_num}) "
                            f"found but is not yet grouped. Scan {scan_log_id} remains ungrouped for now.")
                else:
                    logger.debug(
                        f"Scan {scan_log_id} (Seq {seq_num}): No suitable predecessor S_{predecessor_seq_num} found in DB search.")

            # If a volume ID was determined for the current scan, update it in the DB.
            if target_volume_id_to_assign:
                if args.dry_run:
                    logger.info(
                        f"DRY RUN: Would assign Volume ID {target_volume_id_to_assign.isoformat()} to scan {scan_log_id} (Seq {seq_num})")
                    # Increment count for dry run to simulate a successful assignment for summary purposes
                    scans_volume_id_assigned_count +=1
                else:
                    try:
                        # `update_volume_identifier_for_scans` now returns the count of successful updates.
                        # Since we are calling it with a list of one ID, it will return 1 on success,
                        # or 0 if the UniqueViolation was caught and handled within it for that ID.
                        updated_count = update_volume_identifier_for_scans(conn, [scan_log_id], target_volume_id_to_assign)
                        if updated_count > 0:
                             logger.info(
                                f"Assigned Volume ID {target_volume_id_to_assign.isoformat()} to scan {scan_log_id} (Seq {seq_num})")
                             scans_volume_id_assigned_count += updated_count
                        # else:
                        # The UniqueViolation is now handled and logged as a WARNING inside update_volume_identifier_for_scans.
                        # No specific "skipped_due_to_conflict_count" is needed here as db_manager handles the warning.
                        # The summary will reflect scans_volume_id_assigned_count correctly.

                    except psycopg2.Error as db_err: # Catch other DB errors NOT caught by db_manager's specific UniqueViolation
                        if conn and not conn.closed and conn.status != psycopg2.extensions.STATUS_IN_TRANSACTION:
                             pass
                        elif conn and not conn.closed :
                             conn.rollback()
                        logger.error(f"Database error assigning Volume ID to scan {scan_log_id} (main loop): {db_err}", exc_info=True)
                        error_count +=1
                    except Exception as e:
                        if conn and not conn.closed and conn.status != psycopg2.extensions.STATUS_IN_TRANSACTION:
                             pass
                        elif conn and not conn.closed :
                             conn.rollback()
                        logger.error(f"Unexpected error assigning Volume ID to scan {scan_log_id} (main loop): {e}", exc_info=True)
                        error_count +=1
    except Exception as e:
        logger.exception("An unexpected error occurred during volume grouping's main process:")
        error_count += 1 # Increment general error count
    finally:
        if conn:
            release_connection(conn)
            logger.info("Database connection released.")

    logger.info("--- Volume Grouping Summary ---")
    logger.info(
        f"Scans processed for Volume ID assignment: {len(ungrouped_scans) if 'ungrouped_scans' in locals() else 0}")
    logger.info(f"Scans successfully assigned a Volume ID in this run: {scans_volume_id_assigned_count}")
    # The skipped_due_to_conflict_count is now implicitly part of the difference between processed and assigned,
    # and the warnings are logged by db_manager.
    logger.info(f"Other errors during DB update/processing (main loop): {error_count}")
    logger.info("Volume grouping finished.")
    logger.info(
        f"Note: Scans that could not find a grouped predecessor or conflicted (logged as WARNING by db_manager) "
        f"remain ungrouped and will be re-evaluated in subsequent runs if they fall within the lookback period.")

def cli_process_volumes(args: argparse.Namespace):
    """Handler for the 'process-volumes' command."""
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Corrected Volume Generation ===")

    if not args.version:
        logger.error("A processing version must be specified with --version (e.g., --version v1_0).")
        sys.exit(1)

    logger.info(f"Using processing version: {args.version}")
    logger.info(f"Processing up to {args.limit} volumes in this run.")

    start_dt = None
    end_dt = None
    dt_format = "%Y%m%d_%H%M"
    if args.start:
        try:
            start_dt = datetime.strptime(args.start, dt_format).replace(tzinfo=timezone.utc)
        except ValueError:
            logger.error(f"Invalid start date format. Please use YYYYMMDD_HHMM.")
            sys.exit(1)
    if args.end:
        try:
            end_dt = datetime.strptime(args.end, dt_format).replace(tzinfo=timezone.utc)
        except ValueError:
            logger.error(f"Invalid end date format. Please use YYYYMMDD_HHMM.")
            sys.exit(1)

    if start_dt and end_dt and start_dt >= end_dt:
        logger.error("Start datetime must be before end datetime.")
        sys.exit(1)

    conn = None
    processed_count = 0
    error_count = 0

    try:
        conn = get_connection()
        logger.info("Database connection established.")

        # Pass the version to the database function
        unprocessed_ids = get_unprocessed_volume_identifiers(conn, version=args.version, limit=args.limit,
                                                             start_dt=start_dt, end_dt=end_dt)

        if not unprocessed_ids:
            logger.info(f"No new volumes to process for version '{args.version}' at this time.")
            return

        logger.info(f"Found {len(unprocessed_ids)} volumes to process for version '{args.version}'.")

        iterator_func = tqdm if TQDM_AVAILABLE else tqdm_fallback_iterator
        volume_iterator = iterator_func(unprocessed_ids, desc="Processing Volumes", unit="volume")

        for vol_id in volume_iterator:
            try:
                success = process_volume(volume_identifier=vol_id, version=args.version)
                if success:
                    processed_count += 1
                else:
                    error_count += 1
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing volume {vol_id.isoformat()}: {e}", exc_info=True)
                error_count += 1

    except Exception as e:
        logger.exception("A critical error occurred during the volume processing workflow:")
        error_count += 1
    finally:
        if conn:
            release_connection(conn)
            logger.info("Database connection released.")

    logger.info("--- Volume Processing Summary ---")
    logger.info(f"Successfully processed volumes: {processed_count}")
    logger.info(f"Volumes with processing errors: {error_count}")
    logger.info("Volume processing run finished.")

def cli_plot(args: argparse.Namespace):
    """Handler for the 'plot' command."""
    logger = logging.getLogger(__name__)
    logger.info(f"=== Starting Historical Plot Generation ===")
    dt_format = "%Y%m%d_%H%M"
    try:
        start_dt_utc = datetime.strptime(args.start, dt_format).replace(tzinfo=timezone.utc)
        end_dt_utc = datetime.strptime(args.end, dt_format).replace(tzinfo=timezone.utc)
    except ValueError:
        logger.error(f"Invalid date format. Please use YYYYMMDD_HHMM.")
        sys.exit(1)

    plotting_manager.generate_plots(
        variable=args.variable,
        elevation=args.elevation,
        start_dt=start_dt_utc,
        end_dt=end_dt_utc,
        source=args.source,
        version=args.version,
        plot_extent=args.extent
    )

def main():
    # 1. Load Core Configuration FIRST
    try: load_config(); print("Configuration loaded successfully.")
    except Exception as e: print(f"FATAL: Failed to load configuration: {e}", file=sys.stderr); sys.exit(1)

    # 2. Setup Logging
    log_file = get_setting('app.log_file', 'log/radproc_default.log')
    log_level_file = get_setting('app.log_level_file', 'DEBUG')
    log_level_console = get_setting('app.log_level_console', 'INFO')
    log_max_bytes = get_setting('app.log_max_bytes', 5*1024*1024)
    log_backup_count = get_setting('app.log_backup_count', 5)
    log_dir = os.path.dirname(log_file)
    if log_dir: os.makedirs(log_dir, exist_ok=True)
    _setup_logger(log_file, log_level_file, log_level_console, log_max_bytes, log_backup_count)
    logger = logging.getLogger(__name__)

    # 3. Setup Argparse
    parser = argparse.ArgumentParser(description="Radar Processor CLI.", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(title="Available Commands", dest="subcommand", required=True, help="Use '<command> --help' for details.")

    # --- 'run' command ---
    run_parser = subparsers.add_parser("run", help="Start monitoring for new scans.")
    run_parser.set_defaults(func=cli_run)

    # --- 'timeseries' command ---
    timeseries_parser = subparsers.add_parser("timeseries", help="Generate historical timeseries data.")
    timeseries_parser.add_argument("start", help="Start datetime (YYYYMMDD_HHMM).")
    timeseries_parser.add_argument("end", help="End datetime (YYYYMMDD_HHMM).")
    timeseries_parser.add_argument(
        "--points",
        nargs='*',
        metavar="POINT_NAME",
        default=[],
        help="Specific point names to process. If omitted, all points are processed."
    )
    timeseries_parser.add_argument("--variable", help="Specific variable to extract.")
    timeseries_parser.add_argument(
        "--source",
        choices=['raw', 'corrected'],
        default='raw',
        help="The data source to generate timeseries from (default: raw)."
    )
    timeseries_parser.add_argument(
        "--version",
        metavar="VERSION",
        help="The correction version to use (required for '--source corrected')."
    )
    timeseries_parser.set_defaults(func=cli_timeseries)

    # --- 'accumulate' command ---
    accumulate_parser = subparsers.add_parser("accumulate", help="Calculate accumulated precipitation.")
    accumulate_parser.add_argument("point_name", help="Point name.")
    accumulate_parser.add_argument("start", help="Start datetime (YYYYMMDD_HHMM).")
    accumulate_parser.add_argument("end", help="End datetime (YYYYMMDD_HHMM).")
    accumulate_parser.add_argument("interval", help="Accumulation interval (e.g., '1H').")
    accumulate_parser.add_argument("--variable", default="RATE", help="Input rate variable (default: RATE).")
    accumulate_parser.add_argument("--output-file", help="Output CSV file path.")
    accumulate_parser.add_argument(
        "--source",
        choices=['raw', 'corrected'],
        default='raw',
        help="The data source for the rate variable (default: raw)."
    )
    accumulate_parser.add_argument(
        "--version",
        metavar="VERSION",
        help="The correction version to use (required for '--source corrected')."
    )
    accumulate_parser.set_defaults(func=cli_accumulate)
    accumulate_parser.set_defaults(func=cli_accumulate)

    # --- 'animate' command ---
    animate_parser = subparsers.add_parser("animate", help="Create an animation.")
    animate_parser.add_argument("variable", help="Variable to animate.")
    animate_parser.add_argument("elevation", type=float, help="Target elevation.")
    animate_parser.add_argument("start", help="Start datetime (YYYYMMDD_HHMM).")
    animate_parser.add_argument("end", help="End datetime (YYYYMMDD_HHMM).")
    animate_parser.add_argument("output_file", help="Output animation file path.")
    animate_parser.add_argument("--extent", nargs=4, type=float, metavar=('LONMIN', 'LONMAX', 'LATMIN', 'LATMAX'), help="Set plot extent.")
    animate_parser.add_argument("--no-watermark", action='store_true', help="Exclude watermark.")
    animate_parser.add_argument("--fps", type=int, help="Frames per second.")
    animate_parser.set_defaults(func=cli_animate)

    # --- 'reorg-scans' command ---
    reorg_parser = subparsers.add_parser("reorg-scans", help="Reorganize old scan files.")
    reorg_parser.add_argument("--output-dir", help="Base output directory (defaults to config).")
    reorg_parser.add_argument("--dry-run", action='store_true', help="Simulate without making changes.")
    reorg_parser.add_argument("--delete-empty", action='store_true', help="Delete empty source dirs.")
    reorg_parser.set_defaults(func=cli_reorg_scans)

    # +++ 'index-scans' command +++
    index_parser = subparsers.add_parser(
        "index-scans",
        help="Scan the output directory and populate the radproc_scan_log database table.",
        description="Scans the configured/specified output directory for .scnx.gz files,\n"
                    "extracts their metadata (precise time, elevation, sequence number),\n"
                    "and adds entries to the radproc_scan_log table if they don't already exist."
    )
    index_parser.add_argument(
        "--output-dir",
        metavar="PATH",
        help="Specify the base output directory to scan. Defaults to 'app.output_dir' from config."
    )
    index_parser.add_argument(
        "--dry-run",
        action='store_true',
        help="Simulate the indexing: show what would be added without writing to the database."
    )
    index_parser.set_defaults(func=cli_index_scans) # Link to the handler function

    group_volumes_parser = subparsers.add_parser(
        "group-volumes",
        help="Group scans in radproc_scan_log by assigning a common volume_identifier.",
        description="Identifies scans with sequence_number 0, uses their precise_timestamp as a volume_identifier,\n"
                    "and assigns this ID to other related scans (sequence > 0) that match criteria\n"
                    "(time proximity, elevation) to form a complete volume."
    )
    group_volumes_parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,  # Default lookback period
        help="How many hours back to look for ungrouped scans (default: 24)."
    )
    group_volumes_parser.add_argument(
        "--limit",
        type=int,
        default=1000,  # Default limit on scans fetched at once
        help="Maximum number of ungrouped scans to process in one go (default: 1000)."
    )
    group_volumes_parser.add_argument(
        "--time-window-minutes",
        type=int,
        # Default will be taken from get_setting or hardcoded in the function
        help=f"Time window in minutes around a base scan (_0) to search for other volume members (default: from config or {get_setting('volume_grouping.time_window_minutes', 5)} min)."
    )
    group_volumes_parser.add_argument(
        "--dry-run",
        action='store_true',
        help="Simulate the grouping: show what would be updated without writing to the database."
    )
    group_volumes_parser.set_defaults(func=cli_group_volumes)

    # +++ 'process-volumes' command +++
    process_volumes_parser = subparsers.add_parser(
        "process-volumes",
        help="Find grouped volumes and process them into corrected CfRadial2 files.",
        description="Scans the database for volumes that have been grouped but not yet processed.\n"
                    "For each, it combines all raw scans, applies corrections, and saves a\n"
                    "single volumetric CfRadial2 file, logging the result to the database."
    )
    process_volumes_parser.add_argument(
        "--version",
        required=True,
        metavar="VERSION_STRING",
        help="The processing version to apply and log (e.g., 'v1_0')."
    )
    process_volumes_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of volumes to process in one run (default: 50)."
    )
    process_volumes_parser.add_argument(
        "--start",
        metavar="YYYYMMDD_HHMM",
        help="The start of the datetime range to process volumes."
    )
    process_volumes_parser.add_argument(
        "--end",
        metavar="YYYYMMDD_HHMM",
        help="The end of the datetime range to process volumes."
    )
    process_volumes_parser.set_defaults(func=cli_process_volumes)

    # --- 'plot' command ---
    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate historical PPI plots from raw or corrected data.",
        description="Scans the database for the specified time range and generates PPI plots for a given variable and elevation."
    )
    plot_parser.add_argument("variable", help="The variable to plot (e.g., DBZH).")
    plot_parser.add_argument("elevation", type=float, help="The target elevation angle.")
    plot_parser.add_argument("start", help="Start datetime in YYYYMMDD_HHMM format.")
    plot_parser.add_argument("end", help="End datetime in YYYYMMDD_HHMM format.")
    plot_parser.add_argument(
        "--source",
        choices=['raw', 'corrected'],
        default='raw',
        help="The data source to generate plots from (default: raw)."
    )
    plot_parser.add_argument(
        "--version",
        metavar="VERSION",
        help="The correction version to use (required for '--source corrected')."
    )
    plot_parser.add_argument(
        "--extent",
        nargs=4,
        type=float,
        metavar=('LONMIN', 'LONMAX', 'LATMIN', 'LATMAX'),
        help="Set a custom plot extent."
    )
    plot_parser.set_defaults(func=cli_plot)

    # 4. Parse Arguments
    args = parser.parse_args()

    # 5. Execute Command Function
    logger.info(f"Executing command: {args.subcommand}")
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user (Ctrl+C).")
        print("\nProcess interrupted.", file=sys.stderr)
        stop_worker()
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Command '{args.subcommand}' failed:")
        print(f"Error: Command '{args.subcommand}' failed. Check logs.", file=sys.stderr)
        stop_worker()
        sys.exit(1)
    finally:
         logger.debug("Main process final cleanup.")
         stop_worker()

if __name__ == "__main__":
    main()
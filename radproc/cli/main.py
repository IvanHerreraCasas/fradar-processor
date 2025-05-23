#!/usr/bin/env python3

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
from datetime import datetime, timezone
from pathlib import Path

# Set Agg backend - might be less critical now but safe to keep if CLI might ever trigger plots directly
# However, all plotting is now within core.visualization, which should handle its own backend needs if run non-interactively.
# Let's comment it out unless issues arise.
# matplotlib.use("Agg")

# --- Project Path Setup ---
# If running with 'python -m cli.main' from root, this might not be strictly needed,
# but it provides robustness if run directly for development.
# Consider if this is the best approach vs. installing the package.
# For now, let's keep a simplified version.
# Assuming main.py is in project_root/cli/
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
# --- End Path Setup ---


# --- Import New Core Functions ---
from ..core.config import load_config, get_setting # Load config early
from ..core.file_monitor import start_monitoring
from ..core.processor import generate_historical_plots
from ..core.analysis import generate_point_timeseries, calculate_accumulation
from ..core.visualization.animator import create_animation
from ..core.utils.upload_queue import start_worker, stop_worker
from ..core.data import get_scan_elevation
from ..core.utils.helpers import parse_datetime_from_filename, move_processed_file

# --- End Core Imports ---

# --- Logging Setup ---
def _setup_logger(
    log_file: str,
    log_level_file_str: str = 'DEBUG',
    log_level_console_str: str = 'INFO',
    log_max_bytes: int = 5*1024*1024, # Default 5MB
    log_backup_count: int = 5        # Default 5 backups
    ):
        """Sets up file and console logging using configuration values."""
        # Get the root logger instance. Config changes here apply globally.
        root_logger = logging.getLogger()
        # Set the lowest level captures will process (DEBUG means capture everything)
        root_logger.setLevel(logging.DEBUG)
        # Remove existing handlers to prevent duplication if called multiple times
        # (useful during development/testing)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()


        # Get level objects from strings
        log_level_file = getattr(logging, log_level_file_str.upper(), logging.DEBUG)
        log_level_console = getattr(logging, log_level_console_str.upper(), logging.INFO)

        # --- File Handler ---
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                print(f"Created log directory: {log_dir}") # Info for first run

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=log_max_bytes,
                backupCount=log_backup_count,
                encoding='utf-8' # Explicitly set encoding
            )
            file_handler.setLevel(log_level_file) # Set level for this handler
            # More detailed format for the file
            file_formatter = logging.Formatter(
                 '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            print(f"File logging enabled: Level={logging.getLevelName(log_level_file)}, File={log_file}")

        except PermissionError:
             print(f"ERROR: Permission denied to write log file: {log_file}", file=sys.stderr)
        except Exception as e:
             print(f"ERROR: Failed to set up file logger ({log_file}): {e}", file=sys.stderr)


        # --- Console Handler ---
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level_console) # Set level for this handler
            # Simpler format for the console
            console_formatter = logging.Formatter('%(levelname)s: [%(name)s] %(message)s') # Added logger name
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
            print(f"Console logging enabled: Level={logging.getLevelName(log_level_console)}")
        except Exception as e:
             print(f"ERROR: Failed to set up console logger: {e}", file=sys.stderr)


        logging.getLogger('PIL').setLevel(logging.WARNING)

        # --- Initial Log Message ---
        # Use the root logger directly after setup
        root_logger.info(f"Logging initialized.")

        # Optional: Silence overly verbose libraries if needed
        # logging.getLogger('matplotlib').setLevel(logging.WARNING)
        # logging.getLogger('watchdog').setLevel(logging.INFO)

        return root_logger # Although modifying root logger is usually sufficient

# --- End Logging Setup ---

# --- CLI Command Functions ---

import shutil
from tqdm import tqdm
TQDM_AVAILABLE = True
def cli_run(args: argparse.Namespace):
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Monitor Mode ===")
    worker_started = False # Flag to track if worker needs stopping
    try:
        # --- Start FTP Queue Worker ---
        logger.info("Starting FTP queue worker thread...")
        start_worker() # Assumes start_worker handles not starting duplicates
        worker_started = True # Mark that we need to potentially stop it
        # --- Start File Monitor ---
        logger.info("Starting file monitor...")
        start_monitoring() # This function blocks until stopped (Ctrl+C or error)
        # --- Monitor has stopped ---
        logger.info("Monitor Mode process finished.")
    except KeyboardInterrupt:
         # Already handled within start_monitoring's loop, but log here too
         logger.info("Monitor Mode interrupted by user.")
    except Exception as e:
        logger.exception("!!! Monitor Mode Failed Unexpectedly !!!")
        sys.exit(1) # Exit with error code
    finally:
         # --- Ensure Worker is Signalled to Stop ---
         if worker_started:
             logger.info("Signalling FTP queue worker to stop...")
             stop_worker() # Signal worker thread to terminate gracefully
             logger.info("Exiting Monitor Mode.")
         else:
             logger.info("Exiting Monitor Mode (worker not started or failed).")


def cli_reprocess(args: argparse.Namespace):
    """Handler for the 'reprocess' command."""
    logger = logging.getLogger(__name__)
    logger.info(f"=== Starting Reprocess Mode: {start_dt} -> {end_dt} ===")

    dt_format = "%Y%m%d_%H%M" # Define the expected format

    try:
        start_dt = datetime.strptime(args.start, dt_format)
        end_dt = datetime.strptime(args.end, dt_format)
    except ValueError:
        logger.error(f"Invalid date format. Please use YYYYMMDD_HHMM (e.g., 20231027_1430).")
        sys.exit(1)

    if start_dt >= end_dt:
         logger.error("Start datetime must be before end datetime.")
         sys.exit(1)

    try:
        # Configuration should already be loaded by main()
        # generate_historical_plots will get config via get_config() if needed
        generate_historical_plots(start_dt, end_dt)
        logger.info("Reprocessing finished.")
    except Exception as e:
        logger.exception("An error occurred during reprocessing:")
        sys.exit(1) # Exit with error code

# +++ CLI Handler for Timeseries +++
def cli_timeseries(args: argparse.Namespace):
    """Handler for the 'timeseries' command."""
    logger = logging.getLogger(__name__)
    logger.info(f"=== Starting Historical Timeseries Generation ===")
    logger.info(f"Point: '{args.point_name}', Range: {args.start} -> {args.end}, Variable: {args.variable or 'Default'}")

    # Validate and parse datetimes
    dt_format = "%Y%m%d_%H%M"
    try:
        # Assume naive input, localize to UTC for internal use
        start_dt_naive = datetime.strptime(args.start, dt_format)
        end_dt_naive = datetime.strptime(args.end, dt_format)
        # Make them timezone-aware (UTC)
        start_dt_utc = start_dt_naive.replace(tzinfo=timezone.utc)
        end_dt_utc = end_dt_naive.replace(tzinfo=timezone.utc)
    except ValueError:
        logger.error(f"Invalid date format for start/end. Please use YYYYMMDD_HHMM (e.g., 20231027_1430).")
        sys.exit(1)

    if start_dt_utc >= end_dt_utc:
         logger.error("Start datetime must be before end datetime.")
         sys.exit(1)

    try:
        # Configuration should already be loaded by main()
        # Call the core historical generation function
        generate_point_timeseries(
            point_name=args.point_name,
            start_dt=start_dt_utc,
            end_dt=end_dt_utc,
            variable_override=args.variable # Pass optional variable override
        )
        logger.info("Historical timeseries generation finished.")
    except FileNotFoundError as e:
         # Catch specific errors that might be helpful to the user
         logger.error(f"Error: A required file or directory was not found: {e}")
         sys.exit(1)
    except ValueError as e:
         # Catch potential value errors (e.g., point not found in config)
         logger.error(f"Configuration or Value Error: {e}")
         sys.exit(1)
    except Exception as e:
        logger.exception("An error occurred during historical timeseries generation:")
        sys.exit(1) # Exit with error code
# +++++++++++++++++++++++++++++++++++++++++++++

# +++ CLI Handler for Animation +++
def cli_animate(args: argparse.Namespace):
    """Handler for the 'animate' command."""
    logger = logging.getLogger(__name__)
    logger.info(f"=== Starting Animation Generation ===")
    logger.info(f"Variable: {args.variable}, Elevation: {args.elevation}, Range: {args.start} -> {args.end}")
    logger.info(f"Output File: {args.output_file}")
    if args.extent: logger.info(f"Custom Extent: {args.extent}")
    if args.no_watermark: logger.info("Watermark Disabled.")
    if args.fps: logger.info(f"Custom FPS: {args.fps}")

    # Validate and parse datetimes
    dt_format = "%Y%m%d_%H%M"
    try:
        start_dt_naive = datetime.strptime(args.start, dt_format)
        end_dt_naive = datetime.strptime(args.end, dt_format)
        start_dt_utc = start_dt_naive.replace(tzinfo=timezone.utc)
        end_dt_utc = end_dt_naive.replace(tzinfo=timezone.utc)
    except ValueError:
        logger.error(f"Invalid date format for start/end. Please use YYYYMMDD_HHMM.")
        sys.exit(1)

    if start_dt_utc >= end_dt_utc:
         logger.error("Start datetime must be before end datetime.")
         sys.exit(1)

    # Validate elevation format
    try:
        elevation_float = float(args.elevation)
    except ValueError:
        logger.error(f"Invalid elevation format: '{args.elevation}'. Must be a number.")
        sys.exit(1)

    # Validate output file extension (basic check)
    allowed_formats = ['.gif', '.mp4', '.avi', '.mov'] # Add more as supported by imageio/plugins
    file_format = os.path.splitext(args.output_file)[1].lower()
    if not file_format or file_format not in allowed_formats:
         logger.warning(f"Output file format '{file_format}' might not be supported by imageio. Common formats: {allowed_formats}")
         # Allow proceeding, imageio might handle it or raise an error

    try:
        # Configuration should already be loaded by main()
        success = create_animation(
            variable=args.variable,
            elevation=elevation_float,
            start_dt=start_dt_utc,
            end_dt=end_dt_utc,
            output_filename=args.output_file,
            plot_extent=args.extent, # Pass tuple directly or None
            include_watermark=(not args.no_watermark), # Invert the flag
            fps=args.fps # Pass optional fps override or None
            # file_format determined internally by create_animation based on output_filename
        )

        if success:
            logger.info(f"Animation generation finished successfully. Output: {args.output_file}")
        else:
            logger.error("Animation generation failed. See logs for details.")
            sys.exit(1) # Exit with error if core function reported failure

    except FileNotFoundError as e:
         logger.error(f"Error: A required file or directory was not found: {e}")
         sys.exit(1)
    except ValueError as e:
         logger.error(f"Configuration or Value Error: {e}")
         sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred during animation generation:")
        sys.exit(1)
# +++++++++++++++++++++++++++++++++++++++++++++


# +++ CLI Handler for Accumulation +++
def cli_accumulate(args: argparse.Namespace):
    """Handler for the 'accumulate' command."""
    logger = logging.getLogger(__name__)
    logger.info(f"=== Starting Precipitation Accumulation ===")
    logger.info(f"Point: '{args.point_name}', Range: {args.start} -> {args.end}, Interval: {args.interval}, Variable: {args.variable or 'RATE'}")

    # Validate interval string using pandas
    try:
        pd.Timedelta(args.interval) # Check if it's a valid frequency/offset alias
        # Or potentially: pd.tseries.frequencies.to_offset(args.interval) # More strict frequency check
    except ValueError:
        logger.error(f"Invalid accumulation interval format: '{args.interval}'. Use Pandas frequency string (e.g., '1H', '15min', '1D', '6H').")
        sys.exit(1)

    # Validate and parse datetimes
    dt_format = "%Y%m%d_%H%M"
    try:
        start_dt_naive = datetime.strptime(args.start, dt_format)
        end_dt_naive = datetime.strptime(args.end, dt_format)
        start_dt_utc = start_dt_naive.replace(tzinfo=timezone.utc)
        end_dt_utc = end_dt_naive.replace(tzinfo=timezone.utc)
    except ValueError:
        logger.error(f"Invalid date format for start/end. Please use YYYYMMDD_HHMM.")
        sys.exit(1)

    if start_dt_utc >= end_dt_utc:
         logger.error("Start datetime must be before end datetime.")
         sys.exit(1)

    # Determine output file path
    output_file = args.output_file # Use provided path if given
    if not output_file:
        # Generate default path if not provided
        timeseries_dir = get_setting('app.timeseries_dir')
        if not timeseries_dir:
            logger.error("Configuration Error: 'app.timeseries_dir' must be set to generate default output filename.")
            sys.exit(1)
        rate_variable = args.variable or 'RATE' # Need variable for filename
        default_filename = f"{args.point_name}_{rate_variable}_{args.interval}_acc.csv"
        output_file = os.path.join(timeseries_dir, default_filename)
        logger.info(f"Output file not specified, using default: {output_file}")

    try:
        # Ensure output directory exists before calling core function
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Call the core accumulation function
        success = calculate_accumulation(
            point_name=args.point_name,
            start_dt=start_dt_utc,
            end_dt=end_dt_utc,
            interval=args.interval,
            rate_variable=(args.variable or 'RATE'), # Pass the determined rate variable
            output_file_path=output_file
        )

        if success:
            logger.info(f"Precipitation accumulation finished successfully. Output: {output_file}")
        else:
            logger.error("Precipitation accumulation failed. See previous logs for details.")
            sys.exit(1) # Exit with error if core function reported failure

    except FileNotFoundError as e:
         logger.error(f"Error: A required file or directory was not found: {e}")
         sys.exit(1)
    except ValueError as e:
         logger.error(f"Configuration or Value Error: {e}")
         sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred during precipitation accumulation:")
        sys.exit(1)
# ++++++++++++++++++++++++++++++++++++

# --- End CLI Command Functions ---


# --- Main Execution ---

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
        if os.path.isdir(item_path) and item_name.isdigit() and len(item_name) == 8:
            old_date_dirs_to_process.append(item_path)

    if not old_date_dirs_to_process:
        logger.info("No old-structure YYYYMMDD directories found to process.")
        logger.info("--- Reorganization Summary ---")
        logger.info("No files processed.")
        return

    # Determine the iterator for the outer loop (directories)
    dir_iterator_func = tqdm if TQDM_AVAILABLE else tqdm_fallback_iterator
    dir_iterator = dir_iterator_func(old_date_dirs_to_process, desc="Date Dirs", unit="dir", leave=False)

    for date_dir_path in dir_iterator:
        if TQDM_AVAILABLE:  # Update description for tqdm
            dir_iterator.set_description(f"Processing Dir: {os.path.basename(date_dir_path)}")
        else:  # For fallback, log which directory is being processed
            logger.info(f"Processing Date Directory: {os.path.basename(date_dir_path)}")

        scnx_files_in_dir = [f for f in os.listdir(date_dir_path) if f.endswith(".scnx.gz")]

        if not scnx_files_in_dir:
            logger.info(f"No .scnx.gz files found in {date_dir_path}.")
            if args.delete_empty:
                try:
                    if not os.listdir(date_dir_path):  # Check if truly empty
                        log_msg = f"Would delete empty directory: {date_dir_path}" if args.dry_run else f"Deleting empty directory: {date_dir_path}"
                        logger.info(log_msg)
                        if not args.dry_run:
                            os.rmdir(date_dir_path)
                            deleted_dir_count += 1
                    else:
                        logger.info(f"Directory {date_dir_path} is not empty (contains other files). Will not delete.")
                except OSError as e:
                    logger.error(f"Failed to delete directory {date_dir_path}: {e}")
            continue

        files_successfully_processed_from_dir = 0  # For --delete-empty logic

        file_iterator_func = tqdm if TQDM_AVAILABLE else tqdm_fallback_iterator
        file_iterator = file_iterator_func(scnx_files_in_dir, desc=f"Files in {os.path.basename(date_dir_path)}",
                                           unit="file", leave=False)

        for filename in file_iterator:
            source_filepath = os.path.join(date_dir_path, filename)

            elevation = get_scan_elevation(source_filepath)
            scan_datetime = parse_datetime_from_filename(filename)

            if elevation is None:
                logger.warning(f"Could not get elevation for {source_filepath}. Skipping.")
                skipped_count += 1
                continue
            if scan_datetime is None:
                logger.warning(f"Could not parse datetime from filename {filename}. Skipping.")
                skipped_count += 1
                continue

            elevation_code = f"{int(round(elevation * 100)):03d}"
            date_str = scan_datetime.strftime("%Y%m%d")
            new_target_dir_check = os.path.join(output_dir, elevation_code, date_str)
            new_target_filepath_check = os.path.join(new_target_dir_check, filename)

            if os.path.exists(new_target_filepath_check):
                logger.warning(
                    f"Target file {new_target_filepath_check} already exists. Skipping move for {source_filepath}.")
                skipped_count += 1
                files_successfully_processed_from_dir += 1  # Count as "processed" for deletion
                continue

            log_msg_dry_run = f"DRY RUN: Would move '{source_filepath}' to new structure (Elev: {elevation:.2f}, DT: {scan_datetime.isoformat()})"
            log_msg_action = f"Attempting to move '{source_filepath}'..."
            logger.info(log_msg_dry_run if args.dry_run else log_msg_action)

            if not args.dry_run:
                try:
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
            else:  # Dry run, simulate counts
                moved_count += 1
                files_successfully_processed_from_dir += 1

        # After processing all files in the date directory, check for deletion
        if args.delete_empty and files_successfully_processed_from_dir == len(scnx_files_in_dir):
            try:
                # Check if directory is now truly empty or only contains non-.scnx.gz files
                # For safety, we'll only delete if it's completely empty.
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

    logger.info("--- Reorganization Summary ---")
    logger.info(f"Files processed for moving (includes dry run counts): {moved_count}")
    logger.info(f"Files skipped (target existed, or error getting elev/dt): {skipped_count}")
    logger.info(f"Errors during move operations: {error_count}")
    if args.delete_empty:
        logger.info(f"Old YYYYMMDD directories deleted: {deleted_dir_count}")
    logger.info("Scan file reorganization finished.")

def main():
    # 1. Load Core Configuration FIRST
    try:
        load_config()
        print("Configuration loaded successfully.") # Keep simple startup message
    except Exception as e:
        print(f"FATAL: Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Setup Logging using config values
    log_file = get_setting('app.log_file', 'log/radproc_default.log')
    log_level_file = get_setting('app.log_level_file', 'DEBUG')
    log_level_console = get_setting('app.log_level_console', 'INFO')
    log_max_bytes = get_setting('app.log_max_bytes', 5*1024*1024)
    log_backup_count = get_setting('app.log_backup_count', 5)

    # Create log directory if specified in log_file path and doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
         os.makedirs(log_dir, exist_ok=True)
         
    _setup_logger(log_file, log_level_file, log_level_console, log_max_bytes, log_backup_count)
    # Get logger instance for use within main() if needed, though configuring root is primary
    logger = logging.getLogger(__name__) # Gets 'cli.main' logger

    # 3. Setup Argparse
    parser = argparse.ArgumentParser(
        description="Radar Processor CLI - Monitors and processes radar scans.",
        formatter_class=argparse.RawTextHelpFormatter # Keep formatting in help text
    )
    subparsers = parser.add_subparsers(
        title="Available Commands",
        dest="subcommand",
        required=True, # Make subcommand required
        help="Use '<command> --help' for more information on a specific command."
    )

    # --- 'run' command ---
    run_parser = subparsers.add_parser(
        "run",
        help="Start the monitor to watch the input directory for new radar scans and process them.",
        description="Continuously monitors the configured input directory. When new scan files\n"
                    "matching the pattern appear, they are processed (plots generated)\n"
                    "and optionally moved to the processed data directory.\n"
                    "Processes pre-existing files on startup."
    )
    run_parser.set_defaults(func=cli_run) # Link to the handler function

    # --- 'reprocess' command ---
    reprocess_parser = subparsers.add_parser(
        "reprocess",
        help="Reprocess historical scans stored in the processed data directory within a specific time range.",
        description="Finds already processed scan files (.scnx.gz) within the date-structured\n"
                    "output directory that fall between the specified start and end datetimes.\n"
                    "Regenerates plots for these scans according to the current configuration.\n"
                    "Does NOT move the files again."
    )
    reprocess_parser.add_argument(
        "start",
        help="Start datetime for reprocessing range (Format: YYYYMMDD_HHMM, e.g., '20231027_1000')."
    )
    reprocess_parser.add_argument(
        "end",
        help="End datetime for reprocessing range (Format: YYYYMMDD_HHMM, e.g., '20231027_1200')."
    )
    reprocess_parser.set_defaults(func=cli_reprocess) # Link to the handler function

    # +++ 'timeseries' command +++
    timeseries_parser = subparsers.add_parser(
        "timeseries",
        help="Generate/update historical timeseries CSV for a specific point.",
        description="Scans processed radar files within the output directory for a given date range\n"
                    "and extracts data for the specified point (defined in points.yaml).\n"
                    "Appends new data to the corresponding CSV file in the configured timeseries directory,\n"
                    "avoiding duplicate entries based on timestamps. Useful for backfilling history."
    )
    timeseries_parser.add_argument(
        "point_name",
        help="The unique 'name' of the point defined in config/points.yaml."
    )
    timeseries_parser.add_argument(
        "start",
        help="Start datetime for processing range (Format: YYYYMMDD_HHMM, e.g., '20231027_1000'). Assumed UTC."
    )
    timeseries_parser.add_argument(
        "end",
        help="End datetime for processing range (Format: YYYYMMDD_HHMM, e.g., '20231027_1200'). Assumed UTC."
    )
    timeseries_parser.add_argument(
        "--variable", # Optional argument
        metavar="VAR_NAME",
        help="Extract data for this specific variable (e.g., 'RATE', 'DBZH'), overriding the default in points.yaml."
    )
    timeseries_parser.set_defaults(func=cli_timeseries) # Link to the handler function

    # +++ 'accumulate' command +++
    accumulate_parser = subparsers.add_parser(
        "accumulate",
        help="Calculate accumulated precipitation for a point over a time range.",
        description="Reads an existing RATE timeseries CSV for a point, calculates accumulated\n"
                    "precipitation over a specified interval (e.g., '1H', '15min'), and saves\n"
                    "the results to a new CSV file. Overwrites the output file if it exists."
    )
    accumulate_parser.add_argument(
        "point_name",
        help="The unique 'name' of the point (defined in config/points.yaml)."
    )
    accumulate_parser.add_argument(
        "start",
        help="Start datetime for analysis range (Format: YYYYMMDD_HHMM). Assumed UTC."
    )
    accumulate_parser.add_argument(
        "end",
        help="End datetime for analysis range (Format: YYYYMMDD_HHMM). Assumed UTC."
    )
    accumulate_parser.add_argument(
        "interval",
        help="Accumulation interval (Pandas frequency string, e.g., '1H', '15min', '1D', '6H')."
    )
    accumulate_parser.add_argument(
        "--variable",
        metavar="RATE_VAR",
        default="RATE", # Default to RATE
        help="Specify the input rate variable name (default: RATE)."
    )
    accumulate_parser.add_argument(
        "--output-file",
        metavar="PATH",
        help="Specify the full path for the output CSV file. If omitted, defaults to\n"
             "'<timeseries_dir>/<point_name>_<variable>_<interval>_acc.csv'."
    )
    accumulate_parser.set_defaults(func=cli_accumulate)
    # +++++++++++++++++++++++++++++
    
    # +++ 'animate' command +++
    animate_parser = subparsers.add_parser(
        "animate",
        help="Create an animation (GIF/MP4) from generated plot images for a specific variable and elevation.",
        description="Finds existing plot images within the specified time range, variable, and elevation.\n"
                    "Optionally regenerates frames if custom extent or no watermark is requested.\n"
                    "Stitches valid frames into an animation file (GIF, MP4, etc.)."
    )
    animate_parser.add_argument(
        "variable",
        help="The radar variable name to animate (e.g., 'RATE')."
    )
    animate_parser.add_argument(
        "elevation",
        type=float, # Expect float input
        help="The target elevation angle in degrees (e.g., 0.5)."
    )
    animate_parser.add_argument(
        "start",
        help="Start datetime for animation range (Format: YYYYMMDD_HHMM). Assumed UTC."
    )
    animate_parser.add_argument(
        "end",
        help="End datetime for animation range (Format: YYYYMMDD_HHMM). Assumed UTC."
    )
    animate_parser.add_argument(
        "output_file",
        help="Full path for the output animation file (e.g., 'output/rate_anim.gif', 'results/precip.mp4'). Extension determines format."
    )
    animate_parser.add_argument(
        "--extent",
        nargs=4, # Expect exactly 4 values
        type=float,
        metavar=('LONMIN', 'LONMAX', 'LATMIN', 'LATMAX'),
        help="Set a specific plot extent (geographic coordinates: LonMin LonMax LatMin LatMax). Regenerates frames."
    )
    animate_parser.add_argument(
        "--no-watermark",
        action='store_true', # Flag, default is False (watermark included)
        help="Generate animation frames without the configured watermark. Regenerates frames."
    )
    animate_parser.add_argument(
        "--fps",
        type=int,
        metavar="FRAMES",
        help="Frames per second for the animation (overrides 'animation_fps' in config)."
    )
    animate_parser.set_defaults(func=cli_animate)

    # +++ 'reorg-scans' command +++
    reorg_parser = subparsers.add_parser(
        "reorg-scans",
        help="Reorganize processed scan files from YYYYMMDD/ to ElevationCode/YYYYMMDD/ structure.",
        description="Scans the configured output directory for radar scan files currently stored\n"
                    "in an old format (e.g., output_dir/YYYYMMDD/scan.scnx.gz) and moves them\n"
                    "to the new structured format (output_dir/ElevationCode/YYYYMMDD/scan.scnx.gz).\n"
                    "This command reads each scan to determine its elevation."
    )
    reorg_parser.add_argument(
        "--output-dir",
        metavar="PATH",
        help="Specify the base output directory to reorganize. Defaults to 'app.output_dir' from config."
    )
    reorg_parser.add_argument(
        "--dry-run",
        action='store_true',
        help="Simulate the reorganization: show what would be moved and deleted without making changes."
    )
    reorg_parser.add_argument(
        "--delete-empty",
        action='store_true',
        help="After successfully moving all .scnx.gz files from an old YYYYMMDD directory,\n"
             "delete the source YYYYMMDD directory if it's empty or only contained those files."
    )
    reorg_parser.set_defaults(func=cli_reorg_scans)
    # ++++++++++++++++++++++++++++++

    # +++++++++++++++++++++++++++++

    # 4. Parse Arguments
    args = parser.parse_args()

    # 5. Execute Command Function
    logger.info(f"Executing command: {args.subcommand}")
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user (Ctrl+C).")
        print("\nProcess interrupted.", file=sys.stderr)
        # Ensure worker is stopped if main process is interrupted directly
        stop_worker() # Call stop here too for direct main interruption
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Command '{args.subcommand}' failed with an unhandled exception:")
        print(f"Error: Command '{args.subcommand}' failed. Check logs for details.", file=sys.stderr)
        stop_worker() # Ensure worker stops on other main errors
        sys.exit(1)
    finally:
         # Final safety net to signal worker stop on any exit path from main
         # Note: If run finished normally, stop_worker was already called in cli_run's finally
         # This is mostly for unexpected exits from main itself.
         logger.debug("Main process final cleanup: Ensuring worker stop signal sent.")
         stop_worker() # Call again doesn't hurt



if __name__ == "__main__":
    main()
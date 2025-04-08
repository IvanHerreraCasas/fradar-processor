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
from core.config import load_config, get_setting # Load config early
from core.file_monitor import start_monitoring
from core.processor import generate_historical_plots
from core.analysis import generate_point_timeseries, calculate_accumulation
from core.utils.upload_queue import start_worker, stop_worker
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


        # --- Initial Log Message ---
        # Use the root logger directly after setup
        root_logger.info(f"Logging initialized.")

        # Optional: Silence overly verbose libraries if needed
        # logging.getLogger('matplotlib').setLevel(logging.WARNING)
        # logging.getLogger('watchdog').setLevel(logging.INFO)

        return root_logger # Although modifying root logger is usually sufficient

# --- End Logging Setup ---

# --- CLI Command Functions ---

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
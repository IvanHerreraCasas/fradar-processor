# core/file_monitor.py

import time
import os
import logging
from typing import Dict, Any, Optional
from fnmatch import fnmatch

# Import watchdog components
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
except ImportError:
    Observer = None
    FileSystemEventHandler = None
    FileCreatedEvent = None
    FileMovedEvent = None
    print("WARNING: 'watchdog' library not found. File monitoring will not work.")
    print("Please install it: pip install watchdog")

# Import processor function and config access
from .processor import process_new_scan
from .config import get_config, get_setting

# Setup logger for this module
logger = logging.getLogger(__name__)


class NewScanEventHandler(FileSystemEventHandler):
    """Handles file system events, specifically watching for final scan files."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.file_pattern = get_setting('app.input_file_pattern', '*.scnx.gz')
        logger.info(f"Monitoring for files matching: {self.file_pattern}")
        # Optional: Store recent temp files to avoid double processing if needed
        # self.recently_processed_temp = set()

    # Optional: Keep for debugging if needed, otherwise remove
    # def on_any_event(self, event):
    #      logger.debug(f"[WATCHDOG EVENT] Type: {event.event_type}, Path: {event.src_path}, IsDir: {event.is_directory}")

    def on_created(self, event: FileCreatedEvent):
        """
        Handles direct file creations. May not be the primary trigger if FTP uses temp files.
        Only process if the created file *directly* matches the pattern.
        """
        logger.debug(f"[WATCHDOG CREATED] Path: {event.src_path}, IsDir: {event.is_directory}")
        if event.is_directory:
            return

        filepath = event.src_path
        filename = os.path.basename(filepath)

        # Check if this *created* file itself matches the final pattern
        if fnmatch(filename, self.file_pattern):
            logger.info(f"[WATCHDOG CREATED] Direct creation matching pattern: '{filename}'. Triggering processing...")
            self._trigger_processing(filepath) # Use helper function
        # else: # Ignore temporary files created here (like $$$)
             # logger.debug(f"[WATCHDOG CREATED] File '{filename}' does not match pattern. Likely temporary.")

    def on_moved(self, event: FileMovedEvent):
        """
        Handles file move/rename events. This is often the trigger when FTP
        servers upload to a temp file and then rename to the final name.
        """
        logger.debug(f"[WATCHDOG MOVED] Src: {event.src_path}, Dest: {event.dest_path}, IsDir: {event.is_directory}")
        if event.is_directory:
            return

        dest_filepath = event.dest_path
        dest_filename = os.path.basename(dest_filepath)
        # src_filename = os.path.basename(event.src_path) # Usually the temp name ($$$)

        # Check if the DESTINATION file matches the final pattern
        if fnmatch(dest_filename, self.file_pattern):
            logger.info(f"[WATCHDOG MOVED] Destination file '{dest_filename}' matches pattern. Triggering processing...")
            self._trigger_processing(dest_filepath) # Process the final destination file
        # else: # Ignore moves that don't result in the target pattern
             # logger.debug(f"[WATCHDOG MOVED] Destination '{dest_filename}' does not match pattern.")


    def _trigger_processing(self, filepath: str):
        """Helper function to contain the processing call and error handling."""
        # Debounce check (optional): Avoid processing if recently handled via another event
        # if filepath in self.recently_processed_temp: return
        # Add filepath to debounce set? Need careful management.

        try:
            # Wait a short period to ensure file is fully available after creation/move
            wait_seconds = 2 # Keep configurable or adjust as needed
            logger.debug(f"Waiting {wait_seconds}s before processing '{os.path.basename(filepath)}'...")
            time.sleep(wait_seconds)

            # Verify file still exists after wait (it might be moved again quickly?)
            if not os.path.exists(filepath):
                 logger.warning(f"File disappeared before processing could start: {filepath}")
                 return

            logger.info(f"Calling process_new_scan for: {filepath}")
            process_new_scan(filepath, self.config)
            # Error handling is inside process_new_scan

        except Exception as e:
            logger.error(f"Error triggering processing for {filepath}: {e}", exc_info=True)



def start_monitoring(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Starts the file system monitoring process. Performs an initial scan
    for pre-existing files and then uses Watchdog to monitor for new files.

    Triggers 'process_new_scan' for found files matching the pattern.

    Args:
        config: The application configuration dictionary (optional, uses get_config() if None).
    """
    # --- Basic Setup & Config Loading ---
    if Observer is None:
         logger.error("Watchdog library not installed or failed to import. Cannot start monitoring.")
         return

    if config is None:
        config = get_config()

    input_dir = config.get('app', {}).get('input_dir')
    file_pattern = get_setting('app.input_file_pattern', '*.scnx.gz')
    if not isinstance(file_pattern, str) or not file_pattern:
         logger.warning("Invalid or missing 'app.input_file_pattern' in config. Using '*.scnx.gz'.")
         file_pattern = '*.scnx.gz'

    if not input_dir or not os.path.isdir(input_dir):
        logger.error(f"Input directory '{input_dir}' not found or not configured in 'app.input_dir'. Cannot start monitoring.")
        return

    logger.info(f"Starting file monitor for directory: {input_dir}")
    logger.info(f"Monitoring for files matching: {file_pattern}")


    # --- Initial Scan for Pre-existing Files ---
    logger.info(f"Performing initial scan for existing files in {input_dir}...")
    initial_files_found = 0
    initial_files_processed = 0
    initial_files_failed = 0
    try:
        for dirpath, _, filenames in os.walk(input_dir): # Use os.walk
            logger.debug(f"Scanning directory: {dirpath}")
            for filename in filenames: # Iterate through files in current dirpath
                # --- Check pattern first ---
                if fnmatch(filename, file_pattern):
                    filepath = os.path.join(dirpath, filename) # Construct full path
                    initial_files_found += 1
                    logger.info(f"[Initial Scan] Found matching file: {filepath}. Attempting processing...")
                    try:
                        # Process the pre-existing file
                        success = process_new_scan(filepath, config)
                        if success: initial_files_processed += 1
                        else: initial_files_failed += 1
                    except Exception as e:
                        initial_files_failed += 1
                        logger.error(f"[Initial Scan] Error processing {filepath}: {e}", exc_info=True)
                    # Continue scanning even if one file fails
            # --- Added Else for Debug ---
            # else:
            #     logger.debug(f"Item '{filename}' skipped (is_file={is_file}, matches_pattern={matches_pattern})")
            # --- End Added Else ---

    except FileNotFoundError:
         logger.error(f"Initial scan failed: Input directory '{input_dir}' not found.")
         return # Cannot continue if input dir doesn't exist
    except PermissionError:
         logger.error(f"Initial scan failed: Permission denied for reading directory '{input_dir}'.")
         return # Cannot continue without read permissions
    except Exception as scan_err:
         logger.error(f"Error during initial scan of {input_dir}: {scan_err}", exc_info=True)
         # Decide if you should stop or continue to monitoring phase
         # Let's continue to monitoring for now.

    logger.info(f"Initial scan complete. Found: {initial_files_found}, Processed: {initial_files_processed}, Failed: {initial_files_failed}")


    # --- Start Watchdog Observer for New Files ---
    logger.info("Starting watchdog observer for new files...")
    event_handler = NewScanEventHandler(config) # Pass config to handler
    observer = Observer()
    observer.schedule(event_handler, path=input_dir, recursive=True)
    observer.start()
    logger.info("Watchdog observer started. Monitoring for new file creations. Press Ctrl+C to stop.")

    # --- Keep Main Thread Alive ---
    try:
        while True:
            # Check observer health periodically (optional)
            # if not observer.is_alive():
            #     logger.error("Watchdog observer thread is no longer alive. Stopping.")
            #     break
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping file monitor (KeyboardInterrupt)...")
    except Exception as e:
         logger.error(f"File monitor encountered an error: {e}", exc_info=True)
    finally:
        if observer.is_alive():
            observer.stop()
            logger.info("Watchdog observer stopped.")
        # Wait for the observer thread to fully terminate
        observer.join()
        logger.info("File monitor stopped completely.")



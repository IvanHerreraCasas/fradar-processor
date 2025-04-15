# radproc/huey_app.py
import matplotlib


try:
    matplotlib.use("Agg")
    print("Matplotlib backend set to 'Agg'.")
except Exception as e:
    print(f"Warning: Could not set Matplotlib backend to 'Agg': {e}")
from logging.handlers import RotatingFileHandler
import os
import logging
import sqlite3
import sys
from huey import SqliteHuey

# --- IMPORTANT: Ensure Core Config is loaded ---
# This assumes 'core.config' handles logging setup as part of loading
# Or that the worker environment sets up logging separately.
try:
    # Assuming core.config might be needed by tasks indirectly
    from .core.config import load_config, get_setting
    load_config() # Load config when this module is imported (for worker)
    CONFIG_LOADED = True
except ImportError:
    print("Warning: Could not import core config. Tasks might fail if they rely on get_setting.")
    CONFIG_LOADED = False
    # Define a dummy get_setting if needed to avoid NameError later
    def get_setting(key, default=None):
         print(f"Warning: core.config not loaded, returning default for {key}")
         return default
except Exception as e:
    print(f"FATAL: Failed to load configuration for Huey initialization: {e}")
    # Decide how to handle - raise error to stop worker?
    raise SystemExit(f"Huey cannot start: Config load failed: {e}")

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
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

        # --- Initial Log Message ---
        # Use the root logger directly after setup
        root_logger.info(f"Logging initialized.")

        # Optional: Silence overly verbose libraries if needed
        # logging.getLogger('matplotlib').setLevel(logging.WARNING)
        # logging.getLogger('watchdog').setLevel(logging.INFO)

        return root_logger # Although modifying root logger is usually sufficient


    # 1. Load Core Configuration FIRST

try:
    load_config()
    print("Configuration loaded successfully.") # Keep simple startup message
except Exception as e:
    print(f"FATAL: Failed to load configuration: {e}", file=sys.stderr)
    sys.exit(1)
# 2. Setup Logging using config values
log_file = get_setting('app.huey_log_file', 'log/radproc_default.log')
log_level_file = get_setting('app.log_level_file', 'DEBUG')
log_level_console = get_setting('app.log_level_console', 'INFO')
log_max_bytes = get_setting('app.log_max_bytes', 5*1024*1024)
log_backup_count = get_setting('app.log_backup_count', 5)
# Create log directory if specified in log_file path and doesn't exist
log_dir = os.path.dirname(log_file)
if log_dir:
     os.makedirs(log_dir, exist_ok=True)
     
_setup_logger(log_file, log_level_file, log_level_console, log_max_bytes, log_backup_count)
logger = logging.getLogger(__name__) # Gets 'cli.main' logger


# --- Configure SQLite Queue Path ---
default_db_name = 'huey_queue.db'
# Get project root relative to this file's location (radproc/huey_app.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
default_db_path = os.path.join(project_root, 'data', default_db_name)

# Allow overriding via config if config loaded successfully
huey_db_path = default_db_path # Start with default
if CONFIG_LOADED:
    huey_db_path = get_setting('huey.db_path', default_db_path)
    logger.info("Huey DB path obtained via get_setting.")
else:
    logger.warning(f"Using default Huey DB path as config wasn't loaded: {default_db_path}")


# Ensure the directory for the SQLite file exists
try:
    db_dir = os.path.dirname(huey_db_path)
    if db_dir: # Avoid trying to create if path is just filename in current dir
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Huey SQLite directory ensured: {db_dir}")
except OSError as e:
    logger.error(f"Failed to create directory for Huey DB '{huey_db_path}': {e}")
    raise # Stop if we can't create the directory

logger.info(f"Initializing Huey with SQLite backend at: {huey_db_path}")

# --- Initialize Huey ---
# 'name' helps distinguish if you run multiple Huey queues/apps
# Use results=True (default) to store task results/status needed by API 
huey = SqliteHuey(name='radproc_jobs', filename=huey_db_path, results=True, utc=True)

def apply_huey_pragmas():
    """Applies recommended PRAGMA settings to the Huey SQLite DB."""
    db_path = huey_db_path # Get path directly from the configured Huey instance
    logger.info(f"Attempting to apply PRAGMA settings to Huey DB: {db_path}")
    if not os.path.exists(db_path):
         logger.warning(f"Huey DB file {db_path} does not exist yet. PRAGMAs will be applied on first connection by Huey/consumer or next API start.")
         # Optionally, force creation and application now:
         # try:
         #     with sqlite3.connect(db_path, timeout=10.0) as conn:
         #          logger.info(f"Created Huey DB file: {db_path}")
         # except sqlite3.Error as e:
         #      logger.error(f"Failed to create DB file {db_path} for PRAGMA application: {e}")
         #      return # Abort if creation fails

    # Proceed even if file doesn't exist yet, settings apply on next connection
    try:
        # Connect using standard sqlite3
        # Using a short timeout; locks shouldn't be held long during startup
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            # Set WAL mode - crucial for concurrency
            conn.execute("PRAGMA journal_mode=WAL;")
            logger.info(f"Set journal_mode=WAL for {db_path}.")
            # Set busy timeout - allows waiting for locks briefly
            conn.execute("PRAGMA busy_timeout = 5000;") # 5000ms = 5 seconds
            logger.info(f"Set busy_timeout=5000 for {db_path}.")

            # Optional: Verify settings were applied (useful for debugging)
            # current_journal = conn.execute("PRAGMA journal_mode;").fetchone()
            # current_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()
            # logger.debug(f"Verified settings for {db_path}: journal_mode={current_journal[0]}, busy_timeout={current_timeout[0]}")

        logger.info(f"Successfully applied PRAGMA settings to {db_path}.")
    except sqlite3.OperationalError as e:
         # This might happen if the DB is locked during startup, unlikely but possible
         logger.error(f"Database operational error applying PRAGMA settings to {db_path} (possibly locked?): {e}", exc_info=True)
    except sqlite3.Error as e:
        # Catch other potential SQLite errors during connection/execution
        logger.error(f"SQLite error applying PRAGMA settings to {db_path}: {e}", exc_info=True)
    except Exception as e:
         # Catch any other unexpected errors
         logger.error(f"Unexpected error applying PRAGMA settings to {db_path}: {e}", exc_info=True)

try:
    import radproc.tasks
    logger.info("Successfully imported task module: radproc.tasks")
except ImportError as e:
    logger.error(f"Failed to import task module 'radproc.tasks': {e}. Worker may not find tasks!")
    # Decide if this should be fatal? Probably yes.
    raise SystemExit(f"Huey cannot start: Failed to import tasks module: {e}")
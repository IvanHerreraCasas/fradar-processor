# radproc/api/main.py
import logging
from logging.handlers import RotatingFileHandler
import os
import asyncio
import sys 
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import watchfiles 
import re        

# Import core config loader (to ensure it loads early if not already)
from radproc.core.config import load_config, get_setting, get_config
from radproc.api.dependencies import get_realtime_image_dir, get_core_config

# Import routers
from .routers import status, points, plots, downloads, jobs
from .state import image_update_queue
from radproc.huey_config import apply_huey_pragmas

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


    # 1. Load Core Configuration FIRST

try:
    load_config()
    print("Configuration loaded successfully.") # Keep simple startup message
except Exception as e:
    print(f"FATAL: Failed to load configuration: {e}", file=sys.stderr)
    sys.exit(1)
# 2. Setup Logging using config values
log_file = get_setting('app.api_log_file', 'log/radproc_default.log')
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

# --- File Watcher Task ---
async def start_realtime_watcher():
    """Background task to watch the realtime image directory for changes."""
    realtime_dir = get_setting('app.realtime_image_dir')


    if not realtime_dir:
        logger.error("[Watcher] Cannot start: 'app.realtime_image_dir' is not configured.")
        return # Stop the task if dir isn't set

    if not os.path.isdir(realtime_dir):
        logger.warning(f"[Watcher] Realtime directory not found: {realtime_dir}. Watcher will run but may find no files.")
        # Attempt to create it? Or just let it fail if processor can't write? Let's let it run.
        # os.makedirs(realtime_dir, exist_ok=True) # Optional: create if missing

    # Pattern to match the files we care about
    filename_pattern = re.compile(r"^realtime_[a-zA-Z0-9_-]+_\d{3,4}\.png$")
    watch_filter = lambda change, name: filename_pattern.match(name) is not None

    logger.info(f"[Watcher] Starting to watch directory: {realtime_dir}")
    try:
        # Use awatch to asynchronously monitor the directory
        async for changes in watchfiles.awatch(realtime_dir, ):
            logger.debug(f"[Watcher] Changes *after* filter: {changes}") # See what passed
            # changes is a set of tuples: {(Change.added, 'path'), (Change.modified, 'path'), ...}
            updated_files = set()
            for change, path in changes:
                filename = os.path.basename(path)
                
                if filename_pattern.match(filename):
                    # We care about files being added or modified
                    if change in (watchfiles.Change.added, watchfiles.Change.modified):
                        # Add filename to set to avoid duplicates if added+modified quickly
                        updated_files.add(filename)

            # Put unique updated filenames onto the queue
            for fname in updated_files:
                logger.info(f"[Watcher] Detected change: {fname}. Adding to queue.")
                await image_update_queue.put(fname)

    except FileNotFoundError:
        logger.error(f"[Watcher] Realtime directory disappeared: {realtime_dir}. Watcher stopping.")
    except Exception as e:
        logger.error(f"[Watcher] Error during file watching: {e}", exc_info=True)
    finally:
        logger.info("[Watcher] File watching task finished.")


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # === Startup ===
    print("API Lifespan: Startup sequence beginning...")
    # Apply Huey DB PRAGMAs
    logger.info("Applying Huey database PRAGMAs...")
    apply_huey_pragmas()
    logger.info("API Startup: Loading core configuration...")
    watcher_task = None # Initialize task variable
    try:
        # Explicitly load core config on API startup if not already loaded
        # The dependency will also handle this, but doing it here ensures it happens early.
        load_config()
        config = get_config()
        logger.info("API Startup: Core configuration load initiated.")
    except Exception as e:
        logger.critical(f"API Startup Failed: Could not load core config: {e}", exc_info=True)
        # How to handle this? API can't function. Exit? Or let requests fail?
        # Let requests fail via dependency for now.
        print(f"FATAL: API Startup Failed during core config load: {e}")
        config = None

    # --- Start File Watcher Task ---
    if config and get_setting('app.realtime_image_dir',): # Check if dir is configured
        print("API Lifespan: Starting realtime image watcher task...")
        watcher_task = asyncio.create_task(start_realtime_watcher(), name="RealtimeImageWatcher")
        print("API Lifespan: Realtime image watcher task started.")
    else:
        logger.warning("API Lifespan: Realtime image watcher NOT started (realtime_image_dir not configured).")

    print("API Lifespan: Startup complete. Yielding control.")
    yield # API runs here
    # === Shutdown ===
    print("API Lifespan: Shutdown sequence beginning...")
    # --- Cancel File Watcher Task ---
    if watcher_task:
        print("API Lifespan: Cancelling watcher task...")
        if not watcher_task.done():
            watcher_task.cancel()
            try:
                await watcher_task # Wait for task to acknowledge cancellation
            except asyncio.CancelledError:
                print("API Lifespan: Watcher task cancelled successfully.")
            except Exception as e: # Catch any error during task cleanup
                print(f"API Lifespan: Error during watcher task cancellation/cleanup: {e}")
                logger.error(f"API Shutdown: Error stopping watcher task: {e}", exc_info=True)
        else:
             print("API Lifespan: Watcher task already finished.")
    # -------------------------------------------
    print("API Lifespan: Shutdown complete.")



# --- FastAPI App Initialization ---
app = FastAPI(
    title="RadProc API",
    description="API for interacting with Radar Processor data and functions.",
    version="0.1.0",
    lifespan=lifespan # Use the lifespan context manager
)

# --- CORS Configuration ---
# WARNING: Allow all origins is insecure for production.
# Restrict this to your Flutter app's actual origin(s).
# origins = get_setting("api.cors_origins", default=["*"]) # Example loading from config
origins = ["*"] # Allow all for now (development)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all standard methods
    allow_headers=["*"], # Allows all headers
)

# --- Include Routers ---
api_prefix = "/api/v1" # Optional prefix for versioning

app.include_router(status.router, prefix=api_prefix)
app.include_router(points.router, prefix=api_prefix)
app.include_router(plots.router, prefix=api_prefix)
#app.include_router(downloads.router, prefix=api_prefix)
app.include_router(jobs.router, prefix=api_prefix)
# Add routers for tasks later

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    """Simple endpoint to confirm the API is running."""
    return {"message": "Welcome to the RadProc API!"}

# --- Helper function for file watcher (Phase 1B) ---
# async def start_realtime_watcher():
#    # Implementation will go here
#    pass
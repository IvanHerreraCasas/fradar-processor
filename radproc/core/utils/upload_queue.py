# core/utils/upload_queue.py

import sqlite3
import os
import json # Not strictly needed now if not storing config parts
import logging
import time
from threading import Lock, Thread
from typing import Optional, Dict, Any, Tuple, List

# --- Core Imports ---
from core.config import get_setting # Need access to main config
from core.utils.secrets import get_ftp_password
from core.utils.ftp_client import upload_scan_file, upload_image_file

logger = logging.getLogger(__name__)

# --- Database Setup ---
DB_LOCK = Lock() # Global lock for DB operations
_db_path = None # Store path after first access

def _get_db_path() -> str:
    """Gets the configured DB path, defaulting if not set."""
    global _db_path
    if _db_path is None:
        # Use get_setting with default value
        default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'upload_queue.db'))
        _db_path = get_setting('ftp.queue_db_path', default_path)
        # Ensure directory exists
        db_dir = os.path.dirname(_db_path)
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Using upload queue database at: {_db_path}")
    return _db_path

def _init_db():
    """Initializes the SQLite database schema, creating/altering as needed."""
    db_path = _get_db_path()
    try:
        # Connect (creates DB file if it doesn't exist)
        with DB_LOCK, sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()

            # === Step 1: Ensure the table exists FIRST ===
            logger.debug("Ensuring 'queue' table exists...")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    local_filepath TEXT NOT NULL,
                    server_alias TEXT NOT NULL,
                    -- Define columns potentially added later as nullable initially for easier migration
                    remote_base_dir TEXT,
                    file_type TEXT DEFAULT 'scan', -- Keep default for initial creation
                    status TEXT DEFAULT 'pending',
                    attempts INTEGER DEFAULT 0,
                    last_attempt_ts REAL DEFAULT 0.0,
                    added_ts REAL DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Commit this creation immediately so subsequent checks work
            conn.commit()
            logger.debug("'queue' table check/creation complete.")

            # === Step 2: Check schema and ALTER if necessary ===
            # Now that the table is guaranteed to exist, check its columns
            logger.debug("Checking 'queue' table schema for necessary columns...")
            cursor.execute("PRAGMA table_info(queue)")
            # Use a dictionary for easier column lookup by name
            columns_info = {col[1]: {'type': col[2], 'notnull': col[3], 'default': col[4], 'pk': col[5]}
                            for col in cursor.fetchall()}

            # Add 'file_type' if missing
            if 'file_type' not in columns_info:
                logger.info("Adding 'file_type' column to queue table (default 'scan').")
                try:
                    # Add with the NOT NULL and DEFAULT constraint
                    cursor.execute("ALTER TABLE queue ADD COLUMN file_type TEXT NOT NULL DEFAULT 'scan'")
                    conn.commit() # Commit alter statement
                except sqlite3.OperationalError as e:
                    # Catch if somehow it already exists despite PRAGMA check (unlikely but safe)
                    if "duplicate column name" in str(e).lower():
                        logger.warning(f"Column 'file_type' already exists (detected after PRAGMA check): {e}")
                    else:
                        raise # Re-raise other operational errors

            # Handle 'remote_base_dir' (potentially renamed from 'remote_base_scan_dir')
            if 'remote_base_dir' not in columns_info:
                if 'remote_base_scan_dir' in columns_info:
                    # If the old name exists, rename it
                    logger.info("Renaming 'remote_base_scan_dir' to 'remote_base_dir' in queue table.")
                    try:
                        cursor.execute("ALTER TABLE queue RENAME COLUMN remote_base_scan_dir TO remote_base_dir")
                        conn.commit() # Commit rename
                    except sqlite3.OperationalError as e:
                        logger.warning(f"Error renaming 'remote_base_scan_dir' (might already be renamed): {e}")
                else:
                    # If neither old nor new name exists, add the new one
                    logger.info("Adding 'remote_base_dir' column to queue table.")
                    try:
                        # Add as potentially NULL initially if needed, or add NOT NULL if safe
                        cursor.execute("ALTER TABLE queue ADD COLUMN remote_base_dir TEXT") # Start nullable
                        # Potentially add NOT NULL constraint later if needed and possible
                        conn.commit() # Commit add column
                    except sqlite3.OperationalError as e:
                         if "duplicate column name" in str(e).lower():
                             logger.warning(f"Column 'remote_base_dir' already exists: {e}")
                         else:
                             raise

            # Optional: Ensure NOT NULL constraint on remote_base_dir if added/renamed
            # Re-check schema after potential changes if strictness is needed
            # cursor.execute("PRAGMA table_info(queue)")
            # columns_info_updated = {col[1]: ... for col ...}
            # if 'remote_base_dir' in columns_info_updated and columns_info_updated['remote_base_dir']['notnull'] == 0:
            #    logger.info("Attempting to apply NOT NULL constraint to 'remote_base_dir'. Requires default or update.")
            #    # Applying NOT NULL retroactively in SQLite often requires complex steps
            #    # (new table, copy data, drop old, rename new) - skip for now unless essential

            # === Step 3: Ensure Index Exists ===
            logger.debug("Ensuring index 'idx_status_attempts' exists...")
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status_attempts ON queue (status, attempts, last_attempt_ts)')
            conn.commit() # Commit index creation

            logger.debug(f"Database schema initialization/verification complete at '{db_path}'.")

    except Exception as e:
        logger.error(f"Failed to initialize/update queue database at '{db_path}': {e}", exc_info=True)
        raise # Critical failure if DB setup fails

_init_db() # Initialize DB when module is loaded

# --- Queue Operations ---

def add_scan_to_queue(local_filepath: str, server_config: Dict[str, Any]):
    """Adds a scan file upload task to the persistent queue."""
    db_path = _get_db_path()
    alias = server_config.get("alias")
    remote_base_dir = server_config.get("remote_scan_dir")

    if not alias or not remote_base_dir:
         logger.error(f"Cannot add to queue: Missing 'alias' or 'remote_scan_dir' in server config for {local_filepath}")
         return False # Indicate failure to add

    try:
        with DB_LOCK, sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO queue (local_filepath, server_alias, remote_base_dir, added_ts)
                VALUES (?, ?, ?, ?)
            ''', (local_filepath, alias, remote_base_dir, time.time()))
            conn.commit()
            logger.info(f"Added to upload queue: '{os.path.basename(local_filepath)}' for server '{alias}'.")
            return True
    except Exception as e:
        logger.error(f"Failed to add item to upload queue for '{alias}': {e}", exc_info=True)
        return False

def add_image_to_queue(local_image_path: str, server_config: Dict[str, Any]):
    """Adds a generated image file upload task to the persistent queue."""
    db_path = _get_db_path()
    alias = server_config.get("alias")
    remote_base_dir = server_config.get("remote_image_dir") # <<< Use remote_image_dir

    if not alias or not remote_base_dir:
         logger.error(f"Cannot add image to queue: Missing 'alias' or 'remote_image_dir' in server config for {local_image_path}")
         return False

    try:
        with DB_LOCK, sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()
            # Insert with file_type='image' and the image base dir
            cursor.execute('''
                INSERT INTO queue (local_filepath, server_alias, remote_base_dir, file_type, added_ts)
                VALUES (?, ?, ?, ?, ?)
            ''', (local_image_path, alias, remote_base_dir, 'image', time.time()))
            conn.commit()
            logger.info(f"Added IMAGE to upload queue: '{os.path.basename(local_image_path)}' for server '{alias}'.")
            return True
    except Exception as e:
        logger.error(f"Failed to add image item to upload queue for '{alias}': {e}", exc_info=True)
        return False


def get_pending_task() -> Optional[Tuple[int, str, str, str, str]]: # Added file_type to return tuple
    """Gets the oldest eligible pending upload task, marking it 'processing'."""
    # Rename from get_pending_scan_task for clarity
    db_path = _get_db_path()
    retry_delays_sec = get_setting('ftp.retry_delays_seconds', [60, 300, 900, 3600])
    max_attempts = len(retry_delays_sec)
    logger.debug(f"[Queue] Getting pending task. Max attempts={max_attempts}, Delays={retry_delays_sec}")

    try:
        with DB_LOCK, sqlite3.connect(db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            now = time.time()
            item = None
            for attempt_num in range(max_attempts):
                delay = retry_delays_sec[attempt_num]
                eligible_time = now - delay
                logger.debug(f"[Queue] Checking for pending tasks with attempts={attempt_num}, last_attempt <= {eligible_time:.2f}")
                cursor.execute('''
                    SELECT id, local_filepath, server_alias, remote_base_dir, file_type, last_attempt_ts, attempts
                    FROM queue
                    WHERE status = 'pending' AND attempts = ? AND last_attempt_ts <= ?
                    ORDER BY added_ts ASC
                    LIMIT 1
                ''', (attempt_num, eligible_time))
                row = cursor.fetchone()
                if row:
                    logger.debug(f"[Queue] Found eligible item: id={row['id']}, type={row['file_type']}, attempts={row['attempts']}, last_attempt={row['last_attempt_ts']:.2f}")
                    item = row
                    break

            if not item:
                 logger.debug("[Queue] No eligible pending tasks found.")
                 return None

            logger.debug(f"[Queue] Attempting to mark item {item['id']} as 'processing'.")
            cursor.execute("UPDATE queue SET status = 'processing', last_attempt_ts = ? WHERE id = ?", (now, item['id']))
            conn.commit()
            if cursor.rowcount == 0:
                 logger.warning(f"[Queue] Failed to mark item {item['id']} as 'processing' (race condition?).")
                 return None

            logger.info(f"[Queue] Retrieved item {item['id']} ({item['file_type']}) for processing: {item['local_filepath']} -> {item['server_alias']}")
            # Return id, local path, alias, remote base dir, file type
            return (item['id'], item['local_filepath'], item['server_alias'], item['remote_base_dir'], item['file_type'])

    except Exception as e:
        logger.error(f"Error getting pending task from queue: {e}", exc_info=True)
        return None


def mark_upload_success(item_id: int):
    """Removes a successfully uploaded item from the queue."""
    # No change needed internally, just rename for clarity
    db_path = _get_db_path()
    try:
        with DB_LOCK, sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM queue WHERE id = ?", (item_id,))
            conn.commit()
            if cursor.rowcount > 0:
                 logger.info(f"Removed successfully uploaded item {item_id} from queue.")
            else:
                 logger.warning(f"Tried to remove item {item_id}, but it was not found (maybe removed by another process?).")
    except Exception as e:
        logger.error(f"Error removing item {item_id} from queue: {e}", exc_info=True)


def mark_upload_failure(item_id: int):
    """Marks an item status (pending/failed) and increments attempt count."""
    # No change needed internally, just rename for clarity
    db_path = _get_db_path()
    max_attempts = len(get_setting('ftp.retry_delays_seconds', [60, 300, 900, 3600]))
    logger.debug(f"[Queue] Marking failure for item {item_id}. Max attempts={max_attempts}")
    try:
        with DB_LOCK, sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT attempts FROM queue WHERE id = ? AND status = 'processing'", (item_id,))
            result = cursor.fetchone()
            if result:
                 current_attempts = result[0]
                 next_attempts = current_attempts + 1
                 next_status = 'pending' if next_attempts < max_attempts else 'failed'
                 update_time = time.time()
                 logger.debug(f"[Queue] Item {item_id}: current_attempts={current_attempts}, next_attempts={next_attempts}, next_status='{next_status}'")
                 cursor.execute(f'''
                     UPDATE queue SET status = ?, attempts = ?, last_attempt_ts = ?
                     WHERE id = ?
                 ''', (next_status, next_attempts, update_time, item_id))
                 conn.commit()
                 if cursor.rowcount > 0:
                      logger.warning(f"[Queue] Marked item {item_id} as '{next_status}' after failed attempt {next_attempts}.")
                 else:
                      logger.error(f"[Queue] Failed to update status for item {item_id} after failure (not found or status changed?).")
            else:
                 logger.warning(f"[Queue] Tried to mark item {item_id} as failed, but it wasn't in 'processing' state (maybe already marked?).")
    except Exception as e:
        logger.error(f"Error marking item {item_id} as failed: {e}", exc_info=True)

# --- Background Worker ---
_worker_thread: Optional[Thread] = None
_stop_worker = False

def _handle_local_file_cleanup(local_filepath: str):
    """Deletes the local file if configured to do so after successful upload."""
    try:
        delete_local = get_setting('ftp.delete_local_scan_on_ftp_success', False)
        if delete_local:
            logger.info(f"[Worker] Deleting local file after successful upload: {local_filepath}")
            os.remove(local_filepath)
        else:
            logger.debug(f"[Worker] Keeping local file (delete_local_scan_on_ftp_success=False): {local_filepath}")
    except FileNotFoundError:
         logger.warning(f"[Worker] Local file already gone, cannot delete: {local_filepath}")
    except OSError as e:
         logger.error(f"[Worker] Failed to delete local file {local_filepath}: {e}", exc_info=True)
    except Exception as e:
         logger.error(f"[Worker] Unexpected error during local file cleanup for {local_filepath}: {e}", exc_info=True)

def _handle_local_image_cleanup(local_image_path: str):
    """Deletes the local image file if configured to do so after successful upload."""
    try:
        # Use the specific config setting for images
        delete_local = get_setting('ftp.delete_local_image_on_ftp_success', False)
        if delete_local:
            logger.info(f"[Worker] Deleting local IMAGE after successful upload: {local_image_path}")
            os.remove(local_image_path)
        else:
            logger.debug(f"[Worker] Keeping local image (delete_local_image_on_ftp_success=False): {local_image_path}")
    except FileNotFoundError:
         logger.warning(f"[Worker] Local image already gone, cannot delete: {local_image_path}")
    except OSError as e:
         logger.error(f"[Worker] Failed to delete local image {local_image_path}: {e}", exc_info=True)
    except Exception as e:
         logger.error(f"[Worker] Unexpected error during local image cleanup for {local_image_path}: {e}", exc_info=True)

def _worker_loop():
    """The main loop for the background worker thread."""
    logger.info("FTP Queue Worker thread started.")
    all_server_configs = get_setting('ftp.servers', [])
    server_map = {s.get('alias'): s for s in all_server_configs if s.get('alias')}

    while not _stop_worker:
        logger.debug("[Worker] Loop running, checking for tasks...")
        task_details = None
        try:
            # Use the renamed getter function
            task_details = get_pending_task()
            if task_details:
                # Unpack the tuple including file_type
                item_id, local_path, server_alias, remote_base_dir, file_type = task_details

                logger.info(f"[Worker] Processing queue item {item_id} ({file_type}): Upload '{os.path.basename(local_path)}' to server '{server_alias}'")

                password = get_ftp_password(server_alias)
                if password is None:
                     logger.error(f"[Worker] No password found for alias '{server_alias}'. Failing item {item_id}.")
                     mark_upload_failure(item_id)
                     continue

                server_config = server_map.get(server_alias)
                if not server_config:
                     logger.error(f"[Worker] No server configuration found for alias '{server_alias}'. Failing item {item_id}.")
                     mark_upload_failure(item_id)
                     continue

                # --- Select Upload Function based on file_type ---
                upload_successful = False
                if file_type == 'scan':
                    # Call existing scan upload function
                    upload_successful = upload_scan_file(
                        local_path,
                        server_config,
                        password,
                        remote_base_dir # This IS the scan base dir for this task type
                    )
                elif file_type == 'image':
                    upload_successful = upload_image_file( #
                        local_path,
                        server_config,
                        password,
                        remote_base_dir # This IS the image base dir for this task type
                    )
                else:
                    logger.error(f"[Worker] Unknown file_type '{file_type}' for item {item_id}. Cannot process.")
                    mark_upload_failure(item_id) # Mark as failed
                    continue # Skip processing

                # --- Update Queue and Handle Local File ---
                if upload_successful:
                    logger.info(f"[Worker] Upload SUCCEEDED for item {item_id} ({file_type}).")
                    mark_upload_success(item_id)
                    # Select cleanup function based on type
                    if file_type == 'scan':
                        _handle_local_file_cleanup(local_path) # Original scan cleanup
                    elif file_type == 'image':
                        _handle_local_image_cleanup(local_path) # New image cleanup
                else:
                    logger.warning(f"[Worker] Upload FAILED for item {item_id} ({file_type}).")
                    mark_upload_failure(item_id)

            else:
                sleep_duration = 30
                logger.debug(f"[Worker] No pending tasks found. Sleeping for {sleep_duration}s.")
                time.sleep(sleep_duration)

        except Exception as e:
             logger.error(f"[Worker] Unhandled error in worker loop: {e}", exc_info=True)
             if task_details:
                 try: mark_upload_failure(task_details[0])
                 except Exception as mark_e: logger.error(f"Failed to mark item as failed after worker error: {mark_e}")
             time.sleep(60)

    logger.info("FTP Queue Worker thread stopping.")

def start_worker():
    """Starts the background queue worker thread."""
    global _worker_thread, _stop_worker
    if _worker_thread is None or not _worker_thread.is_alive():
        _stop_worker = False
        _worker_thread = Thread(target=_worker_loop, name="FTPQueueWorker", daemon=True)
        _worker_thread.start()
        logger.info("FTP Queue Worker thread started.")
    else:
        logger.info("FTP Queue Worker thread already running.")


def stop_worker():
    """Signals the background worker thread to stop."""
    global _stop_worker
    if _worker_thread and _worker_thread.is_alive():
        if not _stop_worker: # Avoid multiple stop messages
            logger.info("Signalling FTP Queue Worker thread to stop...")
            _stop_worker = True
    else:
         logger.info("FTP Queue Worker thread not running or already stopped.")
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
from core.utils.ftp_client import upload_scan_file

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
    """Initializes the SQLite database schema if it doesn't exist."""
    db_path = _get_db_path()
    try:
        with DB_LOCK, sqlite3.connect(db_path, timeout=10) as conn: # Added timeout
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    local_filepath TEXT NOT NULL,
                    server_alias TEXT NOT NULL,
                    remote_base_scan_dir TEXT NOT NULL,
                    status TEXT DEFAULT 'pending', -- 'pending', 'processing', 'failed'
                    attempts INTEGER DEFAULT 0,
                    last_attempt_ts REAL DEFAULT 0.0,
                    added_ts REAL DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status_attempts ON queue (status, attempts, last_attempt_ts)')
            conn.commit()
    except Exception as e:
         logger.error(f"Failed to initialize queue database at '{db_path}': {e}", exc_info=True)
         raise # Raise error if DB cannot be initialized

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
                INSERT INTO queue (local_filepath, server_alias, remote_base_scan_dir, added_ts)
                VALUES (?, ?, ?, ?)
            ''', (local_filepath, alias, remote_base_dir, time.time()))
            conn.commit()
            logger.info(f"Added to upload queue: '{os.path.basename(local_filepath)}' for server '{alias}'.")
            return True
    except Exception as e:
        logger.error(f"Failed to add item to upload queue for '{alias}': {e}", exc_info=True)
        return False


def get_pending_scan_task() -> Optional[Tuple[int, str, str, str]]:
    """Gets the oldest eligible pending upload task, marking it 'processing'."""
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
            # Check eligibility based on attempts and delays
            for attempt_num in range(max_attempts):
                # Use last delay if attempts exceed list length? No, stop at max_attempts derived from list length.
                delay = retry_delays_sec[attempt_num]
                eligible_time = now - delay # Item is eligible if last_attempt was *before* this time
                # --- DEBUG LOGGING ---
                logger.debug(f"[Queue] Checking for pending tasks with attempts={attempt_num}, last_attempt <= {eligible_time:.2f} (now={now:.2f}, delay={delay}s)")
                # --- END DEBUG ---
                cursor.execute('''
                    SELECT id, local_filepath, server_alias, remote_base_scan_dir, last_attempt_ts, attempts
                    FROM queue
                    WHERE status = 'pending' AND attempts = ? AND last_attempt_ts <= ?
                    ORDER BY added_ts ASC
                    LIMIT 1
                ''', (attempt_num, eligible_time)) # Use eligible_time here
                row = cursor.fetchone()
                if row:
                    # --- DEBUG LOGGING ---
                    logger.debug(f"[Queue] Found eligible item: id={row['id']}, attempts={row['attempts']}, last_attempt={row['last_attempt_ts']:.2f}")
                    # --- END DEBUG ---
                    item = row
                    break # Found an eligible item

            if not item:
                 # ... (Marking as failed logic remains the same) ...
                 logger.debug("[Queue] No eligible pending tasks found.")
                 return None

            # Mark item as 'processing'
            # --- DEBUG LOGGING ---
            logger.debug(f"[Queue] Attempting to mark item {item['id']} as 'processing'.")
            # --- END DEBUG ---
            cursor.execute("UPDATE queue SET status = 'processing', last_attempt_ts = ? WHERE id = ?", (now, item['id']))
            conn.commit()
            if cursor.rowcount == 0:
                 logger.warning(f"[Queue] Failed to mark item {item['id']} as 'processing' (race condition?).")
                 return None

            logger.info(f"[Queue] Retrieved item {item['id']} for processing: {item['local_filepath']} -> {item['server_alias']}")
            return (item['id'], item['local_filepath'], item['server_alias'], item['remote_base_scan_dir'])

    except Exception as e:
        logger.error(f"Error getting pending task from queue: {e}", exc_info=True)
        return None


def mark_scan_upload_success(item_id: int):
    """Removes a successfully uploaded item from the queue."""
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


def mark_scan_upload_failure(item_id: int):
    """Marks an item status (pending/failed) and increments attempt count."""
    db_path = _get_db_path()
    max_attempts = len(get_setting('ftp.retry_delays_seconds', [60, 300, 900, 3600]))
    # --- DEBUG LOGGING ---
    logger.debug(f"[Queue] Marking failure for item {item_id}. Max attempts={max_attempts}")
    # --- END DEBUG ---
    try:
        with DB_LOCK, sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()
            # Check current attempts before deciding final status
            cursor.execute("SELECT attempts FROM queue WHERE id = ? AND status = 'processing'", (item_id,)) # Ensure it was processing
            result = cursor.fetchone()
            if result:
                 current_attempts = result[0]
                 next_attempts = current_attempts + 1
                 next_status = 'pending' if next_attempts < max_attempts else 'failed'
                 update_time = time.time()
                 # --- DEBUG LOGGING ---
                 logger.debug(f"[Queue] Item {item_id}: current_attempts={current_attempts}, next_attempts={next_attempts}, next_status='{next_status}'")
                 # --- END DEBUG ---
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


def _worker_loop():
    """The main loop for the background worker thread."""
    logger.info("FTP Queue Worker thread started.")
    # Get the full server list once for efficiency if possible, or fetch per item
    all_server_configs = get_setting('ftp.servers', [])
    server_map = {s.get('alias'): s for s in all_server_configs if s.get('alias')}

    while not _stop_worker:
        # +++ Worker Liveness Log +++
        logger.debug("[Worker] Loop running, checking for tasks...")
        # ++++++++++++++++++++++++++++
        task_details = None # Ensure task_details is reset/defined before try block
        try:
            task_details = get_pending_scan_task()
            if task_details:
                item_id, local_path, server_alias, remote_base_dir = task_details

                logger.info(f"[Worker] Processing queue item {item_id}: Upload '{os.path.basename(local_path)}' to server '{server_alias}'")

                # 1. Get Password
                password = get_ftp_password(server_alias)
                if password is None:
                     logger.error(f"[Worker] No password found for alias '{server_alias}'. Failing item {item_id}.")
                     mark_scan_upload_failure(item_id)
                     continue # Skip to next loop iteration

                # 2. Get Full Server Config (host, port, user, passive)
                server_config = server_map.get(server_alias)
                if not server_config:
                     logger.error(f"[Worker] No server configuration found for alias '{server_alias}' in main config. Failing item {item_id}.")
                     mark_scan_upload_failure(item_id)
                     continue

                # 3. Attempt Upload
                upload_successful = upload_scan_file(
                    local_path,
                    server_config,
                    password,
                    remote_base_dir
                )

                # 4. Update Queue and Handle Local File
                if upload_successful:
                    logger.info(f"[Worker] Upload SUCCEEDED for item {item_id}.")
                    mark_scan_upload_success(item_id)
                    _handle_local_file_cleanup(local_path)
                else:
                    logger.warning(f"[Worker] Upload FAILED for item {item_id}.")
                    mark_scan_upload_failure(item_id) # Call failure marking explicitly

            else:
                # No work found, wait before checking again
                # Make sleep duration configurable?
                sleep_duration = 30 # seconds
                logger.debug(f"[Worker] No pending tasks found. Sleeping for {sleep_duration}s.")
                time.sleep(sleep_duration)

        except Exception as e:
             logger.error(f"[Worker] Unhandled error in worker loop: {e}", exc_info=True)
             if task_details: # If error happened *during* task processing
                 try: mark_scan_upload_failure(task_details[0]) # Attempt to mark failed
                 except Exception as mark_e: logger.error(f"Failed to mark item as failed after worker error: {mark_e}")
             time.sleep(60) # Wait longer after an unexpected error

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
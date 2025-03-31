# core/utils/ftp_client.py

import ftplib
import os
import logging
import socket # For socket.timeout
from typing import Dict, Any

# Import helper for parsing date from filename
from .helpers import parse_datetime_from_filename

logger = logging.getLogger(__name__)

def _ensure_remote_dir(ftp: ftplib.FTP, remote_dir: str):
    """
    Ensures a directory exists on the FTP server, creating it if necessary.
    Handles nested directories. Returns True on success, False on critical failure.
    """
    if not remote_dir or remote_dir == '/':
        return True # Root exists

    parts = remote_dir.strip('/').split('/')
    current_path = ""
    for part in parts:
        if not part: continue
        current_path += "/" + part
        try:
            ftp.cwd(current_path)
        except ftplib.error_perm as e:
            if "550" in str(e): # Directory likely doesn't exist
                try:
                    logger.info(f"[FTP Client] Creating remote directory: {current_path}")
                    ftp.mkd(current_path)
                    # Some servers might require changing into it immediately
                    # ftp.cwd(current_path) # Optional, might cause issues if mkd implies cwd
                except ftplib.error_perm as mkd_e:
                    logger.error(f"[FTP Client] Failed to create remote directory '{current_path}': {mkd_e}")
                    return False # Critical failure
            else:
                logger.error(f"[FTP Client] FTP permission error accessing '{current_path}': {e}")
                return False # Critical failure
        except ftplib.all_errors as e:
             logger.error(f"[FTP Client] Unexpected FTP error checking directory '{current_path}': {e}")
             return False # Critical failure
    return True # All directories checked/created successfully


def upload_scan_file(
    local_filepath: str,
    server_config: Dict[str, Any],
    password: str,
    remote_base_dir: str,
) -> bool:
    """
    Uploads a single local scan file (.scnx.gz) to a specified base directory
    on an FTP server, placing it within a YYYYMMDD subdirectory.

    Args:
        local_filepath: Path to the local scan file to upload.
        server_config: Dictionary containing FTP server details (host, port, user, use_passive, alias).
        password: The FTP password (retrieved securely).
        remote_base_dir: The base directory on the FTP server where date subdirs will be created.

    Returns:
        True if upload was confirmed successful (e.g., 226 response), False otherwise.
    """
    if not os.path.exists(local_filepath):
        logger.error(f"[FTP Client] Local file not found: {local_filepath}")
        return False

    # --- Extract details from config ---
    host = server_config.get("host")
    port = server_config.get("port", 21)
    user = server_config.get("user")
    use_passive = server_config.get("use_passive", True)
    alias = server_config.get("alias", host)

    if not host or not user: # Password is required arg, no need to check here
        logger.error(f"[FTP Client] Incomplete FTP config for {alias} (missing host or user).")
        return False

    # --- Determine Remote Target Directory ---
    filename = os.path.basename(local_filepath)
    file_dt = parse_datetime_from_filename(filename)
    if not file_dt:
        logger.error(f"[FTP Client] Cannot parse date from '{filename}' for server {alias}. Cannot determine target directory.")
        return False
    date_str = file_dt.strftime("%Y%m%d")
    # Ensure remote_base_dir doesn't end with / before joining
    remote_target_dir = os.path.join(remote_base_dir.rstrip('/'), date_str).replace("\\", "/")
    remote_filepath = os.path.join(remote_target_dir, filename).replace("\\", "/")

    # --- Connect and Upload ---
    ftp = None
    success = False
    connection_timeout = 30 # seconds
    transfer_timeout = 300 # 5 minutes for transfer

    try:
        logger.info(f"[FTP Client] Connecting to {alias} ({host}:{port}) as user '{user}'...")
        ftp = ftplib.FTP()
        ftp.connect(host=host, port=port, timeout=connection_timeout)
        ftp.login(user=user, passwd=password)
        # Set transfer timeout AFTER connection and login
        ftp.sock.settimeout(transfer_timeout)
        logger.info(f"[FTP Client] Connected and logged in to {alias}.")

        if use_passive:
            logger.debug(f"[FTP Client] Setting passive mode for {alias}.")
            ftp.set_pasv(True)

        # Ensure the target remote directory exists
        if not _ensure_remote_dir(ftp, remote_target_dir):
            logger.error(f"[FTP Client] Failed to ensure remote directory '{remote_target_dir}' exists on {alias}.")
            return False # Stop if directory creation fails

        # Change to the target directory (optional but good practice)
        try:
            ftp.cwd(remote_target_dir)
            logger.debug(f"[FTP Client] Changed remote CWD to: {remote_target_dir}")
        except ftplib.all_errors as cwd_err:
            # Log warning but maybe proceed if dir exists? Depends on server.
            logger.warning(f"[FTP Client] Could not CWD to '{remote_target_dir}' on {alias} (Error: {cwd_err}). Will attempt STOR with full path.")
            # In this case, STOR command might need the full path if CWD failed.
            # Let's stick to using just the filename for STOR assuming CWD or _ensure worked.

        logger.info(f"[FTP Client] Starting upload: '{local_filepath}' -> '{alias}:{remote_filepath}'")
        with open(local_filepath, 'rb') as f:
            # Use storbinary for the scan file
            # Some servers might need 'STOR full/path/to/file', others just 'STOR filename' after CWD
            response = ftp.storbinary(f'STOR {filename}', f) # Assumes CWD worked
            logger.debug(f"[FTP Client] FTP STOR response: {response}")

            # Check for success response codes (226 Transfer complete, 250 Requested file action okay)
            if response.startswith("226") or response.startswith("250"):
                 logger.info(f"[FTP Client] Successfully uploaded '{filename}' to {alias}.")
                 success = True
            else:
                 logger.error(f"[FTP Client] Upload command failed for '{filename}' to {alias}. Response: {response}")

    except ftplib.error_perm as e:
         logger.error(f"[FTP Client] FTP permission error for {alias}: {e}")
    except ftplib.error_temp as e:
         logger.error(f"[FTP Client] FTP temporary error for {alias}: {e}")
    except socket.timeout:
         logger.error(f"[FTP Client] FTP connection or transfer timed out for {alias}.")
    except ConnectionRefusedError:
         logger.error(f"[FTP Client] FTP connection refused by server: {alias}")
    except Exception as e:
        # Catch any other unexpected library or OS errors
        logger.error(f"[FTP Client] An unexpected error occurred during FTP for {alias}: {e}", exc_info=True)
    finally:
        if ftp:
            try:
                ftp.quit()
                logger.debug(f"[FTP Client] Disconnected from {alias}.")
            except Exception: # Ignore errors during quit
                 pass
    return success
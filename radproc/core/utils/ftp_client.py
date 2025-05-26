# core/utils/ftp_client.py

import ftplib
import os
import logging
import socket  # For socket.timeout
import re
from typing import Dict, Any

# Import helper for parsing date from filename AND getting scan elevation
from ..data import extract_scan_key_metadata
from .helpers import parse_datetime_from_filename

logger = logging.getLogger(__name__)


def _ensure_remote_dir(ftp: ftplib.FTP, remote_dir: str):
    """
    Ensures a directory exists on the FTP server, creating it if necessary.
    Handles nested directories. Returns True on success, False on critical failure.
    """
    if not remote_dir or remote_dir == '/':
        return True  # Root exists

    parts = remote_dir.strip('/').split('/')
    current_path = ""
    for part in parts:
        if not part: continue
        current_path += "/" + part
        try:
            ftp.cwd(current_path)
        except ftplib.error_perm as e:
            if "550" in str(e):  # Directory likely doesn't exist
                try:
                    logger.info(f"[FTP Client] Creating remote directory: {current_path}")
                    ftp.mkd(current_path)
                except ftplib.error_perm as mkd_e:
                    logger.error(f"[FTP Client] Failed to create remote directory '{current_path}': {mkd_e}")
                    return False  # Critical failure
            else:
                logger.error(f"[FTP Client] FTP permission error accessing '{current_path}': {e}")
                return False  # Critical failure
        except ftplib.all_errors as e:
            logger.error(f"[FTP Client] Unexpected FTP error checking directory '{current_path}': {e}")
            return False  # Critical failure
    return True  # All directories checked/created successfully


def upload_scan_file(
        local_filepath: str,
        server_config: Dict[str, Any],
        password: str,
        remote_base_dir: str,
) -> bool:
    """
    Uploads a single local scan file (.scnx.gz) to a specified base directory
    on an FTP server, placing it within an {ElevationCode}/{YYYYMMDD} subdirectory.

    Args:
        local_filepath: Path to the local scan file to upload.
        server_config: Dictionary containing FTP server details.
        password: The FTP password.
        remote_base_dir: The base directory on the FTP server.

    Returns:
        True if upload was confirmed successful, False otherwise.
    """
    if not os.path.exists(local_filepath):
        logger.error(f"[FTP Client] Local file not found: {local_filepath}")
        return False

    host = server_config.get("host")
    port = server_config.get("port", 21)
    user = server_config.get("user")
    use_passive = server_config.get("use_passive", True)
    alias = server_config.get("alias", host)

    if not host or not user:
        logger.error(f"[FTP Client] Incomplete FTP config for {alias} (missing host or user).")
        return False

    filename = os.path.basename(local_filepath)
    file_dt = parse_datetime_from_filename(filename)
    if not file_dt:
        logger.error(
            f"[FTP Client] Cannot parse date from '{filename}' for server {alias}. Cannot determine target directory.")
        return False

    # --- Get Scan Elevation ---
    _, scan_elevation, _ = extract_scan_key_metadata(local_filepath)
    if scan_elevation is None:
        logger.error(f"[FTP Client] Cannot get elevation for '{filename}'. Upload aborted.")
        return False

    elevation_code = f"{int(round(scan_elevation * 100)):03d}"
    date_str = file_dt.strftime("%Y%m%d")

    # Construct new remote target directory including elevation code
    remote_target_dir = os.path.join(remote_base_dir.rstrip('/'), elevation_code, date_str).replace("\\", "/")
    remote_filepath = os.path.join(remote_target_dir, filename).replace("\\", "/")

    ftp = None
    success = False
    connection_timeout = 30
    transfer_timeout = 300

    try:
        logger.info(f"[FTP Client] Connecting to {alias} ({host}:{port}) as user '{user}'...")
        ftp = ftplib.FTP()
        ftp.connect(host=host, port=port, timeout=connection_timeout)
        ftp.login(user=user, passwd=password)
        ftp.sock.settimeout(transfer_timeout)
        logger.info(f"[FTP Client] Connected and logged in to {alias}.")

        if use_passive:
            logger.debug(f"[FTP Client] Setting passive mode for {alias}.")
            ftp.set_pasv(True)

        if not _ensure_remote_dir(ftp, remote_target_dir):
            logger.error(f"[FTP Client] Failed to ensure remote directory '{remote_target_dir}' exists on {alias}.")
            return False

        try:
            ftp.cwd(remote_target_dir)
            logger.debug(f"[FTP Client] Changed remote CWD to: {remote_target_dir}")
        except ftplib.all_errors as cwd_err:
            logger.warning(
                f"[FTP Client] Could not CWD to '{remote_target_dir}' on {alias} (Error: {cwd_err}). Will attempt STOR with full path.")
            # Using filename for STOR assumes CWD worked or is not strictly necessary if _ensure_remote_dir is robust.

        logger.info(f"[FTP Client] Starting upload: '{local_filepath}' -> '{alias}:{remote_filepath}'")
        with open(local_filepath, 'rb') as f:
            response = ftp.storbinary(f'STOR {filename}', f)
            logger.debug(f"[FTP Client] FTP STOR response: {response}")

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
        logger.error(f"[FTP Client] An unexpected error occurred during FTP for {alias}: {e}", exc_info=True)
    finally:
        if ftp:
            try:
                ftp.quit()
                logger.debug(f"[FTP Client] Disconnected from {alias}.")
            except Exception:
                pass
    return success


# upload_image_file remains unchanged as its structure already includes variable, date, and elevation code.
def upload_image_file(
        local_image_path: str,
        server_config: Dict[str, Any],
        password: str,
        remote_base_image_dir: str,
) -> bool:
    """
    Uploads a single local image file (.png) to a specified base directory
    on an FTP server, placing it within a nested structure: Variable/YYYYMMDD/ElevationCode/.

    Args:
        local_image_path: Path to the local image file to upload.
        server_config: Dictionary containing FTP server details (host, port, user, use_passive, alias).
        password: The FTP password (retrieved securely).
        remote_base_image_dir: The base directory on the FTP server for images.

    Returns:
        True if upload was confirmed successful, False otherwise.
    """
    if not os.path.exists(local_image_path):
        logger.error(f"[FTP Client - Image] Local image file not found: {local_image_path}")
        return False

    host = server_config.get("host")
    port = server_config.get("port", 21)
    user = server_config.get("user")
    use_passive = server_config.get("use_passive", True)
    alias = server_config.get("alias", host)

    if not host or not user:
        logger.error(f"[FTP Client - Image] Incomplete FTP config for {alias} (missing host or user).")
        return False

    filename = os.path.basename(local_image_path)
    match = re.match(r'([^_]+)_(\d{3})_(\d{8})_\d{4}\.png', filename)
    if not match:
        logger.error(
            f"[FTP Client - Image] Cannot parse image filename '{filename}' for server {alias}. Required format: VAR_ELE_YYYYMMDD_HHMM.png")
        return False
    variable, elevation_code, date_str = match.groups()

    remote_target_dir = os.path.join(remote_base_image_dir.rstrip('/'), variable, date_str, elevation_code).replace(
        "\\", "/")
    remote_filepath = os.path.join(remote_target_dir, filename).replace("\\", "/")

    ftp = None
    success = False
    connection_timeout = 30
    transfer_timeout = 120

    try:
        logger.info(f"[FTP Client - Image] Connecting to {alias} ({host}:{port}) as user '{user}'...")
        ftp = ftplib.FTP()
        ftp.connect(host=host, port=port, timeout=connection_timeout)
        ftp.login(user=user, passwd=password)
        ftp.sock.settimeout(transfer_timeout)
        logger.info(f"[FTP Client - Image] Connected and logged in to {alias}.")

        if use_passive:
            logger.debug(f"[FTP Client - Image] Setting passive mode for {alias}.")
            ftp.set_pasv(True)

        if not _ensure_remote_dir(ftp, remote_target_dir):
            logger.error(
                f"[FTP Client - Image] Failed to ensure remote directory '{remote_target_dir}' exists on {alias}.")
            return False

        try:
            ftp.cwd(remote_target_dir)
            logger.debug(f"[FTP Client - Image] Changed remote CWD to: {remote_target_dir}")
        except ftplib.all_errors as cwd_err:
            logger.warning(
                f"[FTP Client - Image] Could not CWD to '{remote_target_dir}' on {alias} (Error: {cwd_err}). Will attempt STOR with filename.")

        logger.info(f"[FTP Client - Image] Starting upload: '{local_image_path}' -> '{alias}:{remote_filepath}'")
        with open(local_image_path, 'rb') as f:
            response = ftp.storbinary(f'STOR {filename}', f)
            logger.debug(f"[FTP Client - Image] FTP STOR response: {response}")

            if response.startswith("226") or response.startswith("250"):
                logger.info(f"[FTP Client - Image] Successfully uploaded '{filename}' to {alias}.")
                success = True
            else:
                logger.error(
                    f"[FTP Client - Image] Upload command failed for '{filename}' to {alias}. Response: {response}")

    except ftplib.error_perm as e:
        logger.error(f"[FTP Client - Image] FTP permission error for {alias}: {e}")
    except ftplib.error_temp as e:
        logger.error(f"[FTP Client - Image] FTP temporary error for {alias}: {e}")
    except socket.timeout:
        logger.error(f"[FTP Client - Image] FTP connection or transfer timed out for {alias}.")
    except ConnectionRefusedError:
        logger.error(f"[FTP Client - Image] FTP connection refused by server: {alias}")
    except Exception as e:
        logger.error(f"[FTP Client - Image] An unexpected error occurred during image FTP for {alias}: {e}",
                     exc_info=True)
    finally:
        if ftp:
            try:
                ftp.quit()
                logger.debug(f"[FTP Client - Image] Disconnected from {alias}.")
            except Exception:
                pass
    return success
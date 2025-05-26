# core/utils/helpers.py

import os
import re
import shutil
import logging
from datetime import datetime, date, timezone
from typing import Optional

from ..config import get_setting # Import config access

logger = logging.getLogger(__name__)

# --- Filename Sequence Number Parser ---
# Example: extracts '0' from "..._0.scnx.gz" or '10' from "..._10.scnx.gz"
SCAN_SEQUENCE_REGEX = re.compile(r'_(\d{1,2})\.scnx\.gz$')  # Assuming 1 or 2 digits for N


def parse_scan_sequence_number(filename: str) -> Optional[int]:
    """Parses the _N sequence number from a scan filename."""
    match = SCAN_SEQUENCE_REGEX.search(filename)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logger.warning(f"Could not parse sequence number from matched group '{match.group(1)}' in {filename}")
            return None
    logger.debug(f"Scan sequence number pattern not found in filename: {filename}")
    return None

# --- Datetime Parsing Functions ---

def parse_date_from_dirname(dir_name: str) -> Optional[date]:
    """
    Parses a directory name expected to be in YYYYMMDD format into a date object.

    Args:
        dir_name: The name of the directory (e.g., "20231027").

    Returns:
        A date object if parsing is successful, otherwise None.
    """
    try:
        return datetime.strptime(dir_name, "%Y%m%d").date()
    except ValueError:
        logger.debug(f"Could not parse directory name '{dir_name}' as YYYYMMDD date.")
        return None

def parse_datetime_from_filename(filename: str) -> Optional[datetime]:
    """
    Extracts datetime information from a radar scan filename and returns
    a timezone-aware datetime object assuming the time is UTC.

    Looks for _YYYYMMDD_HHMMSS_ pattern anywhere in the name.
    Example: 'prefix_YYYYMMDD_HHMMSS_suffix.extension'

    Args:
        filename: The basename of the file.

    Returns:
        A timezone-aware UTC datetime object if parsing is successful, otherwise None.
    """
    match = re.search(r'_(\d{8})_(\d{6})_', filename)
    if not match:
        logger.debug(f"Filename '{filename}' did not match expected datetime pattern '_YYYYMMDD_HHMMSS_'.")
        return None

    date_str, time_str = match.groups()
    try:
        dt_naive = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        dt_aware_utc = dt_naive.replace(tzinfo=timezone.utc)
        return dt_aware_utc
    except ValueError:
        logger.warning(f"Could not parse extracted datetime string '{date_str}{time_str}' from filename '{filename}'.")
        return None

def parse_datetime_from_image_filename(filename: str) -> Optional[datetime]:
    """
    Extracts datetime information from a plot image filename and returns
    a timezone-aware datetime object assuming the time is UTC.

    Expected format: VAR_ELEV_YYYYMMDD_HHMM.png (e.g., RATE_005_20231027_1530.png)

    Args:
        filename: The basename of the image file.

    Returns:
        A timezone-aware UTC datetime object if parsing is successful, otherwise None.
    """
    match = re.match(r'^([a-zA-Z0-9]+)_(\d{3})_(\d{8})_(\d{4})\.png$', filename, re.IGNORECASE)
    if not match:
        logger.debug(f"Image filename '{filename}' did not match expected pattern 'VAR_ELE_YYYYMMDD_HHMM.png'.")
        return None

    _var, _elev, date_str, time_str = match.groups()
    try:
        dt_naive = datetime.strptime(f"{date_str}{time_str}00", "%Y%m%d%H%M%S")
        dt_aware_utc = dt_naive.replace(tzinfo=timezone.utc)
        return dt_aware_utc
    except ValueError:
        logger.warning(f"Could not parse extracted datetime string '{date_str}{time_str}00' from image filename '{filename}'.")
        return None

# --- File Operations ---

def move_processed_file(
    source_filepath: str,
    processed_base_dir: str,
    elevation: float,
    scan_datetime: datetime
) -> Optional[str]:
    """
    Moves a processed file (e.g., a raw scan) to a structured directory
    based on its elevation and date: {base_dir}/{ElevationCode}/{YYYYMMDD}/.

    Args:
        source_filepath: The full path to the file to be moved.
        processed_base_dir: The base directory where processed files are stored.
        elevation: The elevation angle (in degrees) of the scan.
        scan_datetime: The timezone-aware datetime of the scan.

    Returns:
        The destination filepath if the move was successful, otherwise None.
    Raises:
         FileNotFoundError: If the source file does not exist.
         OSError: If creating directories fails.
         shutil.Error: If the move operation fails.
    """
    if not os.path.isfile(source_filepath):
        logger.error(f"Source file for move not found: {source_filepath}")
        raise FileNotFoundError(f"Source file not found: {source_filepath}")

    filename = os.path.basename(source_filepath)

    try:
        # Calculate Elevation Code
        elevation_code = f"{int(round(elevation * 100)):03d}"

        # Format Date String
        date_str = scan_datetime.strftime("%Y%m%d")

        # Construct Target Path
        target_dir = os.path.join(processed_base_dir, elevation_code, date_str)
        target_filepath = os.path.join(target_dir, filename)

        # Create the target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        # Check if the target file exists (Optional: handle overwriting)
        if os.path.exists(target_filepath):
            logger.warning(f"Target file '{target_filepath}' already exists. Overwriting.")
            # os.remove(target_filepath) # Uncomment to ensure overwrite works with shutil.move

        # Move the file
        shutil.move(source_filepath, target_filepath)
        logger.info(f"Moved '{source_filepath}' to '{target_filepath}'")
        return target_filepath

    except OSError as e:
        logger.error(f"Failed to create target directory '{target_dir}': {e}", exc_info=True)
        raise
    except shutil.Error as e:
        logger.error(f"Failed to move file '{source_filepath}' to '{target_filepath}': {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during file move: {e}", exc_info=True)
        return None
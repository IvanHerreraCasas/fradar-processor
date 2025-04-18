# core/utils/helpers.py

import os
import re
import shutil
import logging
from datetime import datetime, date, timezone
from typing import Optional

from ..config import get_setting # Import config access

logger = logging.getLogger(__name__)

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
        # Parse as naive datetime first
        dt_naive = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        # Make it timezone-aware by assigning UTC timezone info
        dt_aware_utc = dt_naive.replace(tzinfo=timezone.utc) # <<< Attach UTC timezone
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
    # Regex captures: 1=Variable, 2=ElevationCode, 3=Date(YYYYMMDD), 4=Time(HHMM)
    match = re.match(r'^([a-zA-Z0-9]+)_(\d{3})_(\d{8})_(\d{4})\.png$', filename, re.IGNORECASE)
    if not match:
        logger.debug(f"Image filename '{filename}' did not match expected pattern 'VAR_ELE_YYYYMMDD_HHMM.png'.")
        return None

    _var, _elev, date_str, time_str = match.groups()
    try:
        # Parse date and HHMM time, add :00 for seconds
        dt_naive = datetime.strptime(f"{date_str}{time_str}00", "%Y%m%d%H%M%S")
        # Make it timezone-aware UTC
        dt_aware_utc = dt_naive.replace(tzinfo=timezone.utc)
        return dt_aware_utc
    except ValueError:
        logger.warning(f"Could not parse extracted datetime string '{date_str}{time_str}00' from image filename '{filename}'.")
        return None

# --- File Operations ---

def move_processed_file(source_filepath: str, processed_base_dir: str) -> Optional[str]:
    """
    Moves a processed file (e.g., a raw scan) from its source location
    to a structured directory based on its date within a base output directory.
    The target subdirectory will be YYYYMMDD.

    Args:
        source_filepath: The full path to the file to be moved.
        processed_base_dir: The base directory where processed files are stored
                             (e.g., '/path/to/processed_data', fetched from config).

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
    file_dt = parse_datetime_from_filename(filename)

    if not file_dt:
        logger.error(f"Could not parse datetime from filename '{filename}'. Cannot determine target directory.")
        return None # Or raise an error if this is critical

    date_str = file_dt.strftime("%Y%m%d")
    target_dir = os.path.join(processed_base_dir, date_str)
    target_filepath = os.path.join(target_dir, filename)

    try:
        # Create the target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        # Check if the target file exists and remove it if necessary (optional, depends on desired behavior)
        if os.path.exists(target_filepath):
            logger.warning(f"Target file '{target_filepath}' already exists. Overwriting.")
            # Decide if you want to overwrite or skip/rename
            # os.remove(target_filepath) # Uncomment to overwrite

        # Move the file
        shutil.move(source_filepath, target_filepath)
        logger.info(f"Moved '{source_filepath}' to '{target_filepath}'")
        return target_filepath

    except OSError as e:
        logger.error(f"Failed to create target directory '{target_dir}': {e}", exc_info=True)
        raise # Re-raise OS errors
    except shutil.Error as e:
        logger.error(f"Failed to move file '{source_filepath}' to '{target_filepath}': {e}", exc_info=True)
        raise # Re-raise shutil errors
    except Exception as e:
        logger.error(f"An unexpected error occurred during file move: {e}", exc_info=True)
        # Decide whether to raise or return None for unexpected errors
        return None


# You can add other general helper functions here later if needed
# e.g., safe_float_conversion, specific string manipulations etc.
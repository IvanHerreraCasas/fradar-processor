# core/utils/csv_handler.py

import os
import logging
import pandas as pd
from typing import Optional, Dict, Any

import tempfile
import shutil

logger = logging.getLogger(__name__)

def read_timeseries_csv(csv_path: str, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Reads a timeseries CSV file into a pandas DataFrame.

    Args:
        csv_path: The full path to the CSV file.
        timestamp_col: The name of the column containing timestamps.

    Returns:
        A pandas DataFrame with timestamps parsed to timezone-aware UTC datetimes.
        Returns an empty DataFrame with expected columns if the file doesn't exist
        or is empty after header.
    """
    if not os.path.exists(csv_path):
        logger.debug(f"Timeseries file not found, returning empty DataFrame: {csv_path}")
        # Return empty DF with just the timestamp column for consistency in finding max later
        return pd.DataFrame(columns=[timestamp_col])

    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=[timestamp_col], 
            comment='#',  # Ignore lines starting with '#'
        )

        # Ensure timestamp column is timezone-aware (UTC)
        if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            if df[timestamp_col].dt.tz is None:
                logger.debug(f"Localizing timestamp column in {csv_path} to UTC (assuming it was UTC).")
                # If timezone naive, assume UTC and localize. Use errors='raise' to catch issues.
                df[timestamp_col] = df[timestamp_col].dt.tz_localize('UTC', ambiguous='raise', nonexistent='raise')
            else:
                # If timezone aware, convert to UTC just in case it was read as local time somehow
                df[timestamp_col] = df[timestamp_col].dt.tz_convert('UTC')
        else:
            logger.warning(f"Timestamp column '{timestamp_col}' in {csv_path} was not parsed as datetime. Cannot ensure UTC.")
            # Attempt conversion again more robustly?
            try:
                 df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors='coerce')
                 df.dropna(subset=[timestamp_col], inplace=True) # Drop rows where conversion failed
            except Exception as E:
                 logger.error(f"Failed to robustly convert timestamp column {timestamp_col} to datetime: {E}")
                 # Return empty DF if critical parsing fails
                 return pd.DataFrame(columns=[timestamp_col])


        # Drop potential duplicate timestamps, keeping the last entry
        # Sort just in case before dropping duplicates
        df = df.sort_values(by=timestamp_col)
        df = df.drop_duplicates(subset=[timestamp_col], keep='last')

        logger.debug(f"Successfully read {len(df)} rows from {csv_path}")
        return df

    except FileNotFoundError: # Should be caught by os.path.exists, but belt-and-suspenders
        logger.debug(f"Timeseries file not found (race condition?), returning empty DataFrame: {csv_path}")
        return pd.DataFrame(columns=[timestamp_col])
    except pd.errors.EmptyDataError:
         logger.debug(f"Timeseries file is empty (or only header): {csv_path}")
         return pd.DataFrame(columns=[timestamp_col]) # Return empty DF with column
    except KeyError:
        logger.error(f"Timestamp column '{timestamp_col}' not found in CSV: {csv_path}")
        return pd.DataFrame(columns=[timestamp_col]) # Return empty DF
    except Exception as e:
        logger.error(f"Failed to read or parse timeseries CSV {csv_path}: {e}", exc_info=True)
        # Return empty DF on other errors to prevent downstream failures
        return pd.DataFrame(columns=[timestamp_col])

def write_metadata_header_if_needed(csv_path: str, metadata: Dict[str, Any]):
    """
    Ensures the metadata header exists at the beginning of the CSV file.
    - If the file doesn't exist or is empty, creates it with the header.
    - If the file exists with data but no header comments, prepends the header.
    - If the file exists with header comments, does nothing.

    Args:
        csv_path: The full path to the CSV file.
        metadata: A dictionary containing the key-value pairs for the header.
    """
    should_write_or_prepend = False
    mode = 'simple_write' # Modes: 'simple_write', 'prepend', 'skip'
    header_lines_content = [f"# {key}: {value}\n" for key, value in metadata.items()]

    try:
        # --- Determine Action Needed ---
        if not os.path.exists(csv_path):
            logger.info(f"CSV file does not exist. Will create with metadata header: {csv_path}")
            should_write_or_prepend = True
            mode = 'simple_write'
        elif os.path.getsize(csv_path) == 0:
            logger.info(f"CSV file exists but is empty. Will write metadata header: {csv_path}")
            should_write_or_prepend = True
            mode = 'simple_write'
        else:
            # Check for existing '#' comments
            has_comments = False
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    # Check only first few lines needed to confirm header presence/absence
                    for _ in range(len(header_lines_content) + 5): # Check a bit beyond expected header
                        line = f.readline()
                        if not line: break # EOF
                        if line.strip().startswith('#'):
                            has_comments = True
                            break
            except Exception as read_err:
                 logger.warning(f"Could not read start of {csv_path} to check for header comments: {read_err}. Assuming header is missing.")
                 # If we can't read it to check, assume we need to prepend (risky but might be desired)
                 # Or, more safely, skip? Let's assume we should attempt prepend if check fails.
                 has_comments = False # Force prepend attempt if read fails

            if not has_comments:
                logger.info(f"CSV file exists with data but lacks comment header. Will prepend header: {csv_path}")
                should_write_or_prepend = True
                mode = 'prepend'
            else:
                 logger.debug(f"Metadata header comments already found in {csv_path}. Skipping header write.")
                 mode = 'skip'

        # --- Perform Action ---
        if should_write_or_prepend:
            target_dir = os.path.dirname(csv_path)
            if target_dir: # Ensure directory exists
                 os.makedirs(target_dir, exist_ok=True)

            if mode == 'simple_write':
                # Just write the header to a new/empty file
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.writelines(header_lines_content)
                logger.info(f"Successfully wrote new metadata header to {csv_path}")

            elif mode == 'prepend':
                # Prepend header safely using a temporary file
                temp_file_path = None
                try:
                    # Create temp file in the same directory for atomic replace
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=target_dir, prefix=".header_tmp_") as tf:
                        temp_file_path = tf.name
                        # 1. Write new header to temp file
                        tf.writelines(header_lines_content)
                        # 2. Append original content
                        with open(csv_path, 'r', encoding='utf-8') as original_f:
                            shutil.copyfileobj(original_f, tf) # Efficiently copy rest of file

                    # 3. Replace original file with temp file (atomic on most systems)
                    os.replace(temp_file_path, csv_path)
                    logger.info(f"Successfully prepended metadata header to {csv_path}")
                    temp_file_path = None # Prevent deletion in finally block

                except Exception as prepend_err:
                     logger.error(f"Failed to prepend metadata header to {csv_path}: {prepend_err}", exc_info=True)
                     # Don't raise here, allow main process to continue maybe?
                finally:
                     # Clean up temp file if something went wrong before os.replace
                     if temp_file_path and os.path.exists(temp_file_path):
                         try:
                             logger.debug(f"Cleaning up temporary header file: {temp_file_path}")
                             os.remove(temp_file_path)
                         except OSError as rm_err:
                             logger.error(f"Could not remove temporary file {temp_file_path}: {rm_err}")

    except Exception as e:
        logger.error(f"Error during metadata header check/write for {csv_path}: {e}", exc_info=True)
        # Decide if this should halt execution
        # raise

def append_to_timeseries_csv(csv_path: str, new_data_df: pd.DataFrame):
    """
    Appends new data to a timeseries CSV file. Writes the data header row
    if the file is new or empty (e.g., only contains metadata comments).

    Args:
        csv_path: The full path to the CSV file.
        new_data_df: A pandas DataFrame containing the new rows to append.
    """
    if new_data_df.empty:
        logger.debug("No new data provided to append.")
        return

    # Ensure the directory exists before trying to write
    csv_dir = os.path.dirname(csv_path)
    try:
        if csv_dir: # Avoid trying to create directory if path is just a filename
             os.makedirs(csv_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to ensure directory exists for {csv_path}: {e}")
        raise # Re-raise error to signal failure

    try:
        # Determine if the DATA header needs to be written.
        # Write header if the file essentially contains no data rows yet.
        write_data_header = False
        if not os.path.exists(csv_path):
            write_data_header = True # File doesn't exist, definitely need header
        else:
            try:
                 # Check if file contains any non-comment lines
                 has_data_rows = False
                 with open(csv_path, 'r', encoding='utf-8') as f:
                      for line in f:
                           if line.strip() and not line.strip().startswith('#'):
                                has_data_rows = True
                                break
                 if not has_data_rows:
                     write_data_header = True # No data rows found, need header
            except Exception as read_err:
                 logger.warning(f"Could not check existing content of {csv_path} for data header check: {read_err}. Assuming header needed.")
                 write_data_header = True # Be safe

        # Format timestamp for consistent CSV output
        if 'timestamp' in new_data_df.columns:
             if pd.api.types.is_datetime64_any_dtype(new_data_df['timestamp']):
                  if new_data_df['timestamp'].dt.tz is None:
                      ts_col = new_data_df['timestamp'].dt.tz_localize('UTC')
                  else:
                      ts_col = new_data_df['timestamp'].dt.tz_convert('UTC')
                  formatted_df = new_data_df.copy()
                  formatted_df['timestamp'] = ts_col.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
             else:
                  formatted_df = new_data_df
        else:
             formatted_df = new_data_df

        # Append data
        formatted_df.to_csv(
            csv_path,
            mode='a',               # Append mode
            header=write_data_header,# Write data header based on check above
            index=False,            # Don't write pandas index
            encoding='utf-8',       # Specify encoding
            lineterminator='\n'     # Ensure consistent line endings
        )
        logger.debug(f"Appended {len(new_data_df)} rows to {csv_path} (Data Header Written: {write_data_header})")

    except Exception as e:
        logger.error(f"Failed to append data to timeseries CSV {csv_path}: {e}", exc_info=True)
        
def write_timeseries_csv(csv_path: str, data_df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None):
    """
    Writes a DataFrame to a CSV file, optionally prepending a metadata header.
    This function will OVERWRITE the file if it exists.

    Args:
        csv_path: The full path to the output CSV file.
        data_df: The pandas DataFrame containing the data rows.
        metadata: An optional dictionary containing metadata key-value pairs
                  to write as comment lines at the beginning of the file.
    """
    logger.info(f"Writing timeseries data to: {csv_path} (Overwrite mode)")
    # Ensure output directory exists
    csv_dir = os.path.dirname(csv_path)
    try:
        if csv_dir: # Avoid trying to create directory if path is just a filename
             os.makedirs(csv_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to ensure directory exists for {csv_path}: {e}")
        raise # Re-raise error to signal failure

    try:
        # Format timestamp if present
        if 'timestamp' in data_df.columns:
             if pd.api.types.is_datetime64_any_dtype(data_df['timestamp']):
                  if data_df['timestamp'].dt.tz is None:
                      ts_col = data_df['timestamp'].dt.tz_localize('UTC')
                  else:
                      ts_col = data_df['timestamp'].dt.tz_convert('UTC')
                  # Use ISO format with 'Z' indicator for UTC
                  # Create a copy to avoid modifying the original DataFrame
                  output_df_formatted = data_df.copy()
                  output_df_formatted['timestamp'] = ts_col.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
             else:
                  output_df_formatted = data_df # Write as is
        else:
             output_df_formatted = data_df

        # Open in write mode ('w') to overwrite
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            # Write metadata header if provided
            if metadata and isinstance(metadata, dict):
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
                logger.debug(f"Metadata header written to {csv_path}")

            # Write the DataFrame data (including data header)
            output_df_formatted.to_csv(f, index=False, header=True, lineterminator='\n')

        logger.info(f"Successfully wrote {len(data_df)} rows to {csv_path}")

    except Exception as e:
        logger.error(f"Failed to write timeseries CSV {csv_path}: {e}", exc_info=True)
        raise # Re-raise error
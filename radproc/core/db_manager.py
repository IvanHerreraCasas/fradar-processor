# radproc/core/db_manager.py

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import psycopg2
import psycopg2.pool
import psycopg2.extras  # For execute_values, DictCursor
from datetime import datetime, timezone, timedelta  # Ensure timezone is imported

from .config import get_setting

logger = logging.getLogger(__name__)

# --- Connection Pool ---
# The pool should be initialized once when the module is loaded or on first use.
# For simplicity in a non-web-server context that might have multiple entry points (CLI, Huey),
# initializing it lazily can be a good approach.
_pool = None


def _get_db_config() -> Dict[str, Any]:
    """Retrieves database connection configuration."""
    db_conf = get_setting('database.postgresql')
    if not db_conf:
        raise ConnectionError("Database configuration ('database.postgresql') not found.")

    # Critical keys check
    required_keys = ['host', 'port', 'dbname', 'user']
    if not all(key in db_conf for key in required_keys):
        raise ConnectionError(f"Database configuration missing required keys: {required_keys}")
    password_env_var = "RADPROC_DB_PASSWORD"
    db_password = os.environ.get(password_env_var)
    conn_params = {
        "host": db_conf.get('host'),
        "port": db_conf.get('port', 5432),
        "dbname": db_conf.get('dbname'),
        "user": db_conf.get('user'),
    }
    if db_password:  # Only add password if it's set
        conn_params["password"] = db_password
    else:
        logger.warning(f"Env var '{password_env_var}' not set; DB connection might fail if password is required.")
    return conn_params


def _initialize_pool():
    global _pool
    if _pool is None:
        try:
            conn_params = _get_db_config()
            _pool = psycopg2.pool.SimpleConnectionPool(minconn=1, maxconn=5, **conn_params)
            logger.info("Database connection pool initialized.")
        except (psycopg2.Error, ConnectionError) as e:
            logger.critical(f"Failed to initialize database connection pool: {e}", exc_info=True)
            # Subsequent calls to get_connection will fail if pool is None
            _pool = None  # Ensure it stays None if initialization failed
        except Exception as e:
            logger.critical(f"Unexpected error during pool initialization: {e}", exc_info=True)
            _pool = None


def get_connection():
    """Gets a connection from the pool."""
    if _pool is None:
        _initialize_pool()  # Attempt to initialize if not already
        if _pool is None:  # Check again if initialization failed
            raise ConnectionError("Database connection pool is not available.")
    try:
        return _pool.getconn()
    except psycopg2.pool.PoolError as e:
        logger.error(f"Failed to get connection from pool: {e}", exc_info=True)
        raise ConnectionError(f"Failed to get connection from pool: {e}")


def release_connection(conn):
    """Releases a connection back to the pool."""
    if _pool and conn:
        try:
            _pool.putconn(conn)
        except psycopg2.pool.PoolError as e:
            logger.error(f"Failed to release connection to pool: {e}", exc_info=True)
            # If releasing fails, we might have a bigger pool problem.
            # Depending on the error, conn.close() might be an alternative.
            try:
                conn.close()  # Attempt to close it if putconn failed badly
            except Exception:
                pass


def close_pool():
    """Closes all connections in the pool (e.g., on application shutdown)."""
    global _pool
    if _pool:
        try:
            _pool.closeall()
            logger.info("Database connection pool closed.")
        except Exception as e:
            logger.error(f"Error closing database connection pool: {e}", exc_info=True)
        _pool = None


# --- Variable Management ---
def get_or_create_variable_id(conn, variable_name: str) -> Optional[int]:
    """Gets variable ID, creates if new. Units/desc are now optional for creation."""
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("SELECT variable_id FROM radproc_variables WHERE variable_name = %s;", (variable_name,))
        row = cur.fetchone()
        if row:
            return row[0]
        else:
            logger.info(f"Variable '{variable_name}' not found, creating it (with NULL units/description).")
            # Units and description are now nullable in the schema as per user feedback
            cur.execute(
                "INSERT INTO radproc_variables (variable_name, units, description) VALUES (%s, NULL, NULL) RETURNING variable_id;",
                (variable_name,)
            )
            new_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Created variable '{variable_name}' with ID {new_id}.")
            return new_id
    except psycopg2.Error as e:
        logger.error(f"Database error getting/creating variable '{variable_name}': {e}", exc_info=True)
        if conn and not conn.closed: conn.rollback()
        return None
    finally:
        if cur: cur.close()


# --- Point Management ---
def get_point_config_from_db(conn, point_name_or_id: Union[str, int]) -> Optional[Dict[str, Any]]:
    """Fetches point config reflecting new schema (no default_variable_name)."""
    logger.debug(f"Fetching point config for: {point_name_or_id}")
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        sql = """SELECT point_id, \
                        point_name, \
                        latitude, \
                        longitude, \
                        target_elevation,
                        description, \
                        cached_azimuth_index, \
                        cached_range_index
                 FROM radproc_points """  # Removed default_variable_name
        if isinstance(point_name_or_id, str):
            cur.execute(sql + "WHERE point_name = %s;", (point_name_or_id,))
        elif isinstance(point_name_or_id, int):
            cur.execute(sql + "WHERE point_id = %s;", (point_name_or_id,))
        else:
            logger.error("Invalid type for point_name_or_id in get_point_config_from_db.")
            return None
        row = cur.fetchone()
        return dict(row) if row else None
    except psycopg2.Error as e:
        logger.error(f"DB error fetching point config for '{point_name_or_id}': {e}", exc_info=True)
        return None
    finally:
        if cur: cur.close()


def get_all_points_from_db(conn) -> List[Dict[str, Any]]:
    """Fetches all points, reflecting new schema."""
    logger.debug("Fetching all point configurations from database.")
    points_list = []
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
                    SELECT point_id,
                           point_name,
                           latitude,
                           longitude,
                           target_elevation,
                           description,
                           cached_azimuth_index,
                           cached_range_index
                    FROM radproc_points
                    ORDER BY point_name;
                    """)  # Removed default_variable_name
        rows = cur.fetchall()
        for row in rows: points_list.append(dict(row))
        logger.info(f"Fetched {len(points_list)} points from database.")
    except psycopg2.Error as e:
        logger.error(f"DB error fetching all points: {e}", exc_info=True)
    finally:
        if cur: cur.close()
    return points_list


def update_point_cached_indices_in_db(conn, point_id: int, az_idx: Optional[int], rg_idx: Optional[int]) -> bool:
    """
        Updates the cached_azimuth_index and cached_range_index for a given point_id
        in the radproc_points table.
    """
    if point_id is None: return False
    logger.debug(f"Updating cached indices for point_id {point_id}: az={az_idx}, rg={rg_idx}")
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE radproc_points SET cached_azimuth_index = %s, cached_range_index = %s WHERE point_id = %s;",
            (az_idx, rg_idx, point_id)
        )
        conn.commit()
        if cur.rowcount == 0:
            logger.warning(f"No point found with point_id {point_id} to update cached indices.")
            return False
        return True
    except psycopg2.Error as e:
        logger.error(f"DB error updating cached indices for point_id {point_id}: {e}", exc_info=True)
        if conn and not conn.closed: conn.rollback()
        return False
    finally:
        if cur: cur.close()


# --- Timeseries Data Management ---
def get_existing_timestamps_for_multiple_points(
        conn, points_variables_list: List[Tuple[int, int]], start_dt: datetime, end_dt: datetime
) -> Dict[Tuple[int, int], Set[datetime]]:
    """
    Fetches existing timestamps for multiple (point_id, variable_id) pairs within a date range.
    Returns a dictionary mapping (point_id, variable_id) to a set of their existing timestamps.
    """
    logger.debug(f"Fetching existing timestamps for multiple point/variable pairs between {start_dt} and {end_dt}")
    results: Dict[Tuple[int, int], Set[datetime]] = {pv_tuple: set() for pv_tuple in points_variables_list}
    cur = None
    if not points_variables_list: return results
    try:
        cur = conn.cursor()
        # Constructing a query that can handle multiple (point_id, variable_id) pairs efficiently.
        # One way is to use a VALUES list and join, or iterate and do individual queries if the list is small.
        # For larger lists, a more complex single query is better.
        # For now, let's iterate for simplicity, can optimize later if it becomes a bottleneck.
        # A more optimized query might look like:
        # query_template = """
        #     SELECT point_id, variable_id, timestamp FROM timeseries_data
        #     WHERE (point_id, variable_id) IN %s AND timestamp >= %s AND timestamp <= %s;
        # """
        # cur.execute(query_template, (tuple(points_variables_list), start_dt, end_dt))
        # for row in cur.fetchall():
        #     results[(row[0], row[1])].add(row[2])

        # Simpler iterative approach for now:
        for point_id, variable_id in points_variables_list:
            query = """
                    SELECT timestamp \
                    FROM timeseries_data
                    WHERE point_id = %s \
                      AND variable_id = %s \
                      AND timestamp >= %s \
                      AND timestamp <= %s; \
                    """
            cur.execute(query, (point_id, variable_id, start_dt, end_dt))
            for row in cur.fetchall(): results[(point_id, variable_id)].add(row[0])
    except psycopg2.Error as e:
        logger.error(f"DB error fetching existing timestamps for multiple points: {e}", exc_info=True)
    finally:
        if cur: cur.close()
    return results


def batch_insert_timeseries_data(conn, data_list: List[Dict[str, Any]]) -> bool:
    """
    Batch inserts timeseries data. Expects a list of dicts:
    [{'timestamp': dt_obj, 'point_id': int, 'variable_id': int, 'value': float}, ...]
    """
    if not data_list:
        return True  # Nothing to insert

    cur = None
    try:
        cur = conn.cursor()
        data_tuples = [(item['timestamp'], item['point_id'], item['variable_id'], item['value']) for item in data_list]
        insert_query = "INSERT INTO timeseries_data (timestamp, point_id, variable_id, value) VALUES %s ON CONFLICT (timestamp, point_id, variable_id) DO NOTHING;"
        psycopg2.extras.execute_values(cur, insert_query, data_tuples, page_size=1000)
        conn.commit()
        logger.info(f"DB: Batch inserted/skipped {len(data_list)} timeseries rows.")
        return True
    except psycopg2.Error as e:
        logger.error(f"DB error during batch insert of timeseries: {e}", exc_info=True)
        if conn and not conn.closed: conn.rollback()
        return False
    except KeyError as e:
        logger.error(f"Data for batch insert missing key: {e}. Sample: {data_list[0] if data_list else 'N/A'}")
        return False
    finally:
        if cur: cur.close()

def query_timeseries_data_for_point(conn, point_id: int, variable_id: int,
                                    start_dt: datetime, end_dt: datetime,
                                    target_variable_name_for_df: str) -> pd.DataFrame:
    """
    Queries timeseries_data for a specific point/variable/range and returns a pandas DataFrame.
    The value column in the DataFrame will be named after target_variable_name_for_df.
    """
    logger.debug(f"Querying timeseries data for P:{point_id}, V:{variable_id} between {start_dt} and {end_dt}")
    sql = """
          SELECT timestamp, value
          FROM timeseries_data
          WHERE point_id = %s \
            AND variable_id = %s \
            AND timestamp >= %s \
            AND timestamp <= %s
          ORDER BY timestamp; \
          """
    try:
        # pd.read_sql_query handles connection and cursor management internally if conn is passed
        df = pd.read_sql_query(sql, conn, params=(point_id, variable_id, start_dt, end_dt))

        # Ensure timestamp is UTC (psycopg2 usually converts TIMESTAMPTZ to offset-aware datetimes)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')

        # Rename 'value' column to the actual variable name for clarity in pandas operations
        if 'value' in df.columns:
            df.rename(columns={'value': target_variable_name_for_df}, inplace=True)

        logger.info(f"Fetched {len(df)} rows from timeseries_data for P:{point_id}, V:{variable_id}")
        return df
    except psycopg2.Error as e:
        logger.error(f"DB error querying timeseries data for P:{point_id}, V:{variable_id}: {e}", exc_info=True)
        return pd.DataFrame(columns=['timestamp', target_variable_name_for_df])  # Return empty DF on error
    except Exception as e:  # Catch other errors like issues with pd.read_sql_query
        logger.error(f"Unexpected error querying timeseries data for P:{point_id}, V:{variable_id}: {e}", exc_info=True)
        return pd.DataFrame(columns=['timestamp', target_variable_name_for_df])


# --- Scan Log Management ---
def add_scan_to_log(conn, filepath: str, precise_ts: datetime, elevation: float,
                    sequence_num: int, nominal_ts: Optional[datetime] = None,
                    volume_identifier: Optional[datetime] = None) -> Optional[int]:
    """Adds a scan file entry to the radproc_scan_log. Returns scan_log_id or None."""
    logger.debug(f"Adding scan to log: {filepath}")
    cur = None
    try:
        cur = conn.cursor()
        # ON CONFLICT (filepath) DO NOTHING to handle if scan is somehow re-logged
        sql = """
              INSERT INTO radproc_scan_log
              (filepath, precise_timestamp, elevation, scan_sequence_number, nominal_filename_timestamp, \
               volume_identifier, processed_at)
              VALUES (%s, %s, %s, %s, %s, %s, NOW()) ON CONFLICT (filepath) DO NOTHING 
            RETURNING scan_log_id; \
              """
        # If ON CONFLICT happens, RETURNING scan_log_id might not return anything for that specific insert.
        # If we need the ID even if it conflicts, we might need a SELECT first or DO UPDATE.
        # For now, DO NOTHING is fine if we assume re-logging the exact same path is an edge case.
        cur.execute(sql, (filepath, precise_ts, elevation, sequence_num, nominal_ts, volume_identifier))
        row = cur.fetchone()
        conn.commit()
        if row:
            logger.info(f"Added scan {filepath} to log with ID {row[0]}.")
            return row[0]
        else:
            logger.warning(
                f"Scan {filepath} might already exist in log or insert failed silently (ON CONFLICT DO NOTHING).")
            # Optionally, query for existing ID if needed
            cur.execute("SELECT scan_log_id FROM radproc_scan_log WHERE filepath = %s;", (filepath,))
            existing_row = cur.fetchone()
            return existing_row[0] if existing_row else None
    except psycopg2.Error as e:
        logger.error(f"DB error adding scan to log {filepath}: {e}", exc_info=True)
        if conn and not conn.closed: conn.rollback()
        return None
    finally:
        if cur: cur.close()


def query_scan_log_for_timeseries_processing(conn, target_elevation: float,
                                             start_dt: datetime, end_dt: datetime,
                                             elevation_tolerance: float = 0.1
                                             ) -> List[Tuple[str, datetime, int]]:
    """
    Queries radproc_scan_log for scans matching elevation and time range.
    This REPLACES filesystem-based get_filepaths_in_range for this purpose.
    Returns list of (filepath, precise_timestamp, scan_log_id).
    """
    logger.info(f"Querying scan log for Elev~{target_elevation:.2f} between {start_dt} and {end_dt}")
    results: List[Tuple[str, datetime, int]] = []
    cur = None
    try:
        cur = conn.cursor()
        sql = """
              SELECT filepath, precise_timestamp, scan_log_id
              FROM radproc_scan_log
              WHERE elevation >= %s \
                AND elevation <= %s
                AND precise_timestamp >= %s \
                AND precise_timestamp <= %s
              ORDER BY precise_timestamp; \
              """
        cur.execute(sql, (
            target_elevation - elevation_tolerance,
            target_elevation + elevation_tolerance,
            start_dt,
            end_dt
        ))
        for row in cur.fetchall():
            results.append((row[0], row[1], row[2]))  # filepath, precise_ts, scan_log_id
        logger.info(f"Found {len(results)} scans in log matching criteria.")
    except psycopg2.Error as e:
        logger.error(f"DB error querying scan log for timeseries: {e}", exc_info=True)
    finally:
        if cur: cur.close()
    return results


def get_ungrouped_scans_for_volume_assignment(conn,
                                              lookback_hours: int = 24,
                                              limit: Optional[int] = 1000
                                              ) -> List[Dict[str, Any]]:
    """Fetches scan log entries where volume_identifier is NULL, ordered by time."""
    logger.debug(f"Fetching ungrouped scans from past {lookback_hours} hours.")
    results = []
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Look back to avoid processing very old ungrouped scans indefinitely
        time_cutoff = datetime.now(timezone.utc) - pd.Timedelta(hours=lookback_hours)
        sql = """
              SELECT scan_log_id, filepath, precise_timestamp, elevation, scan_sequence_number, nominal_filename_timestamp
              FROM radproc_scan_log
              WHERE volume_identifier IS NULL \
                AND precise_timestamp >= %s
              ORDER BY precise_timestamp ASC \
              """
        if limit:
            sql += f" LIMIT {int(limit)}"  # Basic limit

        cur.execute(sql, (time_cutoff,))
        for row in cur.fetchall():
            results.append(dict(row))
    except psycopg2.Error as e:
        logger.error(f"DB error fetching ungrouped scans: {e}", exc_info=True)
    finally:
        if cur: cur.close()
    return results


def update_volume_identifier_for_scans(conn, scan_log_ids: List[int], volume_identifier: datetime) -> int: # Returns count
    """Updates the volume_identifier for a list of scan_log_ids."""
    if not scan_log_ids: return 0
    logger.info(f"Attempting to update volume ID to {volume_identifier.isoformat()} for {len(scan_log_ids)} scan log entries.")
    cur = None
    successful_updates = 0
    try:
        cur = conn.cursor()
        for scan_log_id in scan_log_ids: # Process one by one
            try:
                cur.execute(
                    "UPDATE radproc_scan_log SET volume_identifier = %s WHERE scan_log_id = %s;",
                    (volume_identifier, scan_log_id)
                )
                if cur.rowcount > 0:
                    successful_updates += 1
                # No need to commit here if we commit after the loop
            except psycopg2.errors.UniqueViolation:
                conn.rollback() # Rollback the current transaction for this single conflicting update
                logger.warning(f"Skipped assigning volume ID to scan_log_id {scan_log_id} due to unique constraint conflict for (volume_identifier, sequence_number, elevation). The conflicting key was for volume {volume_identifier.isoformat()}.")
                # Start a new transaction for the next iteration if default is not autocommit
                # If autocommit is off (default), a new transaction starts automatically after rollback.
            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"DB error updating volume ID for scan_log_id {scan_log_id}: {e}", exc_info=True)
                # Optionally re-raise or handle more specifically if this error should stop the batch
        conn.commit() # Commit all successful non-conflicting updates
        logger.info(f"Successfully updated {successful_updates} of {len(scan_log_ids)} scan log entries with volume ID.")
        return successful_updates
    except Exception as e:
        if conn and not conn.closed: conn.rollback()
        logger.error(f"General error during batch update of volume IDs: {e}", exc_info=True)
        return 0
    finally:
        if cur: cur.close()


def find_latest_scan_for_sequence(conn,
                                  target_sequence_number: int,
                                  before_timestamp: datetime,
                                  time_window_seconds: int,
                                  for_volume_id_timestamp: Optional[datetime] = None
                                  ) -> Optional[Dict[str, Any]]:
    """
    Finds the latest scan entry for a specific sequence_number that occurred
    within a given time window before a reference timestamp.
    Optionally, can try to match based on a known volume_identifier (timestamp of _000 scan)
    if provided, to ensure predecessor belongs to the same tentative volume start.
    """
    cur = None
    logger.debug(
        f"Searching for predecessor seq {target_sequence_number} before {before_timestamp.isoformat()} within {time_window_seconds}s window.")
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        start_search_ts = before_timestamp - timedelta(seconds=time_window_seconds)

        params = [target_sequence_number, start_search_ts, before_timestamp]

        # Base SQL query
        sql = """
              SELECT scan_log_id, \
                     precise_timestamp, \
                     nominal_filename_timestamp, \
                     volume_identifier, \
                     scan_sequence_number, \
                     elevation
              FROM radproc_scan_log
              WHERE scan_sequence_number = %s
                AND precise_timestamp >= %s
                AND precise_timestamp < %s \
              """
        # If we have a candidate volume_id_timestamp (from the current _000 scan being processed),
        # we prefer a predecessor that matches this. If not, we take any.
        # This is more relevant if multiple _000 scans are very close in time.
        # For simpler logic, we might not need for_volume_id_timestamp initially,
        # as the tight MAX_INTER_SCAN_GAP should prevent jumping far.
        # However, if used:
        if for_volume_id_timestamp:
            sql += " AND (volume_identifier = %s OR volume_identifier IS NULL) "  # Allow predecessor to be already part of this forming volume or not yet grouped
            params.append(for_volume_id_timestamp)
            sql += " ORDER BY precise_timestamp DESC, CASE WHEN volume_identifier = %s THEN 0 ELSE 1 END"  # Prefer already matched volume ID
            params.append(for_volume_id_timestamp)
        else:
            sql += " ORDER BY precise_timestamp DESC"

        sql += " LIMIT 1;"

        cur.execute(sql, tuple(params))
        row = cur.fetchone()
        if row:
            logger.debug(f"Found potential predecessor: {dict(row)}")
        else:
            logger.debug(
                f"No predecessor found for seq {target_sequence_number} before {before_timestamp.isoformat()}.")
        return dict(row) if row else None
    except psycopg2.Error as e:
        logger.error(f"DB error finding latest scan for seq {target_sequence_number}: {e}", exc_info=True)
        return None
    finally:
        if cur: cur.close()

# find_potential_volume_members might be better implemented in the CLI command logic itself,
# by querying ungrouped scans and then applying Python logic to group them based on
# sequence number and time proximity around an identified '_0' scan.
# For db_manager, a function to get scans around a certain time for a certain sequence might be:
def get_potential_volume_members_by_time_and_seq(conn,
                                                 start_ts_range: datetime,
                                                 end_ts_range: datetime,
                                                 min_seq_num: Optional[int] = None
                                                 ) -> List[Dict[str, Any]]:
    """Gets ungrouped scans within a time window, optionally from a min sequence number."""
    logger.debug(f"Fetching potential volume members between {start_ts_range} and {end_ts_range}, seq >= {min_seq_num}")
    results = []
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        params = [start_ts_range, end_ts_range]
        sql = """
              SELECT scan_log_id, filepath, precise_timestamp, elevation, scan_sequence_number
              FROM radproc_scan_log
              WHERE volume_identifier IS NULL
                AND precise_timestamp >= %s
                AND precise_timestamp <= %s \
              """
        if min_seq_num is not None:
            sql += " AND scan_sequence_number >= %s"
            params.append(min_seq_num)
        sql += " ORDER BY precise_timestamp ASC, scan_sequence_number ASC;"

        cur.execute(sql, tuple(params))
        for row in cur.fetchall():
            results.append(dict(row))
    except psycopg2.Error as e:
        logger.error(f"DB error fetching potential volume members: {e}", exc_info=True)
    finally:
        if cur: cur.close()
    return results



def add_processed_volume_log(conn, volume_identifier: datetime, filepath: str, version: str) -> Optional[int]:
    """
    Adds a record for a newly created processed volume file (e.g., CfRadial2).

    Args:
        conn: Active database connection.
        volume_identifier: The timestamp identifier of the volume group.
        filepath: The full path to the newly created processed volume file.
        version: The version string of the algorithm used.

    Returns:
        The new `processed_volume_id` if successful, otherwise None.
    """
    logger.debug(f"Logging processed volume for volume_identifier {volume_identifier.isoformat()}, version {version}.")
    cur = None
    try:
        cur = conn.cursor()
        sql = """
            INSERT INTO radproc_processed_volumes
            (volume_identifier, filepath, processing_version)
            VALUES (%s, %s, %s)
            ON CONFLICT (volume_identifier) DO UPDATE SET
                filepath = EXCLUDED.filepath,
                processing_version = EXCLUDED.processing_version,
                processed_at = NOW()
            RETURNING processed_volume_id;
            """
        cur.execute(sql, (volume_identifier, filepath, version))
        row = cur.fetchone()
        conn.commit()
        if row:
            logger.info(f"Logged processed volume {filepath} with ID {row[0]}.")
            return row[0]
        return None
    except psycopg2.Error as e:
        logger.error(f"DB error logging processed volume for volume_identifier {volume_identifier.isoformat()}: {e}", exc_info=True)
        if conn and not conn.closed: conn.rollback()
        return None
    finally:
        if cur: cur.close()


def query_unprocessed_volumes(conn, min_scans_per_volume: int = 10) -> List[datetime]:
    """
    Finds volume_identifiers that are complete and have not yet been processed.

    Args:
        conn: Active database connection.
        min_scans_per_volume: The minimum number of scans required to consider a volume complete.

    Returns:
        A list of `volume_identifier` timestamps that are ready to be processed.
    """
    logger.info("Querying for complete, unprocessed volumes...")
    cur = None
    results = []
    try:
        cur = conn.cursor()
        sql = """
            WITH complete_volumes AS (
                SELECT volume_identifier
                FROM radproc_scan_log
                WHERE volume_identifier IS NOT NULL
                GROUP BY volume_identifier
                HAVING COUNT(scan_log_id) >= %s
            )
            SELECT cv.volume_identifier
            FROM complete_volumes cv
            LEFT JOIN radproc_processed_volumes pv ON cv.volume_identifier = pv.volume_identifier
            WHERE pv.processed_volume_id IS NULL
            ORDER BY cv.volume_identifier;
        """
        cur.execute(sql, (min_scans_per_volume,))
        results = [row[0] for row in cur.fetchall()] # row[0] will be a datetime object
        logger.info(f"Found {len(results)} unprocessed volumes ready for processing.")
    except psycopg2.Error as e:
        logger.error(f"DB error querying for unprocessed volumes: {e}", exc_info=True)
    finally:
        if cur: cur.close()
    return results
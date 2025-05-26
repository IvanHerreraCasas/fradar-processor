# radproc/core/db_manager.py

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psycopg2
import psycopg2.pool
import psycopg2.extras  # For execute_values, DictCursor
from datetime import datetime, timezone  # Ensure timezone is imported

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
              SELECT scan_log_id, filepath, precise_timestamp, elevation, scan_sequence_number
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


def update_volume_identifier_for_scans(conn, scan_log_ids: List[int], volume_identifier: datetime) -> bool:
    """Updates the volume_identifier for a list of scan_log_ids."""
    if not scan_log_ids: return False
    logger.info(f"Updating volume ID to {volume_identifier.isoformat()} for {len(scan_log_ids)} scan log entries.")
    cur = None
    try:
        cur = conn.cursor()
        # Use tuple for IN clause
        sql = "UPDATE radproc_scan_log SET volume_identifier = %s WHERE scan_log_id IN %s;"
        cur.execute(sql, (volume_identifier, tuple(scan_log_ids)))
        conn.commit()
        logger.info(f"Updated {cur.rowcount} scan log entries with volume ID.")
        return cur.rowcount > 0
    except psycopg2.Error as e:
        logger.error(f"DB error updating volume IDs: {e}", exc_info=True)
        if conn and not conn.closed: conn.rollback()
        return False
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
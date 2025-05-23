# radproc/core/db_manager.py

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psycopg2
import psycopg2.pool
import psycopg2.extras  # For execute_values
from datetime import datetime

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
        raise ConnectionError(f"Database configuration is missing one or more required keys: {required_keys}")

    # Password should be loaded from an environment variable
    password_env_var = "RADPROC_DB_PASSWORD"  # Centralize env var name
    db_password = os.environ.get(password_env_var)
    if not db_password:
        logger.warning(
            f"Database password environment variable '{password_env_var}' is not set. "
            "Connection may fail if password is required."
        )
        # Allow proceeding without password if DB allows (e.g. peer auth for local CLI tools)
        # but for application logic, password will likely be needed.

    # Construct DSN or keyword arguments for psycopg2
    # Using keyword arguments is often clearer
    conn_params = {
        "host": db_conf.get('host'),
        "port": db_conf.get('port', 5432),
        "dbname": db_conf.get('dbname'),
        "user": db_conf.get('user'),
    }
    if db_password:  # Only add password if it's set
        conn_params["password"] = db_password

    return conn_params


def _initialize_pool():
    """Initializes the connection pool if it hasn't been already."""
    global _pool
    if _pool is None:
        try:
            conn_params = _get_db_config()
            # Example: min 1, max 5 connections in the pool
            _pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,  # Adjust based on expected concurrency (CLI, Huey workers)
                **conn_params
            )
            logger.info("Database connection pool initialized.")
        except (psycopg2.Error, ConnectionError) as e:
            logger.critical(f"Failed to initialize database connection pool: {e}", exc_info=True)
            # Subsequent calls to get_connection will fail if pool is None
            _pool = None  # Ensure it stays None if initialization failed
        except Exception as e:
            logger.critical(f"An unexpected error occurred during pool initialization: {e}", exc_info=True)
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


# --- Helper to ensure IDs exist and fetch/create them ---

def get_or_create_variable_id(conn, variable_name: str, units: Optional[str] = None,
                              description: Optional[str] = None) -> Optional[int]:
    """
    Gets the ID of a variable if it exists, otherwise creates it and returns the new ID.
    Returns None if operation fails.
    """
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("SELECT variable_id FROM radproc_variables WHERE variable_name = %s;", (variable_name,))
        row = cur.fetchone()
        if row:
            return row[0]
        else:
            # Variable doesn't exist, create it
            logger.info(f"Variable '{variable_name}' not found, creating it.")
            cur.execute(
                "INSERT INTO radproc_variables (variable_name, units, description) VALUES (%s, %s, %s) RETURNING variable_id;",
                (variable_name, units, description)
            )
            new_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Created variable '{variable_name}' with ID {new_id}.")
            return new_id
    except psycopg2.Error as e:
        logger.error(f"Database error getting/creating variable '{variable_name}': {e}", exc_info=True)
        if conn: conn.rollback()  # Rollback on error
        return None
    finally:
        if cur: cur.close()


def get_all_points_from_db(conn) -> List[Dict[str, Any]]:
    """Fetches all point configurations from the radproc_points table, ordered by point_name."""
    logger.debug("Fetching all point configurations from database.")
    points_list = []
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Select columns according to the new radproc_points schema
        cur.execute("""
            SELECT point_id, point_name, latitude, longitude, target_elevation, 
                   description, cached_azimuth_index, cached_range_index 
            FROM radproc_points 
            ORDER BY point_name;
        """)
        rows = cur.fetchall()
        for row in rows:
            points_list.append(dict(row))
        logger.info(f"Fetched {len(points_list)} points from database.")
    except psycopg2.Error as e:
        logger.error(f"Database error fetching all points: {e}", exc_info=True)
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error fetching all points: {e}", exc_info=True)
    finally:
        if cur: cur.close()
    return points_list

def get_point_db_id(conn, point_name: str) -> Optional[int]:
    """Fetches the point_id for a given point_name from radproc_points."""
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("SELECT point_id FROM radproc_points WHERE point_name = %s;", (point_name,))
        row = cur.fetchone()
        if row:
            return row[0]
        else:
            logger.warning(f"Point name '{point_name}' not found in radproc_points table.")
            return None
    except psycopg2.Error as e:
        logger.error(f"Database error fetching point ID for '{point_name}': {e}", exc_info=True)
        return None
    finally:
        if cur: cur.close()


# --- Placeholder for other functions from the plan ---

def get_point_config_from_db(conn, point_name_or_id: Union[str, int]) -> Optional[Dict[str, Any]]:
    logger.debug(f"Fetching point config for: {point_name_or_id}")
    cur = None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Select columns according to the new radproc_points schema
        sql = """SELECT point_id, \
                        point_name, \
                        latitude, \
                        longitude, \
                        target_elevation,
                        description, \
                        cached_azimuth_index, \
                        cached_range_index
                 FROM radproc_points """
        if isinstance(point_name_or_id, str):
            cur.execute(sql + "WHERE point_name = %s;", (point_name_or_id,))
        elif isinstance(point_name_or_id, int):
            cur.execute(sql + "WHERE point_id = %s;", (point_name_or_id,))
        else:
            logger.error("Invalid type for point_name_or_id in get_point_config_from_db.")
            return None

        row = cur.fetchone()
        if row:
            return dict(row)
        else:
            logger.warning(f"Point '{point_name_or_id}' not found in radproc_points table.")
            return None
    except psycopg2.Error as e:
        logger.error(f"Database error fetching point config for '{point_name_or_id}': {e}", exc_info=True)
        return None
    finally:
        if cur: cur.close()


def get_existing_timestamps(conn, point_id: int, variable_id: int, start_dt: datetime, end_dt: datetime) -> Set[
    datetime]:
    # TODO: Implement fetching existing timestamps for a single point/variable
    logger.debug(f"Fetching existing timestamps for P:{point_id} V:{variable_id} between {start_dt} and {end_dt}")
    timestamps = set()
    cur = None
    try:
        cur = conn.cursor()
        query = """
                SELECT DISTINCT timestamp \
                FROM timeseries_data
                WHERE point_id = %s \
                  AND variable_id = %s \
                  AND timestamp >= %s \
                  AND timestamp <= %s; \
                """
        cur.execute(query, (point_id, variable_id, start_dt, end_dt))
        for row in cur.fetchall():
            timestamps.add(row[0])  # Timestamps are TIMESTAMPTZ, psycopg2 handles conversion
        logger.info(f"Found {len(timestamps)} existing timestamps in DB for P:{point_id} V:{variable_id} in range.")
    except psycopg2.Error as e:
        logger.error(f"Database error fetching existing timestamps: {e}", exc_info=True)
        # Return empty set on error, so processing might try to re-insert everything
    finally:
        if cur: cur.close()
    return timestamps


def get_existing_timestamps_for_multiple_points(
        conn,
        points_variables_list: List[Tuple[int, int]],  # List of (point_id, variable_id)
        start_dt: datetime,
        end_dt: datetime
) -> Dict[Tuple[int, int], Set[datetime]]:
    """
    Fetches existing timestamps for multiple (point_id, variable_id) pairs within a date range.
    Returns a dictionary mapping (point_id, variable_id) to a set of their existing timestamps.
    """
    logger.debug(f"Fetching existing timestamps for multiple point/variable pairs between {start_dt} and {end_dt}")
    results: Dict[Tuple[int, int], Set[datetime]] = {pv_tuple: set() for pv_tuple in points_variables_list}
    cur = None

    if not points_variables_list:
        return results

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
            for row in cur.fetchall():
                results[(point_id, variable_id)].add(row[0])
            logger.info(
                f"Found {len(results[(point_id, variable_id)])} existing timestamps for P:{point_id} V:{variable_id} in range.")

    except psycopg2.Error as e:
        logger.error(f"Database error fetching existing timestamps for multiple points: {e}", exc_info=True)
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
        # Using execute_values for efficient batch inserting
        # Ensure columns match the table definition and order in `data_tuple`
        # (timestamp, point_id, variable_id, value)
        data_tuples = [
            (item['timestamp'], item['point_id'], item['variable_id'], item['value'])
            for item in data_list
        ]

        insert_query = """
                       INSERT INTO timeseries_data (timestamp, point_id, variable_id, value)
                       VALUES %s ON CONFLICT (timestamp, point_id, variable_id) DO NOTHING; \
                       """
        # Note: ON CONFLICT requires PostgreSQL 9.5+
        # The primary key (timestamp, point_id, variable_id) serves as the conflict target.

        psycopg2.extras.execute_values(
            cur,
            insert_query,
            data_tuples,
            page_size=1000  # Adjust page_size as needed
        )
        conn.commit()
        logger.info(f"Successfully batch inserted/updated {len(data_list)} timeseries data rows.")
        return True
    except psycopg2.Error as e:
        logger.error(f"Database error during batch insert of timeseries data: {e}", exc_info=True)
        if conn: conn.rollback()
        return False
    except KeyError as e:
        logger.error(
            f"Data for batch insert is missing a required key: {e}. Data sample: {data_list[0] if data_list else 'empty'}")
        return False
    finally:
        if cur: cur.close()

def update_point_cached_indices_in_db(conn, point_id: int, az_idx: Optional[int], rg_idx: Optional[int]) -> bool:
    """
    Updates the cached_azimuth_index and cached_range_index for a given point_id
    in the radproc_points table.
    """
    if point_id is None:
        logger.error("Cannot update cached indices: point_id is None.")
        return False

    logger.debug(f"Attempting to update cached indices for point_id {point_id} to az={az_idx}, rg={rg_idx}")
    cur = None
    try:
        cur = conn.cursor()
        query = """
            UPDATE radproc_points
            SET cached_azimuth_index = %s, cached_range_index = %s
            WHERE point_id = %s;
        """
        cur.execute(query, (az_idx, rg_idx, point_id))
        conn.commit()
        if cur.rowcount == 0:
            logger.warning(f"No point found with point_id {point_id} during cached indices update. Indices not saved to DB.")
            return False # Or perhaps true if we consider it "handled" for a non-existent point
        logger.info(f"Successfully updated cached indices for point_id {point_id} in database.")
        return True
    except psycopg2.Error as e:
        logger.error(f"Database error updating cached indices for point_id {point_id}: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected error updating cached indices for point_id {point_id}: {e}", exc_info=True)
        if conn and not conn.closed: # Check if conn is not closed before rollback
             try:
                 conn.rollback()
             except psycopg2.Error as rb_err:
                 logger.error(f"Rollback failed: {rb_err}")
        return False
    finally:
        if cur:
            cur.close()
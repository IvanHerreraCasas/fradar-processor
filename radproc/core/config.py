# core/config.py
import yaml
import os
import logging
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

_config: Optional[Dict[str, Any]] = None
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))

def _load_yaml(filepath: str) -> Dict[str, Any]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {filepath}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {filepath}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration file {filepath}: {e}")
        return {}

def load_config(config_dir: str = CONFIG_DIR) -> None:
    global _config
    if _config is not None:
        logger.debug("Configuration already loaded.")
        return

    logger.info(f"Loading configuration from: {config_dir}")
    _config = {}
    config_files_map = {
        'app': 'app_config.yaml',
        'ftp': 'ftp_config.yaml',
        'radar': 'radar_params.yaml',
        'styles': 'plot_styles.yaml',
        'database': 'database_config.yaml',
        'corrections': 'corrections_config.yaml'
    }
    required_configs = ['app', 'database'] # 'database' is now also essential

    all_loaded_successfully = True
    for key, filename in config_files_map.items():
        filepath = os.path.join(config_dir, filename)
        logger.debug(f"Attempting to load: {filepath} into key '{key}'")
        loaded_data = _load_yaml(filepath)
        if key in required_configs and not loaded_data:
             logger.error(f"CRITICAL: Required configuration file '{filename}' is empty or could not be loaded.")
             all_loaded_successfully = False
        _config[key] = loaded_data

    if not all_loaded_successfully and any(key in required_configs for key in _config if not _config[key]):
        raise ValueError("One or more essential configuration files (app, database) failed to load.")
    elif not all_loaded_successfully:
        logger.warning("One or more non-essential configuration files failed to load or were empty.")

    logger.info("Configuration loading process complete.")

def get_config() -> Dict[str, Any]:
    if _config is None:
        logger.warning("Configuration accessed before being explicitly loaded. Loading now.")
        load_config()
    if _config is None: # Should be caught by load_config raising error
        logger.error("Configuration is unavailable.")
        return {}
    return _config

def get_setting(key_path: str, default: Any = None) -> Any:
    cfg = get_config()
    keys = key_path.split('.')
    value = cfg
    try:
        for key in keys:
            if isinstance(value, list):
                try:
                    idx = int(key)
                    value = value[idx]
                except (ValueError, IndexError):
                    return default
            elif isinstance(value, dict):
                value = value[key]
            else:
                return default
        return value if value is not None else default # Return value or default
    except (KeyError, IndexError, TypeError):
        return default

# --- Point Configuration Functions (Now Database-Driven) ---

def get_point_config(point_name: str) -> Optional[Dict[str, Any]]:
    """
    Finds the configuration dictionary for a specific point by its name
    by querying the database.
    """
    # Import locally to avoid circular dependency at module load time
    from .db_manager import get_connection, release_connection, get_point_config_from_db
    logger.debug(f"Fetching DB config for point: '{point_name}'")
    conn = None
    try:
        conn = get_connection()
        point_details = get_point_config_from_db(conn, point_name)
        return point_details
    except ConnectionError as ce:
        logger.error(f"DB Connection Error fetching point config for '{point_name}': {ce}",
                     exc_info=False)  # Less noisy for expected errors
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching point config for '{point_name}' from DB: {e}", exc_info=True)
        return None
    finally:
        if conn:
            release_connection(conn)

def get_all_points_config() -> List[Dict[str, Any]]:
    """
    Fetches all point configurations from the radproc_points table in the database.
    Returns an empty list if an error occurs.
    """
    # Import locally
    from .db_manager import get_connection, release_connection, get_all_points_from_db
    logger.debug("Fetching all point configurations from database.")
    conn = None
    try:
        conn = get_connection()
        points_list = get_all_points_from_db(conn)
        return points_list
    except ConnectionError as ce:
        logger.error(f"DB Connection Error fetching all point configs: {ce}", exc_info=False)
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching all point configs from DB: {e}", exc_info=True)
        return []
    finally:
        if conn:
            release_connection(conn)

    # The old get_point_config (reading from YAML via get_setting) is now replaced.
# `load_config` still loads 'points.yaml' into `_config['points_config']`
# This allows the migration script (Phase 4) to access the YAML data via
# `get_setting('points_config.points')` for the initial import. Optional[Dict[str, Any]]:
    """
    Finds the configuration dictionary for a specific point by its name.

    Args:
        point_name: The unique 'name' of the point defined in points.yaml.

    Returns:
        The dictionary containing the point's configuration (name, lat, lon, etc.)
        or None if the point is not found or points config is missing.
    """
    points_data = get_setting('points_config.points') # Access points list safely

    if not isinstance(points_data, list):
        if points_data is not None: # Log if it exists but isn't a list
            logger.warning("'points_config.points' in configuration is not a list. Cannot find point.")
        # Don't log if it's just None (meaning file didn't load or was empty)
        return None

    for point_dict in points_data:
        if isinstance(point_dict, dict) and point_dict.get('name') == point_name:
            # Basic validation for required keys within the found point dict
            if 'latitude' not in point_dict or 'longitude' not in point_dict or 'variable' not in point_dict or "elevation" not in point_dict:
                 logger.warning(f"Point '{point_name}' found, but missing required keys (latitude, longitude, variable, elevation).")
                 return None # Treat incomplete points as invalid
            return point_dict

    logger.warning(f"Point configuration for name '{point_name}' not found in points.yaml.")
    return None
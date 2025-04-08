# core/config.py
import yaml
import os
import logging # Import logging
from typing import Any, Dict, Optional, List # Added List

logger = logging.getLogger(__name__) # Use logger for warnings/errors

_config: Optional[Dict[str, Any]] = None
# Assume CONFIG_DIR is correctly set relative to this file's location
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))

def _load_yaml(filepath: str) -> Dict[str, Any]:
    """Loads a single YAML file."""
    try:
        # Ensure directory exists before trying to open (config dir should always exist)
        # os.makedirs(os.path.dirname(filepath), exist_ok=True) # Less critical for config dir
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data if data is not None else {} # Handle empty files
    except FileNotFoundError:
        # Use logger.warning for non-critical missing files
        # Make it an error only for essential files like app_config
        logger.warning(f"Configuration file not found: {filepath}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {filepath}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration file {filepath}: {e}")
        return {}

def load_config(config_dir: str = CONFIG_DIR) -> None:
    """Loads all YAML configuration files from the specified directory."""
    global _config
    if _config is not None:
        logger.debug("Configuration already loaded.")
        return # Already loaded

    logger.info(f"Loading configuration from: {config_dir}")
    _config = {}
    # Define expected config files and their root keys
    config_files_map = {
        'app': 'app_config.yaml',
        'ftp': 'ftp_config.yaml',
        'radar': 'radar_params.yaml',
        'styles': 'plot_styles.yaml',
        'points_config': 'points.yaml', # <<< NEW: Load points config
    }
    # Define which configs are essential for basic operation
    required_configs = ['app'] # Modify as needed

    all_loaded_successfully = True
    for key, filename in config_files_map.items():
        filepath = os.path.join(config_dir, filename)
        logger.debug(f"Attempting to load: {filepath} into key '{key}'")
        loaded_data = _load_yaml(filepath)

        # Check if required config files loaded successfully
        if key in required_configs and not loaded_data:
             logger.error(f"CRITICAL: Required configuration file '{filename}' is empty or could not be loaded.")
             all_loaded_successfully = False
             # You might want to raise an error here to halt execution if 'app' config is missing
             # raise ValueError(f"Required config file '{filename}' failed to load.")

        _config[key] = loaded_data

    if not all_loaded_successfully:
        # Log a general warning if non-essential files failed, but allow continuation
        logger.warning("One or more non-essential configuration files failed to load or were empty.")
        # If essential files failed, the error/exception should have already been logged/raised


    # --- Basic Validation (Optional but Recommended) ---
    # Example: check essential app paths
    if not get_setting('app.input_dir'): # Use get_setting for safer access
         logger.warning("Configuration check: 'app.input_dir' is not set.")
    if not get_setting('app.output_dir') and get_setting('app.move_file_after_processing'):
         logger.warning("Configuration check: 'app.output_dir' is not set but moving files is enabled.")
    # Add more checks as needed (e.g., for timeseries_dir if enabled)

    logger.info("Configuration loading process complete.")


def get_config() -> Dict[str, Any]:
    """Returns the loaded configuration dictionary."""
    if _config is None:
        # Attempt to load if accessed before explicitly loaded (e.g., in tests or direct script runs)
        logger.warning("Configuration accessed before being explicitly loaded. Loading now.")
        load_config()
    # Check again in case loading failed critically
    if _config is None:
        logger.error("Configuration is unavailable.") # Should not happen if load_config raises error on critical failure
        return {} # Return empty dict to avoid None downstream
    return _config

def get_setting(key_path: str, default: Any = None) -> Any:
    """
    Accesses a configuration setting using dot notation. Handles nested keys.
    Example: get_setting('app.input_dir')
             get_setting('styles.defaults.watermark_zoom')
             get_setting('ftp.servers.0.host') # Access list elements by index
    """
    cfg = get_config()
    keys = key_path.split('.')
    value = cfg
    try:
        for key in keys:
            if isinstance(value, list):
                # Try converting key to integer for list index access
                try:
                    idx = int(key)
                    value = value[idx]
                except (ValueError, IndexError):
                    # logger.debug(f"Config path invalid at list index '{key}' in '{key_path}'.")
                    return default
            elif isinstance(value, dict):
                value = value[key]
            else:
                # Intermediate key is not a dict or list, path is invalid
                # logger.debug(f"Config path invalid at '{key}' in '{key_path}'. Parent is not a dict or list.")
                return default
        # Return value if found, otherwise default (handles case where value is None)
        return value if value is not None else default
    except (KeyError, IndexError, TypeError):
        # Handle missing keys, invalid indices, or trying to index non-subscriptable types
        # logger.debug(f"Config key '{key_path}' not found or path invalid. Returning default: {default}")
        return default

# <<< NEW HELPER FUNCTION >>>
def get_point_config(point_name: str) -> Optional[Dict[str, Any]]:
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
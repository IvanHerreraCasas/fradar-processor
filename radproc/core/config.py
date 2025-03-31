# core/config.py
import yaml
import os
from typing import Any, Dict, Optional

_config: Optional[Dict[str, Any]] = None
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))

def _load_yaml(filepath: str) -> Dict[str, Any]:
    """Loads a single YAML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f: # Added encoding
            data = yaml.safe_load(f)
            return data if data is not None else {} # Handle empty files
    except FileNotFoundError:
        print(f"Warning: Configuration file not found: {filepath}")
        return {}
    except yaml.YAMLError as e: # More specific YAML error
        print(f"Error parsing configuration file {filepath}: {e}")
        return {}
    except Exception as e:
        print(f"Error loading configuration file {filepath}: {e}")
        return {}

def load_config(config_dir: str = CONFIG_DIR) -> None:
    """Loads all YAML configuration files from the specified directory."""
    global _config
    if _config is not None:
        return # Already loaded

    print(f"Loading configuration from: {config_dir}")
    _config = {}
    config_files = {
        'app': 'app_config.yaml',      # Corrected extension if needed
        'radar': 'radar_params.yaml',
        'styles': 'plot_styles.yaml',
        'analysis': 'analysis_params.yaml',
        # +++ ADD FTP CONFIG +++
        'ftp': 'ftp_config.yaml'
        # ++++++++++++++++++++++
    }

    all_loaded_successfully = True
    for key, filename in config_files.items():
        filepath = os.path.join(config_dir, filename)
        loaded_data = _load_yaml(filepath)
        if not loaded_data and key in ['app', 'radar']: # Example: require app/radar config
             print(f"ERROR: Required configuration file '{filename}' is empty or could not be loaded.")
             all_loaded_successfully = False
        _config[key] = loaded_data

    if not all_loaded_successfully:
        # Decide how to handle missing required config - maybe raise error?
        # For now, _config might be partially populated.
        print("Warning: One or more required configuration files failed to load.")
        # raise ValueError("Failed to load required configuration files.")

    # Basic validation (examples) - Check using the loaded _config
    if not _config.get('app', {}).get('input_dir'):
         print("Warning: 'input_dir' not set in app_config.yaml")
    if not _config.get('radar', {}).get('latitude') is None: # Check if key exists and isn't None
         pass # Or check for default value
    else:
         print("Warning: Radar 'latitude' not set in radar_params.yaml")

    print("Configuration loading complete.")


def get_config() -> Dict[str, Any]:
    """Returns the loaded configuration dictionary."""
    if _config is None:
        load_config()
    # Ensure _config isn't None if loading failed critically?
    # load_config might raise error now, or handle partial load
    return _config if _config is not None else {}

def get_setting(key_path: str, default: Any = None) -> Any:
    """
    Accesses a configuration setting using dot notation.
    Example: get_setting('app.input_dir')
             get_setting('styles.defaults.watermark_zoom')
             get_setting('ftp.scan_upload_mode')
    """
    cfg = get_config()
    keys = key_path.split('.')
    value = cfg
    try:
        for key in keys:
            if not isinstance(value, dict): # Check if intermediate value is a dict
                 # print(f"Warning: Config path invalid at '{key}' in '{key_path}'. Not a dictionary.")
                 return default
            value = value[key] # Access using dict key
        # Ensure we don't return None if a non-None default was provided
        return value if value is not None else default
    except KeyError:
        # Key not found at some level
        # print(f"Warning: Config key '{key_path}' not found. Returning default: {default}")
        return default
    except TypeError:
        # Intermediate value was not subscriptable (e.g., None)
        # print(f"Warning: Config path invalid in '{key_path}'. TypeError encountered.")
        return default
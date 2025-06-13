# radproc/core/corrections/__init__.py
import pyart
import logging
from radproc.core.config import get_setting

# Import methods from other files within this package
from .filtering import filter_noise_gatefilter
from .despeckle import despeckle_field_pyart, despeckle_by_azimuth_width
from .attenuation import correct_attenuation_kdp
from .qpe import estimate_rate_composite
from .utils import sanitize_field

logger = logging.getLogger(__name__)


# --- Dispatcher Functions ---
def _dispatch_noise_method(radar, config):
    method = config.get('method', 'none')
    params = config.get('params', {})
    if method == 'gatefilter':
        filter_noise_gatefilter(radar, **params)
    elif method == 'none':
        logger.info("No noise filtering specified.")
    else:
        logger.warning(f"Unknown noise filter method '{method}'.")

def _dispatch_despeckle_method(radar, config):
    method = config.get('method', 'none')
    params = config.get('params', {})
    if method == 'pyart_despeckle':
        despeckle_field_pyart(radar, **params)
    elif method == 'azimuth_width': # Add this new case
        despeckle_by_azimuth_width(radar, **params)
    elif method == 'none':
        logger.info("No despeckle step specified.")
    else:
        logger.warning(f"Unknown despeckle method '{method}'.")


def _dispatch_attenuation_method(radar, config):
    method = config.get('method', 'none')
    params = config.get('params', {})
    if method == 'kdp_based':
        correct_attenuation_kdp(radar, **params)
    elif method == 'none':
        logger.info("No attenuation correction specified.")
    else:
        logger.warning(f"Unknown attenuation method '{method}'.")


def _dispatch_rate_method(radar, config):
    method = config.get('method', 'none')
    params = config.get('params', {})
    if method == 'composite_kdp_z':
        estimate_rate_composite(radar, **params)
    elif method == 'none':
        logger.info("No rate estimation specified.")
    else:
        logger.warning(f"Unknown rate estimation method '{method}'.")


# --- Main Public Orchestrator Function ---
def apply_corrections(radar: pyart.core.Radar, version: str) -> pyart.core.Radar:
    """
    Applies a structured chain of corrections to a Py-ART Radar object
    based on a versioned configuration.
    This is the main entry point for the 'corrections' package.
    """
    logger.info(f"--- Starting scientific corrections for version '{version}' ---")
    config = get_setting(f'corrections.{version}')
    if not config:
        logger.error(f"Configuration for version '{version}' not found. Aborting.")
        return radar

    temp_fields_to_remove = []

    _dispatch_noise_method(radar, config.get('noise_filter', {}))

    _dispatch_despeckle_method(radar, config.get('despeckle', {}))

    # Sanitize all necessary fields once at the beginning ---
    if 'sanitize' in config and isinstance(config['sanitize'], list):
        for s_config in config['sanitize']:
            field_name = s_config.get('field')
            if not field_name: continue

            s_params = s_config.get('params', {}).copy()

            # Use the 'fill_data' parameter from the config, defaulting to True
            fill_data_flag = s_params.pop('fill_data', True)

            sanitized_field_dict = sanitize_field(
                radar, field_name, fill_data=fill_data_flag, **s_params
            )

            if sanitized_field_dict:
                temp_name = f"{field_name}_sanitized"
                radar.add_field(temp_name, sanitized_field_dict, replace_existing=True)
                temp_fields_to_remove.append(temp_name)
                logger.info(f"Created temporary sanitized field: '{temp_name}'")

    try:
        _dispatch_attenuation_method(radar, config.get('attenuation', {}))
        _dispatch_rate_method(radar, config.get('rate_estimation', {}))
    finally:
        # --- Final Step: Clean up all temporary fields ---
        for field in temp_fields_to_remove:
            if field in radar.fields:
                radar.fields.pop(field)
        logger.info(f"Cleaned up {len(temp_fields_to_remove)} temporary fields.")

    logger.info("--- Finished scientific corrections ---")
    return radar
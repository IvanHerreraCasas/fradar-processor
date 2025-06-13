# radproc/core/retrievals/qpe.py
import pyart
import numpy as np
import logging

logger = logging.getLogger(__name__)


def estimate_rate_composite(radar, **params):
    """
    Estimates rain rate using a composite method based on reflectivity (Z-R)
    and specific differential phase (KDP-R).

    This method calculates both R(Z) and R(KDP) and combines them based
    on configurable thresholds for reflectivity and KDP. The final composite
    field is added to the radar object as 'rain_rate'.

    Args:
        radar: The Py-ART Radar object to process.
        **params: A dictionary of parameters from the configuration, including:
            refl_field (str): Name of the reflectivity field (corrected).
            kdp_field (str): Name of the KDP field (sanitized).
            z_a, z_b (float): Parameters for the Z-R relationship R=a*Z^b.
            kdp_a, kdp_b (float): Parameters for the KDP-R relationship R=a*KDP^b.
            z_threshold (float, optional): Reflectivity threshold (dBZ) above
                                           which KDP is considered. Defaults to 35.0.
            kdp_threshold (float, optional): KDP threshold (deg/km) above
                                             which KDP is considered. Defaults to 0.3.
    """
    logger.info("Applying composite KDP-Z rain rate estimation...")
    input_refl_field = params['input_refl_field']
    input_kdp_field = params['input_kdp_field']

    if not input_refl_field in radar.fields or not input_kdp_field in radar.fields:
        logger.error("Input fields for rate composite not found. Skipping.")
        return

    try:
        # Get Z-R and KDP-R relationship parameters from config
        z_a, z_b = float(params['z_a']), float(params['z_b'])
        kdp_a, kdp_b = float(params['kdp_a']), float(params['kdp_b'])

        # Get thresholds, with sensible defaults if not provided
        z_threshold = float(params.get('z_threshold', 35.0))
        kdp_threshold = float(params.get('kdp_threshold', 0.3))
    except (KeyError, ValueError) as e:
        logger.error(f"Missing or invalid parameter for rate estimation in config: {e}. Skipping.")
        return

    # 1. Calculate rain rate from KDP
    # Py-ART's function returns a full field dictionary
    r_kdp_field = pyart.retrieve.est_rain_rate_kdp(
        radar, alpha=kdp_a, beta=kdp_b, kdp_field=input_kdp_field)

    # 2. Calculate rain rate from Reflectivity
    r_z_field = pyart.retrieve.est_rain_rate_z(
        radar, alpha=z_a, beta=z_b, refl_field=input_refl_field)

    # 3. Define the condition where the KDP-based rate is more reliable
    # This condition is met where BOTH reflectivity and KDP are high.
    refl_data = radar.fields[input_refl_field]['data']
    kdp_data = radar.fields[input_kdp_field]['data']

    use_kdp_condition = np.logical_and(
        refl_data >= z_threshold,
        kdp_data >= kdp_threshold
    )

    # 4. Create the composite data array using numpy.where
    # Where the condition is True, use R(KDP); otherwise, use R(Z).
    rain_rate_data = np.where(
        use_kdp_condition,
        r_kdp_field['data'],
        r_z_field['data']
    )

    final_mask = np.ma.getmask(refl_data)

    # Where the mask is True, set the rain rate to 0.0.
    if final_mask is not np.ma.nomask:
        rain_rate_data[final_mask] = 0.0

    # 5. Add the final composite field, overwriting the original RATE field
    # Get the name of the field to overwrite from the config, defaulting to 'RATE'
    output_rate_field = params['output_rate_field']

    final_rate_field = r_z_field.copy()
    final_rate_field['data'] = rain_rate_data
    final_rate_field['long_name'] = 'Corrected composite rain rate'
    final_rate_field['standard_name'] = 'rainfall_rate'
    final_rate_field['comment'] = (
        f'Composite of R(KDP) (a={kdp_a}, b={kdp_b}) and R(Z) (a={z_a}, b={z_b}). '
        f'This field replaces the original {output_rate_field} field.'
    )

    # Use replace_existing=True to ensure the original field is overwritten
    radar.add_field(output_rate_field, final_rate_field, replace_existing=True)
    logger.info(f"Successfully updated field '{output_rate_field}' with composite rain rate.")
# radproc/core/corrections/qpe.py
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


    # 5. Add the final composite field, overwriting the original RATE field
    # Get the name of the field to overwrite from the config, defaulting to 'RATE'
    output_rate_field = params['output_rate_field']

    final_rate_field = r_z_field.copy()
    final_rate_field['data'] = np.ma.masked_where(final_mask, rain_rate_data)
    final_rate_field['long_name'] = 'Corrected composite rain rate'
    final_rate_field['standard_name'] = 'rainfall_rate'
    final_rate_field['comment'] = (
        f'Composite of R(KDP) (a={kdp_a}, b={kdp_b}) and R(Z) (a={z_a}, b={z_b}). '
        f'This field replaces the original {output_rate_field} field.'
    )

    # Use replace_existing=True to ensure the original field is overwritten
    radar.add_field(output_rate_field, final_rate_field, replace_existing=True)
    logger.info(f"Successfully updated field '{output_rate_field}' with composite rain rate.")


def estimate_rate_a_z(radar, **params):
    """
    Estimates rain rate using a specific attenuation-based method (R-A) with a
    fallback to a reflectivity-based method (R-Z).

    This method is designed to be used after attenuation correction, such as the
    Z-PHI method, has been run. It uses the calculated specific attenuation (A)
    to estimate rain rate where A is available and considered reliable. In all
    other regions, it falls back to using the attenuation-corrected
    reflectivity (Z).

    Args:
        radar (pyart.core.Radar): The radar object to process.
        **params (dict): A dictionary of parameters, including:
            a_field (str): Name of the specific attenuation field.
            refl_field (str): Name of the (corrected) reflectivity field.
            output_rate_field (str): Name for the output rain rate field.
            a_a_coef, a_b_coef (float): Coeffs for R=c*A^d relationship.
            z_a_coef, z_b_coef (float): Coeffs for R=a*Z^b relationship.
    """
    logger.info("Applying attenuation-based (R-A) rain rate with R-Z fallback...")

    # --- 1. Unpack Parameters ---
    try:
        a_field = params['a_field']
        refl_field = params['refl_field']
        output_rate_field = params['output_rate_field']
        a_a, a_b = float(params['a_a_coef']), float(params['a_b_coef'])
        z_a, z_b = float(params['z_a_coef']), float(params['z_b_coef'])
    except KeyError as e:
        logger.error(f"Missing required parameter for R(A,Z) estimation: {e}. Skipping.")
        return

    if a_field not in radar.fields or refl_field not in radar.fields:
        logger.error(f"Input fields '{a_field}' or '{refl_field}' not found. Skipping.")
        return

    # --- 2. Calculate R(Z) Everywhere ---
    # This will serve as our base and fallback rate.
    # The est_rain_rate_z function correctly handles the conversion from dBZ.
    r_z_dict = pyart.retrieve.est_rain_rate_z(
        radar, alpha=z_a, beta=z_b, refl_field=refl_field)

    # --- 3. Calculate R(A) Where A is Valid ---
    # R = c * A^d. This calculation is performed only on the valid (unmasked)
    # portions of the specific attenuation field. The result is a MaskedArray.
    r_a_dict = pyart.retrieve.est_rain_rate_a(
        radar, alpha=a_a, beta=a_b, a_field=a_field
    )


    # --- 4. Create the Composite Field ---
    # The logic is simple: where rate_from_a has a valid value, use it.
    # Otherwise, use the value from rate_from_z.
    # np.ma.where is perfect for this, as it respects the masks.
    final_rate_data = np.ma.where(
        ~np.ma.getmask(r_a_dict['data']),  # Condition: where A is NOT masked
        r_a_dict['data'],                 # Value if True
        r_z_dict['data']                  # Value if False (fallback)
    )

    # --- 6. Add the Final Field to the Radar Object ---
    final_rate_field = r_z_dict.copy() # Reuse metadata from the R(Z) calculation
    final_rate_field['data'] = final_rate_data
    final_rate_field['long_name'] = 'Rain rate from R(A) with R(Z) fallback'
    final_rate_field['standard_name'] = 'rainfall_rate'
    final_rate_field['comment'] = (
        f'Composite of R(A) (c={a_a}, d={a_b}) where A is valid, with '
        f'fallback to R(Z) (a={z_a}, b={z_b}).'
    )

    radar.add_field(output_rate_field, final_rate_field, replace_existing=True)
    logger.info(f"Successfully created field '{output_rate_field}' with R(A,Z) method.")

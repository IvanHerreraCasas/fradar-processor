# radproc/core/retrievals/attenuation.py
import pyart
import numpy as np
import logging
from scipy.integrate import cumulative_trapezoid

logger = logging.getLogger(__name__)


def correct_attenuation_kdp(radar, **params):
    """Applies KDP-based attenuation correction using pre-sanitized fields."""
    logger.info("Applying KDP-based attenuation correction...")
    input_refl_field = params['input_refl_field']
    input_kdp_field = params['input_kdp_field']
    output_refl_field = params['output_refl_field']
    a_coef, b_coef = float(params['a_coef']), float(params['b_coef'])


    if input_refl_field not in radar.fields or input_kdp_field not in radar.fields:
        logger.error("Input fields for KDP attenuation not found. Skipping.")
        return

    refl_data = radar.fields[input_refl_field]['data']
    kdp_data = radar.fields[input_kdp_field]['data']

    # Store the mask from the original reflectivity data. This tells us where
    # there was no signal to begin with.
    original_mask = np.ma.getmask(refl_data)



    spec_at_data = a_coef * (kdp_data ** b_coef)
    dr_m = radar.range['data'][1] - radar.range['data'][0]
    pia_data = 2 * cumulative_trapezoid(spec_at_data, dx=dr_m / 1000.0, axis=-1, initial=0)

    # Perform the addition on the unmasked data, filling masked values with 0
    # so that 0 + PIA = PIA in the no-signal regions.
    corrected_data_unmasked = refl_data.filled(0.0) + pia_data

    # Create a new masked array from the result, RE-APPLYING the original mask.
    # This ensures that the original no-signal areas remain masked.
    final_corrected_data = np.ma.array(corrected_data_unmasked, mask=original_mask)

    # Add the result as a new field with the explicit output name
    corrected_refl_dict = radar.fields[input_refl_field].copy()
    corrected_refl_dict['data'] = final_corrected_data
    corrected_refl_dict['comment'] = 'KDP-based attenuation correction applied.'
    radar.add_field(output_refl_field, corrected_refl_dict, replace_existing=True)

    radar.add_field_like(input_kdp_field, 'specific_attenuation', spec_at_data, replace_existing=True)
    radar.add_field_like(input_refl_field, 'path_integrated_attenuation', pia_data, replace_existing=True)
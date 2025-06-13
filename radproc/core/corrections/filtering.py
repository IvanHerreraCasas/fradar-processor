# radproc/core/corrections/filtering.py

import pyart
import numpy as np
import logging

from pyart.util import rolling_window

logger = logging.getLogger(__name__)

# copied from pyart to remove print
def texture(radar, var):
    """Determine a texture field using an 11pt stdev
    texarray=texture(pyradarobj, field)."""
    fld = radar.fields[var]["data"]
    tex = np.ma.zeros(fld.shape)
    for timestep in range(tex.shape[0]):
        ray = np.ma.std(rolling_window(fld[timestep, :], 11), 1)
        tex[timestep, 5:-5] = ray
        tex[timestep, 0:4] = np.ones(4) * ray[0]
        tex[timestep, -5:] = np.ones(5) * ray[-1]
    return tex


def filter_noise_gatefilter(radar, **params):
    """
    Applies a series of filters using Py-ART's GateFilter object.

    This function reads filter conditions from the parameters and applies them
    sequentially. The final combined filter is then used to mask the data
    in a specified list of fields.

    Args:
        radar: The Py-ART Radar object to process.
        **params: A dictionary of parameters from the configuration.
    """
    logger.info("Applying GateFilter noise and clutter filtering...")

    # 1. Initialize the GateFilter with the radar object.
    #    By default, it includes gates that are finite and not masked.
    gatefilter = pyart.correct.GateFilter(radar)

    # 2. Apply filters based on parameters found in the config.

    # --- RhoHV Filter ---
    min_rhohv = params.get('min_rhohv')
    if min_rhohv is not None:
        rhohv_field = params.get('rhohv_field', 'RHOHV')
        if rhohv_field in radar.fields:
            logger.debug(f"Applying RhoHV filter (min_rhohv = {min_rhohv})...")
            gatefilter.exclude_below(rhohv_field, min_rhohv)
        else:
            logger.warning(f"Could not apply RhoHV filter: field '{rhohv_field}' not found.")

        # --- Corrected Texture Filter ---
        max_texture = params.get('max_texture')
        if max_texture is not None:
            texture_source_field = params.get('texture_field', 'DBZH')
            if texture_source_field in radar.fields:
                logger.debug(f"Applying texture filter (max_texture = {max_texture})...")

                # 1. Calculate the texture. The function returns a raw numpy array.
                texture_data = texture(radar, texture_source_field)

                # 2. Create a proper field dictionary for the texture data.
                #    We can copy metadata from the source field as a template.
                texture_field_dict = radar.fields[texture_source_field].copy()
                texture_field_dict['data'] = texture_data
                texture_field_dict['long_name'] = f'Texture of {texture_source_field}'
                texture_field_dict['units'] = 'unitless'  # Texture is a standard deviation

                # 3. Add the correctly formatted texture field to the radar object temporarily.
                temp_texture_field_name = f"{texture_source_field}_texture"
                radar.add_field(temp_texture_field_name, texture_field_dict, replace_existing=True)

                # 4. Use the new texture field to update the gate filter.
                gatefilter.exclude_above(temp_texture_field_name, max_texture)

                # 5. Clean up the temporary texture field.
                radar.fields.pop(temp_texture_field_name)
            else:
                logger.warning(f"Could not apply texture filter: field '{texture_source_field}' not found.")

    # 3. Apply the combined filter mask to the specified fields.
    fields_to_mask = params.get('apply_to_fields', [])
    if not fields_to_mask:
        logger.warning("No fields specified in 'apply_to_fields' for noise filter.")
        return

    logger.info(f"Applying final filter mask to fields: {fields_to_mask}")
    for field_name in fields_to_mask:
        if field_name in radar.fields:
            # Get the existing mask or create a new one if none exists
            current_mask = np.ma.getmask(radar.fields[field_name]['data'])
            # Combine the existing mask with the new filter mask
            combined_mask = np.logical_or(current_mask, gatefilter.gate_excluded)

            # Apply the new, combined mask
            radar.fields[field_name]['data'] = np.ma.masked_where(
                combined_mask, radar.fields[field_name]['data']
            )
        else:
            logger.warning(f"Cannot apply mask: field '{field_name}' not found.")
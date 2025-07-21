# In radproc/core/corrections/despeckle.py
import pyart
import numpy as np
import logging

from scipy.ndimage import find_objects, label

logger = logging.getLogger(__name__)


def despeckle_by_azimuth_width(radar, **params):
    """
    Removes regions of echo that are narrower than a specified number
    of azimuths, based on a reference field. The resulting mask is then
    applied to a list of target fields.

    This is useful for removing long, thin radial "spikes" of noise.

    Args:
        radar: The Py-ART Radar object to process.
        **params: A dictionary of parameters from the configuration.
    """
    # --- 1. Get Parameters ---
    reference_field_name = params.get('reference_field')
    fields_to_filter = params.get('apply_to_fields', [])
    min_width = params.get('min_azimuth_width')
    max_range_km = params.get('max_range_km')

    if not all([reference_field_name, fields_to_filter, min_width]):
        logger.error("Azimuth despeckle config missing 'reference_field', "
                     "'apply_to_fields', or 'min_azimuth_width'.")
        return

    if reference_field_name not in radar.fields:
        logger.warning(f"Reference field '{reference_field_name}' not found for "
                       "azimuth despeckling. Skipping.")
        return

    logger.info(f"Identifying speckles on '{reference_field_name}' with min width {min_width}...")

    # --- 2. Identify Speckles on the Reference Field ---
    ref_field_data = radar.fields[reference_field_name]['data']
    is_valid_echo = ~np.ma.getmaskarray(ref_field_data)
    labeled_array, num_features = label(is_valid_echo)

    # Find the azimuthal width of each labeled region
    labels_to_remove = []
    max_range_m = float(max_range_km) * 1000 if max_range_km is not None else None

    for i in range(1, num_features + 1):
        region_coords = np.where(labeled_array == i)

        if max_range_m is not None:
            min_gate_index = np.min(region_coords[1])
            if radar.range['data'][min_gate_index] > max_range_m:
                continue

        rays_with_label = np.any(labeled_array == i, axis=1)
        azimuth_width = np.sum(rays_with_label)

        if azimuth_width < min_width:
            labels_to_remove.append(i)

    if not labels_to_remove:
        logger.debug("No regions found smaller than the minimum azimuth width.")
        return

    logger.info(f"Identified {len(labels_to_remove)} speckle regions to remove.")

    # --- 3. Create a single speckle mask from the reference field ---
    speckle_mask = np.isin(labeled_array, labels_to_remove)

    # --- 4. Apply the mask to all specified fields ---
    for field_name in fields_to_filter:
        if field_name not in radar.fields:
            logger.warning(f"Field '{field_name}' not found in radar. Skipping its despeckle.")
            continue

        logger.info(f"Applying azimuth speckle mask to '{field_name}'...")
        original_field = radar.fields[field_name]

        # Combine the new speckle mask with the field's existing mask
        new_mask = np.logical_or(np.ma.getmaskarray(original_field['data']), speckle_mask)
        new_masked_data = np.ma.masked_where(new_mask, original_field['data'])

        # Create a new field dictionary and overwrite the original field
        despeckled_field = original_field.copy()
        despeckled_field['data'] = new_masked_data

        comment = despeckled_field.get('comment', '')
        despeckled_field['comment'] = (comment +
                                       f' (azimuth-despeckled with min_width={min_width} based on {reference_field_name})').strip()

        radar.add_field(field_name, despeckled_field, replace_existing=True)


def despeckle_field_pyart(radar, **params):
    """
    Removes small, isolated regions of echo (speckles) from a field.

    This acts as a wrapper around the pyart.correct.despeckle_field function,
    correctly handling its GateFilter return object.

    Args:
        radar: The Py-ART Radar object to process.
        **params: A dictionary of parameters from the configuration.
    """
    field_name = params.get('field_to_despeckle')
    size = params.get('size')

    if not field_name or not size:
        logger.error("Despeckle config missing 'field_to_despeckle' or 'size'. Skipping.")
        return

    if field_name not in radar.fields:
        logger.warning(f"Field '{field_name}' not found for despeckling. Skipping.")
        return

    logger.info(f"Applying despeckle filter to '{field_name}' with min size {size}...")

    # 1. Call despeckle_field to get the GateFilter object
    gatefilter = pyart.correct.despeckle_field(radar, field_name, size=int(size))

    # 2. Get the original field's data and mask
    field_dict = radar.fields[field_name]
    original_data = field_dict['data']

    # 3. Apply the new speckle mask to the data
    # The new mask is True where the original was masked OR where speckle was found.
    new_masked_data = np.ma.masked_where(
        gatefilter.gate_excluded,
        original_data
    )

    # 4. Create a new field dictionary with the despeckled data
    despeckled_field = field_dict.copy()
    despeckled_field['data'] = new_masked_data
    if 'comment' not in despeckled_field:
        despeckled_field['comment'] = ''
    despeckled_field['comment'] += f' (despeckled with size={size})'

    # 5. Overwrite the original field with the new, cleaner version
    radar.add_field(field_name, despeckled_field, replace_existing=True)
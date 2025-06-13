# In radproc/core/corrections/despeckle.py
import pyart
import numpy as np
import logging

from scipy.ndimage import find_objects, label

logger = logging.getLogger(__name__)


def despeckle_by_azimuth_width(radar, **params):
    """
    Removes regions of echo that are narrower than a specified number
    of azimuths.

    This is useful for removing long, thin radial "spikes" of noise.

    Args:
        radar: The Py-ART Radar object to process.
        **params: A dictionary of parameters from the configuration, including:
            field_to_filter (str): The name of the field to clean.
            min_azimuth_width (int): The minimum number of contiguous rays a
                                     region must occupy to be kept.
    """
    field_name = params.get('field_to_filter')
    min_width = params.get('min_azimuth_width')
    max_range_km = params.get('max_range_km')

    if not field_name or not min_width:
        # max_range_km is optional, so we don't check for it here
        logger.error("Azimuth despeckle config missing 'field_to_filter' or 'min_azimuth_width'.")
        return

    if field_name not in radar.fields:
        logger.warning(f"Field '{field_name}' not found for azimuth despeckling. Skipping.")
        return

    logger.info(f"Applying azimuth width filter to '{field_name}' with min width {min_width}...")
    if max_range_km is not None:
        logger.info(f"...only for regions within {max_range_km} km.")

    # Convert max range to meters for comparison
    max_range_m = float(max_range_km) * 1000 if max_range_km is not None else None

    # 1. Get a boolean array of valid data
    field_data = radar.fields[field_name]['data']
    is_valid_echo = ~np.ma.getmaskarray(field_data)

    # 2. Label all connected regions of valid echo
    labeled_array, num_features = label(is_valid_echo)

    # 3. Find the azimuthal width of each labeled region
    labels_to_remove = []
    for i in range(1, num_features + 1):
        # Find all gates belonging to this region
        region_coords = np.where(labeled_array == i)

        # If max_range_km is set, check the range of the region's closest gate
        if max_range_m is not None:
            min_gate_index = np.min(region_coords[1])  # 1 is the gate/range axis
            range_of_closest_gate = radar.range['data'][min_gate_index]

            # If the entire region is outside the radius, skip it and keep it.
            if range_of_closest_gate > max_range_m:
                continue

        # Find all rows (rays) where this label appears
        rays_with_label = np.any(labeled_array == i, axis=1)
        # The width is the number of unique rays it occupies
        azimuth_width = np.sum(rays_with_label)

        if azimuth_width < min_width:
            labels_to_remove.append(i)

    if not labels_to_remove:
        logger.debug("No regions found smaller than the minimum azimuth width.")
        return

    logger.info(f"Identified {len(labels_to_remove)} regions to remove based on azimuth width.")

    # 4. Create a mask for all regions that need to be removed
    speckle_mask = np.isin(labeled_array, labels_to_remove)

    # 5. Apply this new mask to the original field's data
    new_masked_data = np.ma.masked_where(speckle_mask, field_data)

    # 6. Create a new field dictionary and overwrite the original field
    despeckled_field = radar.fields[field_name].copy()
    despeckled_field['data'] = new_masked_data
    if 'comment' not in despeckled_field:
        despeckled_field['comment'] = ''
    despeckled_field['comment'] += f' (azimuth-despeckled with min_width={min_width})'

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
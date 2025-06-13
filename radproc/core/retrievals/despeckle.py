# In radproc/core/retrievals/despeckle.py
import pyart
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
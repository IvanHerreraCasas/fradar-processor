# radproc/core/retrievals/utils.py
import numpy as np
import logging

logger = logging.getLogger(__name__)


# In radproc/core/retrievals/utils.py

def sanitize_field(radar, field_name: str, fill_value: float = 0.0, lower_bound: float = None,
                   fill_data: bool = True) -> dict:
    """
    Creates a sanitized copy of a radar field.

    Args:
        ...
        fill_data (bool): If True, fills masked values with `fill_value`.
                          If False, returns the numpy MaskedArray.
    """
    if field_name not in radar.fields:
        logger.error(f"Field '{field_name}' not found for sanitization.")
        return None

    sanitized_field = radar.fields[field_name].copy()
    data = np.ma.masked_invalid(sanitized_field['data'], copy=True)

    if lower_bound is not None:
        data.mask = np.logical_or(data.mask, data.data < lower_bound)

    # Conditionally fill the data based on the new flag
    if fill_data:
        sanitized_field['data'] = data.filled(fill_value).astype(np.float32)
    else:
        sanitized_field['data'] = data  # Keep it as a MaskedArray

    sanitized_field['comment'] = f'Sanitized version of {field_name}.'

    return sanitized_field
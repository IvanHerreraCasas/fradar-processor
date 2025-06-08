# radproc/core/preprocessing.py

import logging
from typing import Dict, Any

import pyart
import xarray as xr
import numpy as np

from .config import get_setting

logger = logging.getLogger(__name__)


def xarray_to_pyart(dataset: xr.Dataset) -> pyart.Radar:
    """
    Converts a georeferenced xarray.Dataset from this project into a Py-ART Radar object.

    This function manually maps the coordinates and data variables from the
    xarray structure to the dictionary-based structure that Py-ART expects.

    Args:
        dataset: An xarray.Dataset produced by the `radproc.core.data.read_scan`
                 and `radproc.core.utils.geo.georeference_dataset` functions.
                 It must contain 'latitude', 'longitude', 'altitude', 'time',
                 'range', and 'azimuth' coordinates.

    Returns:
        A pyart.Radar object ready for use in Py-ART's correction and retrieval functions.
    """
    # Initialize the Radar object by providing the core sweep and instrument parameters.
    # Py-ART expects these as dictionaries.
    time = {'data': np.atleast_1d(dataset['time'].values), 'units': 'seconds since 1970-01-01T00:00:00Z'}
    _range = {'data': dataset['range'].values, 'meters': 'm', 'spacing_is_constant': 'true'}
    latitude = {'data': np.array([dataset['latitude'].item()], dtype='float64'), 'units': 'degrees_north'}
    longitude = {'data': np.array([dataset['longitude'].item()], dtype='float64'), 'units': 'degrees_east'}
    altitude = {'data': np.array([dataset.get('altitude', 0).item()], dtype='float64'), 'units': 'm'} # Assuming altitude if present

    # Define sweep metadata
    sweep_number = {'data': np.array([0], dtype='int32')}
    sweep_mode = {'data': np.array(['ppi'], dtype='S3')}
    fixed_angle = {'data': np.array([dataset['elevation'].item()], dtype='float32'), 'units': 'degrees'}
    sweep_start_ray_index = {'data': np.array([0], dtype='int32')}
    sweep_end_ray_index = {'data': np.array([len(dataset['azimuth']) - 1], dtype='int32')}

    # Initialize fields dictionary
    fields = {}
    for var_name in dataset.data_vars:
        if dataset[var_name].ndim == 2 and 'azimuth' in dataset[var_name].dims and 'range' in dataset[var_name].dims:
            fields[var_name] = {'data': dataset[var_name].values}
            # Add standard metadata if available (optional but good practice)
            if 'units' in dataset[var_name].attrs:
                fields[var_name]['units'] = dataset[var_name].attrs['units']
            if 'long_name' in dataset[var_name].attrs:
                fields[var_name]['long_name'] = dataset[var_name].attrs['long_name']

    # Create the Radar object
    radar = pyart.core.Radar(
        time,
        _range,
        fields,
        {}, # metadata
        'ppi', # scan_type
        latitude,
        longitude,
        altitude,
        sweep_number,
        sweep_mode,
        fixed_angle,
        sweep_start_ray_index,
        sweep_end_ray_index,
        azimuth={'data': dataset['azimuth'].values, 'units': 'degrees'},
        elevation={'data': np.full_like(dataset['azimuth'].values, dataset['elevation'].item()), 'units': 'degrees'},
    )
    return radar


def apply_corrections(radar: pyart.Radar, version: str) -> pyart.Radar:
    """
    Applies a suite of corrections to a Py-ART Radar object.

    This function acts as a pipeline for the preprocessing steps. The specific
    algorithms and their parameters are controlled by the configuration associated
    with the provided version string.

    Args:
        radar: A pyart.Radar object to be processed.
        version: A string key (e.g., 'v1_0') corresponding to a set of
                 parameters in the 'corrections_config.yaml' file.

    Returns:
        The corrected pyart.Radar object.
    """
    logger.info(f"Applying radar corrections using parameter version: '{version}'")

    # --- Step 1: Fetch Correction Parameters ---
    # We use get_setting to retrieve the entire dictionary for the specified version.
    params = get_setting(f'corrections.{version}')
    if not params:
        logger.error(f"Correction parameters for version '{version}' not found in configuration. Aborting corrections.")
        return radar

    # --- Step 2: Apply Noise Filtering ---
    # In this initial implementation, we are just logging the intent.
    # The actual Py-ART function calls will be added here in the future.
    if 'noise_filter' in params:
        logger.info(f"Applying noise filter with params: {params['noise_filter']}")
        # EXAMPLE future code:
        # gatefilter = pyart.filters.GateFilter(radar)
        # gatefilter.exclude_below('reflectivity', 0)
        # radar = pyart.correct.despeckle(radar, gatefilter=gatefilter, size=params['noise_filter']['despeckle_size'])

    # --- Step 3: Apply Attenuation Correction ---
    if 'attenuation' in params:
        logger.info(f"Applying attenuation correction with params: {params['attenuation']}")
        # EXAMPLE future code:
        # spec_at, pia = pyart.correct.calculate_attenuation(
        #     radar, 0, refl_field=params['attenuation']['refl_field'],
        #     a_coef=params['attenuation']['a_coef'], b_coef=params['attenuation']['b_coef'])
        # radar.add_field('specific_attenuation', spec_at)
        # radar.add_field('path_integrated_attenuation', pia)

    # --- Step 4: Generate New Rate Estimates ---
    if 'rate_estimation' in params:
        logger.info(f"Generating new rainfall estimates with params: {params['rate_estimation']}")
        # EXAMPLE future code:
        # rate_field = pyart.retrieve.est_rain_rate_z(radar, a=params['rate_estimation']['z_a'], b=params['rate_estimation']['z_b'])
        # radar.add_field('rain_rate_corrected', rate_field)

    logger.info("Finished applying corrections pipeline.")
    return radar
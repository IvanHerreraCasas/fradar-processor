# radproc/core/corrections/attenuation.py
import pyart
import numpy as np
import logging
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter, medfilt
from sklearn.isotonic import IsotonicRegression

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


def _find_delta_phidp(reflectivity_ray, phidp_ray, refl_threshold_dbz=25, min_precip_gates=10, median_window=3):
    """Calculates the differential phase shift ($\Delta \Phi_{DP}$) across a
    precipitation region identified in a ray.

    The function first identifies all radar gates with reflectivity above a given
    threshold. It then calculates the difference between the median differential
    phase ($\Phi_{DP}$) at the end of the precipitation path and the median $\Phi_{DP}$
    at the start of the path. This averaging over a small window of gates
    provides a more stable result.

    Args:
        reflectivity_ray (np.ndarray): 1D array of reflectivity data for a ray.
        phidp_ray (np.ndarray): 1D array of differential phase data for the ray.
        refl_threshold_dbz (float): The dBZ value above which a gate is
                                    considered precipitation.
        min_precip_gates (int): The minimum number of total gates required to be
                                considered a valid precipitation segment.
        median_window (int, optional): The number of gates at the start and end
                                     of the precipitation path to average $\Phi_{DP}$
                                     over. Defaults to 3.

    Returns:
        float or None: The calculated delta $\Phi_{DP}$ value ($\Phi_{DP_{end}} - \Phi_{DP_{start}}$).
                       Returns None if a valid precipitation segment cannot be
                       found based on the criteria.
    """
    # Find all gate indices where reflectivity is above the specified threshold.
    above_threshold = np.where(reflectivity_ray > refl_threshold_dbz)[0]

    # If there are not enough gates above the threshold, it's not a valid precip segment.
    if above_threshold.size < min_precip_gates:
        return None

    # From the precipitation gates, select only those with valid (unmasked) PhiDP values
    # to avoid errors in calculation.
    phidp_in_precip = phidp_ray[above_threshold]
    valid_phidp_mask = ~np.ma.getmaskarray(phidp_in_precip)
    valid_precip_indices = above_threshold[valid_phidp_mask]

    # Ensure there are enough valid gates remaining for the start and end windows.
    if valid_precip_indices.size < min_precip_gates or valid_precip_indices.size < median_window * 2:
        return None

    # Define the windows at the start and end of the valid precipitation path.
    start_indices = valid_precip_indices[:median_window]
    end_indices = valid_precip_indices[-median_window:]

    # Calculate the median PhiDP in the start and end windows.
    # Using the median is robust against outliers.
    start_phidp = np.median(phidp_ray[start_indices])
    end_phidp = np.median(phidp_ray[end_indices])

    # Final check to ensure the median values themselves are not masked, which can
    # happen if all values in a window slice were masked.
    if np.ma.is_masked(start_phidp) or np.ma.is_masked(end_phidp):
        return None

    return end_phidp - start_phidp


def _process_phidp_ray(phidp_ray, window_len):
    """
    Unwraps, smooths, and enforces monotonicity on a single ray of
    differential phase ($\Phi_{DP}$) data.

    This function processes a raw $\Phi_{DP}$ ray to make it suitable for attenuation
    correction algorithms. It handles masked data, unwraps the phase, applies a
    median filter for smoothing, and finally uses Isotonic Regression to ensure
    the profile is monotonically non-decreasing, which is physically expected
    in precipitation.

    Args:
        phidp_ray (np.ma.MaskedArray): A single ray of $\Phi_{DP}$ data.
        window_len (int): The kernel size for the median filter.

    Returns:
        np.ma.MaskedArray: The processed, monotonically non-decreasing $\Phi_{DP}$ ray.
    """
    # Get the original mask and valid data points for processing.
    original_mask = np.ma.getmaskarray(phidp_ray)
    valid_mask = ~original_mask
    x_valid = np.where(valid_mask)[0]
    y_valid = phidp_ray[valid_mask].data  # Use .data to get raw numbers

    # If there are too few valid data points, processing is not feasible.
    if len(x_valid) < 5:
        return phidp_ray

    # Create a full x-axis and linearly interpolate to fill masked gaps.
    # This is necessary because unwrapping and filtering require a continuous signal.
    x_full = np.arange(len(phidp_ray))
    interp_phidp = np.interp(x_full, x_valid, y_valid)

    # --- Process the clean, interpolated (non-masked) data ---

    # 1. Unwrap the phase data to remove 360-degree jumps. The `discont`
    #    parameter is set to 180 as $\Phi_{DP}$ typically ranges from 0-360.
    unwrapped_phidp = np.unwrap(interp_phidp, discont=180)

    # 2. Smooth the unwrapped data using a median filter to reduce noise and spikes.
    smoothed_phidp = medfilt(unwrapped_phidp, kernel_size=window_len)

    # 3. Enforce that the profile is non-decreasing using Isotonic Regression.
    #    This is a robust method to ensure physical consistency ($\Phi_{DP}$ should not
    #    decrease with range in rain).
    x_axis = np.arange(len(smoothed_phidp))
    iso_reg = IsotonicRegression(increasing=True, out_of_bounds="clip")
    monotonic_phidp = iso_reg.fit_transform(x_axis, smoothed_phidp)

    # 4. Re-apply the original mask to the final processed data to honor the
    #    original data quality information.
    processed_phidp_ma = np.ma.array(monotonic_phidp, mask=original_mask)

    return processed_phidp_ma


def correct_attenuation_zphi_custom(radar, **params):
    """
    Performs attenuation correction on reflectivity using a custom, robust Z-PHI
    algorithm.

    This algorithm uses the total change in differential phase ($\Delta \Phi_{DP}$) across
    a precipitation path as a constraint to correct reflectivity ($Z$) for attenuation.
    It produces corrected reflectivity, specific attenuation, path-integrated
    attenuation, and two versions of differential phase: a processed version and a
    self-consistent version reconstructed from the attenuation.

    The relationship used is $A = \alpha (\Delta \Phi_{DP})$.

    Args:
        radar (pyart.core.Radar): Radar object to be processed.
        **params (dict): Dictionary of parameters for the correction. Expected keys:
            'input_refl_field', 'input_phidp_field', 'output_refl_field',
            'alpha', 'b_coef', 'window_length', 'delta_phidp_th', etc.

    Returns:
        pyart.core.Radar: The radar object with new corrected fields.
    """
    logger.info("Starting custom Z-PHI correction for Reflectivity and dual PHIDP outputs.")

    # --- 1. Unpack Parameters ---
    refl_field = params['input_refl_field']
    phidp_field = params['input_phidp_field']
    output_refl_field = params['output_refl_field']
    alpha = params['alpha']
    b_coef = params['b_coef']  # Exponent 'b' in Z = a * A^b relation

    window_length = params.get('window_length', 11)
    delta_phidp_th = params.get('delta_phidp_th', 0)

    # Field names for new output data
    output_phidp_c_field = params.get('output_phidp_c_field', 'PHIDP_C')
    output_phidp_sc_field = params.get('output_phidp_sc_field', 'PHIDP_SC')
    output_spec_attn_field = params.get('output_spec_attn_field', 'A')
    output_pia_field = params.get('output_pia_field', 'PIA')

    # --- 2. Prepare Data Structures ---
    refl_data = radar.fields[refl_field]['data']
    phidp_data = radar.fields[phidp_field]['data']

    # Create masked arrays to handle invalid data gracefully.
    refl_ma = np.ma.masked_invalid(refl_data, copy=True)
    phidp_ma = np.ma.masked_invalid(phidp_data, copy=True)

    gate_spacing_km = (radar.range['data'][1] - radar.range['data'][0]) / 1000.0
    original_mask = np.ma.getmaskarray(refl_ma)

    # Initialize arrays for corrected and derived data.
    # Start with a copy of reflectivity filled with 0.0 for calculations.
    corrected_refl_data = refl_ma.filled(0.0).copy()
    spec_attn_data = np.zeros_like(corrected_refl_data, dtype=float)
    pia_data = np.zeros_like(corrected_refl_data, dtype=float)
    phidp_c_data = np.zeros_like(corrected_refl_data, dtype=float)   # For processed PhiDP
    phidp_sc_data = np.zeros_like(corrected_refl_data, dtype=float)  # For self-consistent PhiDP

    # --- 3. Main Attenuation Calculation Loop (Iterate over each ray) ---
    for i in range(radar.nrays):
        # Slices for the current ray
        ray_refl_filled = refl_ma[i, :]          # Original Z with mask
        ray_refl = corrected_refl_data[i, :]     # Unmasked Z for calculation
        raw_ray_phidp = phidp_ma[i, :]           # Original PhiDP with mask

        # Process the raw PhiDP ray to be smooth and monotonic
        processed_ray_phidp = _process_phidp_ray(raw_ray_phidp, window_len=window_length)
        phidp_c_data[i, :] = processed_ray_phidp

        # Calculate the total phase shift across the precipitation in the ray
        delta_phidp = _find_delta_phidp(ray_refl_filled, processed_ray_phidp)

        # If no valid precip path is found or the shift is too small, skip this ray.
        if delta_phidp is None or delta_phidp < delta_phidp_th:
            continue

        # This is the core Z-PHI constraint: total PIA is proportional to total delta_phidp.
        pia_total = alpha * delta_phidp
        # This is the 'c' factor from the Bringi et al. (2001) formulation.
        c_factor = np.exp(0.23 * b_coef * pia_total) - 1.0

        if c_factor < 0:
            continue

        # Convert reflectivity to linear units for calculations.
        refl_linear = 10.0 ** (ray_refl / 10.0)
        refl_linear[ray_refl == 0.0] = 0.0  # Avoid log errors on zero-reflectivity gates

        # This section implements the constrained Z-PHI algorithm.
        integrand = refl_linear ** b_coef
        integral_up_to_r = cumulative_trapezoid(integrand, dx=gate_spacing_km, initial=0)
        integral_full_path = 0.46 * b_coef * integral_up_to_r[-1]
        integral_partial = 0.46 * b_coef * (integral_up_to_r[-1] - integral_up_to_r)

        denominator = integral_full_path + c_factor * integral_partial
        # Avoid division by zero
        denominator[denominator == 0] = np.inf

        # Calculate specific attenuation (A) for each gate in the ray
        specific_attenuation = (refl_linear ** b_coef * c_factor) / denominator
        # Calculate Path Integrated Attenuation (PIA) by integrating A.
        # The factor of 2 accounts for the two-way path of the radar signal.
        cumulative_pia = 2 * cumulative_trapezoid(specific_attenuation, dx=gate_spacing_km, initial=0)

        # Update the fields for the current ray
        corrected_refl_data[i, :] += cumulative_pia
        spec_attn_data[i, :] = specific_attenuation
        pia_data[i, :] = cumulative_pia

        # Reconstruct a "self-consistent" PhiDP from the calculated PIA
        # This shows what PhiDP *should* look like given the final attenuation profile.
        if alpha > 1e-6: # Avoid division by zero
            phidp_sc_data[i, :] = cumulative_pia / alpha

    # --- 4. Final Field Creation and Cleanup ---
    fill_value = pyart.config.get_fillvalue()

    # Clean up non-finite values (NaN, inf) that may result from calculations
    corrected_refl_data[~np.isfinite(corrected_refl_data)] = fill_value
    spec_attn_data[~np.isfinite(spec_attn_data)] = fill_value
    pia_data[~np.isfinite(pia_data)] = fill_value
    phidp_c_data[~np.isfinite(phidp_c_data)] = fill_value
    phidp_sc_data[~np.isfinite(phidp_sc_data)] = fill_value

    # Add corrected reflectivity field to the radar object
    output_field_dict = radar.fields[refl_field].copy()
    output_field_dict['data'] = np.ma.array(corrected_refl_data, mask=original_mask)
    output_field_dict['_FillValue'] = fill_value
    radar.add_field(output_refl_field, output_field_dict, replace_existing=True)

    # Add the "Corrected/Cleaned Input" PHIDP_C field
    if output_phidp_c_field:
        phidp_c_dict = radar.fields[phidp_field].copy()
        phidp_c_dict['data'] = np.ma.array(phidp_c_data, mask=original_mask)
        phidp_c_dict['_FillValue'] = fill_value
        phidp_c_dict['long_name'] = 'Processed differential phase'
        phidp_c_dict['standard_name'] = 'processed_differential_phase'
        phidp_c_dict['notes'] = 'Input differential phase after processing (unwrapping, smoothing, monotonization).'
        radar.add_field(output_phidp_c_field, phidp_c_dict, replace_existing=True)

    # Add the "Self-Consistent" PHIDP_SC field
    if output_phidp_sc_field:
        phidp_sc_dict = radar.fields[phidp_field].copy()
        phidp_sc_dict['data'] = np.ma.array(phidp_sc_data, mask=original_mask)
        phidp_sc_dict['_FillValue'] = fill_value
        phidp_sc_dict['long_name'] = 'Self-consistent differential phase'
        phidp_sc_dict['standard_name'] = 'self_consistent_differential_phase'
        phidp_sc_dict['notes'] = 'Differential phase reconstructed from Path Integrated Attenuation (PIA) for self-consistency.'
        radar.add_field(output_phidp_sc_field, phidp_sc_dict, replace_existing=True)

    # Add specific attenuation field
    if output_spec_attn_field:
        spec_attn_dict = pyart.config.get_metadata('specific_attenuation')
        spec_attn_dict['data'] = np.ma.array(spec_attn_data, mask=original_mask)
        spec_attn_dict['_FillValue'] = fill_value
        radar.add_field(output_spec_attn_field, spec_attn_dict, replace_existing=True)

    # Add path integrated attenuation field
    if output_pia_field:
        pia_dict = pyart.config.get_metadata('path_integrated_attenuation')
        pia_dict['data'] = np.ma.array(pia_data, mask=original_mask)
        pia_dict['_FillValue'] = fill_value
        radar.add_field(output_pia_field, pia_dict, replace_existing=True)

    logger.info(
        f"Custom Z-PHI correction complete. Fields '{output_refl_field}', "
        f"'{output_phidp_c_field}', and '{output_phidp_sc_field}' created."
    )
    return radar



# Messy try to use LP to correct PHIDP for zphi method
# Several problems were faced since LP PRIMAL SOLUTION NOT FOUND: not sure due to clutter (not observable), phase (-180 -- 0)
# or parameters: it was detected that the algorithm worked when const_factor was setted to 10>.
# Yet when a solution was found the process was slow > 8 min, an the result was not close to expected.
# Code is keeped here as a log for future reference. Note that it is not the 'clean' version but the messy one after several tries...
#def _find_precip_segments(reflectivity_ray, refl_threshold_dbz, min_precip_gates):
#    """
#    Identifies continuous segments of precipitation in a single radar ray.
#
#    Args:
#        reflectivity_ray (np.ndarray): 1D array of reflectivity data for a ray.
#        refl_threshold_dbz (float): The dBZ value above which a gate is
#                                    considered precipitation.
#        min_precip_gates (int): The minimum number of contiguous gates to be
#                                considered a valid segment.
#
#    Returns:
#        list: A list of tuples, where each tuple is the (start, end) index
#              of a precipitation segment.
#    """
#    # Find all gates where reflectivity is above the threshold
#    above_threshold = np.where(reflectivity_ray > refl_threshold_dbz)[0]
#    if above_threshold.size < min_precip_gates:
#        return []
#
#    # Find where the continuous segments are broken
#    jumps = np.where(np.diff(above_threshold) > 1)[0]
#
#    # Get the start and end indices of each segment
#    segment_starts = np.insert(above_threshold[jumps + 1], 0, above_threshold[0])
#    segment_ends = np.append(above_threshold[jumps], above_threshold[-1])
#
#    # Filter out segments that are too small
#    segments = [
#        (start, end + 1)  # +1 to make the end index inclusive for slicing
#        for start, end in zip(segment_starts, segment_ends)
#        if (end - start + 1) >= min_precip_gates
#    ]
#
#    return segments
#
#
#def _process_phidp_ray(phidp_ray, window_len, polyorder=2):
#    """
#    Unwraps, smooths, and enforces monotonicity on a single ray of
#    differential phase data.
#
#    Args:
#        phidp_ray (np.ndarray): The 1D array of PHIDP data for a single ray.
#        window_len (int): The length of the filter window for Savitzky-Golay.
#        polyorder (int): The order of the polynomial for Savitzky-Golay.
#
#    Returns:
#        np.ndarray: The processed, smooth, and monotonic PHIDP ray.
#    """
#    # Ensure window length is a positive odd integer, smaller than array if needed
#    window_len = min(window_len, len(phidp_ray) - 1)
#    if window_len % 2 == 0:
#        window_len -= 1
#    if window_len < 3:
#        return phidp_ray  # Not enough points to smooth
#
#    # 1. Unwrap the phase data (for data in degrees)
#    unwrapped_phidp = np.unwrap(phidp_ray, discont=180)
#
#    # 2. Smooth the unwrapped data with a Savitzky-Golay filter
#    smoothed_phidp = savgol_filter(unwrapped_phidp, window_len, polyorder)
#
#    # 3. **NEW STEP**: Enforce that the profile is always non-decreasing.
#    monotonic_phidp = np.maximum.accumulate(smoothed_phidp)
#
#    return monotonic_phidp
#
#
#def correct_attenuation_zphi_custom(radar, **params):
#    """
#    Performs attenuation correction using a custom, robust Z-PHI algorithm.
#    """
#    logger.info("Starting custom Z-PHI attenuation correction.")
#
#    # --- 1. Unpack Parameters ---
#    refl_field = params['input_refl_field']
#    phidp_field = params['input_phidp_field']
#    rhohv_field = params.get('input_rhohv_field', 'RHOHV')
#    output_refl_field = params['output_refl_field']
#    alpha = params['alpha']
#    b_coef = params['b_coef']
#    refl_threshold_dbz = params['refl_threshold_dbz']
#    output_spec_attn_field = params.get('output_spec_attn_field', 'specific_attenuation')
#    output_pia_field = params.get('output_pia_field', 'path_integrated_attenuation')
#
#    # --- 2. Prepare Data (Corrected Section) ---
#    # Get the raw data, which might be a regular or masked array
#    _refl_data = radar.fields[refl_field]['data']
#    _phidp_data = radar.fields[phidp_field]['data']
#
#    # Explicitly convert to masked arrays. This handles cases where the input
#    # is a regular ndarray (i.e., has no masked values), ensuring that
#    # methods like .filled() will always be available.
#    refl_ma = np.ma.masked_invalid(_refl_data, copy=True)
#    phidp_ma = np.ma.masked_invalid(_phidp_data, copy=True)
#
#    gate_spacing_km = (radar.range['data'][1] - radar.range['data'][0]) / 1000.0
#    original_mask = np.ma.getmaskarray(refl_ma)
#
#    # Create a writeable numpy array from the reflectivity data.
#    # .filled() returns a regular ndarray, which we then copy.
#    # This is a robust way to get a modifiable copy of the data.
#    corrected_refl_data = refl_ma.filled(0.0).copy()
#
#    spec_attn_data = np.zeros_like(corrected_refl_data, dtype=float)
#    pia_data = np.zeros_like(corrected_refl_data, dtype=float)
#
#    #radar.fields[phidp_field]['data'] = radar.fields[phidp_field]['data'] + 180
#
#
#    ncp_field_name = pyart.config.get_field_name('normalized_coherent_power')
#    if ncp_field_name not in radar.fields:
#        radar.add_field_like(refl_field, ncp_field_name,
#                                   np.ones_like(radar.fields[refl_field]['data']))
#
#    #phidp_filt_dict, _ = pyart.correct.phase_proc_lp(
#    #    radar,
#    #    offset=0.0,
#    #    refl_field=refl_field,
#    #    phidp_field=phidp_field,
#    #    rhv_field=rhohv_field,
#    #    ncp_field=ncp_field_name,
#    #    LP_solver='cvxopt',
#    #    #sys_phase=-180,
#    #    # overide_sys_phase=True,
#    #    #self_const=100,
#    #    min_rhv=0.9,
#    #    debug=True,
#    #    really_verbose=True,
#    #    window_len=35,
#    #    nowrap=100,
#    #    # low_z=0,
#    #    # high_z=60,
#    #)
#
#    phidp_filt = _phidp_data
#
#    # --- 4. Main Attenuation Calculation Loop ---
#    for i in range(radar.nrays):
#        # This line will now work correctly because refl_ma is guaranteed to be a masked array.
#        ray_refl_filled = refl_ma.filled(np.nan)[i, :]
#        ray_phidp_filt = phidp_filt[i, :]
#
#        segments = _find_precip_segments(ray_refl_filled, refl_threshold_dbz, params.get('min_precip_gates', 10))
#        if not segments:
#            continue
#
#        for r1, r2 in segments:
#            seg_refl = corrected_refl_data[i, r1:r2]
#            seg_phidp = _process_phidp_ray(ray_phidp_filt[r1:r2], window_len=20)
#
#            #if len(seg_phidp) < 10:
#            #    continue
#
#            start_phidp = np.median(seg_phidp[:5])
#            end_phidp = np.median(seg_phidp[-5:])
#            delta_phidp = end_phidp - start_phidp
#
#            if delta_phidp < 0:
#                delta_phidp = 0
#
#            pia_total = alpha * delta_phidp
#            c_factor = np.exp(0.23 * b_coef * pia_total) - 1.0
#            if c_factor < 0:
#                logger.warning(f"Attenuation correction failed for ray {r1}, pia {r2}, c_factor {c_factor}")
#                continue
#
#            refl_linear = 10.0 ** (seg_refl / 10.0)
#            integral_full_path = 0.46 * b_coef * np.sum(refl_linear ** b_coef) * gate_spacing_km
#            integrand = refl_linear ** b_coef
#            integral_partial = 0.46 * b_coef * (np.sum(integrand) - np.cumsum(integrand)) * gate_spacing_km
#            denominator = integral_full_path + c_factor * integral_partial
#            denominator[denominator == 0] = np.inf
#            specific_attenuation = (refl_linear ** b_coef * c_factor) / denominator
#            cumulative_pia = 2 * np.cumsum(specific_attenuation) * gate_spacing_km
#
#            corrected_refl_data[i, r1:r2] += cumulative_pia
#            spec_attn_data[i, r1:r2] = specific_attenuation
#            pia_data[i, r1:r2] = cumulative_pia
#
#    # --- 5. Final Field Creation ---
#    fill_value = pyart.config.get_fillvalue()
#
#    corrected_refl_data[~np.isfinite(corrected_refl_data)] = 0
#    spec_attn_data[~np.isfinite(spec_attn_data)] = 0
#    pia_data[~np.isfinite(pia_data)] = 0
#
#    output_field_dict = radar.fields[refl_field].copy()
#    output_field_dict['data'] = np.ma.array(corrected_refl_data, mask=original_mask)
#    output_field_dict['_FillValue'] = fill_value
#    radar.add_field(output_refl_field, output_field_dict, replace_existing=True)
#
#    if output_spec_attn_field:
#        spec_attn_dict = pyart.config.get_metadata('specific_attenuation')
#        spec_attn_dict['data'] = np.ma.array(spec_attn_data, mask=original_mask)
#        spec_attn_dict['_FillValue'] = fill_value
#        radar.add_field(output_spec_attn_field, spec_attn_dict, replace_existing=True)
#
#    if output_pia_field:
#        pia_dict = pyart.config.get_metadata('path_integrated_attenuation')
#        pia_dict['data'] = np.ma.array(pia_data, mask=original_mask)
#        pia_dict['_FillValue'] = fill_value
#        radar.add_field(output_pia_field, pia_dict, replace_existing=True)
#
#    logger.info(f"Custom Z-PHI attenuation correction complete. Field '{output_refl_field}' created.")
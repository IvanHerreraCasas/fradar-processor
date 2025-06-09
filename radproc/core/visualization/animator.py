# core/visualization/animator.py

import os
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any

# Image/Animation processing
import imageio.v3 as iio
from PIL import Image, UnidentifiedImageError

# Core components
import xarray as xr
from ..config import get_setting
from ..utils.helpers import parse_datetime_from_filename
from ..data import read_ppi_scan
from ..db_manager import get_connection, release_connection, query_scan_log_for_timeseries_processing
from ..utils.geo import georeference_dataset
from ..utils.helpers import parse_datetime_from_image_filename
from .plotter import create_ppi_image
from .style import PlotStyle, get_plot_style # Import PlotStyle dataclass too

logger = logging.getLogger(__name__)




def _validate_image(image_path: str) -> bool:
    """
    Checks if an image file is valid and readable using Pillow.

    Args:
        image_path: Path to the image file.

    Returns:
        True if the image is valid, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            img.verify() # verify() might raise exception on corrupt image
        # Reopen after verify if needed for other checks, but verify is often enough
        # with Image.open(image_path) as img:
        #     img.load() # Try loading pixel data
        logger.debug(f"Image validation successful: {image_path}")
        return True
    except FileNotFoundError:
        logger.warning(f"Image validation failed: File not found - {image_path}")
        return False
    except (UnidentifiedImageError, SyntaxError, OSError, ValueError, TypeError) as e:
        # Catch various Pillow/OS errors related to bad files/formats
        logger.warning(f"Image validation failed: Invalid/corrupt image - {image_path}. Error: {e}")
        return False
    except Exception as e:
         logger.error(f"Unexpected error validating image {image_path}: {e}", exc_info=True)
         return False
     
def _regenerate_frame(
    scan_filepath: str,
    variable: str,
    plot_style: PlotStyle,
    output_dir: str, # Temporary directory for this frame
    frame_idx: int,
    plot_extent: Optional[Tuple[float, float, float, float]] = None,
    include_watermark: bool = True
) -> Optional[str]:
    """
    Reads a scan file, generates a plot image with specific settings,
    and saves it to a temporary location.

    Args:
        scan_filepath: Path to the source .scnx.gz scan file.
        variable: The variable to plot.
        plot_style: The PlotStyle object for the variable.
        output_dir: The temporary directory to save the generated frame.
        frame_idx: The index of this frame (used for filename).
        plot_extent: Optional extent override (LonMin, LonMax, LatMin, LatMax).
        include_watermark: Whether to include the configured watermark.

    Returns:
        The full path to the generated temporary image file, or None if failed.
    """
    logger.debug(f"Regenerating frame {frame_idx} from scan: {scan_filepath}")
    ds = None
    ds_geo = None
    try:
        # 1. Read Scan Data
        # Read only the specific variable needed for plotting
        ds = read_ppi_scan(scan_filepath, variables=[variable])
        if ds is None:
            logger.warning(f"Failed to read scan for frame {frame_idx}: {scan_filepath}")
            return None

        # 2. Georeference
        ds_geo = georeference_dataset(ds)
        if 'x' not in ds_geo.coords: # Basic check
            logger.warning(f"Failed to georeference scan for frame {frame_idx}: {scan_filepath}")
            return None

        # 3. Determine Watermark Path
        watermark_path = None
        if include_watermark:
            watermark_path = get_setting('app.watermark_path')
            if watermark_path and not os.path.exists(watermark_path):
                logger.warning(f"Watermark configured but not found at: {watermark_path}")
                watermark_path = None # Treat as if not configured if missing

        watermark_zoom = get_setting('styles.defaults.watermark_zoom', 0.05)

        # 4. Create Plot Image Bytes using modified plotter function
        image_bytes = create_ppi_image(
            ds=ds_geo,
            variable=variable,
            style=plot_style,
            watermark_path=watermark_path,
            watermark_zoom=watermark_zoom,
            plot_extent=plot_extent # Pass the extent override
            # coverage_radius_km could also be made configurable if needed
        )

        if not image_bytes:
            logger.warning(f"Failed to generate image bytes for frame {frame_idx} from {scan_filepath}")
            return None

        # 5. Save Bytes to Temporary File
        frame_filename = f"frame_{frame_idx:05d}.png" # Consistent frame naming
        frame_filepath = os.path.join(output_dir, frame_filename)

        try:
            with open(frame_filepath, 'wb') as f:
                f.write(image_bytes)
            logger.debug(f"Saved regenerated frame: {frame_filepath}")
            return frame_filepath
        except IOError as e:
            logger.error(f"Failed to save temporary frame {frame_filepath}: {e}")
            return None

    except Exception as e:
        logger.error(f"Unexpected error regenerating frame {frame_idx} from {scan_filepath}: {e}", exc_info=True)
        return None
    finally:
        # Ensure datasets are closed
        if ds_geo is not None:
             try: ds_geo.close()
             except Exception: pass
        if ds is not None:
             try: ds.close()
             except Exception: pass
             
def _get_scan_elevation(scan_filepath: str) -> Optional[float]:
    """Reads a scan file briefly to extract its elevation."""
    ds = None
    try:
        # Read minimal data - might still load coordinates implicitly
        # Pass a known variable likely to exist if needed, or None
        ds = read_ppi_scan(scan_filepath, variables=None) # Try reading without specific variable request
        if ds is not None and 'elevation' in ds.coords:
            elevation = float(ds['elevation'].item())
            return elevation
        elif ds is not None:
            logger.warning(f"Elevation coordinate not found after reading scan: {scan_filepath}")
            return None
        else:
            # read_scan failed, error already logged by it
            return None
    except Exception as e:
        logger.error(f"Failed to read or get elevation from scan {scan_filepath}: {e}", exc_info=True)
        return None
    finally:
        if ds is not None:
            try: ds.close()
            except Exception: pass


def create_animation(
        variable: str,
        elevation: float,  # This is the target_elevation for scans
        start_dt: datetime,
        end_dt: datetime,
        output_filename: str,
        plot_extent: Optional[Tuple[float, float, float, float]] = None,
        include_watermark: bool = True,
        fps: Optional[int] = None,
) -> bool:
    logger.info(f"--- Starting Animation Creation (Scan-Log Driven) ---")
    logger.info(f"Variable: {variable}, Target Elev: {elevation:.1f}, Range: {start_dt} -> {end_dt}")
    logger.info(f"Output File: {output_filename}")

    images_base_dir = get_setting('app.images_dir')
    # processed_scan_dir is no longer directly used to find scans, DB will provide paths
    default_tmp_dir_base = get_setting('app.animation_tmp_dir', 'cache/animation_tmp')
    default_fps = get_setting('app.animation_fps', 5)
    final_fps = fps if fps is not None else default_fps
    elevation_tolerance = get_setting('volume_grouping.elevation_tolerance', 0.1)  # Use a consistent tolerance

    if not images_base_dir:
        logger.error("Animation failed: 'app.images_dir' is not configured.")
        return False

    try:
        target_elevation_for_query = float(elevation)  # Use this for DB query
        elevation_code_for_image_path = f"{int(round(target_elevation_for_query * 100)):03d}"
    except (ValueError, TypeError):
        logger.error(f"Invalid target elevation value provided: {elevation}")
        return False

    plot_style = get_plot_style(variable)
    if plot_style is None: return False

    config_watermark_path = get_setting('app.watermark_path')
    regen_needed_globally = (plot_extent is not None) or \
                            (not include_watermark and config_watermark_path and os.path.exists(config_watermark_path))
    if regen_needed_globally:
        logger.info("Global frame regeneration required (custom extent or no watermark requested).")

    # --- b. Find Scan Files VIA DATABASE ---
    conn = None
    scan_log_entries: List[Tuple[str, datetime, int]] = []  # (filepath, precise_timestamp, scan_log_id)
    try:
        conn = get_connection()
        logger.info(
            f"Querying scan log for animation frames: Elev~{target_elevation_for_query:.2f}, Range: {start_dt} to {end_dt}")
        scan_log_entries = query_scan_log_for_timeseries_processing(
            conn,
            target_elevation=target_elevation_for_query,
            start_dt=start_dt,
            end_dt=end_dt,
            elevation_tolerance=elevation_tolerance
        )
    except Exception as db_err:
        logger.error(f"Database error querying scans for animation: {db_err}", exc_info=True)
        return False  # Cannot proceed without scan list
    finally:
        if conn:
            release_connection(conn)

    if not scan_log_entries:
        logger.error("No scan files found in the database log for the specified criteria.")
        return False
    logger.info(f"Found {len(scan_log_entries)} scan log entries for animation.")
    # scan_log_entries is already sorted by precise_timestamp from the DB query

    # --- c. Prepare Frame List ---
    frame_paths_for_anim = []
    temp_dir_path = None
    processed_scan_count = 0

    try:
        if regen_needed_globally:  # Create temp dir only if needed
            base_tmp = os.path.join(os.getcwd(), default_tmp_dir_base)
            os.makedirs(base_tmp, exist_ok=True)
            temp_dir_path = tempfile.mkdtemp(prefix="radproc_anim_", dir=base_tmp)
            logger.info(f"Created temporary directory for regenerated frames: {temp_dir_path}")

        logger.info("Processing scan log entries: Checking elevation and preparing frames...")
        # scan_log_entries contains (filepath, precise_timestamp, scan_log_id)
        for idx, (scan_path, scan_dt_precise, _scan_log_id) in enumerate(scan_log_entries):
            processed_scan_count += 1
            logger.debug(f"Processing scan log entry {idx + 1}/{len(scan_log_entries)}: {scan_path}")

            # Elevation check already done by DB query with tolerance.
            # We still need scan_elevation from file if we were to re-check,
            # but query_scan_log_for_timeseries_processing handles this.
            # The 'elevation' parameter of this function is the *target* elevation.

            # 2. Determine Expected Image Path
            # Image filenames use floored minute for time and specific elevation code.
            # scan_dt_precise is the exact timestamp from the scan.
            scan_dt_minute_floor = scan_dt_precise.replace(second=0, microsecond=0)
            date_str_for_path = scan_dt_minute_floor.strftime("%Y%m%d")  # For directory structure
            datetime_file_str = scan_dt_minute_floor.strftime("%Y%m%d_%H%M")  # For filename

            # Use elevation_code_for_image_path calculated from the target elevation.
            expected_image_filename = f"{variable}_{elevation_code_for_image_path}_{datetime_file_str}.png"
            expected_image_path = os.path.join(images_base_dir, variable, date_str_for_path,
                                               elevation_code_for_image_path, expected_image_filename)

            # 3. Decide: Use Existing or Regenerate?
            use_existing_image = False
            if not regen_needed_globally:
                if os.path.exists(expected_image_path):
                    if _validate_image(expected_image_path):
                        logger.debug(f"Using existing valid image: {expected_image_path}")
                        frame_paths_for_anim.append(expected_image_path)
                        use_existing_image = True
                    else:
                        logger.warning(f"Existing image is invalid, will attempt regeneration: {expected_image_path}")
                else:
                    logger.info(
                        f"Image not found at {expected_image_path}, attempting regeneration from scan {scan_path}")

            # 4. Regenerate if needed
            if not use_existing_image:
                if temp_dir_path is None:  # Create temp dir if not already created
                    base_tmp = os.path.join(os.getcwd(), default_tmp_dir_base)
                    os.makedirs(base_tmp, exist_ok=True)
                    temp_dir_path = tempfile.mkdtemp(prefix="radproc_anim_", dir=base_tmp)
                    logger.info(f"Created temporary directory (first regeneration): {temp_dir_path}")

                # _regenerate_frame uses the actual scan_path from the DB log
                regenerated_path = _regenerate_frame(
                    scan_filepath=scan_path,
                    variable=variable,
                    plot_style=plot_style,
                    output_dir=temp_dir_path,
                    frame_idx=idx,
                    plot_extent=plot_extent,
                    include_watermark=include_watermark
                )
                if regenerated_path:
                    frame_paths_for_anim.append(regenerated_path)
                else:
                    logger.warning(f"Skipping frame: Failed regeneration for scan {scan_path}")

        # --- d. Create Animation ---
        if not frame_paths_for_anim:
            logger.error("No valid frames available (found/generated) for animation.")
            return False

        logger.info(f"Assembling animation with {len(frame_paths_for_anim)} frames...")
        # ... (rest of animation creation logic with imageio remains the same) ...
        try:
            output_dir_for_anim = os.path.dirname(output_filename)
            if output_dir_for_anim: os.makedirs(output_dir_for_anim, exist_ok=True)

            kwargs_anim = {'fps': final_fps}
            anim_plugin = None  # imageio v3 often infers unless specific needed
            file_format_anim = os.path.splitext(output_filename)[1].lower()

            if file_format_anim == '.mp4':
                # anim_plugin = 'ffmpeg' # Only if auto-detection fails
                kwargs_anim['codec'] = 'libx264'
                kwargs_anim['pixelformat'] = 'yuv420p'
            elif file_format_anim == '.gif':
                # anim_plugin = 'pillow' # Only if auto-detection fails
                kwargs_anim['loop'] = 0  # Loop indefinitely for GIF
                # duration is inverse of fps for GIF if Pillow plugin is used.
                # imageio v3's default GIF writer might handle fps directly.
                # If using Pillow plugin: kwargs_anim['duration'] = 1000 / final_fps (ms per frame)

            frames_data = []
            for p in frame_paths_for_anim:
                try:
                    frames_data.append(iio.imread(p))
                except Exception as read_img_err:
                    logger.warning(f"Could not read frame {p} for animation: {read_img_err}. Skipping this frame.")
                    continue  # Skip problematic frame

            if not frames_data:
                logger.error("No frames could be read for animation after attempting to load them.")
                return False

            iio.imwrite(
                output_filename,
                frames_data,
                plugin=anim_plugin,  # Often None is fine
                **kwargs_anim
            )
            logger.info(f"Animation successfully created: {output_filename}")
            return True

        except ImportError as ie:
            logger.error(
                f"Failed to create animation: Missing imageio plugin or library for format {file_format_anim}. Error: {ie}")
            if file_format_anim == '.mp4':
                logger.error("For MP4 support, ensure 'imageio-ffmpeg' is installed: pip install imageio-ffmpeg")
            return False
        except Exception as anim_err:
            logger.error(f"Failed to create animation: {anim_err}", exc_info=True)
            return False

    finally:
        # --- e. Cleanup ---
        if temp_dir_path and os.path.isdir(temp_dir_path):
            try:
                logger.info(f"Cleaning up temporary animation directory: {temp_dir_path}")
                shutil.rmtree(temp_dir_path)
            except Exception as clean_err:
                logger.error(f"Error removing temporary directory {temp_dir_path}: {clean_err}")
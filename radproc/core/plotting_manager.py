# radproc/core/plotting_manager.py

import os
import logging
from datetime import datetime
from typing import Optional, Tuple

from ..core import db_manager, data
from ..core.visualization import plotter, style
from ..core.config import get_setting
from ..core.utils import geo

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


    def tqdm(iterable, *args, **kwargs):
        return iterable

logger = logging.getLogger(__name__)


def _save_plot_image(image_bytes: bytes, base_dir: str, source: str, version: Optional[str], variable: str,
                     elevation: float, timestamp: datetime) -> bool:
    """
    Saves image bytes to a structured, version-aware directory path.
    Example path: .../images/corrected/v1_0/DBZH/20230101/0300/DBZH_0300_20230101_1200.png
    """
    try:
        # Construct the path
        version_path = version if version else ''
        date_str = timestamp.strftime("%Y%m%d")
        elevation_code = f"{int(round(elevation * 100)):03d}"

        image_sub_dir = os.path.join(base_dir, source, version_path, variable, date_str, elevation_code)
        os.makedirs(image_sub_dir, exist_ok=True)

        datetime_file_str = timestamp.strftime("%Y%m%d_%H%M")
        image_filename = f"{variable}_{elevation_code}_{datetime_file_str}.png"
        image_filepath = os.path.join(image_sub_dir, image_filename)

        # Save the file
        with open(image_filepath, 'wb') as f:
            f.write(image_bytes)
        logger.info(f"Saved plot: {image_filepath}")
        return True
    except IOError as e:
        logger.error(f"Failed to save plot image: {e}")
        return False


def generate_plots(
        variable: str,
        elevation: float,
        start_dt: datetime,
        end_dt: datetime,
        source: str = 'raw',
        version: Optional[str] = None,
        plot_extent: Optional[Tuple[float, float, float, float]] = None,
        output_dir_override: Optional[str] = None
) -> bool:
    """
    Orchestrates the generation of historical PPI plots from either raw or
    corrected data sources.
    """
    logger.info(f"--- Starting Plot Generation from '{source.upper()}' source ---")
    logger.info(f"Params: Var={variable}, Elev={elevation:.2f}, Range: {start_dt} -> {end_dt}")
    if source == 'corrected' and not version:
        logger.error("A --version must be specified when using --source corrected.")
        return False
    if source == 'corrected':
        logger.info(f"Using corrected data version: {version}")

    plot_style = style.get_plot_style(variable)
    if not plot_style:
        logger.error(f"No plot style found for variable '{variable}'. Cannot generate plots.")
        return False

    images_base_dir = output_dir_override or get_setting('app.images_dir')
    if not images_base_dir:
        logger.error("Image output directory not specified or configured in 'app.images_dir'.")
        return False

    conn = db_manager.get_connection()
    if not conn:
        logger.error("Failed to establish database connection.")
        return False

    overall_success = True
    plots_generated = 0



    try:
        if source == 'raw':
            # --- RAW DATA WORKFLOW ---
            scans_to_process = db_manager.query_scan_log_for_timeseries_processing(conn, elevation, start_dt, end_dt)
            logger.info(f"Found {len(scans_to_process)} raw scans to process.")

            scan_iterator = tqdm(scans_to_process, desc="Generating Raw Plots", unit="plot")
            for scan_path, scan_dt, _ in scan_iterator:
                ds = data.read_ppi_scan(scan_path, variables=[variable])
                if ds:
                    ds_geo = geo.georeference_dataset(ds)
                    image_bytes = plotter.create_ppi_image(ds_geo, variable, plot_style, plot_extent=plot_extent)
                    if image_bytes and _save_plot_image(image_bytes, images_base_dir, source, None, variable, elevation,
                                                        scan_dt):
                        plots_generated += 1
                    ds.close()

        elif source == 'corrected':
            volumes_to_process = db_manager.get_processed_volume_paths(conn, start_dt, end_dt, version)
            logger.info(f"Found {len(volumes_to_process)} corrected volumes to process for version '{version}'.")

            volume_iterator = tqdm(volumes_to_process, desc="Generating Corrected Plots", unit="plot")
            for vol_path, vol_id in volume_iterator:
                dtree = data.read_volume_from_cfradial(vol_path)
                if dtree:
                    target_sweep_node = None
                    for sweep_name in dtree.children:
                        sweep_ds = dtree[sweep_name].ds
                        if 'elevation' in sweep_ds.coords and abs(float(sweep_ds.elevation.item()) - elevation) < 0.1:
                            azimuth_step = get_setting('radar.azimuth_step', default=0.5)
                            target_sweep_node = data.preprocess_scan(sweep_ds, azimuth_step)
                            target_sweep_node = geo.georeference_dataset(target_sweep_node)
                            break

                    if target_sweep_node:
                        image_bytes = plotter.create_ppi_image(target_sweep_node, variable, plot_style,
                                                               plot_extent=plot_extent)
                        if image_bytes and _save_plot_image(image_bytes, images_base_dir, source, version, variable,
                                                            elevation, vol_id):
                            plots_generated += 1
                    else:
                        logger.debug(f"No sweep found for elevation {elevation:.2f} in volume {vol_id}")
                    dtree.close()

    except Exception as e:
        logger.error(f"An error occurred during plot generation: {e}", exc_info=True)
        overall_success = False
    finally:
        db_manager.release_connection(conn)

    logger.info(f"--- Plot Generation Finished ---")
    logger.info(f"Successfully generated {plots_generated} plots.")
    return overall_success
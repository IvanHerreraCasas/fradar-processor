# radproc/core/accumulation_map.py
import gc
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple

import numpy as np
import xarray as xr
from tqdm import tqdm

from . import db_manager, data
from .config import get_setting
from .visualization import plotter, style
from .utils.geo import georeference_dataset

logger = logging.getLogger(__name__)


def _create_accumulation_from_files(
        scans_to_process: List[Tuple[str, datetime, int]],
        elevation: float,
        source: str
) -> Optional[xr.Dataset]:
    """
    Creates a gridded accumulated precipitation map from a list of scan files.

    Args:
        scans_to_process: A list of tuples, where each tuple contains
                          (filepath, timestamp, scan_log_id).
        elevation: The target elevation angle.
        source: The data source ('raw' or 'corrected').

    Returns:
        An xarray.Dataset containing the accumulated precipitation grid and
        georeferencing coordinates, or None if processing fails.
    """
    if not scans_to_process:
        logger.warning("No scans provided to _create_accumulation_from_files.")
        return None

    try:
        # 1. Grid Initialization
        first_scan_path, _, _ = scans_to_process[0]

        # Initialize the grid based on the source type
        if source == 'raw':
            with data.read_ppi_scan(first_scan_path, variables=['RATE']) as ds_initial:
                if ds_initial is None:
                    logger.error("Failed to read the first raw scan to initialize grid.")
                    return None
                ds_geo = georeference_dataset(ds_initial)
        elif source == 'corrected':
            with data.read_volume_from_cfradial(first_scan_path) as dtree:
                if dtree is None:
                    logger.error("Failed to read the first corrected volume to initialize grid.")
                    return None
                # Find the correct sweep for grid initialization
                ds_initial = None
                for sweep_name in dtree.children:
                    if 'sweep' in sweep_name:
                        sweep_ds = dtree[sweep_name].ds
                        if 'elevation' in sweep_ds.coords and abs(
                                float(sweep_ds.elevation.values[0]) - elevation) < 0.1:
                            azimuth_step = get_setting('radar.azimuth_step', default=0.5)
                            ds_initial = data.preprocess_scan(sweep_ds, azimuth_step)
                            break
                if ds_initial is None:
                    logger.error(f"No sweep matching elevation {elevation} found in the first corrected volume.")
                    return None
                ds_geo = georeference_dataset(ds_initial)
        else:
            logger.error(f"Unknown source type: {source}")
            return None

        total_accumulation = xr.DataArray(
            np.zeros_like(ds_geo['RATE'].data),
            coords=ds_geo['RATE'].coords,
            dims=ds_geo['RATE'].dims,
            name="precipitation_accumulation",
            attrs={
                "long_name": "Total accumulated precipitation",
                "units": "mm",
                "start_time_utc": scans_to_process[0][1].astimezone(timezone.utc).isoformat(),
                "end_time_utc": scans_to_process[-1][1].astimezone(timezone.utc).isoformat(),
            }
        )

        # 2. Iterative Accumulation
        for i, (scan_path, scan_dt, _) in enumerate(tqdm(scans_to_process, desc="Accumulating Scans")):
            # (Time delta logic remains the same)
            if i + 1 < len(scans_to_process):
                next_scan_dt = scans_to_process[i + 1][1]
                time_delta_seconds = (next_scan_dt - scan_dt).total_seconds()
            else:
                default_interval_minutes = get_setting('app.max_inter_scan_gap_minutes', 5)
                time_delta_seconds = default_interval_minutes * 60
            time_delta_hours = time_delta_seconds / 3600.0

            rate_data = None
            if source == 'raw':
                with data.read_ppi_scan(scan_path, variables=['RATE'], for_volume=True) as ds:
                    if ds:
                        rate_data = ds['RATE']
            elif source == 'corrected':
                with data.read_volume_from_cfradial(scan_path) as dtree:
                    if dtree:
                        for sweep_name in dtree.children:
                            if 'sweep' in sweep_name:
                                sweep_ds = dtree[sweep_name].ds
                                if 'elevation' in sweep_ds.coords and abs(
                                        float(sweep_ds.elevation.values[0]) - elevation) < 0.1:
                                    rate_data = sweep_ds['RATE']
                                    break

            if rate_data is not None:
                incremental_precip = (rate_data.fillna(0) * time_delta_hours)
                total_accumulation += incremental_precip
            else:
                logger.warning(f"Could not extract RATE data for {scan_path} at elevation {elevation}. Skipping.")

            # Proactively clean up any lingering complex objects from the libraries.
            if (i + 1) % 25 == 0:  # Optional: run it every 25 scans to reduce overhead
                gc.collect()

        # 3. Final Computation and Dataset Assembly
        logger.info("Finalizing accumulation computation...")
        final_data = total_accumulation.compute()

        ds_accum = final_data.to_dataset()
        for coord in ['latitude', 'longitude', 'elevation', 'time', 'x', 'y']:
            if coord in ds_geo:
                ds_accum.coords[coord] = ds_geo.coords[coord]

        return ds_accum

    except Exception as e:
        logger.error(f"An error occurred during the core accumulation process: {e}", exc_info=True)
        return None


def generate_accumulation_data(
        scans_to_process: List[Tuple[str, datetime, int]],
        output_file: str,
        elevation: float,
        source: str,
        plot_config: Optional[Dict] = None
) -> bool:
    """
    Generates and saves a gridded accumulated precipitation map from a list of scans.
    """
    logger.info(f"--- Generating Accumulated Precipitation Map from pre-fetched file list ---")

    ds_accum = _create_accumulation_from_files(scans_to_process, elevation, source)

    if ds_accum is None:
        logger.error("Core accumulation processing failed.")
        return False

    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only run makedirs if the path is not empty
            os.makedirs(output_dir, exist_ok=True)
        ds_accum.to_netcdf(output_file)
        logger.info(f"Accumulated precipitation map saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save NetCDF file to {output_file}: {e}", exc_info=True)
        return False

    if plot_config and isinstance(plot_config, dict):
        logger.info("Generating visualization for the accumulation map...")
        plot_output_path = plot_config.get("output_path")
        if plot_output_path:
            accum_style = style.get_plot_style("ACCUM")
            if accum_style:
                image_bytes = plotter.create_accumulation_image(
                    ds=ds_accum,
                    variable="precipitation_accumulation",
                    style=accum_style,
                    watermark_path=get_setting('app.watermark_path'),
                    coverage_radius_km=get_setting('styles.defaults.coverage_radius_km', 70.0),
                    plot_extent=plot_config.get("extent")
                )
                if image_bytes:
                    try:
                        with open(plot_output_path, 'wb') as f:
                            f.write(image_bytes)
                        logger.info(f"Accumulation plot saved to: {plot_output_path}")
                    except IOError as e:
                        logger.error(f"Failed to save accumulation plot: {e}")
                else:
                    logger.error("Failed to generate accumulation plot image bytes.")
            else:
                logger.warning("Could not generate plot: Style for 'ACCUM' not found.")
        else:
            logger.warning("Plotting requested but 'output_path' not provided in plot_config.")

    return True


def orchestrate_accumulation_generation(
        start_dt: datetime,
        end_dt: datetime,
        elevation: float,
        output_file: str,
        source: str = 'raw',
        version: Optional[str] = None,
        plot_config: Optional[Dict] = None
) -> bool:
    """
    Orchestrates the full accumulation generation process, including DB query.
    """
    logger.info(f"--- Orchestrating Accumulated Map Generation ---")
    conn = None
    try:
        conn = db_manager.get_connection()
        scans = []
        if source == 'raw':
            scans = db_manager.query_scan_log_for_timeseries_processing(conn, elevation, start_dt, end_dt)
        elif source == 'corrected':
            if not version:
                logger.error("A 'version' must be specified for 'corrected' source.")
                return False
            # Get paths of processed volumes
            volume_paths = db_manager.get_processed_volume_paths(conn, start_dt, end_dt, version)
            # Adapt to the (filepath, timestamp, id) format
            scans = [(path, ts, 0) for path, ts in volume_paths]  # Use 0 as a placeholder ID

        if not scans:
            logger.warning("No scans or volumes found for the specified criteria.")
            return True

        return generate_accumulation_data(scans, output_file, elevation, source, plot_config)

    except Exception as e:
        logger.error(f"An error occurred during orchestration: {e}", exc_info=True)
        return False
    finally:
        if conn:
            db_manager.release_connection(conn)
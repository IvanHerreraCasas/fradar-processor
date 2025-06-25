# radproc/tasks.py
import logging
import os
from datetime import datetime, timezone
from typing import Optional

# Import the Huey instance defined in huey_config.py
from .huey_config import huey

from .core.analysis import generate_timeseries, calculate_accumulation  # Make sure analysis is imported
from .core.visualization.animator import create_animation

logger = logging.getLogger(__name__)


@huey.task(retries=2, retry_delay=15)
def run_generate_point_timeseries(point_name: str, start_dt_iso: str, end_dt_iso: str, source: str, version: str,
                                  variable_override: Optional[str] = None):
    """
    Huey background task to generate/update timeseries data in the DB.

    Args:
        point_name: Name of the point.
        start_dt_iso: Start datetime in ISO format string (UTC assumed if no offset).
        end_dt_iso: End datetime in ISO format string (UTC assumed if no offset).
        variable_override: Optional variable name to process instead of defaults.

    Returns:
        A dictionary indicating success and parameters, or raises an exception.
    """
    logger.info(
        f"[Task Start] Generating DB timeseries for '{point_name}', {start_dt_iso} to {end_dt_iso}, Var: {variable_override or 'Defaults'}")

    try:
        start_dt = datetime.fromisoformat(start_dt_iso.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_dt_iso.replace('Z', '+00:00'))

        if start_dt.tzinfo is None: start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None: end_dt = end_dt.replace(tzinfo=timezone.utc)

        variables_to_run = [
            variable_override] if variable_override else None  # None means use defaults in analysis function

        # --- Call the Core Logic (DB version) ---
        success = generate_timeseries(
            point_names=[point_name],
            start_dt=start_dt,
            end_dt=end_dt,
            source=source,
            version=version,
            specific_variables=variables_to_run,
        )
        # ----------------------------------------

        if success:
            logger.info(f"[Task Success] DB Timeseries generation complete for '{point_name}'.")
            # Return parameters so the API endpoint can (potentially) know what to query
            return {
                "status": "SUCCESS",
                "message": f"Timeseries DB generation completed for {point_name}.",
                "point_name": point_name,
                "start_dt_iso": start_dt_iso,
                "end_dt_iso": end_dt_iso,
                "variable_processed": variable_override or "defaults"  # Indicate what was processed
            }
        else:
            logger.error(
                f"[Task Failure] Timeseries generation failed for '{point_name}' (core function returned False).")
            raise RuntimeError(f"Core DB timeseries generation failed for point '{point_name}'.")

    except Exception as e:
        logger.exception(f"[Task Error] Exception during timeseries task for point '{point_name}': {e}")
        raise  # Re-raise for Huey



@huey.task(retries=1, retry_delay=10)
def run_calculate_accumulation(point_name: str, start_dt_iso: str, end_dt_iso: str, interval: str, rate_variable: str,
                               output_file_path: str, source: str, version: str):
    """
    Huey background task to calculate precipitation accumulation.
    """
    logger.info(
        f"[Task Start] Calculating accumulation for point '{point_name}', Interval: {interval}, Output: {output_file_path}")
    try:
        start_dt = datetime.fromisoformat(start_dt_iso.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_dt_iso.replace('Z', '+00:00'))
        if start_dt.tzinfo is None: start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None: end_dt = end_dt.replace(tzinfo=timezone.utc)

        success = calculate_accumulation(
            point_name=point_name,
            start_dt=start_dt,
            end_dt=end_dt,
            interval=interval,
            rate_variable=rate_variable,
            output_file_path=output_file_path,
            source=source,
            version=version,
        )

        if success and os.path.exists(output_file_path):
            logger.info(f"[Task Success] Accumulation calculation complete. Output: {output_file_path}")
            return {"status": "SUCCESS", "message": "Accumulation calculated.", "output_path": output_file_path}
        elif success:
            raise FileNotFoundError(f"Accumulation task reported success but output file missing: {output_file_path}")
        else:
            raise RuntimeError(f"Core accumulation calculation failed. Check core logs.")

    except Exception as e:
        logger.exception(f"[Task Error] Exception during accumulation task for point '{point_name}': {e}")
        raise


@huey.task(retries=0, retry_delay=30)
def run_create_animation(variable: str, elevation: float, start_dt_iso: str, end_dt_iso: str, output_filename: str,
                         plot_extent: Optional[list], include_watermark: bool, fps: Optional[int]):
    """
    Huey background task to create an animation.
    """
    logger.info(f"[Task Start] Creating animation: Var='{variable}', Elev={elevation}, Output: {output_filename}")
    try:
        start_dt = datetime.fromisoformat(start_dt_iso.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_dt_iso.replace('Z', '+00:00'))
        if start_dt.tzinfo is None: start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None: end_dt = end_dt.replace(tzinfo=timezone.utc)

        extent_tuple = tuple(plot_extent) if plot_extent else None

        success = create_animation(
            variable=variable,
            elevation=elevation,
            start_dt=start_dt,
            end_dt=end_dt,
            output_filename=output_filename,
            plot_extent=extent_tuple,
            include_watermark=include_watermark,
            fps=fps
        )

        if success and os.path.exists(output_filename):
            logger.info(f"[Task Success] Animation creation complete. Output: {output_filename}")
            return {"status": "SUCCESS", "message": "Animation created.", "output_path": output_filename}
        elif success:
            raise FileNotFoundError(f"Animation task reported success but output file missing: {output_filename}")
        else:
            raise RuntimeError(f"Core animation creation failed. Check core logs.")

    except Exception as e:
        logger.exception(f"[Task Error] Exception during animation task: {e}")
        raise

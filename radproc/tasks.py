# radproc/tasks.py
import logging
import os
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

# Import the Huey instance defined in huey_app.py
from .huey_config import huey

# Import the core logic function we want to run in the background
# (Make sure this function now returns Optional[str] - the csv_path or None)
from .core.analysis import generate_point_timeseries, calculate_accumulation
from .core.visualization.animator import create_animation

logger = logging.getLogger(__name__)

@huey.task(retries=2, retry_delay=15) # Decorator marks this as a Huey background task
def run_generate_point_timeseries(point_name: str, start_dt_iso: str, end_dt_iso: str, variable_override: str = None):
    """
    Huey background task to generate/update timeseries data for a point.

    Args:
        point_name: Name of the point.
        start_dt_iso: Start datetime in ISO format string (UTC assumed if no offset).
        end_dt_iso: End datetime in ISO format string (UTC assumed if no offset).
        variable_override: Optional variable name to override the point's default.

    Returns:
        A dictionary indicating success and the path to the output file.

    Raises:
        Exception: If the core logic fails or an unexpected error occurs,
                   allowing Huey to mark the task as FAILED.
    """
    logger.info(f"[Task Start] Generating timeseries for point '{point_name}', Range: {start_dt_iso} to {end_dt_iso}, Var: {variable_override or 'Default'}")

    try:
        # Convert ISO strings back to timezone-aware datetimes for the core function
        # Use fromisoformat which handles timezone offsets if present, otherwise naive
        start_dt = datetime.fromisoformat(start_dt_iso)
        end_dt = datetime.fromisoformat(end_dt_iso)

        # If parsed datetime is naive, assume it's UTC as per API contract
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
            logger.debug(f"Assumed UTC for naive start_dt: {start_dt_iso}")
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
            logger.debug(f"Assumed UTC for naive end_dt: {end_dt_iso}")

        # --- Call the Core Logic ---
        # This function should now return the csv_path on success, None on failure
        result_path = generate_point_timeseries(
            point_name=point_name,
            start_dt=start_dt,
            end_dt=end_dt,
            variable_override=variable_override
        )
        # -------------------------

        if result_path and os.path.exists(result_path):
            logger.info(f"[Task Success] Timeseries generation complete for '{point_name}'. Output: {result_path}")
            # Return a dictionary that the API can use
            return {
                "status": "SUCCESS",
                "message": f"Timeseries generation completed for {point_name}.",
                "output_path": result_path
            }
        elif result_path:
             # Core function returned a path, but it doesn't exist? Problem.
             logger.error(f"[Task Failure] Core function returned path '{result_path}' but it does not exist.")
             raise FileNotFoundError(f"Task completed but output file missing at {result_path}")
        else:
            # Core function returned None, indicating failure
            logger.error(f"[Task Failure] Timeseries generation failed for '{point_name}' (core function returned None). Check previous logs.")
            # Raise an exception so Huey marks the task as FAILED
            raise RuntimeError(f"Core timeseries generation failed for point '{point_name}'.")

    except Exception as e:
        # Catch any exception from parsing, core logic, or file checks
        logger.exception(f"[Task Error] Exception during timeseries task for point '{point_name}': {e}")
        # Re-raise the exception to ensure Huey marks the task as FAILED
        # Huey will store the exception details.
        raise

@huey.task(retries=1, retry_delay=10) # Fewer retries might be appropriate here
def run_calculate_accumulation(point_name: str, start_dt_iso: str, end_dt_iso: str, interval: str, rate_variable: str, output_file_path: str):
    """
    Huey background task to calculate precipitation accumulation.
    """
    logger.info(f"[Task Start] Calculating accumulation for point '{point_name}', Interval: {interval}, Output: {output_file_path}")
    try:
        # Parse datetimes
        start_dt = datetime.fromisoformat(start_dt_iso)
        end_dt = datetime.fromisoformat(end_dt_iso)
        if start_dt.tzinfo is None: start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None: end_dt = end_dt.replace(tzinfo=timezone.utc)

        # Optional: Validate interval format again within the task?
        try:
            pd.Timedelta(interval) # Basic check
        except ValueError:
             logger.error(f"Invalid interval format received in task: '{interval}'")
             raise ValueError(f"Invalid interval format: {interval}")

        # Call the core function
        success = calculate_accumulation(
            point_name=point_name,
            start_dt=start_dt,
            end_dt=end_dt,
            interval=interval,
            rate_variable=rate_variable,
            output_file_path=output_file_path
        )

        if success and os.path.exists(output_file_path):
            logger.info(f"[Task Success] Accumulation calculation complete. Output: {output_file_path}")
            return {"status": "SUCCESS", "message": "Accumulation calculated.", "output_path": output_file_path}
        elif success:
             # Core function reported success but file doesn't exist? Problem.
             raise FileNotFoundError(f"Accumulation task reported success but output file missing: {output_file_path}")
        else:
             # Core function reported failure
             raise RuntimeError(f"Core accumulation calculation failed for point '{point_name}'. Check core logs.")

    except Exception as e:
        logger.exception(f"[Task Error] Exception during accumulation task for point '{point_name}': {e}")
        raise

@huey.task(retries=0, retry_delay=30) # Animations can be long, maybe no retries or long delay?
def run_create_animation(variable: str, elevation: float, start_dt_iso: str, end_dt_iso: str, output_filename: str, plot_extent: Optional[list], include_watermark: bool, fps: Optional[int]):
    """
    Huey background task to create an animation.
    """
    logger.info(f"[Task Start] Creating animation: Var='{variable}', Elev={elevation}, Output: {output_filename}")
    try:
        # Parse datetimes
        start_dt = datetime.fromisoformat(start_dt_iso)
        end_dt = datetime.fromisoformat(end_dt_iso)
        if start_dt.tzinfo is None: start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None: end_dt = end_dt.replace(tzinfo=timezone.utc)

        # Convert extent list back to tuple if not None
        extent_tuple = tuple(plot_extent) if plot_extent else None

        # Call the core function
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
             raise RuntimeError(f"Core animation creation failed for variable '{variable}'. Check core logs.")

    except Exception as e:
        logger.exception(f"[Task Error] Exception during animation task for var '{variable}', output '{output_filename}': {e}")
        raise
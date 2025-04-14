# radproc/api/dependencies.py
import logging
import os
from typing import Dict, Any
from fastapi import Depends, HTTPException, status

# Import necessary core config functions
from radproc.core.config import load_config, get_config, get_setting

logger = logging.getLogger(__name__)
CONFIG_LOAD_ATTEMPTED = False

async def get_core_config() -> Dict[str, Any]:
    """
    FastAPI dependency to load and provide the core application configuration.
    Ensures config is loaded only once per application lifecycle.
    """
    global CONFIG_LOAD_ATTEMPTED
    config = get_config() # Check if already loaded by core module import

    if config is None or not CONFIG_LOAD_ATTEMPTED:
        try:
            logger.info("API dependency: Loading core configuration...")
            load_config() # Attempt to load
            config = get_config()
            CONFIG_LOAD_ATTEMPTED = True # Mark attempt
            if config is None: # Check again after loading attempt
                 raise ValueError("Core configuration failed to load.")
            logger.info("API dependency: Core configuration loaded successfully.")
        except Exception as e:
            logger.critical(f"API dependency: Failed to load core configuration: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Core application configuration could not be loaded.",
            )
    # Return the hopefully loaded config
    return config

# Example dependency to get a specific required setting
async def get_realtime_image_dir() -> str:
    """Dependency to get the configured realtime_image_dir, raising error if not set."""
    realtime_dir = get_setting('app.realtime_image_dir') # Use injected config
    if not realtime_dir:
        logger.error("API dependency error: 'app.realtime_image_dir' is not configured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Realtime image directory not configured on the server."
        )
    # Ensure it's an absolute path? Or handle relative paths carefully.
    # For now, assume the config value is usable.
    if not os.path.isdir(realtime_dir):
         logger.warning(f"Configured realtime_image_dir does not exist: {realtime_dir}")
         # Depending on endpoint, might still want to proceed or raise 500/404
         # Let's allow proceeding, endpoint should handle non-existence
    return realtime_dir

async def get_timeseries_dir() -> str:
    """Dependency to get the configured timeseries_dir, raising error if not set."""
    timeseries_dir = get_setting('app.timeseries_dir')
    if not timeseries_dir:
        logger.error("API dependency error: 'app.timeseries_dir' is not configured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Timeseries directory not configured on the server."
        )
    return timeseries_dir

async def get_images_dir() -> str:
    """Dependency to get the configured images_dir, raising error if not set."""
    images_dir = get_setting('app.images_dir')
    if not images_dir:
        logger.error("API dependency error: 'app.images_dir' is not configured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Historical images directory not configured on the server."
        )
    # Optional: Check if it exists, but maybe better handled by the endpoint logic
    # if not os.path.isdir(images_dir):
    #     logger.error(f"Configured images_dir does not exist or is not a directory: {images_dir}")
    #     raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Cannot access historical images directory.")
    return images_dir

async def get_animation_output_dir() -> str:
    """Dependency to get the configured animation_output_dir, raising error if not set."""
    anim_dir = get_setting('app.animation_output_dir')
    if not anim_dir:
        logger.error("API dependency error: 'app.animation_output_dir' is not configured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Animation output directory not configured on the server."
        )
    # Optionally ensure the directory exists here, or let the endpoint handle it
    # try:
    #     os.makedirs(anim_dir, exist_ok=True)
    # except OSError as e:
    #     logger.error(f"Failed to ensure animation output directory exists: {anim_dir}, Error: {e}")
    #     raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Cannot access animation output directory.")
    return anim_dir

async def get_api_job_output_dir() -> str:
    """Dependency to get and validate the configured API job output directory."""
    job_output_dir = get_setting('app.api_job_output_dir') # Get from config
    if not job_output_dir:
        logger.error("API dependency error: 'app.api_job_output_dir' is not configured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API job output directory not configured on the server."
        )

    # Resolve to absolute path (assuming relative paths are relative to project root or CWD)
    # This might need adjustment based on how your paths are typically handled
    abs_job_output_dir = os.path.abspath(job_output_dir)

    # Ensure the directory exists and is writable (best effort check)
    try:
        os.makedirs(abs_job_output_dir, exist_ok=True)
        # Basic write test (create and delete a temp file)
        test_file = os.path.join(abs_job_output_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except OSError as e:
        logger.error(f"API job output directory '{abs_job_output_dir}' is not accessible or writable: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"API job output directory is not accessible/writable.",
        )
    except Exception as e:
         logger.error(f"Unexpected error checking API job output directory '{abs_job_output_dir}': {e}")
         raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error accessing API job output directory.")

    logger.debug(f"Using API job output directory: {abs_job_output_dir}")
    return abs_job_output_dir
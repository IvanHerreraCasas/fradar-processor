# radproc/api/routers/plots.py
import os
import re
import asyncio
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from fastapi.responses import FileResponse, StreamingResponse
from typing import AsyncGenerator, Optional, List
from datetime import datetime, timezone, timedelta

from radproc.api.state import image_update_queue
from radproc.api.dependencies import get_realtime_image_dir, get_images_dir
from ..schemas.plots import PlotFramesResponse, PlotFrameInfo

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/plots",
    tags=["Plots"],
)

# --- Filename Validation Pattern ---
# Allows letters, numbers, underscore, hyphen. Starts with 'realtime_'. Ends with '.png'.
REALTIME_FILENAME_PATTERN = re.compile(r"^realtime_[a-zA-Z0-9_-]+_\d{3,4}\.png$")
VALID_VARIABLE_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
# Pattern for expected datetime string in URL path
DATETIME_PATH_PATTERN = re.compile(r"^\d{8}_\d{4}$") # YYYYMMDD_HHMM
DATETIME_PATH_FORMAT = "%Y%m%d_%H%M" # Format string for parsing

@router.get(
    "/frames",
    response_model=PlotFramesResponse,
    summary="List available plot frames for a sequence",
    responses={
        200: {"description": "Successfully retrieved list of available frames."},
        400: {"description": "Invalid input parameters (e.g., bad variable format, invalid date range)."},
        422: {"description": "Validation error for query parameters (e.g., bad datetime format)."},
        500: {"description": "Server configuration error (e.g., images_dir not set)."},
    }
)
async def list_plot_frames(
    variable: str = Query(..., description="Radar variable name (e.g., 'RATE')."),
    elevation: float = Query(..., description="Elevation angle in degrees (e.g., 0.5)."),
    start_dt: datetime = Query(..., description="Start datetime for sequence range (UTC ISO format recommended)."),
    end_dt: datetime = Query(..., description="End datetime for sequence range (UTC ISO format recommended)."),
    images_dir: str = Depends(get_images_dir) # Dependency for base image path
):
    """
    Retrieves a list of identifiers (datetime strings) for existing plot image
    frames that match the specified variable, elevation, and time range.

    This list can be used by clients to fetch individual frames via the
    `/plots/historical/{variable}/{elevation}/{datetime_str}` endpoint.
    Frames are sorted chronologically.
    """
    logger.info(f"Request for plot frames: Var='{variable}', Elev='{elevation}', Range='{start_dt}' to '{end_dt}'")

    # --- Input Validation ---
    if not VALID_VARIABLE_PATTERN.match(variable):
        logger.warning(f"Invalid variable name format requested: {variable}")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid variable name format.")

    # Ensure datetimes from query params are timezone-aware (assume UTC if naive)
    # FastAPI/Pydantic usually handle parsing ISO strings. If naive, make UTC.
    start_dt_utc = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
    end_dt_utc = end_dt if end_dt.tzinfo else end_dt.replace(tzinfo=timezone.utc)

    if start_dt_utc >= end_dt_utc:
        logger.warning(f"Invalid date range: start_dt ({start_dt_utc}) >= end_dt ({end_dt_utc})")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Start datetime must be before end datetime.")

    # --- Calculate Elevation Code ---
    try:
        # Use the same formatting as the plotter/historical endpoint expects
        elevation_code = f"{int(round(elevation * 100)):03d}"
        logger.debug(f"Calculated elevation code: {elevation_code}")
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to format elevation {elevation}: {e}")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid elevation value.")

    # --- Prepare for Search ---
    found_frames_info: List[PlotFrameInfo] = []
    # Regex to extract datetime from valid filenames matching the request
    # Example: RATE_005_20231027_1005.png -> extracts "20231027_1005"
    filename_pattern = re.compile(f"^{re.escape(variable)}_{re.escape(elevation_code)}_(\\d{{8}}_\\d{{4}})\\.png$")
    dt_parse_format = "%Y%m%d_%H%M" # Format expected in filename

    # --- Iterate through dates and search for files ---
    try:
        current_date = start_dt_utc.date()
        end_date = end_dt_utc.date()

        while current_date <= end_date:
            date_str_ymd = current_date.strftime("%Y%m%d")
            target_dir = os.path.join(images_dir, variable, date_str_ymd, elevation_code)

            if os.path.isdir(target_dir):
                logger.debug(f"Scanning directory: {target_dir}")
                for filename in os.listdir(target_dir):
                    match = filename_pattern.match(filename)
                    if match:
                        dt_str_ymd_hm = match.group(1) # Extracted "YYYYMMDD_HHMM"
                        try:
                            # Parse filename datetime and make it UTC aware
                            frame_dt_naive = datetime.strptime(dt_str_ymd_hm, dt_parse_format)
                            frame_dt_utc = frame_dt_naive.replace(tzinfo=timezone.utc)

                            # Check if this frame's time is within the requested range
                            if start_dt_utc <= frame_dt_utc <= end_dt_utc:
                                found_frames_info.append(PlotFrameInfo(datetime_str=dt_str_ymd_hm))

                        except ValueError:
                            logger.warning(f"Could not parse datetime from valid filename pattern: {filename}")
                        except Exception as parse_err:
                            logger.error(f"Unexpected error processing filename {filename}: {parse_err}")
            # else: # Optional: log if a date directory is skipped
                 # logger.debug(f"Skipping non-existent directory: {target_dir}")

            current_date += timedelta(days=1) # Move to the next day

    except Exception as e:
         logger.exception(f"Error occurred while searching for plot frames: {e}")
         # Don't expose internal error details generally
         raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to search for plot frames.")

    # --- Sort and Return ---
    # Sort frames chronologically by the datetime string
    found_frames_info.sort(key=lambda x: x.datetime_str)

    logger.info(f"Found {len(found_frames_info)} matching plot frames for request.")
    return PlotFramesResponse(frames=found_frames_info)

@router.get(
    "/historical/{variable}/{elevation}/{datetime_str}",
    summary="Get a specific historical plot image by variable, elevation, and datetime",
    responses={
        200: {"content": {"image/png": {}}, "description": "Successful Response"},
        400: {"description": "Invalid variable, elevation, or datetime format, or path traversal attempt"},
        404: {"description": "Historical plot image not found for the specified parameters"},
        500: {"description": "Server configuration error (image directory not set)"},
    }
)
async def get_historical_plot(
    variable: str,
    elevation: float,
    datetime_str: str, # Expect YYYYMMDD_HHMM
    images_dir: str = Depends(get_images_dir) # <<< Use new dependency
):
    """
    Serves a specific historical plot image identified by its variable,
    elevation angle, and exact datetime string (YYYYMMDD_HHMM).

    - **variable**: Name of the radar variable (e.g., RATE, DBZH).
    - **elevation**: Elevation angle in degrees (e.g., 0.5, 1.5).
    - **datetime_str**: The exact datetime the plot represents, in 'YYYYMMDD_HHMM' format (UTC).
    """
    logger.info(f"Request for historical plot: Var='{variable}', Elev='{elevation}', Time='{datetime_str}'")

    # --- Input Validation ---
    if not VALID_VARIABLE_PATTERN.match(variable):
        logger.warning(f"Invalid variable name format requested: {variable}")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid variable name format.")

    if not DATETIME_PATH_PATTERN.match(datetime_str):
        logger.warning(f"Invalid datetime format requested: {datetime_str}")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Invalid datetime format. Expected '{DATETIME_PATH_FORMAT}'.")

    # Parse datetime string to validate and extract components
    try:
        # Assume the datetime_str represents UTC, parse as naive then make aware
        dt_naive = datetime.strptime(datetime_str, DATETIME_PATH_FORMAT)
        dt_utc = dt_naive.replace(tzinfo=timezone.utc) # Add UTC timezone info
    except ValueError:
        # This shouldn't happen if regex matches, but safety check
        logger.error(f"Could not parse datetime string '{datetime_str}' after regex match.")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Invalid datetime value '{datetime_str}'.")

    # --- Construct Path Components ---
    try:
        # Use the exact elevation code formatting from core processor
        elevation_code = f"{int(round(elevation * 100)):03d}" # Includes zero-padding

        # Extract date part (YYYYMMDD) for directory structure
        date_str_ymd = dt_utc.strftime("%Y%m%d")

        # Datetime part for filename (YYYYMMDD_HHMM) - already have this as datetime_str
        datetime_file_str = datetime_str

        # Construct the expected filename
        filename = f"{variable}_{elevation_code}_{datetime_file_str}.png"

        # Construct the full expected path
        expected_path = os.path.join(images_dir, variable, date_str_ymd, elevation_code, filename)
        logger.debug(f"Constructed expected historical path: {expected_path}")

    except Exception as format_err:
        logger.error(f"Error formatting path components for historical plot request: {format_err}", exc_info=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Error processing request parameters.")

    # --- Security Check & Path Validation ---
    try:
        base_dir_abs = os.path.abspath(images_dir)
        filepath_abs = os.path.abspath(expected_path)

        # Check if constructed path is within the allowed base directory
        if not filepath_abs.startswith(base_dir_abs + os.sep) and filepath_abs != base_dir_abs:
            logger.error(f"Path traversal attempt detected for historical plot: Request params resulted in path '{filepath_abs}' outside base directory '{base_dir_abs}'.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request parameters.",
            )
        filepath = filepath_abs

    except Exception as path_err:
        logger.error(f"Error constructing/validating path for historical plot: {path_err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request causing path error.",
        )

    # --- Check Existence ---
    if not os.path.isfile(filepath):
        logger.warning(f"Historical plot file not found: {filepath}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Historical plot not found for Var='{variable}', Elev='{elevation}', Time='{datetime_str}'.",
        )

    # --- Return File ---
    logger.info(f"Serving historical plot: {filepath}")
    return FileResponse(filepath, media_type='image/png', filename=filename)

@router.get(
    "/realtime/{variable}/{elevation}",
    summary="Get latest plot image for a specific variable and elevation",
    responses={
        200: {"content": {"image/png": {}}, "description": "Successful Response"},
        400: {"description": "Invalid variable name or elevation format, or path traversal attempt"},
        404: {"description": "Realtime plot image not found for this variable/elevation"},
        500: {"description": "Server configuration error (realtime directory not set)"},
    }
)
async def get_realtime_plot(
    variable: str,    # <<< Path parameter
    elevation: float, # <<< Path parameter (FastAPI handles float conversion)
    realtime_dir: str = Depends(get_realtime_image_dir) # Use dependency
):
    """
    Serves the latest generated plot image for a given variable and elevation.

    - **variable**: Name of the radar variable (e.g., RATE, DBZH).
    - **elevation**: Elevation angle in degrees (e.g., 0.5, 1.5).
    """
    logger.info(f"Request received for realtime plot: Variable='{variable}', Elevation='{elevation}'")
    # Elevation is validated as float by FastAPI path parameter type hint

    # --- Construct Filename ---
    try:
        # Format elevation code (e.g., 2.5 -> 250, 12.5 -> 1250) - Handle potential precision issues
        elevation_code = f"{int(round(elevation * 100))}" # Multiply by 100 for two decimal place
        filename = f"realtime_{variable}_{elevation_code}.png"
        logger.debug(f"Constructed filename: {filename}")
    except (ValueError, TypeError):
        logger.error(f"Error formatting elevation code for elevation: {elevation}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid elevation value for formatting.",
        )

    # --- Security Check & Path Construction ---
    try:
        base_dir_abs = os.path.abspath(realtime_dir)
        filepath_abs = os.path.abspath(os.path.join(base_dir_abs, filename))

        if not filepath_abs.startswith(base_dir_abs + os.sep) and filepath_abs != base_dir_abs:
            logger.error(f"Path traversal attempt detected: Variable/Elevation '{variable}/{elevation}' resulted in path '{filepath_abs}' outside base directory '{base_dir_abs}'.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid variable/elevation combination.", # User doesn't need to know details
            )
        filepath = filepath_abs

    except Exception as path_err:
        logger.error(f"Error constructing path for Variable/Elevation '{variable}/{elevation}': {path_err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid variable/elevation causing path error.",
        )

    # --- Check Existence ---
    if not os.path.isfile(filepath):
        logger.warning(f"Realtime plot file not found or is not a file: {filepath}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Realtime plot for variable '{variable}' at elevation '{elevation}' not found.",
        )

    # --- Return File ---
    logger.info(f"Serving realtime plot: {filepath}")
    # Filename in response can be the constructed one, or a more descriptive one
    return FileResponse(filepath, media_type='image/png', filename=filename)

# +++ SSE Endpoint +++
@router.get("/stream/updates", summary="Stream real-time plot update notifications (SSE)")
async def stream_plot_updates(request: Request): # Inject Request object
    """
    Establishes a Server-Sent Events connection to notify clients when
    a realtime plot image is updated.
    """
    logger.info("SSE client connected.")

    async def event_generator() -> AsyncGenerator[str, None]:
        # Send an initial connected message? Optional.
        # yield f"event: connected\ndata: You are connected\n\n"
        while True:
            try:
                # Check if client disconnected before waiting
                if await request.is_disconnected():
                     logger.info("SSE client disconnected before getting queue item.")
                     break

                # Wait for a filename from the queue (with timeout for heartbeats)
                try:
                    filename = await asyncio.wait_for(image_update_queue.get(), timeout=25.0) # Wait up to 25s
                    # Format and yield the SSE message
                    sse_data = f'{{"updated_image": "{filename}"}}' # Simple JSON payload
                    yield f"event: plot_update\ndata: {sse_data}\n\n"
                    image_update_queue.task_done()
                    logger.debug(f"SSE event sent for: {filename}")
                except asyncio.TimeoutError:
                    # No update received, send a heartbeat comment to keep connection alive
                    yield ": heartbeat\n\n"
                    logger.debug("SSE heartbeat sent.")

            except asyncio.CancelledError:
                 logger.info("SSE generator task cancelled (client likely disconnected).")
                 break # Exit loop on cancellation
            except Exception as e:
                 # Handle potential errors during queue get or yielding (e.g., client disconnect mid-yield)
                 logger.error(f"Error in SSE generator: {e}", exc_info=True)
                 break # Exit loop on other errors

    # Return the streaming response using the async generator
    # Add headers to prevent caching and keep connection alive
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no", # Often needed for Nginx buffering issues
    }
    return StreamingResponse(event_generator(), headers=headers)
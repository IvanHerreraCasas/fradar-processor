# radproc/api/routers/jobs.py
import io
import os
import logging
import pandas as pd
from fastapi import APIRouter, HTTPException, status, Body, Depends, Query, Response
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from datetime import datetime, timezone
from huey.exceptions import TaskException # Import Huey's base exception
from mimetypes import guess_type

from typing import Optional, Literal, Any, Tuple

from ...huey_config import huey
from ...tasks import run_generate_point_timeseries, run_calculate_accumulation, run_create_animation
from ..schemas.jobs import (
    TimeseriesJobRequest,
    AccumulationJobRequest,
    AnimationJobRequest,
    JobSubmissionResponse,
    JobStatusResponse,
    JobStatus,
)
from ..dependencies import get_core_config
from ...core.config import get_point_config, get_setting
from ...core.utils.csv_handler import read_timeseries_csv

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/jobs",
    tags=["Background Jobs"],
)

# --- POST /jobs/timeseries ---
@router.post(
    "/timeseries",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a job to generate/update historical timeseries data",
)
async def queue_timeseries_job(
    request_body: TimeseriesJobRequest = Body(...),
    _config: dict = Depends(get_core_config),
):
    point_name = request_body.point_name
    start_dt = request_body.start_dt
    end_dt = request_body.end_dt
    variable = request_body.variable

    logger.info(
        f"Received request to queue timeseries job for point: '{point_name}', "
        f"Range: {start_dt.isoformat()} to {end_dt.isoformat()}, Var: {variable or 'Default'}"
    )

    # Optional: Early Validation
    if not get_point_config(point_name):
        logger.warning(f"Timeseries job request failed: Point '{point_name}' not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Point '{point_name}' not found in configuration.",
        )
    if start_dt >= end_dt:
        logger.warning(f"Timeseries job request failed: Start datetime must be before end datetime.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Start datetime must be before end datetime.",
        )

    try:
        task = run_generate_point_timeseries(
            request_body.point_name,
            start_dt.isoformat(),
            end_dt.isoformat(),
            variable,
        )
        task_id = task.id
        logger.info(f"Timeseries job queued successfully for point '{point_name}'. Task ID: {task_id}")
        return JobSubmissionResponse(
            message="Timeseries generation job queued.",
            job_type="timeseries",
            task_id=task_id,
        )
    except Exception as e:
        logger.exception(f"Failed to queue timeseries job for point '{point_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue job: An internal error occurred.",
        )

# --- POST /accumulation endpoint ---
@router.post(
    "/accumulation",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a job to calculate precipitation accumulation",
)
async def queue_accumulation_job(
    request_body: AccumulationJobRequest = Body(...),
    _config: dict = Depends(get_core_config),
):
    """
    Submits a background job to calculate accumulated precipitation for a point.
    """
    point_name = request_body.point_name
    start_dt = request_body.start_dt
    end_dt = request_body.end_dt
    interval = request_body.interval
    rate_variable = request_body.rate_variable or "RATE" # Use default if None
    output_file = request_body.output_file

    logger.info(f"Received request to queue accumulation job for point: '{point_name}', Interval: {interval}")

    # --- Validation ---
    if not get_point_config(point_name):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Point '{point_name}' not found.")
    if start_dt >= end_dt:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Start datetime must be before end datetime.")
    # Interval format validated by Pydantic

    # Determine default output path if not provided
    if not output_file:
        timeseries_dir = get_setting('app.timeseries_dir')
        if not timeseries_dir:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Timeseries directory not configured for default output path.")
        safe_interval = interval.replace(':','-').replace('/','_') # Make interval filename-safe
        default_filename = f"{point_name}_{rate_variable}_{safe_interval}_acc.csv"
        output_file = os.path.join(timeseries_dir, default_filename)
        logger.info(f"Using default output file for accumulation job: {output_file}")

    # Ensure output directory exists before queueing (task might fail otherwise)
    try:
         output_dir = os.path.dirname(output_file)
         if output_dir: os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
         logger.error(f"Cannot create output directory {output_dir} for accumulation job: {e}")
         raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Cannot create output directory.")

    # --- Enqueue Task ---
    try:
        task = run_calculate_accumulation(
            point_name,
            start_dt.isoformat(),
            end_dt.isoformat(),
            interval,
            rate_variable,
            output_file # Pass the determined output path
        )
        task_id = task.id
        logger.info(f"Accumulation job queued successfully for point '{point_name}'. Task ID: {task_id}")
        return JobSubmissionResponse(
            message="Accumulation calculation job queued.",
            job_type="accumulation",
            task_id=task_id,
        )
    except Exception as e:
        logger.exception(f"Failed to queue accumulation job for point '{point_name}': {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to queue job.")

# --- POST /animation endpoint ---
@router.post(
    "/animation",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a job to generate an animation",
)
async def queue_animation_job(
    request_body: AnimationJobRequest = Body(...),
    _config: dict = Depends(get_core_config),
):
    """
    Submits a background job to generate an animation from plot images.
    """
    variable=request_body.variable
    elevation=request_body.elevation
    start_dt=request_body.start_dt
    end_dt=request_body.end_dt
    output_file=request_body.output_file
    extent=request_body.extent # Will be tuple or None
    no_watermark=request_body.no_watermark or False # Default to False
    fps=request_body.fps

    logger.info(f"Received request to queue animation job: Var='{variable}', Elev={elevation}, Output='{output_file}'")

    # --- Validation ---
    if start_dt >= end_dt:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Start datetime must be before end datetime.")
    # Output file extension validated loosely by Pydantic

    # Ensure output directory exists before queueing
    try:
         output_dir = os.path.dirname(output_file)
         if output_dir: os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
         logger.error(f"Cannot create output directory {output_dir} for animation job: {e}")
         raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Cannot create output directory.")

    # --- Enqueue Task ---
    try:
        # Convert extent tuple to list for serialization if not None
        extent_list = list(extent) if extent else None

        task = run_create_animation(
            variable,
            elevation,
            start_dt.isoformat(),
            end_dt.isoformat(),
            output_file,
            extent_list, # Pass list
            not no_watermark, # Pass include_watermark flag
            fps
        )
        task_id = task.id
        logger.info(f"Animation job queued successfully for var '{variable}'. Task ID: {task_id}")
        return JobSubmissionResponse(
            message="Animation generation job queued.",
            job_type="animation",
            task_id=task_id,
        )
    except Exception as e:
        logger.exception(f"Failed to queue animation job for var '{variable}': {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to queue job.")

@router.get(
    "/{task_id}/status",
    response_model=JobStatusResponse,
    summary="Get the status of a background job",
    responses={
        200: {"description": "Status retrieved successfully."},
        404: {"description": "Job ID not found."},
        500: {"description": "Error checking job status."},
    }
)
async def get_job_status(task_id: str):
    """    
    Retrieves the current status for a given job Task ID provided by Huey.

    Possible statuses:
    - **SUCCESS**: Task completed successfully. Result is available.
    - **FAILURE**: Task failed during execution. Error info is available.
    - **REVOKED**: Task was cancelled before execution could complete.
    - **PENDING**: Task is waiting in the queue, currently running, OR the Task ID is unknown/expired (Huey does not definitively distinguish these via this API).

    Note: If a Task ID is completely invalid or has expired from the result store,
    this endpoint will likely return a PENDING status. A 404 may occur only if
    the status check itself encounters certain errors.
    """
    logger.debug(f"Request received for status of job ID: {task_id}")

    task_status_str = "UNKNOWN"
    result_obj: Any = None
    error_info: Optional[str] = None
    last_update: Optional[datetime] = None # Huey doesn't easily expose this

    try:
        # --- Attempt to retrieve the result or status ---
        # huey.result() might return:
        # - The actual result value if SUCCESSFUL and finished.
        # - An Exception object if FAILED and finished.
        # - None if PENDING, RUNNING, or potentially UNKNOWN ID.
        retrieved_value = None
        task_failed_exception = None
        logger.debug(f"huey.result() for {task_id} returned: {type(retrieved_value)} - {retrieved_value}")
        
        # --- Try to get the result, catching TaskException specifically ---
        try:
            retrieved_value = huey.result(task_id, preserve=True)
            logger.debug(f"huey.result() for {task_id} returned type: {type(retrieved_value)}")

        except TaskException as te:
            # Huey signals failure by raising TaskException when result() is called
            task_status_str = "FAILURE"
            task_failed_exception = te # Store the TaskException
            logger.warning(f"Job ID '{task_id}' indicated failure via TaskException.")
        except Exception as e:
             logger.exception(f"Unexpected error calling huey.result() for job ID '{task_id}': {e}")
             # If result() itself fails unexpectedly, treat as 500 (or maybe 404?)
             raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Error communicating with task queue backend.")


        # --- Determine status based on outcome ---
        if task_status_str == "FAILURE":
             # Extract original error from TaskException metadata or the exception itself
             original_error = getattr(task_failed_exception, 'metadata', task_failed_exception)
             error_info = f"{type(original_error).__name__}: {str(original_error)}"
             logger.warning(f"Job ID '{task_id}' failure detail: {error_info}")
             result_obj = None

        elif retrieved_value is None:
             # Result not ready or task unknown. Check revoked status.
             if huey.is_revoked(task_id):
                 task_status_str = "REVOKED"
                 logger.info(f"Job ID '{task_id}' is revoked.")
             else:
                 # Assume PENDING (covers pending, running, or unknown/expired ID)
                 task_status_str = "PENDING"
                 logger.debug(f"Job ID '{task_id}' is pending, running, or unknown (result not ready).")

        # Safety check: If result() returned an Exception that wasn't TaskException
        elif isinstance(retrieved_value, Exception):
             task_status_str = "FAILURE"
             original_error = retrieved_value
             error_info = f"{type(original_error).__name__}: {str(original_error)}"
             logger.warning(f"Job ID '{task_id}' reported failure (retrieved direct Exception): {error_info}")
             result_obj = None
        else:
             # Success - retrieved_value is the actual result
             task_status_str = "SUCCESS"
             result_obj = retrieved_value
             if not isinstance(result_obj, (str, int, float, bool, list, dict, type(None))):
                 result_obj = str(result_obj) # Sanitize
             logger.debug(f"Job ID '{task_id}' finished successfully.")

    except HTTPException:
        raise # Re-raise FastAPI exceptions
    except Exception as e:
        # Catch-all for unexpected errors during the status check logic itself
        logger.exception(f"General error retrieving status for job ID '{task_id}': {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to retrieve job status.")

    # --- Construct the Response ---
    # If status is still UNKNOWN after checks, treat as Not Found (though less likely now)
    if task_status_str == "UNKNOWN":
         logger.error(f"Could not determine final status for Task ID '{task_id}', treating as Not Found.")
         raise HTTPException(status.HTTP_404_NOT_FOUND, f"Could not determine status for Job ID '{task_id}'.")

    status_details = JobStatus(
        status=task_status_str,
        last_update=last_update,
        result=result_obj if task_status_str == "SUCCESS" else None,
        error_info=error_info if task_status_str == "FAILURE" else None,
    )

    # TODO: Store/retrieve job type associated with task_id
    job_type_placeholder = "unknown"

    return JobStatusResponse(
        task_id=task_id,
        job_type=job_type_placeholder,
        status_details=status_details
    )

# --- You will need similar logic adjustment in get_timeseries_job_data ---
# --- Refactor GET /jobs/timeseries/{task_id}/data (Adjusted Logic) ---
@router.get(
    "/timeseries/{task_id}/data",
    summary="Retrieve the data generated by a completed timeseries job",
    # ... (responses remain the same) ...
)
async def get_timeseries_job_data(
    task_id: str,
    format: Literal["json", "csv"] = Query("json", description="Output data format"),
    start_dt: Optional[datetime] = Query(None, description="Filter data >= this datetime (UTC ISO format recommended)"),
    end_dt: Optional[datetime] = Query(None, description="Filter data <= this datetime (UTC ISO format recommended)"),
    interval: Optional[str] = Query(None, description="Resample data to this Pandas interval (e.g., '1H', '15min')"),
):
    logger.info(f"Request for timeseries data from job ID: {task_id}")
    job_status, job_result_payload, error_info = await _get_job_outcome(task_id) # Use helper

    if job_status == "PENDING":
        return Response(status_code=status.HTTP_202_ACCEPTED, content=f"Job {task_id} is pending.")
    elif job_status == "FAILURE":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Job {task_id} failed: {error_info}")
    elif job_status == "REVOKED":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Job {task_id} was revoked.")
    elif job_status == "SUCCESS":
        # --- Process SUCCESS ---
        try:
            if not isinstance(job_result_payload, dict):
                 raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Job completed with unexpected result format.")
            output_path = job_result_payload.get("output_path")
            if not output_path:
                 raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Job completed but output path is missing.")
            if not os.path.isfile(output_path):
                 raise HTTPException(status.HTTP_404_NOT_FOUND, f"Generated data file not found for job {task_id}.")

            # Read, Filter, Resample, Return (same logic as before)
            df = read_timeseries_csv(output_path)
            # ... filtering logic ...
            filter_start_dt = start_dt.replace(tzinfo=timezone.utc) if start_dt and start_dt.tzinfo is None else start_dt
            filter_end_dt = end_dt.replace(tzinfo=timezone.utc) if end_dt and end_dt.tzinfo is None else end_dt
            if not df.empty:
                if filter_start_dt: df = df[df["timestamp"] >= filter_start_dt]
                if filter_end_dt: df = df[df["timestamp"] <= filter_end_dt]
            # ... resampling logic ...
            if interval and not df.empty:
                try:
                    df_indexed = df.set_index("timestamp")
                    df_resampled = df_indexed.resample(interval).mean()
                    df = df_resampled.reset_index()
                except ValueError as resample_err:
                     raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid resampling interval: {interval}")
                except Exception as resample_err:
                     logger.error(f"Error resampling data for job {task_id}: {resample_err}", exc_info=True)
                     raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to resample data.")

            # ... return formatting logic (CSV/JSON) ...
            if format == "csv":
                 # ... CSV StreamingResponse ...
                 stream = io.StringIO()
                 df_out = df.copy()
                 if "timestamp" in df_out.columns and pd.api.types.is_datetime64_any_dtype(df_out["timestamp"]):
                      df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                 df_out.to_csv(stream, index=False)
                 content = stream.getvalue()
                 stream.close()
                 return Response(content=content, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="job_{task_id}_data.csv"'})
            else: # JSON
                 # ... JSON JSONResponse ...
                 if df.empty: return JSONResponse(content=[])
                 df_out = df.copy()
                 if "timestamp" in df_out.columns and pd.api.types.is_datetime64_any_dtype(df_out["timestamp"]):
                      df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                 df_out = df_out.fillna(value=pd.NA).replace({pd.NA: None})
                 json_data = df_out.to_dict(orient="records")
                 return JSONResponse(content=json_data)

        except HTTPException: raise
        except FileNotFoundError as e:
             logger.error(f"Generated file missing during data read for job {task_id}: {e}")
             raise HTTPException(status.HTTP_404_NOT_FOUND, "Generated data file could not be read.")
        except Exception as e:
             logger.exception(f"Failed to read or process data file for job {task_id}: {e}")
             raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Error processing generated data.")
    else:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Unexpected job status '{job_status}' encountered.")


@router.get(
    "/accumulation/{task_id}/data",
    summary="Retrieve the CSV data from a completed accumulation job",
    response_class=Response, # Generic response class for dynamic content type
    responses={
        200: {"content": {"text/csv": {}}, "description": "CSV data retrieved successfully."},
        202: {"description": "Job is still pending or running."},
        400: {"description": "Job failed or was revoked."},
        404: {"description": "Job ID not found or generated data file missing."},
        500: {"description": "Internal server error."},
    },
)
async def get_accumulation_job_data(task_id: str):
    logger.info(f"Request for data from accumulation job ID: {task_id}")
    job_status, job_result_payload, error_info = await _get_job_outcome(task_id) # Use helper

    if job_status == "PENDING": # ... (return 202) ...
        return Response(status_code=status.HTTP_202_ACCEPTED, content=f"Job {task_id} is pending.")
    elif job_status == "FAILURE": # ... (return 400) ...
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Job {task_id} failed: {error_info}")
    elif job_status == "REVOKED": # ... (return 400) ...
         raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Job {task_id} was revoked.")
    elif job_status == "SUCCESS": # ... (process SUCCESS) ...
        try:
            if not isinstance(job_result_payload, dict): # ... (check result format) ...
                 raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Job completed with unexpected result format.")
            output_path = job_result_payload.get("output_path") # ... (get output path) ...
            if not output_path: # ... (check path exists in result) ...
                 raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Job completed but output path is missing.")
            if not os.path.isfile(output_path): # ... (check file exists on disk) ...
                 raise HTTPException(status.HTTP_404_NOT_FOUND, f"Generated accumulation file not found for job {task_id}.")

            filename = os.path.basename(output_path)
            return FileResponse(path=output_path, media_type='text/csv', filename=filename) # ... (return FileResponse) ...
        except HTTPException: raise
        except Exception as e: # ... (handle errors) ...
            logger.exception(f"Error serving accumulation file for job {task_id}: {e}")
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Error serving generated file.")
    else: # ... (handle unexpected status) ...
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Unexpected job status '{job_status}'.")


# --- NEW: GET /jobs/animation/{task_id}/data endpoint ---
@router.get(
    "/animation/{task_id}/data",
    summary="Retrieve the animation file from a completed animation job",
    response_class=Response,
    responses={
        200: {"description": "Animation file retrieved successfully."},
        202: {"description": "Job is still pending or running."},
        400: {"description": "Job failed or was revoked."},
        404: {"description": "Job ID not found or generated data file missing."},
        500: {"description": "Internal server error."},
    },
)
async def get_animation_job_data(task_id: str):
    """
    Retrieves the animation file (e.g., GIF, MP4) generated by a successful job.
    """
    logger.info(f"Request for data from animation job ID: {task_id}")
    job_status, job_result_payload, error_info = await _get_job_outcome(task_id) # Use helper

    if job_status == "PENDING":
        return Response(status_code=status.HTTP_202_ACCEPTED, content=f"Job {task_id} is pending.")
    elif job_status == "FAILURE":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Job {task_id} failed: {error_info}")
    elif job_status == "REVOKED":
         raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Job {task_id} was revoked.")
    elif job_status == "SUCCESS":
        try:
            if not isinstance(job_result_payload, dict):
                 raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Job completed with unexpected result format.")
            output_path = job_result_payload.get("output_path")
            if not output_path:
                 raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Job completed but output path is missing.")
            if not os.path.isfile(output_path):
                 raise HTTPException(status.HTTP_404_NOT_FOUND, f"Generated animation file not found for job {task_id}.")

            # Return the animation file directly, guessing MIME type
            filename = os.path.basename(output_path)
            media_type, _ = guess_type(filename)
            return FileResponse(
                path=output_path,
                media_type=media_type or 'application/octet-stream', # Default if guess fails
                filename=filename
            )
        except HTTPException: raise
        except Exception as e:
            logger.exception(f"Error serving animation file for job {task_id}: {e}")
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Error serving generated file.")
    else: # Should not happen
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Unexpected job status.")

# --- Helper function to consolidate status checking logic ---
async def _get_job_outcome(task_id: str) -> Tuple[str, Any, Optional[str]]:
    """
    Checks job status via Huey and returns (status_str, result_payload, error_info).
    status_str: "PENDING", "SUCCESS", "FAILURE", "REVOKED"
    Raises HTTPException 404/500 for check errors.
    """
    logger.debug(f"Helper _get_job_outcome checking task: {task_id}")
    try:
        retrieved_value = None
        task_failed_exception = None

        try:
            retrieved_value = huey.result(task_id, preserve=True)
            logger.debug(f"_get_job_outcome: huey.result() for {task_id} returned type: {type(retrieved_value)}")
        except TaskException as te:
            task_failed_exception = te
            logger.warning(f"_get_job_outcome: Job ID '{task_id}' indicated failure via TaskException.")
        except Exception as e:
             logger.exception(f"_get_job_outcome: Unexpected error calling huey.result() for job ID '{task_id}': {e}")
             raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Error communicating with task queue backend.")

        # --- Determine outcome ---
        if task_failed_exception:
            original_error = getattr(task_failed_exception, 'metadata', task_failed_exception)
            error_info = f"{type(original_error).__name__}: {str(original_error)}"
            return "FAILURE", None, error_info
        elif retrieved_value is None:
            if huey.is_revoked(task_id):
                return "REVOKED", None, None
            else:
                # NOTE: Cannot distinguish PENDING from NOT_FOUND/EXPIRED here.
                return "PENDING", None, None
        elif isinstance(retrieved_value, Exception):
             # Safety catch if non-TaskException was stored somehow
             original_error = retrieved_value
             error_info = f"{type(original_error).__name__}: {str(original_error)}"
             return "FAILURE", None, error_info
        else:
             # Success
             return "SUCCESS", retrieved_value, None

    except Exception as e:
        logger.exception(f"_get_job_outcome: General error processing status for job ID '{task_id}': {e}")
        # If something else went wrong, treat as if task not found or failed check
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Failed to check status or Job ID '{task_id}' not found.")
# radproc/api/routers/downloads.py
import os
import re
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from mimetypes import guess_type

from ..dependencies import get_timeseries_dir, get_animation_output_dir

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/downloads",
    tags=["Downloads"],
)

# Basic validation patterns (adapt as needed for your exact naming)
TIMESERIES_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]+\.csv$") # e.g., point_A_RATE.csv
ACCUMULATION_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+_acc\.csv$") # e.g., point_A_RATE_1H_acc.csv
ANIMATION_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+\.(gif|mp4|avi|mov|webm)$", re.IGNORECASE) # Basic check

def _safe_join(base_dir: str, filename: str) -> str:
    """Safely join directory and filename, preventing traversal."""
    if ".." in filename or filename.startswith(("/", "\\")):
        raise ValueError("Invalid filename contains traversal elements.")
    filepath = os.path.abspath(os.path.join(base_dir, filename))
    if not filepath.startswith(os.path.abspath(base_dir)):
        raise ValueError("Resulting path is outside the base directory.")
    return filepath

@router.get("/timeseries/{filename}", summary="Download a generated timeseries CSV file")
async def download_timeseries(
    filename: str,
    timeseries_dir: str = Depends(get_timeseries_dir) # Use dependency
):
    """Downloads a specific timeseries CSV file by its filename."""
    if not TIMESERIES_FILENAME_PATTERN.match(filename):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid timeseries filename format.")

    try:
        filepath = _safe_join(timeseries_dir, filename)
    except ValueError as e:
        logger.warning(f"Invalid download filename requested: {filename}, Error: {e}")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid filename.")

    if not os.path.isfile(filepath):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Timeseries file '{filename}' not found.")

    return FileResponse(filepath, media_type='text/csv', filename=filename) # Suggest original filename

@router.get("/accumulation/{filename}", summary="Download a generated accumulation CSV file")
async def download_accumulation(
    filename: str,
    timeseries_dir: str = Depends(get_timeseries_dir) # Accumulation often stored with timeseries
):
    """Downloads a specific accumulation CSV file by its filename."""
    if not ACCUMULATION_FILENAME_PATTERN.match(filename):
         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid accumulation filename format.")

    try:
        filepath = _safe_join(timeseries_dir, filename)
    except ValueError as e:
        logger.warning(f"Invalid download filename requested: {filename}, Error: {e}")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid filename.")

    if not os.path.isfile(filepath):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Accumulation file '{filename}' not found.")

    return FileResponse(filepath, media_type='text/csv', filename=filename)

@router.get("/animation/{filename}", summary="Download a generated animation file")
async def download_animation(
    filename: str,
    animation_dir: str = Depends(get_animation_output_dir) # <<< Use dependency
    # config: dict = Depends(get_core_config) # No longer need general config here
):
    """Downloads a specific animation file by its filename from the configured directory."""
    if not ANIMATION_FILENAME_PATTERN.match(filename):
         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid animation filename or format.")

    try:
        # Use the injected animation_dir
        filepath = _safe_join(animation_dir, filename)
    except ValueError as e:
        logger.warning(f"Invalid download filename requested: {filename}, Error: {e}")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid filename.")

    if not os.path.isfile(filepath):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Animation file '{filename}' not found.")

    media_type, _ = guess_type(filename) # Guess MIME type from extension
    return FileResponse(filepath, media_type=media_type or 'application/octet-stream', filename=filename)
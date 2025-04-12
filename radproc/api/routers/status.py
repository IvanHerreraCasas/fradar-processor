# radproc/api/routers/status.py
import logging
from fastapi import APIRouter, Depends # <<< Make sure APIRouter is imported
from typing import Dict, Any

# Assuming dependencies.py is one level up relative to routers/
from ..dependencies import get_core_config
from ..schemas.status import StatusResponse # Import your status response model

logger = logging.getLogger(__name__)

# <<< Define the router instance >>>
router = APIRouter(
    prefix="/status",
    tags=["Status"], # Assign tags here for better docs grouping
)

@router.get(
    "/",
    summary="Get basic API and configuration status",
    response_model=StatusResponse # <<< Apply the response model here
)
async def get_status(config: Dict[str, Any] = Depends(get_core_config)):
    """
    Returns a simple status message indicating the API is running
    and that the core configuration was accessible.
    """
    logger.info("Status endpoint requested.")
    # The dependency `get_core_config` ensures config loaded or raises 500
    return {
        "status": "ok",
        "message": "RadProc API is running."
    } # <<< Ensure this matches StatusResponse fields
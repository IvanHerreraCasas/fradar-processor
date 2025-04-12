# radproc/api/routers/points.py
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional

from radproc.api.dependencies import get_core_config
from radproc.core.config import get_setting, get_point_config

from ..schemas.point import Point

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/points",
    tags=["Points of Interest"],
)

@router.get(
    "/",
    summary="List defined points of interest",
    response_model=List[Point] 
)
async def list_points() -> List[Dict[str, Any]]:
    """Retrieves the list of points defined in the points.yaml configuration."""
    logger.info("Request received to list points.")
    points_data = get_setting('points_config.points', default=[])

    if not isinstance(points_data, list):
        logger.error("Server configuration error: 'points_config.points' is not a list.")
        # This case shouldn't happen if config loading works, but safety check
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: Points data is not a list."
        )

    # Basic validation of list items before returning
    # (FastAPI will do more thorough validation against the schema)
    valid_points = []
    required_keys = {'name', 'latitude', 'longitude', 'variable', 'elevation'} # Check keys from config
    for idx, point_dict in enumerate(points_data):
        if isinstance(point_dict, dict) and required_keys.issubset(point_dict.keys()):
            valid_points.append(point_dict)
        else:
             logger.warning(f"Item at index {idx} in 'points_config.points' is not a valid point dictionary or is missing keys. Skipping.")

    logger.info(f"Returning {len(valid_points)} valid points.")
    # FastAPI will validate that each item in valid_points matches the Point schema
    return valid_points # Return the filtered list of valid dicts

@router.get(
    "/{point_name}",
    summary="Get details for a specific point",
    response_model=Point
)
async def get_point_details(point_name: str) -> Dict[str, Any]:
    """
    Retrieves the configuration details for a single point by its unique name.
    Searches the 'points_config.points' list in the configuration.
    """
    logger.info(f"Request received for details of point: '{point_name}'")
    points_data = get_setting('points_config.points', default=[])
    if not isinstance(points_data, list):
        logger.error("Server configuration error: 'points_config.points' is not a list.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: Points data is not a list."
        )

    found_point = get_point_config(point_name) # Use the helper function to find the point

    if found_point:
        logger.info(f"Found details for point '{point_name}'.")
        # FastAPI validates the returned dict against the Point schema
        # Pydantic v2+ uses model_config["from_attributes"] = True (or orm_mode in v1)
        # It will map the 'elevation' key from the dict to the 'target_elevation' field
        # Let's ensure the 'elevation' key *does* exist in the dict before returning
        if 'elevation' not in found_point:
             logger.error(f"Point '{point_name}' config dictionary is missing the 'elevation' key required for schema mapping.")
             # This indicates an issue with the config or the earlier key check
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal data inconsistency for point '{point_name}'.")

        return found_point # Return the raw dictionary
    else:
        logger.warning(f"Point '{point_name}' not found in configuration.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Point '{point_name}' not found in configuration."
        )
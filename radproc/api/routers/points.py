# radproc/api/routers/points.py
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

# --- Updated Imports ---
from ...core.db_manager import get_connection, release_connection, get_all_points_from_db, get_point_config_from_db
from ..schemas.point import PointDB  # Use the updated schema
# --------------------

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/points",
    tags=["Points of Interest"],
)

@router.get(
    "",
    summary="List defined points of interest from Database",
    response_model=List[PointDB] # Use updated schema
)
async def list_points_from_db() -> List[PointDB]:
    """Retrieves the list of all points defined in the radproc_points database table."""
    logger.info("Request received to list points from DB.")
    conn = None
    try:
        conn = get_connection()
        points_list_dicts = get_all_points_from_db(conn)
        if not points_list_dicts:
            logger.info("No points found in the database.")
            return [] # Return empty list if none found


        points_list_models = [PointDB(**point_dict) for point_dict in points_list_dicts]

        logger.info(f"Returning {len(points_list_models)} points from DB.")
        return points_list_models  # <<< Return the list of models
    except Exception as e:
        logger.error(f"Failed to retrieve points from database: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve points from database."
        )
    finally:
        if conn:
            release_connection(conn)

@router.get(
    "/{point_name}",
    summary="Get details for a specific point from Database",
    response_model=PointDB # Use updated schema
)
async def get_point_details_from_db(point_name: str) -> PointDB:
    """
    Retrieves the configuration details for a single point by its unique name
    by querying the radproc_points database table.
    """
    logger.info(f"Request received for DB details of point: '{point_name}'")
    conn = None
    try:
        conn = get_connection()
        point_details_dict = get_point_config_from_db(conn, point_name)

        if point_details_dict:
            logger.info(f"Found DB details for point '{point_name}'.")
            point_model = PointDB(**point_details_dict)
            return point_model # <<< Return the model instance
        else:
            logger.warning(f"Point '{point_name}' not found in database.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Point '{point_name}' not found in database."
            )
    except HTTPException:
        raise # Re-raise known HTTP exceptions
    except Exception as e:
        logger.error(f"Failed to retrieve point '{point_name}' from database: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve point '{point_name}' from database."
        )
    finally:
        if conn:
            release_connection(conn)
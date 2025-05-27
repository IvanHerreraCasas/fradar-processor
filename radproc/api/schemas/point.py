# radproc/api/schemas/point.py
from pydantic import BaseModel, Field
from typing import Optional

class PointDB(BaseModel):
    """
    Represents a point of interest as stored in and retrieved from the database.
    Reflects the radproc_points table structure.
    """
    point_id: int = Field(..., description="Unique database ID for the point")
    point_name: str = Field(..., description="Unique identifier name for the point")
    latitude: float = Field(..., description="Latitude in decimal degrees (WGS84)")
    longitude: float = Field(..., description="Longitude in decimal degrees (WGS84)")
    target_elevation: float = Field(..., description="Target elevation angle in degrees")
    description: Optional[str] = Field(None, description="Optional description of the point")
    cached_azimuth_index: Optional[int] = Field(None, description="Cached azimuth index for faster lookups")
    cached_range_index: Optional[int] = Field(None, description="Cached range index for faster lookups")

    # If using Pydantic v2+:
    model_config = {
        "from_attributes": True, # Allows mapping from DB objects/dicts
        "json_schema_extra": {
            "example": {
                "point_id": 1,
                "point_name": "point_A",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "target_elevation": 0.5,
                "description": "Example Point A near NYC",
                "cached_azimuth_index": 120,
                "cached_range_index": 55
            }
        }
    }
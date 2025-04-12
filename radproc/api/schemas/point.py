# radproc/api/schemas/point.py
from pydantic import BaseModel, Field
from typing import Optional

class Point(BaseModel):
    """Represents a configured point of interest."""
    name: str = Field(..., description="Unique identifier for the point")
    latitude: float = Field(..., description="Latitude in decimal degrees (WGS84)")
    longitude: float = Field(..., description="Longitude in decimal degrees (WGS84)")
    variable: str = Field(..., description="Default radar variable extracted for this point")
    elevation: float = Field(..., description="Target elevation angle in degrees") # Renamed from 'elevation' for clarity vs scan elevation
    description: Optional[str] = Field(None, description="Optional description of the point")

    # If using Pydantic v1:
    # class Config:
    #     orm_mode = True

    # If using Pydantic v2+:
    model_config = {
        "from_attributes": True, # Use this instead of orm_mode in Pydantic v2
        "json_schema_extra": { # Example for adding schema examples in docs
            "example": {
                "name": "point_A",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "variable": "RATE",
                "elevation": 0.5,
                "description": "Example Point A near NYC"
            }
        }
    }
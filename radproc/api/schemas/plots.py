# radproc/api/schemas/plots.py
from pydantic import BaseModel, Field
from typing import List

class PlotFrameInfo(BaseModel):
    """Information about a single available plot frame."""
    datetime_str: str = Field(
        ...,
        description="Timestamp identifier for the frame (YYYYMMDD_HHMM format, UTC)",
        example="20231027_1005"
    )
    # Note: We only need datetime_str for the client to construct the
    # URL for the /plots/historical endpoint. Variable and elevation
    # are known from the request parameters used to call /plots/frames.

    # Add model_config for examples in OpenAPI docs (Pydantic v2+)
    model_config = {
        "json_schema_extra": {
            "example": {
                "datetime_str": "20240115_1430"
            }
        }
    }
    # For Pydantic v1, use:
    # class Config:
    #     schema_extra = { ... example ... }

class PlotFramesResponse(BaseModel):
    """Response containing a list of available plot frames for a sequence."""
    frames: List[PlotFrameInfo] = Field(
        ...,
        description="List of available frame information, sorted chronologically."
    )

    # Add model_config for examples in OpenAPI docs (Pydantic v2+)
    model_config = {
        "json_schema_extra": {
            "example": {
                "frames": [
                    {"datetime_str": "20240115_1430"},
                    {"datetime_str": "20240115_1435"},
                    {"datetime_str": "20240115_1440"},
                ]
            }
        }
    }
    # For Pydantic v1, use:
    # class Config:
    #     schema_extra = { ... example ... }
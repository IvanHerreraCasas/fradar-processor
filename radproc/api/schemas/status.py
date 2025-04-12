# radproc/api/schemas/status.py
from pydantic import BaseModel, Field

class StatusResponse(BaseModel):
    """Basic status response model."""
    status: str = Field(..., example="ok")
    message: str = Field(..., example="RadProc API is running.")

    # If using Pydantic v2+:
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "message": "RadProc API is running."
            }
        }
    }
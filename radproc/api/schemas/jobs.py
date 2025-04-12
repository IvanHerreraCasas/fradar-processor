# radproc/api/schemas/jobs.py
import os
import pandas as pd


from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone


class TimeseriesJobRequest(BaseModel):
    """Request body for submitting a timeseries generation job."""
    point_name: str = Field(..., description="Unique name of the point from points.yaml", example="point_A")
    start_dt: datetime = Field(..., description="Start datetime for timeseries range (UTC recommended)", example="2023-10-27T10:00:00Z")
    end_dt: datetime = Field(..., description="End datetime for timeseries range (UTC recommended)", example="2023-10-28T12:00:00Z")
    variable: Optional[str] = Field(None, description="Override default variable for the point", example="RATE")

    @field_validator('start_dt', 'end_dt')
    @classmethod
    def ensure_timezone(cls, v: datetime) -> datetime:
        """Ensure datetime has timezone info, default to UTC if naive."""
        # This validator should only run AFTER Pydantic successfully parses the string.
        if isinstance(v, datetime) and v.tzinfo is None:
             # Use timezone.utc for compatibility with Python < 3.11
             return v.replace(tzinfo=timezone.utc)
        return v

    model_config = {
         "json_schema_extra": {
             "example": {
                 "point_name": "point_A",
                 "start_dt": "2023-10-27T10:00:00Z",
                 "end_dt": "2023-10-27T18:00:00Z",
                 "variable": "RATE"
             }
         }
    }

class JobSubmissionResponse(BaseModel):
    """Response after successfully submitting a job."""
    message: str = Field(..., example="Job queued successfully.")
    job_type: str = Field(..., example="timeseries")
    task_id: str = Field(..., description="Unique ID for the submitted job (provided by Huey)")

    # Pydantic v2+ model_config
    model_config = {
         "json_schema_extra": {
              "example": {
                   "message": "Timeseries generation job queued.",
                   "job_type": "timeseries",
                   "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
              }
         }
    }

class JobStatus(BaseModel):
    """Holds the status and potentially result/error of a job."""
    status: str = Field(..., description="Current job status (PENDING, RUNNING, SUCCESS, FAILURE, REVOKED)") # Adjusted RUNNING possibility
    last_update: Optional[datetime] = Field(None, description="Timestamp of the last status update (if available)") # Optional info
    result: Optional[Any] = Field(None, description="Result if task succeeded (structure depends on task)")
    error_info: Optional[str] = Field(None, description="Error message if task failed")

class JobStatusResponse(BaseModel):
    """Response when querying the status of a specific job."""
    task_id: str
    job_type: Optional[str] = Field(None, description="Type of job (e.g., timeseries, animation) - Needs storage") # Optional: Requires storing job type with ID
    status_details: JobStatus

     # Pydantic v2+ model_config
    model_config = {
         "json_schema_extra": {
              "example": {
                   "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                   "job_type": "timeseries",
                   "status_details": {
                        "status": "SUCCESS",
                        "last_update": "2023-10-28T15:30:00Z",
                        "result": {"status": "SUCCESS", "message": "...", "output_path": "/path/to/file.csv"},
                        "error_info": None
                   }
              }
         }
    }

class AccumulationJobRequest(BaseModel):
    """Request body for submitting an accumulation job."""
    point_name: str = Field(..., description="Unique name of the point from points.yaml", example="point_A")
    start_dt: datetime = Field(..., description="Start datetime for accumulation range (UTC recommended)", example="2023-10-27T00:00:00Z")
    end_dt: datetime = Field(..., description="End datetime for accumulation range (UTC recommended)", example="2023-10-28T00:00:00Z")
    interval: str = Field(..., description="Pandas frequency string for accumulation interval (e.g., '1H', '15min', '1D')", example="1H")
    rate_variable: Optional[str] = Field("RATE", description="Source rate variable name in timeseries CSV", example="RATE")
    output_file: Optional[str] = Field(None, description="Optional full path for the output CSV. If omitted, defaults to '<timeseries_dir>/<point>_<rate_var>_<interval>_acc.csv'.", example="/data/output/point_A_acc.csv")

    # --- Define the validator directly within this class ---
    @field_validator('start_dt', 'end_dt')
    @classmethod # Keep classmethod if preferred, though not strictly needed here
    def ensure_datetime_timezone(cls, v: Any) -> datetime:
        """Ensure datetime has timezone info, default to UTC if naive."""
        # Pydantic v2 passes the value directly. We expect datetime here
        # after initial parsing from string.
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v
        # If v is not a datetime at this point, Pydantic's initial parsing
        # likely failed, and it should raise a validation error before this.
        # However, adding robustness:
        elif isinstance(v, str):
             # This case shouldn't ideally be hit if Pydantic is parsing first.
             # If it does, attempt parsing here or raise error.
             try:
                 dt = datetime.fromisoformat(v.replace('Z', '+00:00')) # Handle Z
                 if dt.tzinfo is None:
                     return dt.replace(tzinfo=timezone.utc)
                 return dt
             except ValueError:
                 raise ValueError("Invalid datetime format provided.")
        # Handle other unexpected types if necessary
        raise TypeError("Expected datetime object.")



    @field_validator('interval')
    @classmethod
    def validate_interval(cls, v: str) -> str:
        """Validate if the interval is a recognizable Pandas frequency."""
        try:
            pd.Timedelta(v) # Check if it's a valid offset alias like '1H', '15min' etc.
            # More strict check if needed: pd.tseries.frequencies.to_offset(v)
        except ValueError:
            raise ValueError(f"Invalid accumulation interval format: '{v}'. Use Pandas frequency string (e.g., '1H', '15min', '1D').")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "point_name": "point_A",
                "start_dt": "2023-10-27T00:00:00Z",
                "end_dt": "2023-10-28T00:00:00Z",
                "interval": "6H",
                "rate_variable": "RATE",
                "output_file": None
            }
        }
    }

class AnimationJobRequest(BaseModel):
    """Request body for submitting an animation job."""
    variable: str = Field(..., description="Radar variable to animate", example="RATE")
    elevation: float = Field(..., description="Target elevation angle in degrees", example=0.5)
    start_dt: datetime = Field(..., description="Start datetime for animation range (UTC recommended)", example="2023-10-27T10:00:00Z")
    end_dt: datetime = Field(..., description="End datetime for animation range (UTC recommended)", example="2023-10-27T12:00:00Z")
    output_file: str = Field(..., description="Full path for the output animation file (e.g., /data/output/rate_anim.mp4)", example="/data/output/rate_anim.mp4")
    extent: Optional[Tuple[float, float, float, float]] = Field(None, description="Optional plot extent override (LonMin, LonMax, LatMin, LatMax)", example=[-75, -73, 40, 41])
    no_watermark: Optional[bool] = Field(False, description="Set true to exclude watermark")
    fps: Optional[int] = Field(None, description="Override default frames per second")

    # --- Define the validator directly within this class ---
    @field_validator('start_dt', 'end_dt')
    @classmethod # Keep classmethod if preferred, though not strictly needed here
    def ensure_datetime_timezone(cls, v: Any) -> datetime:
        """Ensure datetime has timezone info, default to UTC if naive."""
        # Pydantic v2 passes the value directly. We expect datetime here
        # after initial parsing from string.
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v
        # If v is not a datetime at this point, Pydantic's initial parsing
        # likely failed, and it should raise a validation error before this.
        # However, adding robustness:
        elif isinstance(v, str):
             # This case shouldn't ideally be hit if Pydantic is parsing first.
             # If it does, attempt parsing here or raise error.
             try:
                 dt = datetime.fromisoformat(v.replace('Z', '+00:00')) # Handle Z
                 if dt.tzinfo is None:
                     return dt.replace(tzinfo=timezone.utc)
                 return dt
             except ValueError:
                 raise ValueError("Invalid datetime format provided.")
        # Handle other unexpected types if necessary
        raise TypeError("Expected datetime object.")

    @field_validator('output_file')
    @classmethod
    def validate_output_extension(cls, v: str) -> str:
        """Basic check for common animation extensions."""
        allowed_formats = ['.gif', '.mp4', '.avi', '.mov', '.webm']
        ext = os.path.splitext(v)[1].lower()
        if not ext or ext not in allowed_formats:
             # Just warn, don't raise error, let imageio handle it
             # print(f"Warning: Animation output file extension '{ext}' might not be supported.")
             pass
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "variable": "RATE",
                "elevation": 0.5,
                "start_dt": "2023-10-27T10:00:00Z",
                "end_dt": "2023-10-27T12:00:00Z",
                "output_file": "/data/output/animations/rate_10-12_utc.mp4",
                "extent": None,
                "no_watermark": False,
                "fps": 10
            }
        }
    }

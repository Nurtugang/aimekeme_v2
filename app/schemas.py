"""Request / response models for the API.

Pydantic gives us validation, serialization and auto-generated OpenAPI docs.
"""

from typing import Literal

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    frames: list[str] = Field(
        ...,
        description="Exactly 16 base64-encoded JPEG frames, in chronological order.",
        examples=[["<base64_jpg>", "...", "<base64_jpg>"]],
    )


class DetectionResponse(BaseModel):
    label: Literal["fight", "normal"] = Field(
        ..., description="Predicted class for the clip."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of the predicted label (0..1).",
    )
    processing_ms: float = Field(
        ...,
        description="Server-side wall-clock time spent on decode + inference, in ms.",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "loading"]
    model_ready: bool
    device: str
    version: str

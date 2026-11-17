"""Request and response schemas for the inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import VALID_MONTHS, VALID_VISITOR_TYPES

Month = Literal["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VisitorType = Literal["Returning_Visitor", "New_Visitor", "Other"]


class ShopperFeatures(BaseModel):
    """One session's features, matching the training columns."""

    administrative: int = Field(ge=0, description="Administrative pages visited")
    administrative_duration: float = Field(ge=0)
    informational: int = Field(ge=0)
    informational_duration: float = Field(ge=0)
    product_related: int = Field(ge=0)
    product_related_duration: float = Field(ge=0)
    bounce_rates: float = Field(ge=0, le=1)
    exit_rates: float = Field(ge=0, le=1)
    page_values: float = Field(ge=0)
    special_day: float = Field(ge=0, le=1)
    month: Month
    operating_systems: int = Field(ge=1)
    browser: int = Field(ge=1)
    region: int = Field(ge=1)
    traffic_type: int = Field(ge=1)
    visitor_type: VisitorType
    weekend: bool = False

    @field_validator("month")
    @classmethod
    def known_month(cls, value: str) -> str:
        if value not in VALID_MONTHS:
            raise ValueError(f"month must be one of {VALID_MONTHS}")
        return value

    @field_validator("visitor_type")
    @classmethod
    def known_visitor_type(cls, value: str) -> str:
        if value not in VALID_VISITOR_TYPES:
            raise ValueError(f"visitor_type must be one of {VALID_VISITOR_TYPES}")
        return value

    def as_record(self) -> dict:
        record = self.model_dump()
        record["weekend"] = int(self.weekend)
        return record


class Prediction(BaseModel):
    prediction: int = Field(description="1 when the session is likely to convert")
    label: Literal["purchase", "no_purchase"]
    probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_path: str


class ModelInfo(BaseModel):
    model_path: str
    sklearn_version: str
    features: list[str]
    trained_at: str | None = None
    dataset_sha256: str | None = None

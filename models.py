from typing import List, Literal
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Represents a named entity extracted from text."""
    text: str = Field(..., description="The entity text")
    label: str = Field(..., description="The entity label (e.g., PERSON, ORG, PRODUCT)")
    start: int = Field(..., description="Start position in text")
    end: int = Field(..., description="End position in text")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)


class Ticket(BaseModel):
    """Represents a validated customer support ticket."""
    customer_id: str = Field(..., description="Customer identifier")
    product: str = Field(..., description="Product mentioned in the ticket")
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        ..., description="Sentiment analysis result"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        ..., description="Urgency classification"
    )
    entities: List[Entity] = Field(..., description="Named entities extracted from text")
    summary: str = Field(..., description="Brief summary of the ticket")
    next_action: str = Field(..., description="Recommended next action")
    confidence_score: float = Field(
        ..., description="Overall confidence in the classification", ge=0.0, le=1.0
    )
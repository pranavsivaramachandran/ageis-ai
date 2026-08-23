"""
Prediction contracts for AEGIS AI.

Defines the canonical prediction direction and immutable prediction
result model used by the Prediction Engine and downstream event layer.

This module contains no broker logic, execution logic, or model-training
logic.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from aegis.interfaces.market_data import Timeframe


class PredictionDirection(str, Enum):
    """Canonical prediction direction."""

    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class PredictionResult(BaseModel):
    """
    Canonical prediction output produced by the Prediction Engine.

    This represents an analytical prediction only.

    It does NOT represent:
    - an executable order
    - a broker instruction
    - a guaranteed trading signal
    - a profitability claim
    """

    symbol: str = Field(
        ...,
        min_length=1,
        description="Instrument symbol.",
    )

    timestamp: datetime = Field(
        ...,
        description="UTC timestamp associated with the prediction.",
    )

    timeframe: Timeframe = Field(
        ...,
        description="Timeframe used for the prediction.",
    )

    direction: PredictionDirection = Field(
        ...,
        description="Predicted market direction.",
    )

    confidence: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Prediction confidence in the range 0.0 to 1.0.",
    )

    reasoning: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of the baseline prediction.",
    )

    model_name: str = Field(
        default="baseline_development_predictor",
        min_length=1,
        description="Name of the prediction model producing this result.",
    )

    model_config = {
        "frozen": True,
    }

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        """Require a timezone-aware UTC timestamp."""

        if value.tzinfo is None:
            raise ValueError(
                "Prediction timestamp must be timezone-aware UTC"
            )

        if value.utcoffset().total_seconds() != 0:
            raise ValueError(
                "Prediction timestamp must represent UTC"
            )

        return value

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        """Reject symbols containing only whitespace."""

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Prediction symbol cannot be empty"
            )

        return normalized

    def to_prediction_event(self, trace_id: str = "") -> "PredictionEvent":
        """
        Convert this PredictionResult to a PredictionEvent for the event bus.

        This adapter bridges the prediction layer with the existing event
        architecture without modifying aegis/events/contracts.py.

        Args:
            trace_id: Optional trace ID for event correlation.

        Returns:
            A PredictionEvent compatible with the existing event system.
        """
        from aegis.events.contracts import PredictionEvent

        return PredictionEvent(
            symbol=self.symbol,
            direction=self.direction.value,
            confidence=self.confidence,
            timeframe=self.timeframe,
            trace_id=trace_id,
        )
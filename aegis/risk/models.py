"""
Risk contracts for AEGIS AI.

Defines the canonical risk decision model used by the Risk Management Engine.
This module contains no execution logic or broker state.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from aegis.interfaces.market_data import Timeframe
from aegis.prediction.models import PredictionDirection


class RiskStatus(str, Enum):
    """Canonical risk approval status."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class RiskDecision(BaseModel):
    """
    Canonical output of the Risk Management Engine.

    This represents whether a prediction is acceptable from a risk perspective.
    It does NOT represent an executable order.
    """

    symbol: str = Field(
        ...,
        min_length=1,
        description="Instrument symbol.",
    )

    timestamp: datetime = Field(
        ...,
        description="UTC timestamp associated with the decision.",
    )

    timeframe: Timeframe = Field(
        ...,
        description="Timeframe used for the underlying prediction.",
    )

    prediction_direction: PredictionDirection = Field(
        ...,
        description="Original predicted market direction.",
    )

    confidence: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Original prediction confidence.",
    )

    status: RiskStatus = Field(
        ...,
        description="Whether the risk was approved or rejected.",
    )

    reason: Optional[str] = Field(
        default=None,
        description="Reason for rejection or approval notes.",
    )

    risk_amount: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Calculated risk amount, if applicable.",
    )

    position_size: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Calculated position size, if applicable.",
    )

    model_config = {
        "frozen": True,
    }

    @model_validator(mode='after')
    def validate_approved_status(self) -> 'RiskDecision':
        if self.status == RiskStatus.APPROVED:
            if self.risk_amount is None or self.position_size is None:
                raise ValueError("risk_amount and position_size must be set when status is APPROVED.")
        return self

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        """Require a timezone-aware UTC timestamp."""

        if value.tzinfo is None:
            raise ValueError(
                "Risk decision timestamp must be timezone-aware UTC"
            )

        if value.utcoffset().total_seconds() != 0:
            raise ValueError(
                "Risk decision timestamp must represent UTC"
            )

        return value

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        """Reject symbols containing only whitespace."""

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Risk decision symbol cannot be empty"
            )

        return normalized

    def to_risk_decision_event(self, trace_id: str = "") -> "RiskDecisionEvent":
        """
        Convert this RiskDecision to a RiskDecisionEvent for the event bus.

        Args:
            trace_id: Optional trace ID for event correlation.

        Returns:
            A RiskDecisionEvent compatible with the existing event system.
        """
        from aegis.events.contracts import RiskDecisionEvent

        return RiskDecisionEvent(
            symbol=self.symbol,
            direction=self.prediction_direction.value,
            confidence=self.confidence,
            timeframe=self.timeframe,
            risk_status=self.status.value,
            risk_reason=self.reason,
            risk_amount=self.risk_amount,
            position_size=self.position_size,
            trace_id=trace_id,
        )

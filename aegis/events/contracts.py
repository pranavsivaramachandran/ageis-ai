"""
Internal event contracts for AEGIS AI inter-agent communication.

These are typed in-memory message models (Pydantic) — distinct from the
SystemEvent ORM model used for database persistence. Each event can be
converted to a SystemEvent via to_system_event() for audit logging.
"""
import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from aegis.db.models.system_event import SystemEvent
from aegis.interfaces.market_data import Tick, OHLC, Timeframe
from aegis.interfaces.broker import OrderRequest


class AlertSeverity(str, Enum):
    """Severity levels for system alerts."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class BaseEvent(BaseModel):
    """
    Base contract for all internal AEGIS events.
    Every event carries an event_type, timestamp, and trace_id.
    """
    event_type: str = Field(..., min_length=1, description="Discriminator for event routing")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the event was created"
    )
    trace_id: str = Field(default="", description="Trace ID for correlation")

    model_config = {"frozen": True}

    def to_system_event(self) -> SystemEvent:
        """Convert this in-memory event to a persistable SystemEvent ORM instance."""
        return SystemEvent(
            event_type=self.event_type,
            detail=self.model_dump_json(),
            trace_id=self.trace_id,
        )


class MarketDataEvent(BaseEvent):
    """Emitted when new market data arrives from a provider."""
    event_type: str = Field(default="MARKET_DATA", frozen=True)
    symbol: str = Field(..., min_length=1)
    tick: Tick


class AnalysisRequestEvent(BaseEvent):
    """Request for analysis agents to process candle data."""
    event_type: str = Field(default="ANALYSIS_REQUEST", frozen=True)
    symbol: str = Field(..., min_length=1)
    timeframe: Timeframe
    candles: list[OHLC]


class PredictionEvent(BaseEvent):
    """Result of an analysis agent's prediction."""
    event_type: str = Field(default="PREDICTION", frozen=True)
    symbol: str = Field(..., min_length=1)
    direction: str = Field(..., description="Predicted direction: BUY, SELL, or NEUTRAL")
    confidence: Decimal = Field(..., ge=0, le=1, description="Confidence score 0.0-1.0")
    timeframe: Timeframe


class RiskCheckEvent(BaseEvent):
    """Request for the risk layer to evaluate a proposed action."""
    event_type: str = Field(default="RISK_CHECK", frozen=True)
    symbol: str = Field(..., min_length=1)
    proposed_action: str = Field(..., description="The proposed trading action")
    confidence: Decimal = Field(..., ge=0, le=1)


class ExecutionRequestEvent(BaseEvent):
    """Request for the execution layer to process an order."""
    event_type: str = Field(default="EXECUTION_REQUEST", frozen=True)
    order: OrderRequest
    risk_approved: bool = Field(..., description="Whether risk layer approved this action")


class AgentHeartbeatEvent(BaseEvent):
    """Periodic health signal from an agent to the supervisor."""
    event_type: str = Field(default="HEARTBEAT", frozen=True)
    agent_name: str = Field(..., min_length=1)
    status: str = Field(..., description="Agent status: HEALTHY, UNHEALTHY, STARTING, STOPPED")


class SystemAlertEvent(BaseEvent):
    """System-wide alert for exceptional conditions."""
    event_type: str = Field(default="ALERT", frozen=True)
    severity: AlertSeverity
    source: str = Field(..., min_length=1, description="Component that raised the alert")
    message: str = Field(..., min_length=1, description="Human-readable alert description")

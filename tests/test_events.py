from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from aegis.events.contracts import (
    AlertSeverity,
    AgentHeartbeatEvent,
    AnalysisRequestEvent,
    BaseEvent,
    ExecutionRequestEvent,
    MarketDataEvent,
    PredictionEvent,
    RiskCheckEvent,
    SystemAlertEvent,
)
from aegis.interfaces.broker import (
    OrderRequest,
    OrderSide,
    OrderType,
)
from aegis.interfaces.market_data import OHLC, Tick, Timeframe


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_tick() -> Tick:
    return Tick(
        symbol="EUR/USD",
        timestamp=utc_now(),
        bid=Decimal("1.10000"),
        ask=Decimal("1.10020"),
    )


def make_candle() -> OHLC:
    return OHLC(
        symbol="EUR/USD",
        timestamp=utc_now(),
        timeframe=Timeframe.M15,
        open=Decimal("1.10000"),
        high=Decimal("1.10200"),
        low=Decimal("1.09800"),
        close=Decimal("1.10100"),
        volume=Decimal("1000"),
    )


def make_order() -> OrderRequest:
    return OrderRequest(
        symbol="EUR/USD",
        side=OrderSide.BUY,
        volume=Decimal("0.10"),
        order_type=OrderType.MARKET,
    )


class TestBaseEvent:

    def test_base_event_creation(self):
        event = BaseEvent(
            event_type="TEST_EVENT",
            trace_id="trace-001",
        )

        assert event.event_type == "TEST_EVENT"
        assert event.trace_id == "trace-001"
        assert event.timestamp.tzinfo is not None

    def test_event_type_required(self):
        with pytest.raises(ValidationError):
            BaseEvent(event_type="")

    def test_event_is_immutable(self):
        event = BaseEvent(
            event_type="TEST_EVENT",
            trace_id="trace-001",
        )

        with pytest.raises(ValidationError):
            event.trace_id = "changed"

    def test_event_serialization(self):
        event = BaseEvent(
            event_type="TEST_EVENT",
            trace_id="trace-001",
        )

        data = event.model_dump_json()

        assert "TEST_EVENT" in data
        assert "trace-001" in data

    def test_to_system_event(self):
        event = BaseEvent(
            event_type="TEST_EVENT",
            trace_id="trace-001",
        )

        system_event = event.to_system_event()

        assert system_event.event_type == "TEST_EVENT"
        assert system_event.trace_id == "trace-001"
        assert system_event.detail is not None


class TestMarketDataEvent:

    def test_valid_market_data_event(self):
        event = MarketDataEvent(
            symbol="EUR/USD",
            tick=make_tick(),
            trace_id="trace-market",
        )

        assert event.event_type == "MARKET_DATA"
        assert event.symbol == "EUR/USD"
        assert event.tick.symbol == "EUR/USD"
        assert event.trace_id == "trace-market"

    def test_market_data_event_requires_symbol(self):
        with pytest.raises(ValidationError):
            MarketDataEvent(
                symbol="",
                tick=make_tick(),
            )


class TestAnalysisRequestEvent:

    def test_valid_analysis_request(self):
        event = AnalysisRequestEvent(
            symbol="EUR/USD",
            timeframe=Timeframe.M15,
            candles=[make_candle()],
            trace_id="trace-analysis",
        )

        assert event.event_type == "ANALYSIS_REQUEST"
        assert event.timeframe == Timeframe.M15
        assert len(event.candles) == 1

    def test_analysis_request_requires_symbol(self):
        with pytest.raises(ValidationError):
            AnalysisRequestEvent(
                symbol="",
                timeframe=Timeframe.M15,
                candles=[make_candle()],
            )


class TestPredictionEvent:

    def test_valid_prediction(self):
        event = PredictionEvent(
            symbol="EUR/USD",
            direction="BUY",
            confidence=Decimal("0.85"),
            timeframe=Timeframe.M15,
            trace_id="trace-prediction",
        )

        assert event.event_type == "PREDICTION"
        assert event.direction == "BUY"
        assert event.confidence == Decimal("0.85")

    def test_confidence_cannot_exceed_one(self):
        with pytest.raises(ValidationError):
            PredictionEvent(
                symbol="EUR/USD",
                direction="BUY",
                confidence=Decimal("1.01"),
                timeframe=Timeframe.M15,
            )

    def test_confidence_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            PredictionEvent(
                symbol="EUR/USD",
                direction="BUY",
                confidence=Decimal("-0.01"),
                timeframe=Timeframe.M15,
            )


class TestRiskCheckEvent:

    def test_valid_risk_check(self):
        event = RiskCheckEvent(
            symbol="EUR/USD",
            proposed_action="BUY",
            confidence=Decimal("0.80"),
            trace_id="trace-risk",
        )

        assert event.event_type == "RISK_CHECK"
        assert event.proposed_action == "BUY"

    def test_invalid_confidence_rejected(self):
        with pytest.raises(ValidationError):
            RiskCheckEvent(
                symbol="EUR/USD",
                proposed_action="BUY",
                confidence=Decimal("2"),
            )


class TestExecutionRequestEvent:

    def test_valid_execution_request(self):
        event = ExecutionRequestEvent(
            order=make_order(),
            risk_approved=False,
            trace_id="trace-execution",
        )

        assert event.event_type == "EXECUTION_REQUEST"
        assert event.order.symbol == "EUR/USD"
        assert event.risk_approved is False

    def test_risk_approval_is_required(self):
        with pytest.raises(ValidationError):
            ExecutionRequestEvent(
                order=make_order(),
            )


class TestAgentHeartbeatEvent:

    def test_valid_heartbeat(self):
        event = AgentHeartbeatEvent(
            agent_name="orchestrator",
            status="HEALTHY",
            trace_id="trace-heartbeat",
        )

        assert event.event_type == "HEARTBEAT"
        assert event.agent_name == "orchestrator"
        assert event.status == "HEALTHY"

    def test_agent_name_required(self):
        with pytest.raises(ValidationError):
            AgentHeartbeatEvent(
                agent_name="",
                status="HEALTHY",
            )


class TestSystemAlertEvent:

    def test_valid_alert(self):
        event = SystemAlertEvent(
            severity=AlertSeverity.WARNING,
            source="risk_engine",
            message="Risk threshold approaching",
            trace_id="trace-alert",
        )

        assert event.event_type == "ALERT"
        assert event.severity == AlertSeverity.WARNING
        assert event.source == "risk_engine"

    def test_alert_severity_is_typed(self):
        with pytest.raises(ValidationError):
            SystemAlertEvent(
                severity="INVALID",
                source="risk_engine",
                message="Test",
            )

    def test_alert_message_required(self):
        with pytest.raises(ValidationError):
            SystemAlertEvent(
                severity=AlertSeverity.CRITICAL,
                source="risk_engine",
                message="",
            )
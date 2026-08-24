"""
Sprint 5 — Risk Models tests.

Tests for RiskDecision covering validation, properties, and data preservation.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from aegis.interfaces.market_data import Timeframe
from aegis.prediction.models import PredictionDirection
from aegis.risk.models import RiskDecision, RiskStatus


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TestRiskDecisionValidation:
    """Verify validation rules of RiskDecision."""

    def test_valid_approved_decision(self):
        decision = RiskDecision(
            symbol="EURUSD",
            timestamp=_utc_now(),
            timeframe=Timeframe.H1,
            prediction_direction=PredictionDirection.BUY,
            confidence=Decimal("0.85"),
            status=RiskStatus.APPROVED,
            reason="Passes all risk checks.",
            risk_amount=Decimal("100.0"),
            position_size=Decimal("1.5"),
        )
        assert decision.status == RiskStatus.APPROVED

    def test_valid_rejected_decision(self):
        decision = RiskDecision(
            symbol="EURUSD",
            timestamp=_utc_now(),
            timeframe=Timeframe.H1,
            prediction_direction=PredictionDirection.SELL,
            confidence=Decimal("0.4"),
            status=RiskStatus.REJECTED,
            reason="Confidence too low.",
        )
        assert decision.status == RiskStatus.REJECTED

    def test_invalid_negative_risk_amount(self):
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            RiskDecision(
                symbol="EURUSD",
                timestamp=_utc_now(),
                timeframe=Timeframe.H1,
                prediction_direction=PredictionDirection.BUY,
                confidence=Decimal("0.85"),
                status=RiskStatus.APPROVED,
                risk_amount=Decimal("-10.0"),
            )

    def test_invalid_negative_position_size(self):
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            RiskDecision(
                symbol="EURUSD",
                timestamp=_utc_now(),
                timeframe=Timeframe.H1,
                prediction_direction=PredictionDirection.BUY,
                confidence=Decimal("0.85"),
                status=RiskStatus.APPROVED,
                position_size=Decimal("-1.0"),
            )

    def test_invalid_negative_confidence(self):
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            RiskDecision(
                symbol="EURUSD",
                timestamp=_utc_now(),
                timeframe=Timeframe.H1,
                prediction_direction=PredictionDirection.BUY,
                confidence=Decimal("-0.1"),
                status=RiskStatus.APPROVED,
            )

    def test_invalid_confidence_above_one(self):
        with pytest.raises(ValueError, match="less than or equal to 1"):
            RiskDecision(
                symbol="EURUSD",
                timestamp=_utc_now(),
                timeframe=Timeframe.H1,
                prediction_direction=PredictionDirection.BUY,
                confidence=Decimal("1.1"),
                status=RiskStatus.APPROVED,
            )

    def test_timezone_naive_timestamp_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            RiskDecision(
                symbol="EURUSD",
                timestamp=datetime.now(),
                timeframe=Timeframe.H1,
                prediction_direction=PredictionDirection.BUY,
                confidence=Decimal("0.85"),
                status=RiskStatus.APPROVED,
            )

    def test_empty_symbol_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            RiskDecision(
                symbol="   ",
                timestamp=_utc_now(),
                timeframe=Timeframe.H1,
                prediction_direction=PredictionDirection.BUY,
                confidence=Decimal("0.85"),
                status=RiskStatus.APPROVED,
            )


class TestDataPreservation:
    """Verify data is preserved exactly as input."""

    def test_symbol_preservation(self):
        decision = RiskDecision(
            symbol="BTCUSD",
            timestamp=_utc_now(),
            timeframe=Timeframe.H1,
            prediction_direction=PredictionDirection.BUY,
            confidence=Decimal("0.85"),
            status=RiskStatus.APPROVED,
        )
        assert decision.symbol == "BTCUSD"

    def test_timestamp_preservation(self):
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        decision = RiskDecision(
            symbol="BTCUSD",
            timestamp=ts,
            timeframe=Timeframe.H1,
            prediction_direction=PredictionDirection.BUY,
            confidence=Decimal("0.85"),
            status=RiskStatus.APPROVED,
        )
        assert decision.timestamp == ts

    def test_timeframe_preservation(self):
        decision = RiskDecision(
            symbol="BTCUSD",
            timestamp=_utc_now(),
            timeframe=Timeframe.D1,
            prediction_direction=PredictionDirection.BUY,
            confidence=Decimal("0.85"),
            status=RiskStatus.APPROVED,
        )
        assert decision.timeframe == Timeframe.D1

    def test_prediction_direction_preservation(self):
        decision = RiskDecision(
            symbol="BTCUSD",
            timestamp=_utc_now(),
            timeframe=Timeframe.H1,
            prediction_direction=PredictionDirection.SELL,
            confidence=Decimal("0.85"),
            status=RiskStatus.APPROVED,
        )
        assert decision.prediction_direction == PredictionDirection.SELL

    def test_immutable_model(self):
        decision = RiskDecision(
            symbol="BTCUSD",
            timestamp=_utc_now(),
            timeframe=Timeframe.H1,
            prediction_direction=PredictionDirection.BUY,
            confidence=Decimal("0.85"),
            status=RiskStatus.APPROVED,
        )
        with pytest.raises(Exception):
            decision.symbol = "ETHUSD"

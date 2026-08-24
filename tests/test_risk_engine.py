"""
Sprint 5 — Risk Engine tests.

Tests for RiskManagementEngine covering rules, limits, sizing, determinism,
and safety guarantees.
"""

import inspect
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from aegis.core.config import config
from aegis.interfaces.market_data import Timeframe
from aegis.prediction.models import PredictionDirection, PredictionResult
from aegis.risk.engine import RiskManagementEngine
from aegis.risk.models import RiskDecision, RiskStatus


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _valid_buy_prediction(confidence: Decimal = Decimal("0.8")) -> PredictionResult:
    return PredictionResult(
        symbol="RELIANCE",
        timestamp=_utc_now(),
        timeframe=Timeframe.H1,
        direction=PredictionDirection.BUY,
        confidence=confidence,
    )


def _valid_sell_prediction(confidence: Decimal = Decimal("0.8")) -> PredictionResult:
    return PredictionResult(
        symbol="TCS",
        timestamp=_utc_now(),
        timeframe=Timeframe.H1,
        direction=PredictionDirection.SELL,
        confidence=confidence,
    )


class TestRiskEngineBaseRules:
    """Verify base acceptance and rejection rules."""

    def test_valid_prediction_accepted(self):
        engine = RiskManagementEngine()
        pred = _valid_buy_prediction()
        decision = engine.evaluate_prediction(pred, risk_distance=Decimal("10.0"))
        assert decision.status == RiskStatus.APPROVED

    def test_neutral_prediction_rejected(self):
        engine = RiskManagementEngine()
        pred = PredictionResult(
            symbol="INFY",
            timestamp=_utc_now(),
            timeframe=Timeframe.H1,
            direction=PredictionDirection.NEUTRAL,
            confidence=Decimal("0"),
        )
        decision = engine.evaluate_prediction(pred)
        assert decision.status == RiskStatus.REJECTED
        assert "NEUTRAL" in decision.reason

    def test_low_confidence_rejected(self):
        engine = RiskManagementEngine()
        # default min confidence is 0.7
        pred = _valid_buy_prediction(confidence=Decimal("0.5"))
        decision = engine.evaluate_prediction(pred)
        assert decision.status == RiskStatus.REJECTED
        assert "Confidence" in decision.reason


class TestRiskEngineLossLimits:
    """Verify daily, weekly, monthly loss limits."""

    def test_daily_loss_limit_rejection(self):
        engine = RiskManagementEngine()
        pred = _valid_buy_prediction()
        
        # Test just below limit
        decision = engine.evaluate_prediction(pred, current_daily_loss=config.RISK_MAX_DAILY_LOSS - Decimal("1"), risk_distance=Decimal("10.0"))
        assert decision.status == RiskStatus.APPROVED
        
        # Test exactly at limit
        decision = engine.evaluate_prediction(pred, current_daily_loss=config.RISK_MAX_DAILY_LOSS)
        assert decision.status == RiskStatus.REJECTED
        assert "daily" in decision.reason

    def test_weekly_loss_limit_rejection(self):
        engine = RiskManagementEngine()
        pred = _valid_buy_prediction()
        
        # Test exactly at limit
        decision = engine.evaluate_prediction(pred, current_weekly_loss=config.RISK_MAX_WEEKLY_LOSS)
        assert decision.status == RiskStatus.REJECTED
        assert "weekly" in decision.reason

    def test_monthly_loss_limit_rejection(self):
        engine = RiskManagementEngine()
        pred = _valid_buy_prediction()
        
        # Test exactly at limit
        decision = engine.evaluate_prediction(pred, current_monthly_loss=config.RISK_MAX_MONTHLY_LOSS)
        assert decision.status == RiskStatus.REJECTED
        assert "monthly" in decision.reason


class TestRiskEnginePositionSizing:
    """Verify mathematical calculation of position sizing."""

    def test_valid_position_size(self):
        engine = RiskManagementEngine()
        pred = _valid_buy_prediction()
        
        # risk_amount = 100, risk_distance = 20 -> position_size = 5
        decision = engine.evaluate_prediction(pred, risk_distance=Decimal("20.0"))
        
        assert decision.status == RiskStatus.APPROVED
        assert decision.risk_amount == Decimal("100.0")
        assert decision.position_size == Decimal("5.0")

    def test_zero_risk_distance_rejected(self):
        engine = RiskManagementEngine()
        pred = _valid_buy_prediction()
        
        decision = engine.evaluate_prediction(pred, risk_distance=Decimal("0.0"))
        assert decision.status == RiskStatus.REJECTED
        assert "strictly positive" in decision.reason

    def test_negative_risk_distance_rejected(self):
        engine = RiskManagementEngine()
        pred = _valid_buy_prediction()
        
        decision = engine.evaluate_prediction(pred, risk_distance=Decimal("-5.0"))
        assert decision.status == RiskStatus.REJECTED
        assert "strictly positive" in decision.reason

    def test_max_position_size_enforcement(self):
        engine = RiskManagementEngine()
        pred = _valid_buy_prediction()
        
        # risk_amount = 100. If distance is 5, size would be 20. Max is 10.
        decision = engine.evaluate_prediction(pred, risk_distance=Decimal("5.0"))
        
        assert decision.status == RiskStatus.REJECTED
        assert "exceeds maximum" in decision.reason


class TestRiskEngineDeterminism:
    """Verify engine output is strictly deterministic based on inputs."""
    
    def test_identical_input_identical_output(self):
        engine = RiskManagementEngine()
        pred = _valid_buy_prediction()
        
        dec1 = engine.evaluate_prediction(pred, risk_distance=Decimal("20.0"))
        dec2 = engine.evaluate_prediction(pred, risk_distance=Decimal("20.0"))
        
        assert dec1.status == dec2.status
        assert dec1.position_size == dec2.position_size
        assert dec1.reason == dec2.reason


class TestRiskEngineSafety:
    """Verify safety boundaries are respected."""
    
    def test_engine_does_not_import_broker(self):
        import aegis.risk.engine as engine_module
        source = inspect.getsource(engine_module)
        assert "from aegis.interfaces.broker" not in source
        assert "import aegis.interfaces.broker" not in source

    def test_engine_no_external_api_access(self):
        import aegis.risk.engine as engine_module
        source = inspect.getsource(engine_module)
        assert "requests" not in source
        assert "urllib" not in source
        assert "httpx" not in source
        
    def test_execution_mode_not_modified(self):
        from aegis.core.config import config, ExecutionMode
        original_mode = config.SYSTEM_MODE
        
        engine = RiskManagementEngine()
        engine.evaluate_prediction(_valid_buy_prediction())
        
        assert config.SYSTEM_MODE == original_mode
        assert config.SYSTEM_MODE == ExecutionMode.PREDICTION_ONLY.value

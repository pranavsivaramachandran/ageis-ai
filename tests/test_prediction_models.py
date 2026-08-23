"""
Sprint 4.1 — Prediction contract tests.

Tests for PredictionDirection and PredictionResult to verify that the
prediction contracts enforce all validation rules, immutability, and
boundary conditions.

These tests validate contracts only — no engine, broker, or execution logic.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from aegis.prediction.models import PredictionDirection, PredictionResult
from aegis.interfaces.market_data import Timeframe


# ===================================================================
# Helpers
# ===================================================================

def _utc_now() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def _valid_prediction(**overrides) -> PredictionResult:
    """Build a valid PredictionResult with sensible defaults, allowing overrides."""
    defaults = {
        "symbol": "RELIANCE",
        "timestamp": _utc_now(),
        "timeframe": Timeframe.H1,
        "direction": PredictionDirection.BUY,
        "confidence": Decimal("0.75"),
    }
    defaults.update(overrides)
    return PredictionResult(**defaults)


# ===================================================================
# PredictionDirection
# ===================================================================

class TestPredictionDirection:
    """Verify all canonical prediction directions exist."""

    def test_buy_exists(self):
        assert PredictionDirection.BUY == "BUY"

    def test_sell_exists(self):
        assert PredictionDirection.SELL == "SELL"

    def test_neutral_exists(self):
        assert PredictionDirection.NEUTRAL == "NEUTRAL"


# ===================================================================
# PredictionResult — Valid construction
# ===================================================================

class TestPredictionResultValid:
    """Verify that correctly-formed PredictionResults are accepted."""

    def test_valid_result(self):
        result = _valid_prediction()
        assert result.symbol == "RELIANCE"
        assert result.direction == PredictionDirection.BUY
        assert result.confidence == Decimal("0.75")

    def test_confidence_zero_accepted(self):
        result = _valid_prediction(confidence=Decimal("0"))
        assert result.confidence == Decimal("0")

    def test_confidence_one_accepted(self):
        result = _valid_prediction(confidence=Decimal("1"))
        assert result.confidence == Decimal("1")

    def test_valid_utc_timestamp(self):
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _valid_prediction(timestamp=ts)
        assert result.timestamp == ts

    def test_valid_timeframe(self):
        for tf in Timeframe:
            result = _valid_prediction(timeframe=tf)
            assert result.timeframe == tf

    def test_default_model_name(self):
        result = _valid_prediction()
        assert result.model_name == "baseline_development_predictor"

    def test_custom_model_name(self):
        result = _valid_prediction(model_name="custom_model_v2")
        assert result.model_name == "custom_model_v2"

    def test_optional_reasoning_none(self):
        result = _valid_prediction()
        assert result.reasoning is None

    def test_optional_reasoning_provided(self):
        result = _valid_prediction(reasoning="RSI oversold, MACD bullish crossover")
        assert result.reasoning == "RSI oversold, MACD bullish crossover"


# ===================================================================
# PredictionResult — Validation rejection
# ===================================================================

class TestPredictionResultRejection:
    """Verify that invalid inputs are rejected by Pydantic validation."""

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            _valid_prediction(confidence=Decimal("1.01"))

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            _valid_prediction(confidence=Decimal("-0.01"))

    def test_blank_symbol_rejected(self):
        with pytest.raises(ValidationError):
            _valid_prediction(symbol="")

    def test_whitespace_only_symbol_rejected(self):
        with pytest.raises(ValidationError):
            _valid_prediction(symbol="   ")

    def test_timezone_naive_timestamp_rejected(self):
        naive_ts = datetime(2025, 6, 15, 12, 0, 0)  # no tzinfo
        with pytest.raises(ValidationError, match="timezone-aware"):
            _valid_prediction(timestamp=naive_ts)

    def test_non_utc_timestamp_rejected(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        non_utc_ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=ist)
        with pytest.raises(ValidationError, match="UTC"):
            _valid_prediction(timestamp=non_utc_ts)


# ===================================================================
# PredictionResult — Immutability
# ===================================================================

class TestPredictionResultImmutability:
    """Verify that PredictionResult is frozen (immutable)."""

    def test_symbol_immutable(self):
        result = _valid_prediction()
        with pytest.raises(ValidationError):
            result.symbol = "CHANGED"

    def test_confidence_immutable(self):
        result = _valid_prediction()
        with pytest.raises(ValidationError):
            result.confidence = Decimal("0.5")

    def test_direction_immutable(self):
        result = _valid_prediction()
        with pytest.raises(ValidationError):
            result.direction = PredictionDirection.SELL

    def test_timestamp_immutable(self):
        result = _valid_prediction()
        with pytest.raises(ValidationError):
            result.timestamp = _utc_now()

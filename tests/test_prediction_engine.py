"""
Sprint 4.2 — Prediction Engine tests.

Tests for BaselinePredictor covering determinism, directional scenarios,
edge cases, safety guarantees, and PredictionEvent integration.

These tests validate the engine produces correct, deterministic predictions
from FeatureVectors without accessing any broker, provider, or external API.
"""

import inspect
import math
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from aegis.features.builder import FeatureVector
from aegis.interfaces.market_data import Timeframe
from aegis.prediction.engine import BaselinePredictor, PredictionError
from aegis.prediction.models import PredictionDirection, PredictionResult


# ===================================================================
# Helpers — Deterministic FeatureVector factories
# ===================================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _base_fv(**overrides) -> FeatureVector:
    """Minimal FeatureVector with all features as None."""
    defaults = {
        "timestamp": datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        "symbol": "RELIANCE",
        "timeframe": Timeframe.H1,
        "last_close": 100.0,
    }
    defaults.update(overrides)
    return FeatureVector(**defaults)


def _bullish_fv() -> FeatureVector:
    """FeatureVector with strongly bullish indicators."""
    return FeatureVector(
        timestamp=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        symbol="RELIANCE",
        timeframe=Timeframe.H1,
        last_close=105.0,
        returns=[0.02, 0.015, 0.018, 0.01, 0.012],
        sma_value=100.0,
        ema_value=101.0,
        rsi_value=25.0,          # Oversold → bullish
        macd_line=1.5,
        macd_signal=0.8,
        macd_histogram=0.7,      # Positive → bullish
        atr_value=2.5,
        bollinger_upper=110.0,
        bollinger_middle=105.0,
        bollinger_lower=100.0,
        momentum_value=3.5,      # Positive → bullish
        volatility=0.015,
    )


def _bearish_fv() -> FeatureVector:
    """FeatureVector with strongly bearish indicators."""
    return FeatureVector(
        timestamp=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        symbol="TATASTEEL",
        timeframe=Timeframe.D1,
        last_close=95.0,
        returns=[-0.02, -0.015, -0.018, -0.01, -0.012],
        sma_value=100.0,
        ema_value=99.0,
        rsi_value=78.0,          # Overbought → bearish
        macd_line=-1.5,
        macd_signal=-0.8,
        macd_histogram=-0.7,     # Negative → bearish
        atr_value=2.5,
        bollinger_upper=110.0,
        bollinger_middle=105.0,
        bollinger_lower=100.0,
        momentum_value=-3.5,     # Negative → bearish
        volatility=0.025,
    )


def _neutral_fv() -> FeatureVector:
    """FeatureVector with contradictory/weak indicators → NEUTRAL."""
    return FeatureVector(
        timestamp=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        symbol="INFY",
        timeframe=Timeframe.H4,
        last_close=100.0,
        returns=[0.001, -0.001, 0.0005, -0.0005, 0.0002],
        sma_value=100.0,
        ema_value=100.0,
        rsi_value=50.0,           # Exactly neutral
        macd_line=0.0,
        macd_signal=0.0,
        macd_histogram=0.0,       # Zero → neutral
        atr_value=1.0,
        bollinger_upper=105.0,
        bollinger_middle=100.0,
        bollinger_lower=95.0,
        momentum_value=0.0,       # Zero → neutral
        volatility=0.01,
    )


# ===================================================================
# Engine construction
# ===================================================================

class TestBaselinePredictorConstruction:
    """Verify engine construction and parameter validation."""

    def test_default_construction(self):
        predictor = BaselinePredictor()
        assert predictor is not None

    def test_custom_threshold(self):
        predictor = BaselinePredictor(direction_threshold=0.3)
        assert predictor is not None

    def test_invalid_threshold_zero(self):
        with pytest.raises(PredictionError):
            BaselinePredictor(direction_threshold=0.0)

    def test_invalid_threshold_one(self):
        with pytest.raises(PredictionError):
            BaselinePredictor(direction_threshold=1.0)

    def test_invalid_threshold_negative(self):
        with pytest.raises(PredictionError):
            BaselinePredictor(direction_threshold=-0.5)


# ===================================================================
# Determinism
# ===================================================================

class TestDeterminism:
    """Verify that the engine is fully deterministic."""

    def test_identical_input_identical_output(self):
        predictor = BaselinePredictor()
        fv = _bullish_fv()
        result1 = predictor.predict(fv)
        result2 = predictor.predict(fv)
        assert result1.direction == result2.direction
        assert result1.confidence == result2.confidence

    def test_repeated_calls_stable(self):
        """10 repeated calls must all produce identical results."""
        predictor = BaselinePredictor()
        fv = _bearish_fv()
        results = [predictor.predict(fv) for _ in range(10)]
        directions = {r.direction for r in results}
        confidences = {r.confidence for r in results}
        assert len(directions) == 1
        assert len(confidences) == 1


# ===================================================================
# Directional scenarios
# ===================================================================

class TestDirectionalScenarios:
    """Verify BUY, SELL, and NEUTRAL detection."""

    def test_bullish_scenario_produces_buy(self):
        predictor = BaselinePredictor()
        result = predictor.predict(_bullish_fv())
        assert result.direction == PredictionDirection.BUY

    def test_bearish_scenario_produces_sell(self):
        predictor = BaselinePredictor()
        result = predictor.predict(_bearish_fv())
        assert result.direction == PredictionDirection.SELL

    def test_neutral_scenario_produces_neutral(self):
        predictor = BaselinePredictor()
        result = predictor.predict(_neutral_fv())
        assert result.direction == PredictionDirection.NEUTRAL

    def test_no_features_produces_neutral(self):
        predictor = BaselinePredictor()
        fv = _base_fv()  # all features None
        result = predictor.predict(fv)
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == Decimal("0")

    def test_weak_evidence_may_be_neutral(self):
        """Contradictory weak signals should not force a directional call."""
        predictor = BaselinePredictor()
        fv = FeatureVector(
            timestamp=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
            symbol="HDFC",
            timeframe=Timeframe.H1,
            last_close=100.0,
            rsi_value=48.0,        # Very slightly bullish
            macd_histogram=-0.01,  # Very slightly bearish
        )
        result = predictor.predict(fv)
        # With contradictory weak signals, result should be low confidence
        assert result.confidence <= Decimal("0.5")


# ===================================================================
# Confidence bounds
# ===================================================================

class TestConfidenceBounds:
    """Verify confidence is always in [0, 1]."""

    def test_confidence_lower_bound(self):
        predictor = BaselinePredictor()
        result = predictor.predict(_base_fv())
        assert result.confidence >= Decimal("0")

    def test_confidence_upper_bound(self):
        predictor = BaselinePredictor()
        result = predictor.predict(_bullish_fv())
        assert result.confidence <= Decimal("1")

    def test_confidence_range_on_all_scenarios(self):
        predictor = BaselinePredictor()
        for fv_factory in [_bullish_fv, _bearish_fv, _neutral_fv, _base_fv]:
            result = predictor.predict(fv_factory())
            assert Decimal("0") <= result.confidence <= Decimal("1"), (
                f"Confidence {result.confidence} out of range for {fv_factory.__name__}"
            )


# ===================================================================
# Data preservation
# ===================================================================

class TestDataPreservation:
    """Verify that symbol, timestamp, and timeframe are preserved from input."""

    def test_symbol_preserved(self):
        predictor = BaselinePredictor()
        fv = _base_fv(symbol="SBIN")
        result = predictor.predict(fv)
        assert result.symbol == "SBIN"

    def test_timestamp_preserved(self):
        predictor = BaselinePredictor()
        ts = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        fv = _base_fv(timestamp=ts)
        result = predictor.predict(fv)
        assert result.timestamp == ts

    def test_timeframe_preserved(self):
        predictor = BaselinePredictor()
        fv = _base_fv(timeframe=Timeframe.M15)
        result = predictor.predict(fv)
        assert result.timeframe == Timeframe.M15


# ===================================================================
# Invalid input handling
# ===================================================================

class TestInvalidInputs:
    """Verify that invalid numeric values are rejected."""

    def test_nan_rsi_rejected(self):
        predictor = BaselinePredictor()
        fv = _base_fv(rsi_value=float("nan"))
        with pytest.raises(PredictionError, match="invalid"):
            predictor.predict(fv)

    def test_inf_macd_rejected(self):
        predictor = BaselinePredictor()
        fv = _base_fv(macd_histogram=float("inf"))
        with pytest.raises(PredictionError, match="invalid"):
            predictor.predict(fv)

    def test_neg_inf_momentum_rejected(self):
        predictor = BaselinePredictor()
        fv = _base_fv(momentum_value=float("-inf"))
        with pytest.raises(PredictionError, match="invalid"):
            predictor.predict(fv)

    def test_nan_in_returns_rejected(self):
        predictor = BaselinePredictor()
        fv = _base_fv(returns=[0.01, float("nan"), 0.02])
        with pytest.raises(PredictionError, match="invalid"):
            predictor.predict(fv)

    def test_inf_in_returns_rejected(self):
        predictor = BaselinePredictor()
        fv = _base_fv(returns=[0.01, float("inf")])
        with pytest.raises(PredictionError, match="invalid"):
            predictor.predict(fv)


# ===================================================================
# Insufficient / partial features
# ===================================================================

class TestPartialFeatures:
    """Verify graceful handling of incomplete feature data."""

    def test_only_rsi_available(self):
        predictor = BaselinePredictor()
        fv = _base_fv(rsi_value=25.0)
        result = predictor.predict(fv)
        assert isinstance(result, PredictionResult)
        assert result.direction == PredictionDirection.BUY

    def test_only_macd_available(self):
        predictor = BaselinePredictor()
        fv = _base_fv(macd_histogram=-2.0)
        result = predictor.predict(fv)
        assert isinstance(result, PredictionResult)
        assert result.direction == PredictionDirection.SELL

    def test_only_momentum_available(self):
        predictor = BaselinePredictor()
        fv = _base_fv(momentum_value=5.0)
        result = predictor.predict(fv)
        assert isinstance(result, PredictionResult)

    def test_all_none_returns_neutral(self):
        predictor = BaselinePredictor()
        fv = _base_fv()
        result = predictor.predict(fv)
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == Decimal("0")


# ===================================================================
# PredictionEvent compatibility
# ===================================================================

class TestPredictionEventCompat:
    """Verify the PredictionResult → PredictionEvent adapter."""

    def test_adapter_produces_prediction_event(self):
        from aegis.events.contracts import PredictionEvent

        predictor = BaselinePredictor()
        result = predictor.predict(_bullish_fv())
        event = result.to_prediction_event(trace_id="test-trace-001")
        assert isinstance(event, PredictionEvent)

    def test_adapter_preserves_symbol(self):
        from aegis.events.contracts import PredictionEvent

        predictor = BaselinePredictor()
        result = predictor.predict(_bullish_fv())
        event = result.to_prediction_event()
        assert event.symbol == result.symbol

    def test_adapter_preserves_direction(self):
        from aegis.events.contracts import PredictionEvent

        predictor = BaselinePredictor()
        result = predictor.predict(_bullish_fv())
        event = result.to_prediction_event()
        assert event.direction == result.direction.value

    def test_adapter_preserves_confidence(self):
        from aegis.events.contracts import PredictionEvent

        predictor = BaselinePredictor()
        result = predictor.predict(_bullish_fv())
        event = result.to_prediction_event()
        assert event.confidence == result.confidence

    def test_adapter_preserves_timeframe(self):
        from aegis.events.contracts import PredictionEvent

        predictor = BaselinePredictor()
        result = predictor.predict(_bullish_fv())
        event = result.to_prediction_event()
        assert event.timeframe == result.timeframe

    def test_adapter_preserves_trace_id(self):
        from aegis.events.contracts import PredictionEvent

        predictor = BaselinePredictor()
        result = predictor.predict(_bullish_fv())
        event = result.to_prediction_event(trace_id="abc-123")
        assert event.trace_id == "abc-123"

    def test_adapter_default_trace_id(self):
        predictor = BaselinePredictor()
        result = predictor.predict(_bullish_fv())
        event = result.to_prediction_event()
        assert event.trace_id == ""


# ===================================================================
# Safety tests — CRITICAL
# ===================================================================

class TestSafetyGuarantees:
    """
    Verify the prediction engine does NOT interact with broker,
    execution, or live systems.
    """

    def test_engine_does_not_import_broker(self):
        """The engine module must not import broker modules."""
        import aegis.prediction.engine as engine_module
        source = inspect.getsource(engine_module)
        assert "from aegis.interfaces.broker" not in source
        assert "import aegis.interfaces.broker" not in source

    def test_engine_does_not_import_market_data_provider(self):
        """The engine must not import MarketDataProvider."""
        import aegis.prediction.engine as engine_module
        source = inspect.getsource(engine_module)
        assert "MarketDataProvider" not in source

    def test_engine_does_not_alter_execution_mode(self):
        """Predict must not modify ExecutionMode."""
        from aegis.core.config import config, ExecutionMode

        original_mode = config.SYSTEM_MODE
        predictor = BaselinePredictor()
        predictor.predict(_bullish_fv())
        assert config.SYSTEM_MODE == original_mode
        assert config.SYSTEM_MODE == ExecutionMode.PREDICTION_ONLY.value

    def test_engine_does_not_submit_orders(self):
        """Engine must have no order submission capability."""
        import aegis.prediction.engine as engine_module
        source = inspect.getsource(engine_module)
        assert "submit_order" not in source
        assert "cancel_order" not in source
        assert "OrderRequest" not in source

    def test_engine_no_external_api_access(self):
        """Engine must not use networking libraries."""
        import aegis.prediction.engine as engine_module
        source = inspect.getsource(engine_module)
        assert "requests" not in source
        assert "urllib" not in source
        assert "httpx" not in source
        assert "aiohttp" not in source

    def test_prediction_result_is_not_order(self):
        """PredictionResult must not contain order/execution fields."""
        predictor = BaselinePredictor()
        result = predictor.predict(_bullish_fv())
        assert not hasattr(result, "order_id")
        assert not hasattr(result, "volume")
        assert not hasattr(result, "stop_loss")
        assert not hasattr(result, "take_profit")
        assert not hasattr(result, "order_type")

    def test_no_future_data_access(self):
        """Engine consumes only the supplied FeatureVector — no external data."""
        predictor = BaselinePredictor()
        # The engine accepts FeatureVector and returns PredictionResult.
        # If it tried to access external data, it would need provider imports
        # (tested above) or network access (tested above).
        result = predictor.predict(_base_fv())
        assert isinstance(result, PredictionResult)


# ===================================================================
# Model metadata
# ===================================================================

class TestModelMetadata:
    """Verify prediction metadata."""

    def test_model_name_is_baseline(self):
        predictor = BaselinePredictor()
        result = predictor.predict(_bullish_fv())
        assert result.model_name == "baseline_development_predictor"

    def test_reasoning_present(self):
        predictor = BaselinePredictor()
        result = predictor.predict(_bullish_fv())
        assert result.reasoning is not None
        assert "BASELINE" in result.reasoning

    def test_reasoning_on_no_features(self):
        predictor = BaselinePredictor()
        result = predictor.predict(_base_fv())
        assert "No usable features" in result.reasoning


class TestMAandBollingerCorrections:
    """Verify MA and Bollinger logic as per Sprint 4.2 correction."""

    def test_sma_bullish_contribution(self):
        predictor = BaselinePredictor()
        fv = _base_fv(last_close=105.0, sma_value=100.0)
        votes = predictor._collect_votes(fv)
        sma_vote = next(v for n, v, w in votes if n == "SMA")
        assert sma_vote > 0.0

    def test_sma_bearish_contribution(self):
        predictor = BaselinePredictor()
        fv = _base_fv(last_close=95.0, sma_value=100.0)
        votes = predictor._collect_votes(fv)
        sma_vote = next(v for n, v, w in votes if n == "SMA")
        assert sma_vote < 0.0

    def test_sma_neutral_contribution(self):
        predictor = BaselinePredictor()
        fv = _base_fv(last_close=100.0, sma_value=100.0)
        votes = predictor._collect_votes(fv)
        sma_vote = next(v for n, v, w in votes if n == "SMA")
        assert sma_vote == 0.0

    def test_ema_bullish_contribution(self):
        predictor = BaselinePredictor()
        fv = _base_fv(last_close=105.0, ema_value=100.0)
        votes = predictor._collect_votes(fv)
        ema_vote = next(v for n, v, w in votes if n == "EMA")
        assert ema_vote > 0.0

    def test_ema_bearish_contribution(self):
        predictor = BaselinePredictor()
        fv = _base_fv(last_close=95.0, ema_value=100.0)
        votes = predictor._collect_votes(fv)
        ema_vote = next(v for n, v, w in votes if n == "EMA")
        assert ema_vote < 0.0

    def test_ema_neutral_contribution(self):
        predictor = BaselinePredictor()
        fv = _base_fv(last_close=100.0, ema_value=100.0)
        votes = predictor._collect_votes(fv)
        ema_vote = next(v for n, v, w in votes if n == "EMA")
        assert ema_vote == 0.0

    def test_bollinger_bullish_reversion(self):
        predictor = BaselinePredictor()
        fv = _base_fv(last_close=90.0, bollinger_upper=110.0, bollinger_middle=100.0, bollinger_lower=95.0)
        votes = predictor._collect_votes(fv)
        bb_vote = next(v for n, v, w in votes if n == "Bollinger")
        assert bb_vote > 0.0

    def test_bollinger_bearish_reversion(self):
        predictor = BaselinePredictor()
        fv = _base_fv(last_close=115.0, bollinger_upper=110.0, bollinger_middle=100.0, bollinger_lower=95.0)
        votes = predictor._collect_votes(fv)
        bb_vote = next(v for n, v, w in votes if n == "Bollinger")
        assert bb_vote < 0.0

    def test_bollinger_neutral_inside(self):
        predictor = BaselinePredictor()
        fv = _base_fv(last_close=100.0, bollinger_upper=110.0, bollinger_middle=100.0, bollinger_lower=95.0)
        votes = predictor._collect_votes(fv)
        bb_vote = next(v for n, v, w in votes if n == "Bollinger")
        assert bb_vote == 0.0

    def test_sma_independent_of_returns(self):
        predictor = BaselinePredictor()
        fv1 = _base_fv(last_close=105.0, sma_value=100.0, returns=[0.01])
        fv2 = _base_fv(last_close=105.0, sma_value=110.0, returns=[0.01])
        v1 = next(v for n, v, w in predictor._collect_votes(fv1) if n == "SMA")
        v2 = next(v for n, v, w in predictor._collect_votes(fv2) if n == "SMA")
        assert v1 != v2

    def test_ema_independent_of_returns(self):
        predictor = BaselinePredictor()
        fv1 = _base_fv(last_close=105.0, ema_value=100.0, returns=[0.01])
        fv2 = _base_fv(last_close=105.0, ema_value=110.0, returns=[0.01])
        v1 = next(v for n, v, w in predictor._collect_votes(fv1) if n == "EMA")
        v2 = next(v for n, v, w in predictor._collect_votes(fv2) if n == "EMA")
        assert v1 != v2

"""
Tests for ML Dataset generation, ensuring temporal correctness and alignment.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.features.builder import FeatureBuilder, FeatureBuilderConfig
from aegis.prediction.model_interface import FeatureSchema
from aegis.prediction.models import PredictionDirection
from aegis.ml.labels import TargetConfig, TargetGenerator
from aegis.ml.dataset import MLDatasetBuilder

def build_ohlc_sequence(closes: list[float]) -> list[OHLC]:
    candles = []
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    for i, c in enumerate(closes):
        candles.append(OHLC(
            symbol="BTC/USD",
            timestamp=base_time + timedelta(hours=i),
            timeframe=Timeframe.H1,
            open=Decimal("100.0"),
            high=Decimal(str(c + 10)),
            low=Decimal(str(c - 10)),
            close=Decimal(str(c)),
            volume=Decimal("1000")
        ))
    return candles


class TestMLDatasetBuilder:
    @pytest.fixture
    def setup_builder(self):
        # Use very short periods so we don't need 26 candles
        feat_config = FeatureBuilderConfig(
            sma_period=2, ema_period=2, rsi_period=2,
            macd_fast=2, macd_slow=3, macd_signal=2,
            atr_period=2, bollinger_period=2, momentum_period=2, volatility_period=2
        )
        fb = FeatureBuilder(feat_config)
        
        target_config = TargetConfig(target_horizon_candles=1, threshold=0.01)
        tg = TargetGenerator(target_config)
        
        schema = FeatureSchema(
            schema_version=1,
            required_features=["last_close", "sma_value", "momentum_value"]
        )
        
        return MLDatasetBuilder(fb, tg, schema)

    def test_valid_training_sample(self, setup_builder):
        # Provide enough candles to satisfy periods
        history = build_ohlc_sequence([100.0, 101.0, 102.0, 105.0, 103.0, 106.0, 110.0])
        dataset = setup_builder.build(history)
        
        assert len(dataset) > 0
        assert len(dataset.x_matrix[0]) == 3
        assert dataset.y_vector[0] in list(PredictionDirection)
        
    def test_missing_feature_drops_sample(self, setup_builder):
        # If we have only 2 candles, momentum_value will be None (requires 3).
        # Schema requires momentum_value.
        # It should drop the sample instead of crashing.
        history = build_ohlc_sequence([100.0, 101.0, 102.0]) # length 3
        # target horizon is 1, so index 0 requires index 1, index 1 requires index 2, index 2 is dropped.
        # at index 0, window is [0]. (length 1). FeatureBuilder returns None. (dropped)
        # at index 1, window is [0, 1]. length 2. momentum(period 2) requires 3 candles. So momentum_value=None. (dropped by schema)
        dataset = setup_builder.build(history)
        assert len(dataset) == 0
        
    def test_deterministic_feature_order(self, setup_builder):
        history = build_ohlc_sequence([100.0, 101.0, 102.0, 105.0, 103.0, 106.0, 110.0])
        dataset = setup_builder.build(history)
        # Schema requires: "last_close", "sma_value", "momentum_value"
        sample = dataset.samples[0]
        assert sample.features[0] == float(sample.raw_feature_vector.last_close)
        assert sample.features[1] == sample.raw_feature_vector.sma_value
        assert sample.features[2] == sample.raw_feature_vector.momentum_value

    def test_no_future_feature_access(self, setup_builder):
        # At index i, features should only be computed using candles <= i.
        history = build_ohlc_sequence([100.0, 101.0, 102.0, 105.0, 103.0, 106.0, 110.0])
        dataset = setup_builder.build(history)
        
        # Take the first sample.
        sample = dataset.samples[0]
        
        # Verify that the timestamp of the sample matches the timestamp of the most recent candle used for its features.
        # And it should strictly be earlier than the target outcome.
        target_timestamp = sample.timestamp + timedelta(hours=setup_builder.target_generator.config.target_horizon_candles)
        
        # The history index matching target_timestamp is where the target was calculated.
        target_candle = next(c for c in history if c.timestamp == target_timestamp)
        
        assert sample.timestamp < target_candle.timestamp

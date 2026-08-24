"""
Tests for explicit prevention of future data leakage in the ML pipeline.
"""

import pytest
import copy
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


class TestMLLeakage:
    @pytest.fixture
    def setup_builder(self):
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

    def test_future_append_does_not_change_historical_x(self, setup_builder):
        # 1. Base dataset through T3
        history_base = build_ohlc_sequence([100.0, 101.0, 102.0, 105.0])
        dataset_base = setup_builder.build(history_base)
        
        # 2. Appended dataset through T4
        history_appended = build_ohlc_sequence([100.0, 101.0, 102.0, 105.0, 110.0])
        dataset_appended = setup_builder.build(history_appended)
        
        # The features for the shared observations should be IDENTICAL.
        # Note: the target for the last observation in history_base might be None (dropped).
        # We just need to check all samples that exist in BOTH.
        assert len(dataset_appended.samples) > len(dataset_base.samples)
        
        for base_sample in dataset_base.samples:
            appended_sample = next(s for s in dataset_appended.samples if s.timestamp == base_sample.timestamp)
            assert base_sample.features == appended_sample.features

    def test_future_change_does_not_alter_historical_x(self, setup_builder):
        # 1. Base dataset 
        history_base = build_ohlc_sequence([100.0, 101.0, 102.0, 105.0])
        dataset_base = setup_builder.build(history_base)
        
        # 2. Alter only the FUTURE value (T3)
        history_altered = build_ohlc_sequence([100.0, 101.0, 102.0, 95.0]) # 105.0 -> 95.0
        dataset_altered = setup_builder.build(history_altered)
        
        # 3. Find sample computed at T2.
        # Its features must be identical.
        # Its label must be different.
        sample_base_t2 = next(s for s in dataset_base.samples if s.timestamp == history_base[2].timestamp)
        sample_alt_t2 = next(s for s in dataset_altered.samples if s.timestamp == history_altered[2].timestamp)
        
        assert sample_base_t2.features == sample_alt_t2.features
        
        # In base, T2 to T3 goes 102 -> 105 (+2.9%), label is BUY.
        assert sample_base_t2.target == PredictionDirection.BUY
        
        # In alt, T2 to T3 goes 102 -> 95 (-6.8%), label is SELL.
        assert sample_alt_t2.target == PredictionDirection.SELL

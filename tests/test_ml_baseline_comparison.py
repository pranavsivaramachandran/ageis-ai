"""
Tests for comparing ML predictions with Baseline predictions to ensure
they can operate on the same data and produce comparable structures.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.features.builder import FeatureBuilder, FeatureBuilderConfig
from aegis.prediction.model_interface import FeatureSchema
from aegis.ml.labels import TargetConfig, TargetGenerator
from aegis.ml.dataset import MLDatasetBuilder
from aegis.ml.training import TrainerConfig, Trainer
from aegis.prediction.engine import BaselinePredictor

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


class TestMLBaselineComparison:
    def test_ml_and_baseline_integration(self):
        closes = []
        val = 100.0
        for i in range(40):
            if i % 3 == 0:
                val *= 1.01
            elif i % 3 == 1:
                val *= 0.99
            else:
                val *= 1.00
            closes.append(val)
            
        history = build_ohlc_sequence(closes)
        
        # Train ML Model
        feat_config = FeatureBuilderConfig()
        fb = FeatureBuilder(feat_config)
        target_config = TargetConfig(target_horizon_candles=1, threshold=0.005)
        tg = TargetGenerator(target_config)
        
        # We use a standard baseline schema-compatible requirement
        schema = FeatureSchema(schema_version=1, required_features=["sma_value", "ema_value", "rsi_value"])
        
        builder = MLDatasetBuilder(fb, tg, schema)
        dataset = builder.build(history)
        
        trainer = Trainer(TrainerConfig(random_state=42), schema)
        ml_model = trainer.train(dataset, model_id="ml_test", version=1)
        
        # Get baseline model
        baseline_model = BaselinePredictor()
        
        # Pick the last sample
        fv = dataset.samples[-1].raw_feature_vector
        
        # Both models should be able to predict on the SAME FeatureVector
        baseline_result = baseline_model.predict(fv)
        ml_result = ml_model.predict(fv)
        
        # Verify structure
        assert baseline_result.symbol == ml_result.symbol == "BTC/USD"
        assert baseline_result.timeframe == ml_result.timeframe
        assert baseline_result.timestamp == ml_result.timestamp
        
        # ML Model has different ID
        assert baseline_result.model_name == "baseline_development_predictor"
        assert ml_result.model_name == "ml_test"

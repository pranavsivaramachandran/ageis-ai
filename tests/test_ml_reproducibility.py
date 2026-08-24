"""
Tests proving the end-to-end reproducibility of the ML pipeline.
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


class TestMLReproducibility:
    def test_end_to_end_reproducibility(self):
        closes = []
        val = 100.0
        # Need enough data for 10 samples.
        for i in range(30):
            if i % 3 == 0:
                val *= 1.01
            elif i % 3 == 1:
                val *= 0.99
            else:
                val *= 1.00
            closes.append(val)
            
        history = build_ohlc_sequence(closes)
        
        feat_config = FeatureBuilderConfig(sma_period=3, momentum_period=3)
        fb = FeatureBuilder(feat_config)
        target_config = TargetConfig(target_horizon_candles=1, threshold=0.005)
        tg = TargetGenerator(target_config)
        schema = FeatureSchema(schema_version=1, required_features=["last_close", "sma_value", "momentum_value"])
        
        # Pipeline 1
        builder1 = MLDatasetBuilder(fb, tg, schema)
        dataset1 = builder1.build(history)
        trainer1 = Trainer(TrainerConfig(random_state=999), schema)
        model1 = trainer1.train(dataset1)
        
        # Pipeline 2
        builder2 = MLDatasetBuilder(fb, tg, schema)
        dataset2 = builder2.build(history)
        trainer2 = Trainer(TrainerConfig(random_state=999), schema)
        model2 = trainer2.train(dataset2)
        
        # Compare every single prediction
        for sample in dataset1.samples:
            fv = sample.raw_feature_vector
            p1 = model1.predict(fv)
            p2 = model2.predict(fv)
            
            assert p1.direction == p2.direction
            assert p1.confidence == p2.confidence

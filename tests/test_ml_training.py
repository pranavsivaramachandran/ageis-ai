"""
Tests for ML Trainer, ensuring it validates inputs and successfully trains models.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import dataclasses

from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.features.builder import FeatureBuilder, FeatureBuilderConfig
from aegis.prediction.model_interface import FeatureSchema
from aegis.prediction.models import PredictionDirection
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

class TestMLTrainer:
    @pytest.fixture
    def setup_data(self):
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
        builder = MLDatasetBuilder(fb, tg, schema)
        
        # Need at least 10 valid samples. With minimum period 2, and horizon 1, we drop some edges.
        # Let's provide 20 alternating closes to generate BUY and SELL labels.
        closes = []
        val = 100.0
        for i in range(20):
            if i % 2 == 0:
                val *= 1.02
            else:
                val *= 0.98
            closes.append(val)
            
        history = build_ohlc_sequence(closes)
        dataset = builder.build(history)
        
        trainer_config = TrainerConfig(random_state=42)
        trainer = Trainer(trainer_config, schema)
        return trainer, dataset
        
    def test_valid_training(self, setup_data):
        trainer, dataset = setup_data
        model = trainer.train(dataset, "test_model", 1)
        
        assert model.is_ready()
        assert model.model_id == "test_model"
        assert model.version == 1
        
    def test_insufficient_samples(self, setup_data):
        trainer, dataset = setup_data
        # truncate dataset to 9 samples (since dataclass is frozen, we can't reassign directly without a bypass, 
        # but MLDataset is a dataclass without frozen=True on the list reference, but it's frozen=True on the object.
        # So we create a new MLDataset)
        from aegis.ml.dataset import MLDataset
        short_dataset = MLDataset(samples=dataset.samples[:9])
        
        with pytest.raises(ValueError, match="Insufficient training samples"):
            trainer.train(short_dataset)
            
    def test_one_class_behavior(self, setup_data):
        trainer, dataset = setup_data
        # Force all targets to BUY
        from aegis.ml.dataset import MLSample, MLDataset
        new_samples = []
        for s in dataset.samples:
            new_samples.append(MLSample(s.timestamp, s.symbol, s.timeframe, s.features, PredictionDirection.BUY, s.raw_feature_vector))
            
        one_class_dataset = MLDataset(samples=new_samples)
        
        with pytest.raises(ValueError, match="must contain at least 2 distinct classes"):
            trainer.train(one_class_dataset)
            
    def test_deterministic_seed(self, setup_data):
        trainer1 = Trainer(TrainerConfig(random_state=42), setup_data[0].schema)
        trainer2 = Trainer(TrainerConfig(random_state=42), setup_data[0].schema)
        
        model1 = trainer1.train(setup_data[1])
        model2 = trainer2.train(setup_data[1])
        
        # Check that the models produce identical predictions on the first sample
        fv = setup_data[1].samples[0].raw_feature_vector
        pred1 = model1.predict(fv)
        pred2 = model2.predict(fv)
        
        assert pred1.direction == pred2.direction
        assert pred1.confidence == pred2.confidence

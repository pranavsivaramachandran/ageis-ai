import sys
from datetime import datetime, timezone, timedelta
from decimal import Decimal

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
print([s.target.name for s in dataset.samples])
print("y_vector:", dataset.y_vector)
import numpy as np
print("unique:", np.unique(np.array(dataset.y_vector)))

trainer_config = TrainerConfig(random_state=42)
trainer = Trainer(trainer_config, schema)
try:
    trainer.train(dataset, "test_model", 1)
    print("Train succeeded!")
except Exception as e:
    print(f"Train failed: {e}")

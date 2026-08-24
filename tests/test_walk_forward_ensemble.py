import pytest
from datetime import datetime, timezone, timedelta
from typing import Sequence

from aegis.interfaces.market_data import OHLC
from aegis.evaluation.walk_forward import WalkForwardEvaluator, WindowGenerator
from aegis.evaluation.models import WalkForwardConfig, WalkForwardStrategy
from aegis.prediction.models import PredictionDirection
from aegis.backtest.models import SimulationConfig
from aegis.ml.training import TrainerConfig

# Simple inline mocks
class MockFeatureBuilder:
    minimum_candles = 5
    def build(self, data):
        from aegis.features.builder import FeatureVector
        from aegis.interfaces.market_data import Timeframe
        return FeatureVector(
            timestamp=data[-1].timestamp,
            symbol="BTC/USD",
            timeframe=Timeframe.H1,
            last_close=float(data[-1].close),
            atr_value=1.5,
            rsi_value=50.0,
            ema_value=100.0,
            macd_line=1.0,
            macd_signal=0.5,
            macd_histogram=0.5
        )

class MockTargetGenerator:
    def get_target(self, idx, history):
        from aegis.prediction.models import PredictionDirection
        return PredictionDirection.BUY if idx % 2 == 0 else PredictionDirection.SELL

class MockSchema:
    required_features = ["rsi_value", "ema_value"]
    def validate_features(self, fv):
        pass

class MockRiskEngine:
    def process(self, *args, **kwargs):
        pass
    def evaluate_prediction(self, *args, **kwargs):
        from unittest.mock import MagicMock
        return MagicMock(status="REJECTED")

def test_walk_forward_ensemble_integration():
    """Verify WalkForwardEvaluator orchestrates models properly."""
    # Setup minimal valid history
    history = []
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(100):
        val = 100.0 + (i % 2) # Alternating
        history.append(OHLC(symbol="BTC/USD", timestamp=base_time + timedelta(hours=i), timeframe="H1",
                            open=val, high=val+1, low=val-1, close=val, volume=100))
                            
    config = WalkForwardConfig(
        strategy=WalkForwardStrategy.ROLLING,
        train_size=40,
        validation_size=20,
        test_size=20,
        step_size=20,
        minimum_train_samples=5
    )
    
    sim_config = SimulationConfig()
    trainer_config = TrainerConfig(max_iter=10)
    
    evaluator = WalkForwardEvaluator(
        feature_builder=MockFeatureBuilder(),
        target_generator=MockTargetGenerator(),
        schema=MockSchema(),
        trainer_config=trainer_config,
        risk_engine=MockRiskEngine(),
        simulation_config=sim_config
    )
    
    report = evaluator.evaluate(history, config, "test_ensemble_experiment")
    
    assert report.total_windows > 0
    assert report.successful_windows > 0
    
    # Check that ensemble metrics are populated
    for window in report.windows:
        if not window.error:
            assert window.metrics.ensemble_pnl is not None
            assert window.metrics.ensemble_total_trades is not None
            assert window.metrics.ensemble_max_drawdown is not None
            assert window.selection_metric == "f1_macro"
            assert "calibrated" in window.ml_model_id

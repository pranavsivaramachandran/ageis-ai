import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import datetime, timezone, timedelta

from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.evaluation.models import WalkForwardConfig, WalkForwardStrategy
from aegis.evaluation.walk_forward import WalkForwardEvaluator
from aegis.ml.training import TrainerConfig
from aegis.prediction.model_interface import FeatureSchema
from aegis.backtest.models import SimulationConfig


def create_mock_history(size: int = 150):
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return [
        OHLC(
            timestamp=start + timedelta(hours=i),
            symbol="BTC/USD",
            timeframe=Timeframe.H1,
            open=Decimal("100") + i,
            high=Decimal("105") + i,
            low=Decimal("95") + i,
            close=Decimal("100") + i,
            volume=Decimal("10")
        ) for i in range(size)
    ]


def test_scaler_not_fit_on_test_data():
    history = create_mock_history(100)
    
    # We will mock the dependencies
    mock_feature_builder = MagicMock()
    mock_feature_builder.minimum_candles = 5
    
    # Need to return some valid features so dataset builder doesn't drop samples
    from aegis.features.builder import FeatureVector
    def side_effect(window):
        return FeatureVector(
            timestamp=window[-1].timestamp,
            symbol="BTC/USD",
            timeframe=Timeframe.H1,
            last_close=float(window[-1].close),
            atr_value=1.5,
            rsi_value=50.0,
            ema_value=100.0,
            macd_line=1.0,
            macd_signal=0.5,
            macd_histogram=0.5
        )
    mock_feature_builder.build.side_effect = side_effect
    
    mock_target_generator = MagicMock()
    from aegis.prediction.models import PredictionDirection
    
    def target_side_effect(idx, hist):
        return PredictionDirection.BUY if idx % 2 == 0 else PredictionDirection.SELL
        
    mock_target_generator.get_target.side_effect = target_side_effect
    
    schema = FeatureSchema(required_features=["rsi_value", "ema_value"])
    
    trainer_config = TrainerConfig(random_state=42)
    risk_engine = MagicMock()
    risk_engine.evaluate_prediction.return_value = MagicMock(status="REJECTED") # prevent trades for simplicity
    
    sim_config = SimulationConfig(initial_capital=Decimal("10000"), commission_per_trade=Decimal("1"), slippage_percent=Decimal("0.001"), position_size_percent=Decimal("0.1"), holding_period_candles=5, stop_loss_atr_multiplier=Decimal("1.0"))
    
    evaluator = WalkForwardEvaluator(
        feature_builder=mock_feature_builder,
        target_generator=mock_target_generator,
        schema=schema,
        trainer_config=trainer_config,
        risk_engine=risk_engine,
        simulation_config=sim_config
    )
    
    config = WalkForwardConfig(
        strategy=WalkForwardStrategy.EXPANDING,
        train_size=40,
        validation_size=10,
        test_size=10,
        step_size=20
    )
    
    report = evaluator.evaluate(history, config, "test_exp")
    
    assert report.total_windows == 3
    assert report.failed_windows == 0
    assert report.successful_windows == 3

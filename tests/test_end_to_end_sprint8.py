"""
Sprint 8 End-to-End Prediction Architecture Test.

Validates the full pipeline:
Dataset -> FeatureBuilder -> PredictionEngine(BaselinePredictor) -> RiskEngine -> BacktestEngine.
No brokers, no live execution, strictly deterministic.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal

from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.features.builder import FeatureBuilder
from aegis.prediction.engine import BaselinePredictor, PredictionEngine
from aegis.prediction.registry import ModelRegistry
from aegis.risk.engine import RiskManagementEngine
from aegis.core.config import config as global_config
from aegis.backtest.engine import BacktestEngine
from aegis.backtest.models import SimulationConfig
from aegis.experiments.dataset import ChronologicalSplitter
from unittest.mock import patch


def make_candle(dt: datetime, val: float) -> OHLC:
    return OHLC(
        symbol="EUR/USD",
        timestamp=dt,
        timeframe=Timeframe.H1,
        open=Decimal(str(val)),
        high=Decimal(str(val + 0.5)),
        low=Decimal(str(val - 0.5)),
        close=Decimal(str(val)),
        volume=Decimal("100")
    )


def test_full_pipeline_end_to_end():
    # 1. Generate Dataset
    dt = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    # 50 candles trending up
    history = [make_candle(dt + timedelta(hours=i), 100.0 + i * 0.1) for i in range(50)]
    
    # 2. Setup Architecture
    fb = FeatureBuilder()
    
    # Register and wrap predictor
    registry = ModelRegistry()
    baseline = BaselinePredictor(direction_threshold=0.05)
    registry.register(baseline)
    
    model = registry.get("baseline", 1)
    pe = PredictionEngine(model)
    
    
    # Backtest Config
    config = SimulationConfig(initial_capital=Decimal("10000"), holding_period_candles=3)
    
    with patch('aegis.core.config.config.RISK_MIN_CONFIDENCE', Decimal("0.0")), \
         patch('aegis.core.config.config.RISK_MAX_POSITION_SIZE', Decimal("1000.0")):
        re = RiskManagementEngine()
        backtester = BacktestEngine(fb, pe, re, config)
        
        # 3. Execute
        report = backtester.run(history)
    
    # 4. Verify
    assert report.total_trades > 0
    assert report.symbol == "EUR/USD"
    
    # The pipeline must run completely end-to-end without failing.

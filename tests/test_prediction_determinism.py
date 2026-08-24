"""
Tests for BaselinePredictor determinism.
"""

from datetime import datetime, timezone
from aegis.interfaces.market_data import Timeframe
from aegis.features.builder import FeatureVector
from aegis.prediction.engine import BaselinePredictor


def test_baseline_predictor_determinism():
    fv1 = FeatureVector(
        timestamp=datetime.now(timezone.utc),
        symbol="EUR/USD",
        timeframe=Timeframe.H1,
        last_close=100.0,
        sma_value=105.0,
        rsi_value=25.0
    )
    
    fv2 = FeatureVector(
        timestamp=fv1.timestamp,
        symbol="EUR/USD",
        timeframe=Timeframe.H1,
        last_close=100.0,
        sma_value=105.0,
        rsi_value=25.0
    )
    
    predictor = BaselinePredictor()
    
    res1 = predictor.predict(fv1)
    res2 = predictor.predict(fv2)
    
    assert res1.direction == res2.direction
    assert res1.confidence == res2.confidence
    assert res1.reasoning == res2.reasoning

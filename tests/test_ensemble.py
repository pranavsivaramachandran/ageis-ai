import pytest
from datetime import datetime, timezone
from pydantic import BaseModel
from decimal import Decimal

from aegis.prediction.models import PredictionResult, PredictionDirection
from aegis.prediction.model_interface import PredictionModel
from aegis.ml.ensemble import EnsemblePredictionModel

class MockPredictionModel(PredictionModel):
    def __init__(self, fixed_probs: dict[PredictionDirection, float], model_id: str):
        self.fixed_probs = fixed_probs
        self._model_id = model_id
        self._version = 1
        
    @property
    def model_id(self) -> str:
        return self._model_id
        
    @property
    def version(self) -> int:
        return self._version
        
    @property
    def schema(self):
        class MockSchema:
            pass
        return MockSchema()

    def is_ready(self) -> bool:
        return True
        
    def predict(self, feature_vector: BaseModel) -> PredictionResult:
        # Get highest prob
        direction = max(self.fixed_probs, key=self.fixed_probs.get)
        confidence = self.fixed_probs[direction]
        
        return PredictionResult(
            symbol=feature_vector.symbol,
            timestamp=feature_vector.timestamp,
            timeframe=feature_vector.timeframe,
            direction=direction,
            confidence=confidence,
            model_name=self.model_id,
            reasoning=None
        )

    def predict_proba(self, feature_vector: BaseModel) -> dict[PredictionDirection, float]:
        return self.fixed_probs


class MockFeatureVector(BaseModel):
    symbol: str = "BTC/USD"
    timestamp: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timeframe: str = "H1"
    

def test_ensemble_initialization():
    model1 = MockPredictionModel({PredictionDirection.BUY: 0.8, PredictionDirection.SELL: 0.2}, "m1")
    model2 = MockPredictionModel({PredictionDirection.BUY: 0.6, PredictionDirection.SELL: 0.4}, "m2")
    
    # Valid
    ensemble = EnsemblePredictionModel([model1, model2], [0.5, 0.5])
    assert len(ensemble._models) == 2
    
    # Invalid weights sum
    with pytest.raises(ValueError, match="Weights must sum to 1"):
        EnsemblePredictionModel([model1, model2], [0.5, 0.6])
        
    # Invalid weights length
    with pytest.raises(ValueError, match="Number of models and weights must match"):
        EnsemblePredictionModel([model1, model2], [1.0])
        
    # Invalid weight negative
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        EnsemblePredictionModel([model1, model2], [-0.1, 1.1])
        
def test_ensemble_probability_averaging():
    model1 = MockPredictionModel({
        PredictionDirection.BUY: 0.8, 
        PredictionDirection.SELL: 0.1, 
        PredictionDirection.NEUTRAL: 0.1
    }, "m1")
    
    model2 = MockPredictionModel({
        PredictionDirection.BUY: 0.2, 
        PredictionDirection.SELL: 0.6, 
        PredictionDirection.NEUTRAL: 0.2
    }, "m2")
    
    # Equal weights
    ensemble = EnsemblePredictionModel([model1, model2], [0.5, 0.5])
    
    fv = MockFeatureVector()
    probs = ensemble.predict_proba(fv)
    
    # Expected: BUY (0.8+0.2)/2 = 0.5
    # SELL (0.1+0.6)/2 = 0.35
    # NEUTRAL (0.1+0.2)/2 = 0.15
    assert abs(probs[PredictionDirection.BUY] - 0.5) < 1e-6
    assert abs(probs[PredictionDirection.SELL] - 0.35) < 1e-6
    assert abs(probs[PredictionDirection.NEUTRAL] - 0.15) < 1e-6
    
    pred = ensemble.predict(fv)
    assert pred.direction == PredictionDirection.BUY
    assert abs(float(pred.confidence) - 0.5) < 1e-6
    assert pred.model_name == "ensemble_m1_m2"
    
def test_ensemble_weighted_averaging():
    model1 = MockPredictionModel({
        PredictionDirection.BUY: 0.8, 
        PredictionDirection.SELL: 0.1, 
        PredictionDirection.NEUTRAL: 0.1
    }, "m1")
    
    model2 = MockPredictionModel({
        PredictionDirection.BUY: 0.2, 
        PredictionDirection.SELL: 0.6, 
        PredictionDirection.NEUTRAL: 0.2
    }, "m2")
    
    # Weight model 2 more heavily
    ensemble = EnsemblePredictionModel([model1, model2], [0.2, 0.8])
    
    fv = MockFeatureVector()
    probs = ensemble.predict_proba(fv)
    
    # Expected: BUY 0.8*0.2 + 0.2*0.8 = 0.16 + 0.16 = 0.32
    # SELL 0.1*0.2 + 0.6*0.8 = 0.02 + 0.48 = 0.50
    # NEUTRAL 0.1*0.2 + 0.2*0.8 = 0.02 + 0.16 = 0.18
    assert abs(probs[PredictionDirection.BUY] - 0.32) < 1e-6
    assert abs(probs[PredictionDirection.SELL] - 0.50) < 1e-6
    assert abs(probs[PredictionDirection.NEUTRAL] - 0.18) < 1e-6
    
    pred = ensemble.predict(fv)
    assert pred.direction == PredictionDirection.SELL
    assert abs(float(pred.confidence) - 0.50) < 1e-6

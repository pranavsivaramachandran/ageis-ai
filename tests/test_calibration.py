import pytest
import numpy as np
from datetime import datetime, timezone
from pydantic import BaseModel

from aegis.prediction.models import PredictionResult, PredictionDirection
from aegis.prediction.model_interface import PredictionModel
from aegis.ml.calibration import CalibratedPredictionModel, IsotonicCalibrator

class MockPredictionModel(PredictionModel):
    def __init__(self, fixed_probs: dict[PredictionDirection, float]):
        self.fixed_probs = fixed_probs
        self._model_id = "mock_model"
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
        # Mock probabilities
        return PredictionResult(
            symbol=feature_vector.symbol,
            timestamp=feature_vector.timestamp,
            timeframe=feature_vector.timeframe,
            direction=PredictionDirection.BUY,
            confidence=self.fixed_probs[PredictionDirection.BUY],
            model_name=self.model_id,
            reasoning=None
        )

    def predict_proba(self, feature_vector: BaseModel) -> dict[PredictionDirection, float]:
        return self.fixed_probs


class MockFeatureVector(BaseModel):
    symbol: str = "BTC/USD"
    timestamp: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timeframe: str = "H1"
    

def test_isotonic_calibrator():
    calibrator = IsotonicCalibrator()
    
    # Train calibrator
    raw_probs = [0.1, 0.4, 0.9, 0.8]
    true_labels = [0, 0, 1, 1]
    
    calibrator.fit(raw_probs, true_labels)
    
    assert calibrator.is_fitted
    
    # Predict
    calibrated = calibrator.predict([0.2, 0.85])
    assert len(calibrated) == 2
    assert 0.0 <= calibrated[0] <= 1.0
    assert 0.0 <= calibrated[1] <= 1.0
    
def test_calibrated_prediction_model():
    base_model = MockPredictionModel({
        PredictionDirection.BUY: 0.8,
        PredictionDirection.SELL: 0.1,
        PredictionDirection.NEUTRAL: 0.1
    })
    
    # Create calibrators
    buy_cal = IsotonicCalibrator()
    buy_cal.fit([0.2, 0.8], [0, 1])
    
    sell_cal = IsotonicCalibrator()
    sell_cal.fit([0.1, 0.9], [0, 1])
    
    neutral_cal = IsotonicCalibrator()
    neutral_cal.fit([0.1, 0.9], [0, 1])
    
    calibrators = {
        PredictionDirection.BUY: buy_cal,
        PredictionDirection.SELL: sell_cal,
        PredictionDirection.NEUTRAL: neutral_cal
    }
    
    calibrated_model = CalibratedPredictionModel(base_model, calibrators)
    
    fv = MockFeatureVector()
    pred = calibrated_model.predict(fv)
    
    assert isinstance(pred, PredictionResult)
    assert pred.model_name.endswith("_calibrated")
    assert 0.0 <= pred.confidence <= 1.0
    
def test_unfitted_calibrator_raises():
    base_model = MockPredictionModel({PredictionDirection.BUY: 0.8})
    calibrators = {PredictionDirection.BUY: IsotonicCalibrator()}
    
    calibrated_model = CalibratedPredictionModel(base_model, calibrators)
    fv = MockFeatureVector()
    
    with pytest.raises(ValueError, match="Calibrator for .* is not fitted"):
        calibrated_model.predict(fv)

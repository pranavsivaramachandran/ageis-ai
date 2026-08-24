"""
Probability calibration for ML models.
"""
from typing import Dict, List, Any
import numpy as np
from sklearn.isotonic import IsotonicRegression

from pydantic import BaseModel

from aegis.prediction.models import PredictionDirection, PredictionResult
from aegis.prediction.model_interface import PredictionModel


class IsotonicCalibrator:
    """Wrapper around scikit-learn IsotonicRegression."""
    
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.is_fitted = False
        
    def fit(self, raw_probs: List[float], true_labels: List[int]):
        """
        Fit the calibrator.
        Args:
            raw_probs: Raw probabilities from the model.
            true_labels: Binary true labels (1 if this class, 0 otherwise).
        """
        self.calibrator.fit(raw_probs, true_labels)
        self.is_fitted = True
        
    def predict(self, raw_probs: List[float]) -> List[float]:
        """Apply calibration."""
        if not self.is_fitted:
            raise ValueError("Calibrator is not fitted.")
        calibrated = self.calibrator.predict(raw_probs)
        return [float(p) for p in calibrated]


class CalibratedPredictionModel(PredictionModel):
    """
    Wraps a base PredictionModel and applies probability calibration 
    independently per class.
    """
    
    def __init__(self, base_model: PredictionModel, calibrators: Dict[PredictionDirection, IsotonicCalibrator]):
        """
        Args:
            base_model: The underlying model.
            calibrators: A fitted calibrator per class direction.
        """
        self.base_model = base_model
        self.calibrators = calibrators
        
        self._model_id = f"{base_model.model_id}_calibrated"
        self._version = base_model.version
        
    @property
    def model_id(self) -> str:
        return self._model_id
        
    @property
    def version(self) -> int:
        return self._version
        
    @property
    def schema(self):
        return self.base_model.schema

    def is_ready(self) -> bool:
        return self.base_model.is_ready()
        
    def predict(self, feature_vector: BaseModel) -> PredictionResult:
        """Predicts and returns calibrated probabilities."""
        calibrated_probs = self.predict_proba(feature_vector)
        
        # Select best direction
        best_direction = max(calibrated_probs, key=calibrated_probs.get) # type: ignore
        best_confidence = calibrated_probs[best_direction]
        
        return PredictionResult(
            symbol=feature_vector.symbol, # type: ignore
            timestamp=feature_vector.timestamp, # type: ignore
            timeframe=feature_vector.timeframe, # type: ignore
            direction=best_direction,
            confidence=best_confidence,
            model_name=self.model_id,
            reasoning="Calibrated"
        )
        
    def predict_proba(self, feature_vector: BaseModel) -> Dict[PredictionDirection, float]:
        """Returns calibrated probabilities for all classes."""
        raw_probs = self.base_model.predict_proba(feature_vector)
        
        calibrated = {}
        for direction, raw_prob in raw_probs.items():
            if direction not in self.calibrators:
                # If no calibrator, just use raw prob (though usually we have one per class)
                calibrated[direction] = raw_prob
                continue
                
            calibrator = self.calibrators[direction]
            if not calibrator.is_fitted:
                raise ValueError(f"Calibrator for {direction} is not fitted.")
                
            calibrated[direction] = calibrator.predict([raw_prob])[0]
            
        # Optional: Normalize to sum to 1? Calibration doesn't strictly guarantee sum=1 for multi-class OVA.
        # But for probabilities it's good practice.
        total = sum(calibrated.values())
        if total > 0:
            calibrated = {d: p / total for d, p in calibrated.items()}
            
        return calibrated

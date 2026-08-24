"""
Ensemble models for prediction.
"""
from typing import List, Dict

from pydantic import BaseModel

from aegis.prediction.models import PredictionDirection, PredictionResult
from aegis.prediction.model_interface import PredictionModel


class EnsemblePredictionModel(PredictionModel):
    """
    An ensemble of multiple PredictionModels using weighted probability averaging.
    """
    
    def __init__(self, models: List[PredictionModel], weights: List[float]):
        """
        Args:
            models: List of PredictionModel instances.
            weights: List of weights (must sum to 1).
        """
        if len(models) != len(weights):
            raise ValueError("Number of models and weights must match")
            
        for w in weights:
            if w < 0:
                raise ValueError("Weights must be non-negative")
                
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1, got {total_weight}")
            
        self._models = models
        self._weights = weights
        
        # Build model ID deterministically from components
        model_names = [m.model_id for m in models]
        self._model_id = f"ensemble_{'_'.join(model_names)}"
        self._version = 1
        
    @property
    def model_id(self) -> str:
        return self._model_id
        
    @property
    def version(self) -> int:
        return self._version
        
    @property
    def schema(self):
        return self._models[0].schema

    def is_ready(self) -> bool:
        return all(m.is_ready() for m in self._models)
        
    def predict(self, feature_vector: BaseModel) -> PredictionResult:
        """
        Predict direction using highest ensemble probability.
        """
        ensemble_probs = self.predict_proba(feature_vector)
        
        best_direction = max(ensemble_probs, key=ensemble_probs.get) # type: ignore
        best_confidence = ensemble_probs[best_direction]
        
        return PredictionResult(
            symbol=feature_vector.symbol, # type: ignore
            timestamp=feature_vector.timestamp, # type: ignore
            timeframe=feature_vector.timeframe, # type: ignore
            direction=best_direction,
            confidence=best_confidence,
            model_name=self.model_id,
            reasoning=f"Ensemble averaging ({len(self._models)} models)"
        )

    def predict_proba(self, feature_vector: BaseModel) -> Dict[PredictionDirection, float]:
        """
        Calculate weighted average probabilities.
        """
        ensemble_probs: Dict[PredictionDirection, float] = {
            PredictionDirection.BUY: 0.0,
            PredictionDirection.SELL: 0.0,
            PredictionDirection.NEUTRAL: 0.0
        }
        
        for model, weight in zip(self._models, self._weights):
            probs = model.predict_proba(feature_vector)
            for direction in PredictionDirection:
                # Default to 0 if model doesn't output this class
                ensemble_probs[direction] += probs.get(direction, 0.0) * weight
                
        return ensemble_probs

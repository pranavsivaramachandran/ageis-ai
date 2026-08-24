"""
ML Prediction Model for AEGIS AI.

Wraps a trained scikit-learn model and scaler to implement the 
PredictionModel interface safely.
"""

from decimal import Decimal
import numpy as np

from aegis.features.builder import FeatureVector
from aegis.prediction.model_interface import PredictionModel, FeatureSchema
from aegis.prediction.models import PredictionResult, PredictionDirection


class MLPredictionModel(PredictionModel):
    """
    Supervised ML model implementing the PredictionModel interface.
    
    This model wraps a trained scikit-learn classifier and scaler.
    It does not train itself or touch the execution flow.
    """
    
    def __init__(
        self,
        model_id: str,
        version: int,
        schema: FeatureSchema,
        classifier,
        scaler,
        classes_mapping: list[PredictionDirection],
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            model_id: Deterministic model ID.
            version: Model version integer.
            schema: The feature schema required by this model.
            classifier: A fitted scikit-learn classifier (e.g. LogisticRegression).
            scaler: A fitted scikit-learn scaler (e.g. StandardScaler).
            classes_mapping: List mapping classifier.classes_ indices to PredictionDirection.
        """
        self._model_id = model_id
        self._version = version
        self._schema = schema
        self._classifier = classifier
        self._scaler = scaler
        self._classes_mapping = classes_mapping
        self._confidence_threshold = confidence_threshold
        
        # Verify the model is actually trained
        # In scikit-learn, fitted classifiers usually have a classes_ attribute
        if not hasattr(self._classifier, "classes_"):
            raise ValueError("Classifier must be fitted before wrapping in MLPredictionModel")
            
        if len(self._classes_mapping) != len(self._classifier.classes_):
            raise ValueError("classes_mapping length must match classifier classes count")

    @property
    def model_id(self) -> str:
        return self._model_id
        
    @property
    def version(self) -> int:
        return self._version
        
    @property
    def schema(self) -> FeatureSchema:
        return self._schema
        
    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold
        
    @confidence_threshold.setter
    def confidence_threshold(self, value: float):
        if not (0.0 <= value <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        self._confidence_threshold = value
        
    @property
    def classifier(self):
        return self._classifier
        
    @property
    def scaler(self):
        return self._scaler

    @property
    def classes_mapping(self) -> list[PredictionDirection]:
        return self._classes_mapping

    def is_ready(self) -> bool:
        # For this offline architecture, if it's instantiated it's ready.
        return True

    def _extract_features(self, fv: FeatureVector) -> list[float]:
        """Extract ordered features based on schema."""
        self._schema.validate_features(fv)
        
        extracted = []
        for feature_name in self._schema.required_features:
            val = getattr(fv, feature_name)
            extracted.append(float(val))
        return extracted

    def predict_proba(self, fv: FeatureVector) -> dict[PredictionDirection, float]:
        """Returns raw probabilities mapped to PredictionDirection."""
        if not self.is_ready():
            raise RuntimeError(f"Model {self.model_id} is not ready.")
            
        raw_features = self._extract_features(fv)
        X = np.array(raw_features).reshape(1, -1)
        if self._scaler is not None:
            X = self._scaler.transform(X)
            
        proba = self._classifier.predict_proba(X)[0]
        
        result = {}
        for idx, direction in enumerate(self._classes_mapping):
            result[direction] = float(proba[idx])
            
        return result

    def predict(self, fv: FeatureVector) -> PredictionResult:
        """
        Produce a deterministic prediction from a FeatureVector.
        """
        probs = self.predict_proba(fv)
        
        # 4. Map to PredictionDirection and Confidence
        direction = max(probs, key=probs.get) # type: ignore
        confidence_val = probs[direction]
        confidence_val = min(max(confidence_val, 0.0), 1.0)
        
        if confidence_val < self._confidence_threshold:
            direction = PredictionDirection.NEUTRAL
            reasoning = f"ML Classification ({self.model_id} v{self.version}): Confidence {confidence_val:.2f} below threshold {self._confidence_threshold:.2f}, predicting NEUTRAL"
        else:
            reasoning = f"ML Classification ({self.model_id} v{self.version}): Confidence {confidence_val:.2f} for {direction.value}"
        
        return PredictionResult(
            symbol=fv.symbol,
            timestamp=fv.timestamp,
            timeframe=fv.timeframe,
            direction=direction,
            confidence=Decimal(str(round(confidence_val, 4))),
            model_name=self.model_id,
            reasoning=reasoning
        )

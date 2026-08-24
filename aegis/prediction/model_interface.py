"""
Model interface and feature schema for AEGIS AI prediction architecture.

Provides a clean abstraction for prediction models (Baseline, ML, Ensembles)
without exposing them to risk, backtesting, or execution logic.
"""

from abc import ABC, abstractmethod
from typing import Any
import math
from pydantic import BaseModel, Field

from aegis.features.builder import FeatureVector
from aegis.prediction.models import PredictionResult


class FeatureSchema(BaseModel):
    """
    Deterministic schema definition for a prediction model's required inputs.
    """
    
    schema_version: int = Field(default=1, gt=0)
    required_features: list[str] = Field(default_factory=list)
    
    model_config = {"frozen": True}
    
    def validate_features(self, fv: FeatureVector) -> None:
        """
        Verify that a given FeatureVector satisfies this schema.
        
        Args:
            fv: The FeatureVector to validate.
            
        Raises:
            ValueError: If a required feature is missing or contains an invalid value (NaN/Infinity).
        """
        # First, ensure all required fields are present and not None
        for feature in self.required_features:
            if not hasattr(fv, feature):
                raise ValueError(f"Feature '{feature}' is missing from FeatureVector entirely")
                
            value = getattr(fv, feature)
            if value is None:
                raise ValueError(f"Required feature '{feature}' is None")
                
        # Next, enforce strict numerical correctness on ALL numerical fields present,
        # whether required or optional, to protect the model from math errors.
        
        # We manually list the known numerical fields of FeatureVector.
        numeric_fields = [
            "sma_value", "ema_value", "rsi_value", "macd_line", "macd_signal", 
            "macd_histogram", "atr_value", "bollinger_upper", "bollinger_middle", 
            "bollinger_lower", "momentum_value", "volatility"
        ]
        
        for name in numeric_fields:
            if hasattr(fv, name):
                value = getattr(fv, name)
                if value is not None and (math.isnan(value) or math.isinf(value)):
                    raise ValueError(f"FeatureVector contains invalid {name}: {value}")
                    
        # Returns list check
        if fv.returns is not None:
            for i, r in enumerate(fv.returns):
                if r is not None and (math.isnan(r) or math.isinf(r)):
                    raise ValueError(f"FeatureVector contains invalid return at index {i}: {r}")

    def is_compatible_with(self, other: "FeatureSchema") -> bool:
        """
        Check if this schema is compatible with another schema.
        Exact match is required for deterministic behavior.
        """
        return self.schema_version == other.schema_version and set(self.required_features) == set(other.required_features)


class PredictionModel(ABC):
    """
    Abstract Base Class for all prediction models.
    
    A PredictionModel is strictly a stateless (or read-only stateful) 
    inference component. It does NOT:
    - Train itself
    - Manage risk
    - Access brokers
    - Modify global state
    """
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Deterministic identity for this model (e.g., 'baseline')."""
        pass
        
    @property
    @abstractmethod
    def version(self) -> int:
        """Version of this model."""
        pass
        
    @property
    @abstractmethod
    def schema(self) -> FeatureSchema:
        """The FeatureSchema expected by this model."""
        pass
        
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the model is fully initialized and ready to predict."""
        pass

    @abstractmethod
    def predict(self, fv: FeatureVector) -> PredictionResult:
        """
        Produce a deterministic prediction from a FeatureVector.
        
        Args:
            fv: Validated FeatureVector.
            
        Returns:
            PredictionResult containing direction and confidence.
            
        Raises:
            RuntimeError: If is_ready() is False.
            ValueError: If the FeatureVector violates the schema.
        """
        pass

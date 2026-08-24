"""
Model registry for AEGIS AI.

Provides deterministic in-process registration and retrieval of prediction models.
"""

from typing import Dict, Optional
from aegis.prediction.model_interface import PredictionModel


class ModelRegistry:
    """
    In-memory deterministic registry for prediction models.
    
    Prevents duplicate registrations and provides safe retrieval.
    """
    
    def __init__(self):
        self._models: Dict[str, PredictionModel] = {}
        
    def register(self, model: PredictionModel) -> None:
        """
        Register a PredictionModel instance.
        
        Args:
            model: The PredictionModel to register.
            
        Raises:
            ValueError: If the model_id is already registered.
        """
        full_id = f"{model.model_id}-v{model.version}"
        if full_id in self._models:
            raise ValueError(f"Model identity '{full_id}' is already registered")
            
        self._models[full_id] = model
        
    def get(self, model_id: str, version: int) -> PredictionModel:
        """
        Retrieve a registered PredictionModel.
        
        Args:
            model_id: The base ID of the model.
            version: The version number.
            
        Returns:
            The registered PredictionModel.
            
        Raises:
            KeyError: If the model is not found.
        """
        full_id = f"{model_id}-v{version}"
        if full_id not in self._models:
            raise KeyError(f"Model identity '{full_id}' not found in registry")
            
        return self._models[full_id]
        
    def list_models(self) -> list[str]:
        """Return a list of all registered model identities."""
        return sorted(list(self._models.keys()))

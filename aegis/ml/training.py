"""
ML Trainer for AEGIS AI.

Trains the MLPredictionModel using scikit-learn without leaking 
future data into the preprocessing steps.
"""

import hashlib

import numpy as np
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aegis.ml.dataset import MLDataset
from aegis.ml.models import MLPredictionModel
from aegis.prediction.model_interface import FeatureSchema
from aegis.prediction.models import PredictionDirection


class TrainerConfig(BaseModel):
    """Configuration for ML model training."""
    
    random_state: int = Field(default=42, description="Seed for reproducible training.")
    max_iter: int = Field(default=1000, description="Max iterations for the solver.")
    c_param: float = Field(default=1.0, description="Inverse of regularization strength.")
    
    model_config = {"frozen": True}
    
    @property
    def identity(self) -> str:
        """Deterministic identity for this trainer config."""
        content = f"{self.random_state}|{self.max_iter}|{self.c_param}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


class Trainer:
    """Trains classical ML models cleanly."""
    
    def __init__(self, config: TrainerConfig, schema: FeatureSchema):
        self.config = config
        self.schema = schema
        
    def train(self, dataset: MLDataset, model_id: str = "ml_baseline", version: int = 1) -> MLPredictionModel:
        """
        Train a model on the provided dataset.
        
        Args:
            dataset: The training dataset.
            model_id: Deterministic ID for the resulting model.
            version: Model version.
            
        Returns:
            A trained MLPredictionModel ready for inference.
            
        Raises:
            ValueError: If dataset is insufficient or has only 1 class.
        """
        if len(dataset) < 10:
            raise ValueError("Insufficient training samples (minimum 10 required).")
            
        X = np.array(dataset.x_matrix)
        y = np.array([d.value for d in dataset.y_vector])
        
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("Training dataset must contain at least 2 distinct classes.")
            
        # Fit scaler on TRAIN ONLY
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit classifier
        classifier = LogisticRegression(
            random_state=self.config.random_state,
            max_iter=self.config.max_iter,
            C=self.config.c_param,
            class_weight="balanced"  # To handle class imbalance typical in finance
        )
        classifier.fit(X_scaled, y)
        
        # Determine classes mapping
        classes_mapping = [PredictionDirection(c) for c in classifier.classes_]
        
        return MLPredictionModel(
            model_id=model_id,
            version=version,
            schema=self.schema,
            classifier=classifier,
            scaler=scaler,
            classes_mapping=classes_mapping
        )

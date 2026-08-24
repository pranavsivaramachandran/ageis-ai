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


class ExpectedWindowFailure(Exception):
    """Raised when a walk-forward window is expected to fail (e.g., insufficient data)."""
    pass


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
        Train a model on the provided dataset. (Maintains backward compatibility by returning LogisticRegression).
        """
        candidates = self.train_candidates(dataset, base_model_id=model_id, version=version)
        # Default to the first candidate (Logistic Regression) for backward compatibility
        return candidates[0]

    def train_candidates(self, dataset: MLDataset, base_model_id: str = "ml", version: int = 1) -> list[MLPredictionModel]:
        """
        Train multiple candidate models on the provided dataset.
        
        Args:
            dataset: The training dataset.
            base_model_id: Deterministic base ID for the resulting models.
            version: Model version.
            
        Returns:
            A list of trained MLPredictionModel candidates ready for inference.
            
        Raises:
            ExpectedWindowFailure: If dataset is insufficient or has only 1 class.
        """
        from sklearn.ensemble import RandomForestClassifier
        
        if len(dataset) < 10:
            raise ExpectedWindowFailure("Insufficient training samples (minimum 10 required).")
            
        X = np.array(dataset.x_matrix)
        y = np.array([d.value for d in dataset.y_vector])
        
        classes = np.unique(y)
        if len(classes) < 2:
            raise ExpectedWindowFailure("Training dataset must contain at least 2 distinct classes.")
            
        # Fit scaler on TRAIN ONLY
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        candidates = []
        
        # Candidate 1: Logistic Regression
        lr = LogisticRegression(
            random_state=self.config.random_state,
            max_iter=self.config.max_iter,
            C=self.config.c_param,
            class_weight="balanced"
        )
        lr.fit(X_scaled, y)
        lr_classes_mapping = [PredictionDirection(c) for c in lr.classes_]
        
        candidates.append(MLPredictionModel(
            model_id=f"{base_model_id}_lr",
            version=version,
            schema=self.schema,
            classifier=lr,
            scaler=scaler,
            classes_mapping=lr_classes_mapping
        ))
        
        # Candidate 2: Random Forest
        rf = RandomForestClassifier(
            random_state=self.config.random_state,
            n_estimators=100,
            max_depth=5,
            class_weight="balanced"
        )
        rf.fit(X_scaled, y)
        rf_classes_mapping = [PredictionDirection(c) for c in rf.classes_]
        
        candidates.append(MLPredictionModel(
            model_id=f"{base_model_id}_rf",
            version=version,
            schema=self.schema,
            classifier=rf,
            scaler=scaler,
            classes_mapping=rf_classes_mapping
        ))
        
        return candidates

"""
ML Dataset generation for AEGIS AI.

Orchestrates the combination of historical data, feature generation,
and target generation into a structured, leakage-free dataset.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from aegis.features.builder import FeatureBuilder, FeatureVector
from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.ml.labels import TargetGenerator
from aegis.prediction.model_interface import FeatureSchema
from aegis.prediction.models import PredictionDirection


@dataclass(frozen=True)
class MLSample:
    """A single training sample."""
    timestamp: datetime
    symbol: str
    timeframe: Timeframe
    features: list[float]
    target: PredictionDirection
    raw_feature_vector: FeatureVector


@dataclass(frozen=True)
class MLDataset:
    """A collection of ML samples."""
    samples: list[MLSample]
    
    @property
    def class_distribution(self) -> dict[PredictionDirection, int]:
        """Get the count of samples per class."""
        counts = Counter(s.target for s in self.samples)
        # Ensure all classes are present in the dict even if 0
        for direction in PredictionDirection:
            if direction not in counts:
                counts[direction] = 0
        return dict(counts)
    
    @property
    def x_matrix(self) -> list[list[float]]:
        """Get the feature matrix (list of lists)."""
        return [s.features for s in self.samples]
        
    @property
    def y_vector(self) -> list[PredictionDirection]:
        """Get the target vector."""
        return [s.target for s in self.samples]
    
    def __len__(self) -> int:
        return len(self.samples)


class MLDatasetBuilder:
    """Builds an MLDataset from historical data."""
    
    def __init__(
        self,
        feature_builder: FeatureBuilder,
        target_generator: TargetGenerator,
        schema: FeatureSchema
    ):
        self.feature_builder = feature_builder
        self.target_generator = target_generator
        self.schema = schema

    def extract_features(self, fv: FeatureVector) -> list[float]:
        """
        Extract ordered features based on schema.
        Raises ValueError if a feature is missing or invalid.
        """
        self.schema.validate_features(fv)
        
        extracted = []
        for feature_name in self.schema.required_features:
            val = getattr(fv, feature_name)
            extracted.append(float(val))
        return extracted

    def build(self, history: list[OHLC]) -> MLDataset:
        """
        Build a dataset avoiding future leakage.
        
        Args:
            history: Canonical OHLC history.
            
        Returns:
            An MLDataset containing aligned features and targets.
        """
        samples = []
        
        # We need at least enough candles for features + horizon
        min_candles = self.feature_builder.minimum_candles
        
        for i in range(min_candles - 1, len(history)):
            # 1. Check if we have a target (drops tail samples)
            target = self.target_generator.get_target(i, history)
            if target is None:
                continue
                
            # 2. Get features up to index i (strict chronology)
            # Candles used: 0 up to i (inclusive)
            # This is safe because slicing is exclusive on the right, so we do i+1
            window = history[:i + 1]
            fv = self.feature_builder.build(window)
            
            if fv is None:
                continue
                
            # 3. Extract mapped features using schema
            try:
                features_list = self.extract_features(fv)
            except ValueError:
                # Missing or invalid features -> drop row
                continue
                
            # 4. Construct sample
            sample = MLSample(
                timestamp=fv.timestamp,
                symbol=fv.symbol,
                timeframe=fv.timeframe,
                features=features_list,
                target=target,
                raw_feature_vector=fv
            )
            samples.append(sample)
            
        return MLDataset(samples=samples)

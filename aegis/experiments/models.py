"""
Experiment models for AEGIS AI.

Provides deterministic data structures for experiments, including
metadata, configuration, splits, and results.
"""

from datetime import datetime
import pydantic
from pydantic import BaseModel, Field
from typing import Optional
import hashlib

from aegis.interfaces.market_data import Timeframe, OHLC
from aegis.backtest.models import BacktestReport


class DatasetMetadata(BaseModel):
    """
    Metadata representing a canonical dataset identity.
    Note: The identity property represents metadata uniqueness, not a full content hash.
    """
    
    symbol: str = Field(..., min_length=1)
    timeframe: Timeframe
    start_timestamp: datetime
    end_timestamp: datetime
    observation_count: int = Field(..., gt=0)
    
    model_config = {"frozen": True}
    
    @property
    def identity(self) -> str:
        """Deterministic identity for this dataset metadata."""
        content = f"{self.symbol}|{self.timeframe.value}|{self.start_timestamp.isoformat()}|{self.end_timestamp.isoformat()}|{self.observation_count}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


class TemporalSplit(BaseModel):
    """
    A chronologically split dataset representing Train, Validation, and Test sets.
    
    Validation and Test sets may be empty depending on the configuration,
    but Train must always contain data.
    """
    
    train: list[OHLC] = Field(..., min_length=1)
    validation: list[OHLC] = Field(default_factory=list)
    test: list[OHLC] = Field(default_factory=list)
    
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    def __post_init__(self):
        # Additional chronological validation (since it's an arbitrary generic list, Pydantic's model_validator is better, 
        # but TemporalSplit might be constructed manually. We will use a model_validator)
        pass

    @pydantic.model_validator(mode='after')
    def validate_chronology(self) -> 'TemporalSplit':
        if self.validation and self.train:
            if self.train[-1].timestamp >= self.validation[0].timestamp:
                raise ValueError("Train data overlaps with or succeeds Validation data chronologically.")
        
        if self.test and self.validation:
            if self.validation[-1].timestamp >= self.test[0].timestamp:
                raise ValueError("Validation data overlaps with or succeeds Test data chronologically.")
        elif self.test and self.train:
            if self.train[-1].timestamp >= self.test[0].timestamp:
                raise ValueError("Train data overlaps with or succeeds Test data chronologically.")
        return self


class ExperimentConfig(BaseModel):
    """Configuration for an experiment."""
    
    experiment_name: str = Field(..., min_length=1)
    dataset_identity: str = Field(..., min_length=1)
    model_identity: str = Field(..., min_length=1)
    feature_config_identity: str = Field(default="baseline_features")
    
    model_config = {"frozen": True}
    
    @property
    def identity(self) -> str:
        """Deterministic identity for this experiment configuration."""
        content = f"{self.experiment_name}|{self.dataset_identity}|{self.model_identity}|{self.feature_config_identity}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


class ExperimentResult(BaseModel):
    """The final result of an experiment, combining setup and evaluation."""
    
    experiment_identity: str = Field(..., min_length=1)
    dataset_identity: str = Field(..., min_length=1)
    model_identity: str = Field(..., min_length=1)
    
    # We can store metrics or just rely on the backtest report
    backtest_result: BacktestReport
    
    model_config = {"frozen": True}

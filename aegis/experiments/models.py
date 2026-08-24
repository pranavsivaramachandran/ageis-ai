"""
Experiment models for AEGIS AI.

Provides deterministic data structures for experiments, including
metadata, configuration, splits, and results.
"""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
import hashlib

from aegis.interfaces.market_data import Timeframe, OHLC
from aegis.backtest.models import BacktestReport


class DatasetMetadata(BaseModel):
    """Metadata representing a canonical dataset identity."""
    
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

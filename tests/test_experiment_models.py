"""
Tests for experiment models and identity generation.
"""

from datetime import datetime, timezone
from aegis.interfaces.market_data import Timeframe
from aegis.experiments.models import DatasetMetadata, ExperimentConfig


def test_dataset_identity_determinism():
    dt = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    dt2 = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    
    meta1 = DatasetMetadata(
        symbol="EUR/USD",
        timeframe=Timeframe.H1,
        start_timestamp=dt,
        end_timestamp=dt2,
        observation_count=100
    )
    
    meta2 = DatasetMetadata(
        symbol="EUR/USD",
        timeframe=Timeframe.H1,
        start_timestamp=dt,
        end_timestamp=dt2,
        observation_count=100
    )
    
    # Must be deterministic
    assert meta1.identity == meta2.identity
    
    # Changing symbol must change identity
    meta3 = DatasetMetadata(
        symbol="GBP/USD",
        timeframe=Timeframe.H1,
        start_timestamp=dt,
        end_timestamp=dt2,
        observation_count=100
    )
    assert meta1.identity != meta3.identity
    
    # Changing count must change identity
    meta4 = DatasetMetadata(
        symbol="EUR/USD",
        timeframe=Timeframe.H1,
        start_timestamp=dt,
        end_timestamp=dt2,
        observation_count=101
    )
    assert meta1.identity != meta4.identity


def test_experiment_config_determinism():
    config1 = ExperimentConfig(
        experiment_name="baseline_test",
        dataset_identity="abc123def456",
        model_identity="baseline-v1"
    )
    
    config2 = ExperimentConfig(
        experiment_name="baseline_test",
        dataset_identity="abc123def456",
        model_identity="baseline-v1"
    )
    
    assert config1.identity == config2.identity
    
    config3 = ExperimentConfig(
        experiment_name="baseline_test",
        dataset_identity="xyz987def456",
        model_identity="baseline-v1"
    )
    
    assert config1.identity != config3.identity

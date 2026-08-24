"""
Tests for prediction model interface and schema validation.
"""

import pytest
from aegis.prediction.model_interface import FeatureSchema
from aegis.features.builder import FeatureVector
from datetime import datetime, timezone
from aegis.interfaces.market_data import Timeframe
import math

def test_feature_schema_validation():
    schema = FeatureSchema(schema_version=1, required_features=["sma_value", "rsi_value"])
    
    # Missing optional is fine, but required must be present
    fv_missing = FeatureVector(
        timestamp=datetime.now(timezone.utc),
        symbol="EUR/USD",
        timeframe=Timeframe.H1,
        last_close=100.0,
        sma_value=None,  # Missing required
        rsi_value=50.0
    )
    with pytest.raises(ValueError, match="Required feature 'sma_value' is None"):
        schema.validate_features(fv_missing)
        
    fv_valid = FeatureVector(
        timestamp=datetime.now(timezone.utc),
        symbol="EUR/USD",
        timeframe=Timeframe.H1,
        last_close=100.0,
        sma_value=105.0,
        rsi_value=50.0
    )
    schema.validate_features(fv_valid)


def test_feature_schema_nan_inf_rejection():
    schema = FeatureSchema(schema_version=1, required_features=["sma_value"])
    
    fv_nan = FeatureVector(
        timestamp=datetime.now(timezone.utc),
        symbol="EUR/USD",
        timeframe=Timeframe.H1,
        last_close=100.0,
        sma_value=math.nan
    )
    with pytest.raises(ValueError, match="FeatureVector contains invalid sma_value"):
        schema.validate_features(fv_nan)
        
    fv_inf = FeatureVector(
        timestamp=datetime.now(timezone.utc),
        symbol="EUR/USD",
        timeframe=Timeframe.H1,
        last_close=100.0,
        sma_value=105.0,
        macd_line=math.inf  # Optional field but still invalid
    )
    with pytest.raises(ValueError, match="FeatureVector contains invalid macd_line"):
        schema.validate_features(fv_inf)


def test_feature_schema_compatibility():
    s1 = FeatureSchema(schema_version=1, required_features=["a", "b"])
    s2 = FeatureSchema(schema_version=1, required_features=["a", "b"])
    s3 = FeatureSchema(schema_version=1, required_features=["a"])
    s4 = FeatureSchema(schema_version=2, required_features=["a", "b"])
    
    assert s1.is_compatible_with(s2) is True
    assert s1.is_compatible_with(s3) is False
    assert s1.is_compatible_with(s4) is False

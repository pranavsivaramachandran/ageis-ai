"""
Tests for ML Prediction Model wrapper.
"""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone

from aegis.interfaces.market_data import Timeframe
from aegis.features.builder import FeatureVector
from aegis.prediction.model_interface import FeatureSchema
from aegis.prediction.models import PredictionDirection
from aegis.ml.models import MLPredictionModel


class DummyClassifier:
    def __init__(self, classes):
        self.classes_ = classes
        
    def predict_proba(self, X):
        # Dummy mock: always predicts 60% confidence for the first class (index 0)
        # assuming X is well formed
        return np.array([[0.6, 0.4]])


class DummyScaler:
    def transform(self, X):
        return X * 2


class TestMLPredictionModel:
    @pytest.fixture
    def setup_model(self):
        schema = FeatureSchema(
            schema_version=1,
            required_features=["last_close", "sma_value"]
        )
        classifier = DummyClassifier(classes=[PredictionDirection.BUY, PredictionDirection.SELL])
        scaler = DummyScaler()
        
        model = MLPredictionModel(
            model_id="test_model",
            version=1,
            schema=schema,
            classifier=classifier,
            scaler=scaler,
            classes_mapping=[PredictionDirection.BUY, PredictionDirection.SELL]
        )
        return model, schema
        
    def test_model_identity_and_version(self, setup_model):
        model, _ = setup_model
        assert model.model_id == "test_model"
        assert model.version == 1
        assert model.is_ready()
        
    def test_schema_compatibility(self, setup_model):
        model, schema = setup_model
        assert model.schema.is_compatible_with(schema)
        
    def test_prediction_output(self, setup_model):
        model, _ = setup_model
        fv = FeatureVector(
            timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            symbol="BTC/USD",
            timeframe=Timeframe.H1,
            last_close=100.0,
            sma_value=101.0
        )
        
        result = model.predict(fv)
        
        # Based on DummyClassifier, we expect BUY with 0.6 confidence
        assert result.direction == PredictionDirection.BUY
        assert result.confidence == Decimal("0.6000")
        assert result.symbol == "BTC/USD"
        assert result.model_name == "test_model"
        assert "ML Classification" in result.reasoning
        
    def test_missing_feature_fails(self, setup_model):
        model, _ = setup_model
        # Missing sma_value
        fv = FeatureVector(
            timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            symbol="BTC/USD",
            timeframe=Timeframe.H1,
            last_close=100.0,
            sma_value=None
        )
        with pytest.raises(ValueError, match="Required feature 'sma_value' is None"):
            model.predict(fv)
            
    def test_unfitted_classifier_fails(self):
        # Classifier without classes_ attribute
        class UnfittedClassifier:
            pass
            
        schema = FeatureSchema(schema_version=1, required_features=[])
        with pytest.raises(ValueError, match="Classifier must be fitted"):
            MLPredictionModel(
                model_id="test", version=1, schema=schema,
                classifier=UnfittedClassifier(), scaler=None, classes_mapping=[]
            )

    def test_nan_feature_fails(self, setup_model):
        model, _ = setup_model
        fv = FeatureVector(
            timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            symbol="BTC/USD",
            timeframe=Timeframe.H1,
            last_close=100.0,
            sma_value=float('nan')
        )
        with pytest.raises(ValueError, match="FeatureVector contains invalid sma_value"):
            model.predict(fv)

    def test_infinity_feature_fails(self, setup_model):
        model, _ = setup_model
        fv = FeatureVector(
            timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            symbol="BTC/USD",
            timeframe=Timeframe.H1,
            last_close=100.0,
            sma_value=float('inf')
        )
        with pytest.raises(ValueError, match="FeatureVector contains invalid sma_value"):
            model.predict(fv)

    def test_incompatible_schema_fails(self, setup_model):
        model, _ = setup_model
        # Provide a FeatureVector that is entirely missing a required field
        # Note: Pydantic models typically enforce types, but FeatureVector is a dataclass.
        # If a required feature doesn't even exist as an attribute (e.g. incompatible schema).
        # We can simulate this by mocking or using a different schema.
        # Here we test schema validation which raises ValueError.
        fv = FeatureVector(
            timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            symbol="BTC/USD",
            timeframe=Timeframe.H1,
            last_close=100.0,
            # sma_value is None, which violates the schema.
        )
        with pytest.raises(ValueError, match="Required feature 'sma_value' is None"):
            model.predict(fv)

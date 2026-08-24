"""
Tests for ModelRegistry.
"""

import pytest
from aegis.prediction.registry import ModelRegistry
from aegis.prediction.engine import BaselinePredictor


def test_registry_registration_and_retrieval():
    registry = ModelRegistry()
    model = BaselinePredictor()
    
    registry.register(model)
    
    retrieved = registry.get("baseline", 1)
    assert retrieved is model
    assert registry.list_models() == ["baseline-v1"]


def test_registry_duplicate_rejection():
    registry = ModelRegistry()
    model = BaselinePredictor()
    
    registry.register(model)
    
    with pytest.raises(ValueError, match="already registered"):
        registry.register(model)


def test_registry_unknown_model():
    registry = ModelRegistry()
    
    with pytest.raises(KeyError, match="not found in registry"):
        registry.get("unknown", 1)

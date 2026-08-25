import pytest
from aegis.prediction.registry import ModelRegistry
from aegis.prediction.model_interface import PredictionModel, FeatureSchema, PredictionResult
from aegis.features.builder import FeatureVector
from aegis.governance.models import GovernanceStatus, PromotionDecision, PromotionDecisionType
from datetime import datetime, timezone

class DummyModel(PredictionModel):
    def __init__(self, model_id, version):
        self._model_id = model_id
        self._version = version
    @property
    def model_id(self): return self._model_id
    @property
    def version(self): return self._version
    @property
    def schema(self): return FeatureSchema()
    def is_ready(self): return True
    def predict(self, fv): return None

def test_registry_initial_status():
    registry = ModelRegistry()
    m = DummyModel("m1", 1)
    registry.register(m)
    assert registry.get_status("m1", 1) == GovernanceStatus.CANDIDATE

def test_registry_single_champion_invariant():
    registry = ModelRegistry()
    m1 = DummyModel("m1", 1)
    m2 = DummyModel("m2", 1)
    
    registry.register(m1)
    registry.register(m2)
    
    decision1 = PromotionDecision(model_identity="m1", model_version=1, decision=PromotionDecisionType.PROMOTE, reason="test", policy_identity="test")
    registry.promote("m1", 1, decision1)
    
    assert registry.get_champion().model_id == "m1"
    assert registry.get_status("m1", 1) == GovernanceStatus.CHAMPION
    
    # Promote m2 -> m1 should be superseded
    decision2 = PromotionDecision(model_identity="m2", model_version=1, decision=PromotionDecisionType.PROMOTE, reason="test", policy_identity="test")
    registry.promote("m2", 1, decision2)
    
    assert registry.get_champion().model_id == "m2"
    assert registry.get_status("m1", 1) == GovernanceStatus.SUPERSEDED
    assert registry.get_status("m2", 1) == GovernanceStatus.CHAMPION

def test_registry_reject():
    registry = ModelRegistry()
    m = DummyModel("m1", 1)
    registry.register(m)
    
    decision = PromotionDecision(model_identity="m1", model_version=1, decision=PromotionDecisionType.REJECT, reason="test", policy_identity="test")
    registry.reject("m1", 1, decision)
    assert registry.get_status("m1", 1) == GovernanceStatus.REJECTED

def test_registry_retire():
    registry = ModelRegistry()
    m = DummyModel("m1", 1)
    registry.register(m)
    decision = PromotionDecision(model_identity="m1", model_version=1, decision=PromotionDecisionType.PROMOTE, reason="test", policy_identity="test")
    registry.promote("m1", 1, decision)
    
    assert registry.get_champion().model_id == "m1"
    
    registry.retire("m1", 1)
    assert registry.get_status("m1", 1) == GovernanceStatus.RETIRED
    assert registry.get_champion() is None

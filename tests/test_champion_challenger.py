import pytest
from aegis.prediction.registry import ModelRegistry
from aegis.prediction.model_interface import PredictionModel, FeatureSchema
from aegis.governance.evaluator import GovernanceEvaluator
from aegis.governance.models import (
    PromotionPolicy, ExperimentRecord, PromotionDecisionType, GovernanceStatus, ExperimentStatus
)

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

def test_challenger_evaluation_does_not_mutate_champion():
    registry = ModelRegistry()
    champ = DummyModel("champ", 1)
    registry.register(champ)
    
    # promote directly for testing setup
    from aegis.governance.models import PromotionDecision
    registry.promote("champ", 1, PromotionDecision(
        model_identity="champ", model_version=1, decision=PromotionDecisionType.PROMOTE,
        reason="init", policy_identity="test"
    ))
    
    evaluator = GovernanceEvaluator(registry)
    policy = PromotionPolicy(policy_id="test_policy")
    
    experiment = ExperimentRecord(
        experiment_id="exp-1", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        candidate_model_ids=["challenger"],
        status=ExperimentStatus.COMPLETED,
        overall_metrics={
            "walk_forward_windows": 1, "mean_f1": 0.0, "max_drawdown": 0.0, "beats_baseline": False
        }
    )
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, is_reproducible=True, is_safe=True)
    
    # Ensure champion is still active and unchanged
    assert registry.get_champion().model_id == "champ"
    assert registry.get_status("champ", 1) == GovernanceStatus.CHAMPION

def test_challenger_can_be_rejected():
    registry = ModelRegistry()
    evaluator = GovernanceEvaluator(registry)
    policy = PromotionPolicy(policy_id="test", min_mean_f1=0.8)
    
    experiment = ExperimentRecord(
        experiment_id="exp-1", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        candidate_model_ids=["challenger"],
        status=ExperimentStatus.COMPLETED,
        overall_metrics={"mean_f1": 0.5}
    )
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, is_reproducible=True, is_safe=True)
    assert decision.decision == PromotionDecisionType.REJECT

def test_challenger_can_be_promoted_and_transitions():
    registry = ModelRegistry()
    champ = DummyModel("champ", 1)
    challenger = DummyModel("challenger", 1)
    registry.register(champ)
    registry.register(challenger)
    
    from aegis.governance.models import PromotionDecision
    registry.promote("champ", 1, PromotionDecision(
        model_identity="champ", model_version=1, decision=PromotionDecisionType.PROMOTE,
        reason="init", policy_identity="test"
    ))
    
    evaluator = GovernanceEvaluator(registry)
    policy = PromotionPolicy(policy_id="test", min_mean_f1=0.5, requires_baseline_beat=False)
    
    experiment = ExperimentRecord(
        experiment_id="exp-1", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        candidate_model_ids=["challenger"],
        status=ExperimentStatus.COMPLETED,
        overall_metrics={"mean_f1": 0.8, "max_drawdown": 0.05, "walk_forward_windows": 5}
    )
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, is_reproducible=True, is_safe=True)
    assert decision.decision == PromotionDecisionType.PROMOTE
    
    registry.promote("challenger", 1, decision)
    
    assert registry.get_champion().model_id == "challenger"
    assert registry.get_status("champ", 1) == GovernanceStatus.SUPERSEDED

import pytest
from aegis.governance.evaluator import GovernanceEvaluator
from aegis.governance.models import PromotionPolicy, ExperimentRecord, PromotionDecisionType, ExperimentStatus
from aegis.prediction.registry import ModelRegistry
from aegis.governance.reproducibility import ReproducibilityReceipt, VerificationStatus
import os

def test_governance_safety_failure():
    registry = ModelRegistry()
    evaluator = GovernanceEvaluator(registry)
    policy = PromotionPolicy(policy_id="test", require_safety_check=True)
    
    experiment = ExperimentRecord(
        experiment_id="exp-1", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        candidate_model_ids=["challenger"],
        status=ExperimentStatus.COMPLETED,
        overall_metrics={"mean_f1": 0.8, "max_drawdown": 0.05, "walk_forward_windows": 5, "beats_baseline": True}
    )
    
    # Simulating safety check failure
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, reproducibility_receipt=ReproducibilityReceipt(experiment_id="exp-1", model_identity="m1", dataset_identity="d1", feature_identity="f1", target_identity="t1", training_config_identity="tc1", expected_fingerprint="fp1", observed_fingerprint="fp1", status=VerificationStatus.VERIFIED), is_safe=False)
    assert decision.decision == PromotionDecisionType.REJECT
    assert "Safety check failed" in decision.reason

def test_promotion_does_not_change_execution_mode():
    registry = ModelRegistry()
    evaluator = GovernanceEvaluator(registry)
    policy = PromotionPolicy(
        policy_id="test", min_mean_f1=0.5, requires_baseline_beat=False
    )
    
    experiment = ExperimentRecord(
        experiment_id="exp-1", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        candidate_model_ids=["challenger"],
        status=ExperimentStatus.COMPLETED,
        overall_metrics={"mean_f1": 0.8, "max_drawdown": 0.05, "walk_forward_windows": 5}
    )
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, reproducibility_receipt=ReproducibilityReceipt(experiment_id="exp-1", model_identity="m1", dataset_identity="d1", feature_identity="f1", target_identity="t1", training_config_identity="tc1", expected_fingerprint="fp1", observed_fingerprint="fp1", status=VerificationStatus.VERIFIED), is_safe=True)
    
    assert decision.decision == PromotionDecisionType.PROMOTE

import pytest
from aegis.governance.models import PromotionPolicy, ExperimentRecord, PromotionDecisionType, ExperimentStatus
from aegis.governance.evaluator import GovernanceEvaluator
from aegis.prediction.registry import ModelRegistry

def test_promotion_policy_valid_candidate_promoted():
    registry = ModelRegistry()
    evaluator = GovernanceEvaluator(registry)
    
    policy = PromotionPolicy(
        policy_id="test_policy",
        min_walk_forward_windows=5,
        min_mean_f1=0.55,
        max_drawdown_pct=0.15,
        requires_baseline_beat=True
    )
    
    experiment = ExperimentRecord(
        experiment_id="exp-1", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        candidate_model_ids=["candidate-id-1"],
        status=ExperimentStatus.COMPLETED,
        overall_metrics={
            "walk_forward_windows": 6,
            "mean_f1": 0.60,
            "max_drawdown": 0.10,
            "beats_baseline": True
        }
    )
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, is_reproducible=True, is_safe=True)
    assert decision.decision == PromotionDecisionType.PROMOTE

def test_promotion_policy_insufficient_evidence_rejected():
    registry = ModelRegistry()
    evaluator = GovernanceEvaluator(registry)
    
    policy = PromotionPolicy(
        policy_id="test_policy",
        min_walk_forward_windows=5,
        min_mean_f1=0.55,
        max_drawdown_pct=0.15
    )
    
    # Missing windows
    experiment = ExperimentRecord(
        experiment_id="exp-1", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        candidate_model_ids=["candidate-id-1"],
        status=ExperimentStatus.COMPLETED,
        overall_metrics={
            "walk_forward_windows": 4, # Too few
            "mean_f1": 0.60,
            "max_drawdown": 0.10,
            "beats_baseline": True
        }
    )
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, is_reproducible=True, is_safe=True)
    assert decision.decision == PromotionDecisionType.REJECT
    assert "Insufficient walk-forward windows" in decision.reason

def test_promotion_policy_failed_experiment_cannot_promote():
    registry = ModelRegistry()
    evaluator = GovernanceEvaluator(registry)
    policy = PromotionPolicy(policy_id="test_policy")
    experiment = ExperimentRecord(
        experiment_id="exp-1", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        status=ExperimentStatus.FAILED,
        overall_metrics={}
    )
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, is_reproducible=True, is_safe=True)
    assert decision.decision == PromotionDecisionType.REJECT
    assert "FAILED status" in decision.reason

def test_promotion_policy_non_reproducible_rejected():
    registry = ModelRegistry()
    evaluator = GovernanceEvaluator(registry)
    policy = PromotionPolicy(policy_id="test_policy", require_reproducibility=True)
    experiment = ExperimentRecord(
        experiment_id="exp-1", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        status=ExperimentStatus.COMPLETED,
        overall_metrics={
            "walk_forward_windows": 5, "mean_f1": 0.60, "max_drawdown": 0.10, "beats_baseline": True
        }
    )
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, is_reproducible=False, is_safe=True)
    assert decision.decision == PromotionDecisionType.REJECT
    assert "Reproducibility check failed" in decision.reason

def test_promotion_policy_safety_failure_rejected():
    registry = ModelRegistry()
    evaluator = GovernanceEvaluator(registry)
    policy = PromotionPolicy(policy_id="test_policy", require_safety_check=True)
    experiment = ExperimentRecord(
        experiment_id="exp-1", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        status=ExperimentStatus.COMPLETED,
        overall_metrics={
            "walk_forward_windows": 5, "mean_f1": 0.60, "max_drawdown": 0.10, "beats_baseline": True
        }
    )
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, is_reproducible=True, is_safe=False)
    assert decision.decision == PromotionDecisionType.REJECT
    assert "Safety check failed" in decision.reason

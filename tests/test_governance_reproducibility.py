import pytest
from aegis.governance.evaluator import GovernanceEvaluator
from aegis.governance.models import PromotionPolicy, ExperimentRecord, PromotionDecisionType, ExperimentStatus
from aegis.prediction.registry import ModelRegistry

def test_governance_reproducibility_rejection():
    registry = ModelRegistry()
    evaluator = GovernanceEvaluator(registry)
    policy = PromotionPolicy(policy_id="test", require_reproducibility=True)
    
    experiment = ExperimentRecord(
        experiment_id="exp-1", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1",
        candidate_model_ids=["challenger"],
        status=ExperimentStatus.COMPLETED,
        overall_metrics={"mean_f1": 0.8, "max_drawdown": 0.05, "walk_forward_windows": 5, "beats_baseline": True}
    )
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, is_reproducible=False, is_safe=True)
    assert decision.decision == PromotionDecisionType.REJECT
    assert "Reproducibility" in decision.reason

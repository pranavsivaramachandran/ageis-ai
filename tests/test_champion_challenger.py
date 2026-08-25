import pytest
from aegis.prediction.registry import ModelRegistry
from aegis.governance.reproducibility import ReproducibilityReceipt, VerificationStatus
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
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, reproducibility_receipt=ReproducibilityReceipt(experiment_id="exp-1", model_identity="m1", dataset_identity="d1", feature_identity="f1", target_identity="t1", training_config_identity="tc1", expected_fingerprint="fp1", observed_fingerprint="fp1", status=VerificationStatus.VERIFIED), is_safe=True)
    
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
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, reproducibility_receipt=ReproducibilityReceipt(experiment_id="exp-1", model_identity="m1", dataset_identity="d1", feature_identity="f1", target_identity="t1", training_config_identity="tc1", expected_fingerprint="fp1", observed_fingerprint="fp1", status=VerificationStatus.VERIFIED), is_safe=True)
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
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, reproducibility_receipt=ReproducibilityReceipt(experiment_id="exp-1", model_identity="m1", dataset_identity="d1", feature_identity="f1", target_identity="t1", training_config_identity="tc1", expected_fingerprint="fp1", observed_fingerprint="fp1", status=VerificationStatus.VERIFIED), is_safe=True)
    assert decision.decision == PromotionDecisionType.PROMOTE
    
    registry.promote("challenger", 1, decision)
    
    assert registry.get_champion().model_id == "challenger"
    assert registry.get_status("champ", 1) == GovernanceStatus.SUPERSEDED

def test_repeated_promotion_and_rollback():
    registry = ModelRegistry()
    champ_a = DummyModel("champ_a", 1)
    champ_b = DummyModel("champ_b", 1)
    champ_c = DummyModel("champ_c", 1)
    
    registry.register(champ_a)
    registry.register(champ_b)
    registry.register(champ_c)
    
    from aegis.governance.models import PromotionDecision, PromotionDecisionType
    
    # 1. Champion A promoted
    registry.promote("champ_a", 1, PromotionDecision(
        model_identity="champ_a", model_version=1, decision=PromotionDecisionType.PROMOTE,
        reason="init", policy_identity="test"
    ))
    
    assert registry.get_status("champ_a", 1) == GovernanceStatus.CHAMPION
    
    # 2. Challenger B promoted
    # 3. A becomes SUPERSEDED
    # 4. B becomes CHAMPION
    registry.promote("champ_b", 1, PromotionDecision(
        model_identity="champ_b", model_version=1, decision=PromotionDecisionType.PROMOTE,
        reason="upgrade", policy_identity="test"
    ))
    
    assert registry.get_status("champ_a", 1) == GovernanceStatus.SUPERSEDED
    assert registry.get_status("champ_b", 1) == GovernanceStatus.CHAMPION
    
    # 5. Exactly one active Champion exists
    assert registry.get_champion().model_id == "champ_b"
    
    # 7. promoted_at is populated
    from aegis.db.session import get_db
    from aegis.db.models.governance import ModelRegistration
    with get_db() as db:
        reg_a = db.query(ModelRegistration).filter_by(id="champ_a-v1").first()
        reg_b = db.query(ModelRegistration).filter_by(id="champ_b-v1").first()
        
        assert reg_a.promoted_at is not None
        assert reg_a.superseded_at is not None
        assert reg_b.promoted_at is not None
        assert reg_b.superseded_at is None
        
        # 6. Both records remain persistent
        assert reg_a.governance_status == GovernanceStatus.SUPERSEDED
        assert reg_b.governance_status == GovernanceStatus.CHAMPION

    # 8. Repeated promotion still behaves correctly
    registry.promote("champ_c", 1, PromotionDecision(
        model_identity="champ_c", model_version=1, decision=PromotionDecisionType.PROMOTE,
        reason="upgrade 2", policy_identity="test"
    ))
    
    assert registry.get_status("champ_a", 1) == GovernanceStatus.SUPERSEDED
    assert registry.get_status("champ_b", 1) == GovernanceStatus.SUPERSEDED
    assert registry.get_status("champ_c", 1) == GovernanceStatus.CHAMPION
    
    with get_db() as db:
        reg_b = db.query(ModelRegistration).filter_by(id="champ_b-v1").first()
        assert reg_b.superseded_at is not None

def test_promotion_rollback_on_failure():
    registry = ModelRegistry()
    champ_x = DummyModel("champ_x", 1)
    champ_y = DummyModel("champ_y", 1)
    
    registry.register(champ_x)
    registry.register(champ_y)
    
    from aegis.governance.models import PromotionDecision, PromotionDecisionType
    registry.promote("champ_x", 1, PromotionDecision(
        model_identity="champ_x", model_version=1, decision=PromotionDecisionType.PROMOTE,
        reason="init", policy_identity="test"
    ))
    
    # 9. Failed promotion rolls back both state changes
    import pytest
    from unittest.mock import patch
    with pytest.raises(Exception):
        with patch('aegis.prediction.registry.GovernanceAuditRecord', side_effect=ValueError("Simulated Error")):
            registry.promote("champ_y", 1, PromotionDecision(
                model_identity="champ_y", model_version=1, decision=PromotionDecisionType.PROMOTE,
                reason="upgrade", policy_identity="test"
            ))
            
    assert registry.get_status("champ_x", 1) == GovernanceStatus.CHAMPION
    assert registry.get_status("champ_y", 1) == GovernanceStatus.CANDIDATE
    
    from aegis.db.session import get_db
    from aegis.db.models.governance import ModelRegistration
    with get_db() as db:
        reg_x = db.query(ModelRegistration).filter_by(id="champ_x-v1").first()
        reg_y = db.query(ModelRegistration).filter_by(id="champ_y-v1").first()
        assert reg_x.governance_status == GovernanceStatus.CHAMPION
        assert reg_x.superseded_at is None
        assert reg_y.governance_status == GovernanceStatus.CANDIDATE


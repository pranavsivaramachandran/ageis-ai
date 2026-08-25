import pytest
from aegis.governance.evaluator import GovernanceEvaluator
from aegis.governance.models import PromotionPolicy, ExperimentRecord, PromotionDecisionType, ExperimentStatus
from aegis.prediction.registry import ModelRegistry
from aegis.governance.reproducibility import ReproducibilityReceipt, VerificationStatus, ReproducibilityVerifier

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
    
    decision = evaluator.evaluate_candidate(experiment, policy, {}, 1, reproducibility_receipt=ReproducibilityReceipt(experiment_id="exp-1", model_identity="m1", dataset_identity="d1", feature_identity="f1", target_identity="t1", training_config_identity="tc1", expected_fingerprint="fp1", observed_fingerprint="missing", status=VerificationStatus.FAILED), is_safe=True)
    assert decision.decision == PromotionDecisionType.REJECT
    assert "Reproducibility" in decision.reason

def test_reproducibility_verifier_independent_derivation():
    # 1. valid authoritative evidence -> VERIFIED
    base_kwargs = {
        "experiment_id": "exp_a",
        "model_identity": "m1",
        "dataset_identity": "d1",
        "feature_identity": "f1",
        "target_identity": "t1",
        "training_config_identity": "tc1",
        "seed": 42
    }
    
    # Let's see what it derives first
    # We create a dummy receipt with a random expected fingerprint to get the actual derived one
    receipt_dummy = ReproducibilityVerifier.verify(**base_kwargs, expected_fingerprint="dummy")
    actual_fingerprint = receipt_dummy.observed_fingerprint
    
    receipt_valid = ReproducibilityVerifier.verify(**base_kwargs, expected_fingerprint=actual_fingerprint)
    assert receipt_valid.status == VerificationStatus.VERIFIED
    
    # 2. changed model configuration -> FAILED
    bad_model = base_kwargs.copy()
    bad_model["model_identity"] = "m2"
    assert ReproducibilityVerifier.verify(**bad_model, expected_fingerprint=actual_fingerprint).status == VerificationStatus.FAILED
    
    # 3. changed dataset identity -> FAILED
    bad_ds = base_kwargs.copy()
    bad_ds["dataset_identity"] = "d2"
    assert ReproducibilityVerifier.verify(**bad_ds, expected_fingerprint=actual_fingerprint).status == VerificationStatus.FAILED
    
    # 4. changed feature identity -> FAILED
    bad_feat = base_kwargs.copy()
    bad_feat["feature_identity"] = "f2"
    assert ReproducibilityVerifier.verify(**bad_feat, expected_fingerprint=actual_fingerprint).status == VerificationStatus.FAILED
    
    # 5. changed target identity -> FAILED
    bad_target = base_kwargs.copy()
    bad_target["target_identity"] = "t2"
    assert ReproducibilityVerifier.verify(**bad_target, expected_fingerprint=actual_fingerprint).status == VerificationStatus.FAILED
    
    # 6. changed training configuration -> FAILED
    bad_tr = base_kwargs.copy()
    bad_tr["training_config_identity"] = "tc2"
    assert ReproducibilityVerifier.verify(**bad_tr, expected_fingerprint=actual_fingerprint).status == VerificationStatus.FAILED
    
    # 8. caller cannot force VERIFIED by passing matching arbitrary strings
    # (Since `observed_fingerprint` is no longer a parameter, caller can't force it)
    with pytest.raises(TypeError):
        ReproducibilityVerifier.verify(**base_kwargs, expected_fingerprint="abc", observed_fingerprint="abc")

def test_receipt_identity_deterministic():
    base_kwargs = {
        "experiment_id": "exp_a",
        "model_identity": "m1",
        "dataset_identity": "d1",
        "feature_identity": "f1",
        "target_identity": "t1",
        "training_config_identity": "tc1",
        "expected_fingerprint": "fake"
    }
    r1 = ReproducibilityVerifier.verify(**base_kwargs)
    r2 = ReproducibilityVerifier.verify(**base_kwargs)
    assert r1.identity == r2.identity

def test_receipt_immutable():
    base_kwargs = {
        "experiment_id": "exp_a",
        "model_identity": "m1",
        "dataset_identity": "d1",
        "feature_identity": "f1",
        "target_identity": "t1",
        "training_config_identity": "tc1",
        "expected_fingerprint": "fake"
    }
    r = ReproducibilityVerifier.verify(**base_kwargs)
    
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        r.model_identity = "m2"

import hashlib
import json
from enum import Enum
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class VerificationStatus(Enum):
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    INDETERMINATE = "INDETERMINATE"

class ReproducibilityReceipt(BaseModel):
    receipt_id: str = Field(default_factory=lambda: "rcpt-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    experiment_id: str
    model_identity: str
    dataset_identity: str
    feature_identity: str
    target_identity: str
    training_config_identity: str
    expected_fingerprint: str
    observed_fingerprint: str
    status: VerificationStatus
    verification_method: str = "local_hash_match"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": True}

    @property
    def identity(self) -> str:
        """Deterministic logical identity of the reproducibility receipt."""
        components = {
            "experiment_id": self.experiment_id,
            "model_identity": self.model_identity,
            "dataset_identity": self.dataset_identity,
            "feature_identity": self.feature_identity,
            "target_identity": self.target_identity,
            "training_config_identity": self.training_config_identity,
            "expected_fingerprint": self.expected_fingerprint,
            "observed_fingerprint": self.observed_fingerprint,
            "status": self.status.value,
        }
        canonical_str = json.dumps(components, sort_keys=True)
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()[:16]

class ReproducibilityVerifier:
    @staticmethod
    def verify(
        experiment_id: str,
        model_identity: str,
        dataset_identity: str,
        feature_identity: str,
        target_identity: str,
        training_config_identity: str,
        expected_fingerprint: str,
        seed: Optional[int] = None,
        calibration_config_identity: Optional[str] = None,
        selection_config_identity: Optional[str] = None,
        ensemble_config_identity: Optional[str] = None
    ) -> ReproducibilityReceipt:
        # Independently derive the fingerprint based on authoritative evidence
        components = {
            "model_identity": model_identity,
            "dataset_identity": dataset_identity,
            "feature_identity": feature_identity,
            "target_identity": target_identity,
            "training_config_identity": training_config_identity,
        }
        
        if seed is not None:
            components["seed"] = seed
        if calibration_config_identity is not None:
            components["calibration_config_identity"] = calibration_config_identity
        if selection_config_identity is not None:
            components["selection_config_identity"] = selection_config_identity
        if ensemble_config_identity is not None:
            components["ensemble_config_identity"] = ensemble_config_identity

        canonical_str = json.dumps(components, sort_keys=True)
        observed_fingerprint = hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()[:16]
        
        if expected_fingerprint == observed_fingerprint:
            status = VerificationStatus.VERIFIED
        else:
            status = VerificationStatus.FAILED

        return ReproducibilityReceipt(
            experiment_id=experiment_id,
            model_identity=model_identity,
            dataset_identity=dataset_identity,
            feature_identity=feature_identity,
            target_identity=target_identity,
            training_config_identity=training_config_identity,
            expected_fingerprint=expected_fingerprint,
            observed_fingerprint=observed_fingerprint,
            status=status
        )

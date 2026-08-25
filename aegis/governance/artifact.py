import hashlib
import os
from enum import Enum
from typing import Optional
from aegis.governance.models import ArtifactMetadata
from aegis.prediction.model_interface import PredictionModel
import joblib

class ArtifactLoadResultStatus(Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    MISSING = "MISSING"

class ArtifactLoadResult:
    def __init__(self, status: ArtifactLoadResultStatus, model: Optional[PredictionModel], reason: str, expected_hash: str, observed_hash: Optional[str]):
        self.status = status
        self.model = model
        self.reason = reason
        self.expected_hash = expected_hash
        self.observed_hash = observed_hash

class ArtifactVerifier:
    @staticmethod
    def compute_file_hash(filepath: str) -> Optional[str]:
        if not os.path.exists(filepath):
            return None
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

class ArtifactLoader:
    @staticmethod
    def load(filepath: str, expected_metadata: ArtifactMetadata) -> ArtifactLoadResult:
        # 1. Verify existence and hash
        observed_hash = ArtifactVerifier.compute_file_hash(filepath)
        
        if observed_hash is None:
            return ArtifactLoadResult(
                status=ArtifactLoadResultStatus.MISSING,
                model=None,
                reason="Artifact file not found",
                expected_hash=expected_metadata.integrity_hash,
                observed_hash=None
            )
            
        if observed_hash != expected_metadata.integrity_hash:
            return ArtifactLoadResult(
                status=ArtifactLoadResultStatus.INVALID,
                model=None,
                reason="Artifact integrity hash mismatch",
                expected_hash=expected_metadata.integrity_hash,
                observed_hash=observed_hash
            )
            
        # 2. Load the model
        try:
            model = joblib.load(filepath)
            
            # Simple identity check post-load if possible
            if hasattr(model, 'model_id') and model.model_id != expected_metadata.model_identity:
                return ArtifactLoadResult(
                    status=ArtifactLoadResultStatus.INVALID,
                    model=None,
                    reason="Model identity mismatch in loaded artifact",
                    expected_hash=expected_metadata.integrity_hash,
                    observed_hash=observed_hash
                )
                
            return ArtifactLoadResult(
                status=ArtifactLoadResultStatus.VALID,
                model=model,
                reason="Artifact verified and loaded successfully",
                expected_hash=expected_metadata.integrity_hash,
                observed_hash=observed_hash
            )
        except Exception as e:
            return ArtifactLoadResult(
                status=ArtifactLoadResultStatus.INVALID,
                model=None,
                reason=f"Failed to deserialize artifact: {str(e)}",
                expected_hash=expected_metadata.integrity_hash,
                observed_hash=observed_hash
            )

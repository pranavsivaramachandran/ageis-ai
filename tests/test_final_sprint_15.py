"""
AEGIS AI - Sprint 15 Final Negative Path & Recovery Tests
"""

import os
import shutil
import tempfile
import hashlib
import joblib
import pytest
from datetime import datetime, timezone

from aegis.db import session as db_session_module
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aegis.governance.models import ArtifactMetadata, GovernanceStatus, PromotionDecision, PromotionDecisionType
from aegis.prediction.registry import ModelRegistry
from aegis.prediction.model_interface import FeatureSchema, PredictionModel

class DummySprint15Model(PredictionModel):
    def __init__(self, model_id, version):
        self._model_id = model_id
        self._version = version
    @property
    def model_id(self): return self._model_id
    @property
    def version(self): return int(self._version)
    @property
    def schema(self): return FeatureSchema(schema_version=1, required_features=[])
    def is_ready(self): return True
    def predict(self, fv): return None

def compute_hash(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for b in iter(lambda: f.read(4096), b""):
            sha.update(b)
    return sha.hexdigest()

@pytest.fixture
def isolated_db_and_registry():
    fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    test_engine = create_engine(f"sqlite:///{temp_db_path}")
    db_session_module.engine = test_engine
    db_session_module.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db_session_module.init_db()
    
    artifact_dir = tempfile.mkdtemp(prefix="aegis_artifacts_")
    
    yield temp_db_path, artifact_dir
    
    shutil.rmtree(artifact_dir)
    test_engine.dispose()
    os.remove(temp_db_path)

def test_sprint_15_champion_restart_recovery(isolated_db_and_registry):
    temp_db_path, artifact_dir = isolated_db_and_registry
    
    # Process A
    registry1 = ModelRegistry(artifact_dir=artifact_dir)
    model = DummySprint15Model("final_model", 1)
    
    artifact_path = os.path.join(artifact_dir, "final_model-v1.joblib")
    joblib.dump(model, artifact_path)
    hsh = compute_hash(artifact_path)
    
    meta = ArtifactMetadata(
        artifact_format="joblib", model_identity="final_model", version=1,
        fingerprint="fp1", training_experiment="exp1", feature_schema_version=1,
        dataset_identity="ds1", training_date=datetime.now(timezone.utc),
        random_seed=42, integrity_hash=hsh
    )
    registry1.register(model, GovernanceStatus.CANDIDATE, metadata=meta)
    
    registry1.promote("final_model", 1, PromotionDecision(
        model_identity="final_model", model_version=1, decision=PromotionDecisionType.PROMOTE,
        reason="Test", policy_identity="test_policy"
    ))
    
    # Validate Process A
    champ_a = registry1.get_champion()
    assert champ_a.model_id == "final_model"
    
    # Process B (Restart)
    registry2 = ModelRegistry(artifact_dir=artifact_dir)
    champ_b = registry2.get_champion()
    
    assert champ_b is not None
    assert champ_b.model_id == "final_model"
    assert champ_b.version == 1

def test_sprint_15_artifact_tampering_blocked(isolated_db_and_registry):
    temp_db_path, artifact_dir = isolated_db_and_registry
    
    registry = ModelRegistry(artifact_dir=artifact_dir)
    model = DummySprint15Model("tamper_model", 1)
    
    artifact_path = os.path.join(artifact_dir, "tamper_model-v1.joblib")
    joblib.dump(model, artifact_path)
    hsh = compute_hash(artifact_path)
    
    meta = ArtifactMetadata(
        artifact_format="joblib", model_identity="tamper_model", version=1,
        fingerprint="fp1", training_experiment="exp1", feature_schema_version=1,
        dataset_identity="ds1", training_date=datetime.now(timezone.utc),
        random_seed=42, integrity_hash=hsh
    )
    registry.register(model, GovernanceStatus.CANDIDATE, metadata=meta)
    
    # Corrupt on disk
    with open(artifact_path, "ab") as f:
        f.write(b"corrupt_bytes")
        
    # Attempt load
    with pytest.raises(RuntimeError, match="Artifact integrity hash mismatch"):
        registry.get("tamper_model", 1)

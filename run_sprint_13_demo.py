"""
AEGIS AI - Sprint 13 Correction Demonstration
Governance Hardening, Reproducibility Receipts, Artifact Verification, Persistent Registry
"""

import os
import shutil
import tempfile
import hashlib
import joblib
import logging
from datetime import datetime, timezone
import json

from aegis.core.config import ExecutionMode
from aegis.db import session as db_session_module
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aegis.governance.models import ArtifactMetadata, GovernanceStatus, PromotionDecision, PromotionDecisionType
from aegis.governance.reproducibility import ReproducibilityVerifier
from aegis.prediction.registry import ModelRegistry
from aegis.prediction.model_interface import PredictionModel, FeatureSchema
from aegis.db.models.governance import GovernanceAuditRecord, ModelRegistration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sprint_13_demo")

class DummyMLModel(PredictionModel):
    def __init__(self, model_id, version):
        self._model_id = model_id
        self._version = version
    @property
    def model_id(self): return self._model_id
    @property
    def version(self): return int(self._version)
    @property
    def schema(self): return FeatureSchema()
    def is_ready(self): return True
    def predict(self, fv): return None

def compute_hash(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for b in iter(lambda: f.read(4096), b""):
            sha.update(b)
    return sha.hexdigest()

def run_demo():
    logger.info("Starting AEGIS AI Sprint 13 Correction Demonstration")
    logger.info(f"Execution Mode: {ExecutionMode.PREDICTION_ONLY.name}")
    
    # 1. Initialize persistent DB (File-backed temporary SQLite)
    fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    test_engine = create_engine(f"sqlite:///{temp_db_path}")
    db_session_module.engine = test_engine
    db_session_module.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db_session_module.init_db()
    
    logger.info(f"Initialized isolated persistent database at: {temp_db_path}")
    
    # 2. Setup Artifacts
    artifact_dir = tempfile.mkdtemp(prefix="aegis_artifacts_")
    registry1 = ModelRegistry(artifact_dir=artifact_dir)
    
    # PROCESS A: Register and Promote
    logger.info("--- PROCESS A: REGISTRATION & PROMOTION ---")
    
    # Simulate reproducibility derivation
    base_kwargs = {
        "model_identity": "rf_classifier_a",
        "dataset_identity": "ds_a",
        "feature_identity": "f1",
        "target_identity": "t1",
        "training_config_identity": "tr1",
    }
    canonical_str = json.dumps(base_kwargs, sort_keys=True)
    actual_fingerprint = hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()[:16]
    
    model_a = DummyMLModel("rf_classifier_a", 1)
    artifact_a_path = os.path.join(artifact_dir, "rf_classifier_a-v1.joblib")
    joblib.dump(model_a, artifact_a_path)
    hash_a = compute_hash(artifact_a_path)
    
    meta_a = ArtifactMetadata(
        artifact_format="joblib",
        model_identity="rf_classifier_a",
        version=1,
        fingerprint=actual_fingerprint,
        training_experiment="exp_a",
        feature_schema_version=1,
        dataset_identity="ds_a",
        training_date=datetime.now(timezone.utc),
        random_seed=42,
        integrity_hash=hash_a
    )
    
    logger.info(f"Registering Model A (rf_classifier_a-v1) with hash: {hash_a[:8]}...")
    registry1.register(model_a, GovernanceStatus.CANDIDATE, metadata=meta_a)
    
    # Generate Reproducibility Receipt for A (Valid)
    logger.info("Generating Reproducibility Receipt for A...")
    receipt_a = ReproducibilityVerifier.verify(
        experiment_id="exp_a", 
        expected_fingerprint=actual_fingerprint,
        **base_kwargs
    )
    logger.info(f"Receipt A Status: {receipt_a.status.name}")
    
    # Generate failure case
    logger.info("Generating Reproducibility Receipt with tampered evidence...")
    bad_kwargs = base_kwargs.copy()
    bad_kwargs["dataset_identity"] = "ds_b"
    receipt_bad = ReproducibilityVerifier.verify(
        experiment_id="exp_a", 
        expected_fingerprint=actual_fingerprint,
        **bad_kwargs
    )
    logger.info(f"Receipt Bad Status: {receipt_bad.status.name}")
    
    # Promote A
    logger.info("Promoting A to CHAMPION...")
    registry1.promote("rf_classifier_a", 1, PromotionDecision(
        model_identity="rf_classifier_a",
        model_version=1,
        decision=PromotionDecisionType.PROMOTE,
        reason="Passed checks",
        policy_identity="strict_v1"
    ))
    
    # Process B: Restart Recovery
    logger.info("--- PROCESS B: RESTART RECOVERY & CACHE TAMPERING ---")
    logger.info("Simulating process restart (new registry instance)...")
    registry2 = ModelRegistry(artifact_dir=artifact_dir)
    
    champ = registry2.get_champion()
    logger.info(f"Successfully recovered CHAMPION: {champ.model_id}-v{champ.version}")
    
    # Verify cached model works initially
    retrieved = registry2.get("rf_classifier_a", 1)
    logger.info(f"Successfully retrieved cached model: {retrieved.model_id}-v{retrieved.version}")
    
    # Corrupt artifact
    logger.info("Corrupting Model A's physical artifact bytes on disk...")
    with open(artifact_a_path, "ab") as f:
        f.write(b"corruption_bytes")
        
    # Attempt to load A (now corrupt on disk) using cached model bypass protection
    try:
        logger.info("Attempting to get() corrupted Model A (should fail despite cache)...")
        registry2.get("rf_classifier_a", 1)
    except RuntimeError as e:
        logger.info(f"Integrity check successfully prevented cache bypass: {e}")

    try:
        logger.info("Attempting to get_champion() corrupted Model A (should fail despite cache)...")
        registry2.get_champion()
    except RuntimeError as e:
        logger.info(f"Integrity check successfully prevented cache bypass: {e}")
    
    # 3. Print Audit Trail
    logger.info("--- GOVERNANCE AUDIT TRAIL ---")
    with db_session_module.get_db() as db:
        audits = db.query(GovernanceAuditRecord).order_by(GovernanceAuditRecord.id).all()
        for a in audits:
            logger.info(f"[{a.timestamp.isoformat()}] {a.action} on {a.model_identity}-v{a.model_version}: {a.reason}")
            
    # Cleanup
    shutil.rmtree(artifact_dir)
    test_engine.dispose()
    os.remove(temp_db_path)
    logger.info("Demo complete. DB and artifacts cleaned up.")

if __name__ == "__main__":
    run_demo()

import pytest
import os
import tempfile
import joblib
import hashlib
from datetime import datetime, timezone
from aegis.prediction.registry import ModelRegistry
from aegis.prediction.model_interface import PredictionModel
from aegis.prediction.models import PredictionResult, PredictionDirection
from aegis.governance.models import GovernanceStatus, ArtifactMetadata, PromotionDecision, PromotionDecisionType
from aegis.governance.artifact import ArtifactLoader
from aegis.db.session import get_db
from aegis.db.models.governance import ModelRegistration

class DummyModel(PredictionModel):
    def __init__(self, model_id, version):
        self._model_id = model_id
        self._version = version
    
    @property
    def model_id(self) -> str:
        return self._model_id
        
    @property
    def version(self) -> int:
        return self._version
        
    @property
    def schema(self):
        from aegis.prediction.model_interface import FeatureSchema
        return FeatureSchema()
        
    def is_ready(self) -> bool:
        return True
        
    def predict(self, features) -> PredictionResult:
        return PredictionResult(
            symbol="BTC/USD",
            confidence=0.9,
            direction=PredictionDirection.BUY,
            timestamp=datetime.now(timezone.utc),
            timeframe="1h"
        )

@pytest.fixture
def temp_artifact_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d

@pytest.mark.integration
def test_registry_integration_persistence_and_cache_tampering(integration_db, temp_artifact_dir):
    """
    Tests:
    - Persistence in realistic file-backed DB
    - Recovery after restart
    - Cache tampering rejection
    - promoted_at assignment
    - Champion invariant
    """
    model_id = "integration-test-model"
    version = 1
    model = DummyModel(model_id, version)
    
    # 1. Create a valid artifact
    filepath = os.path.join(temp_artifact_dir, f"{model_id}-v{version}.joblib")
    joblib.dump(model, filepath)
    
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for b in iter(lambda: f.read(4096), b""):
            sha.update(b)
    
    meta = ArtifactMetadata(
        artifact_format="joblib",
        model_identity=model_id,
        version=version,
        fingerprint="fp1",
        training_experiment="exp1",
        feature_schema_version=1,
        dataset_identity="ds1",
        training_date=datetime.now(timezone.utc),
        random_seed=42,
        integrity_hash=sha.hexdigest()
    )
    
    # 2. Register model
    registry1 = ModelRegistry(artifact_dir=temp_artifact_dir)
    registry1.register(model, metadata=meta)
    
    # Verify cached model works
    retrieved1 = registry1.get(model_id, version)
    assert retrieved1.model_id == model_id
    
    # 3. Promote candidate
    decision = PromotionDecision(
        model_identity=model_id,
        model_version=version,
        decision=PromotionDecisionType.PROMOTE,
        reason="Looks good",
        policy_identity="pol1"
    )
    registry1.promote(model_id, version, decision)
    
    # Verify promoted_at is set
    with get_db() as db:
        reg = db.query(ModelRegistration).filter_by(id=f"{model_id}-v{version}").first()
        assert reg.governance_status == GovernanceStatus.CHAMPION
        assert reg.promoted_at is not None
        assert reg.promoted_at.tzinfo is not None
    
    # 4. Simulate process restart -> new registry instance
    registry2 = ModelRegistry(artifact_dir=temp_artifact_dir)
    champ = registry2.get_champion()
    assert champ is not None
    assert champ.model_id == model_id
    assert champ.version == version
    
    # 5. Tamper physical artifact bytes
    with open(filepath, 'ab') as f:
        f.write(b'tampered')
    
    # 6. Call registry2.get() and get_champion() and EXPECT FAILURE despite cache
    with pytest.raises(RuntimeError, match="Artifact integrity hash mismatch"):
        registry2.get(model_id, version)
        
    with pytest.raises(RuntimeError, match="Artifact integrity hash mismatch"):
        registry2.get_champion()

@pytest.mark.integration
def test_registry_single_champion_constraint(integration_db):
    registry = ModelRegistry()
    m1 = DummyModel("m1", 1)
    m2 = DummyModel("m2", 1)
    
    registry.register(m1)
    registry.register(m2)
    
    d1 = PromotionDecision(model_identity="m1", model_version=1, decision=PromotionDecisionType.PROMOTE, reason="r", policy_identity="p")
    registry.promote("m1", 1, d1)
    
    d2 = PromotionDecision(model_identity="m2", model_version=1, decision=PromotionDecisionType.PROMOTE, reason="r", policy_identity="p")
    registry.promote("m2", 1, d2)
    
    with get_db() as db:
        champs = db.query(ModelRegistration).filter_by(governance_status=GovernanceStatus.CHAMPION).all()
        assert len(champs) == 1
        assert champs[0].model_identity == "m2"
        
        m1_reg = db.query(ModelRegistration).filter_by(id="m1-v1").first()
        assert m1_reg.governance_status == GovernanceStatus.SUPERSEDED

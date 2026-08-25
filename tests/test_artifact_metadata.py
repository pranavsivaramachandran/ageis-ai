import pytest
from datetime import datetime, timezone
from aegis.governance.models import ArtifactMetadata

def test_artifact_metadata_valid():
    meta = ArtifactMetadata(
        artifact_format="joblib",
        model_identity="m1",
        version=1,
        fingerprint="fp1",
        training_experiment="exp1",
        feature_schema_version=1,
        dataset_identity="ds1",
        training_date=datetime.now(timezone.utc),
        random_seed=42,
        integrity_hash="hash1"
    )
    assert meta.model_identity == "m1"
    assert meta.integrity_hash == "hash1"

def test_artifact_metadata_immutable():
    meta = ArtifactMetadata(
        artifact_format="joblib",
        model_identity="m1",
        version=1,
        fingerprint="fp1",
        training_experiment="exp1",
        feature_schema_version=1,
        dataset_identity="ds1",
        training_date=datetime.now(timezone.utc),
        random_seed=42,
        integrity_hash="hash1"
    )
    with pytest.raises(Exception):
        meta.version = 2

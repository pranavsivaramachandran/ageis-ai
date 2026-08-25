import pytest
from aegis.governance.models import ModelIdentity

def test_model_identity_determinism():
    id1 = ModelIdentity(
        model_type="rf",
        feature_schema_id="fs1",
        feature_config_id="fc1",
        target_config_id="tc1",
        training_config_id="tr1",
        seed=42
    )
    id2 = ModelIdentity(
        model_type="rf",
        feature_schema_id="fs1",
        feature_config_id="fc1",
        target_config_id="tc1",
        training_config_id="tr1",
        seed=42
    )
    assert id1.identity == id2.identity

def test_model_identity_changes():
    id1 = ModelIdentity(
        model_type="rf", feature_schema_id="fs1", feature_config_id="fc1",
        target_config_id="tc1", training_config_id="tr1", seed=42
    )
    id2 = ModelIdentity(
        model_type="xgb", feature_schema_id="fs1", feature_config_id="fc1",
        target_config_id="tc1", training_config_id="tr1", seed=42
    )
    id3 = ModelIdentity(
        model_type="rf", feature_schema_id="fs2", feature_config_id="fc1",
        target_config_id="tc1", training_config_id="tr1", seed=42
    )
    id4 = ModelIdentity(
        model_type="rf", feature_schema_id="fs1", feature_config_id="fc1",
        target_config_id="tc2", training_config_id="tr1", seed=42
    )
    id5 = ModelIdentity(
        model_type="rf", feature_schema_id="fs1", feature_config_id="fc1",
        target_config_id="tc1", training_config_id="tr2", seed=42
    )
    id6 = ModelIdentity(
        model_type="rf", feature_schema_id="fs1", feature_config_id="fc1",
        target_config_id="tc1", training_config_id="tr1", seed=43
    )
    id7 = ModelIdentity(
        model_type="rf", feature_schema_id="fs1", feature_config_id="fc2",
        target_config_id="tc1", training_config_id="tr1", seed=42
    )
    id8 = ModelIdentity(
        model_type="rf", feature_schema_id="fs1", feature_config_id="fc1",
        target_config_id="tc1", training_config_id="tr1", calibration_config_id="cal1", seed=42
    )

    assert len({id1.identity, id2.identity, id3.identity, id4.identity, id5.identity, id6.identity, id7.identity, id8.identity}) == 8

def test_model_identity_immutable():
    id1 = ModelIdentity(
        model_type="rf", feature_schema_id="fs1", feature_config_id="fc1",
        target_config_id="tc1", training_config_id="tr1", seed=42
    )
    with pytest.raises(Exception): # Pydantic ValidationError for frozen=True
        id1.seed = 43

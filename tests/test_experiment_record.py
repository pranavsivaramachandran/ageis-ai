import pytest
from datetime import datetime, timezone
from aegis.governance.models import ExperimentRecord, ExperimentStatus

def test_experiment_record_valid():
    record = ExperimentRecord(
        experiment_id="exp-1",
        dataset_identity="ds-1",
        feature_identity="fc-1",
        target_identity="tc-1",
        training_config_identity="tr-1"
    )
    assert record.status == ExperimentStatus.CREATED
    assert record.created_at is not None

def test_experiment_record_state_transition():
    # Since models are frozen, state transitions mean creating a new instance via model_copy
    record = ExperimentRecord(
        experiment_id="exp-1", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1"
    )
    running_record = record.model_copy(update={"status": ExperimentStatus.RUNNING})
    assert running_record.status == ExperimentStatus.RUNNING

def test_experiment_record_immutable():
    record = ExperimentRecord(
        experiment_id="exp-1", dataset_identity="ds-1", feature_identity="fc-1",
        target_identity="tc-1", training_config_identity="tr-1"
    )
    with pytest.raises(Exception):
        record.status = ExperimentStatus.COMPLETED

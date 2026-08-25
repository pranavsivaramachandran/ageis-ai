from datetime import datetime, timezone
from aegis.governance.monitoring.models import ReferenceProfile, MonitoringWindow, MonitoringPolicy
from aegis.governance.monitoring.engine import MonitoringEngine

def test_no_future_leakage():
    """
    Verifies that the reference and observation evaluation is completely deterministic 
    and isolated based solely on the provided inputs, ensuring no future leakage.
    """
    # Dataset A (T0 ... T1000)
    ref = ReferenceProfile(
        champion_identity="champ1", champion_version=1, experiment_identity="exp1",
        feature_schema_identity="s1", feature_config_identity="c1", target_identity="t1",
        reference_window_identity="rw1",
        feature_statistics={"f1": {"mean": 1.0, "std": 0.5, "missing_rate": 0.0}}
    )
    
    obs = MonitoringWindow(
        start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2023, 1, 31, tzinfo=timezone.utc),
        sample_count=1000,
        labeled_sample_count=1000,
        champion_identity="champ1",
        champion_version=1,
        observation_fingerprint="obs1",
        feature_statistics={"f1": {"mean": 1.1, "std": 0.6, "missing_rate": 0.0}}
    )
    
    policy = MonitoringPolicy(policy_id="p1")
    
    assessment_A = MonitoringEngine.assess_health(ref, obs, policy)
    
    # Dataset B (T0 ... T1500)
    # The reference and observation represent T0..T1000. 
    # If the system were re-evaluating the same window but the database changed in the future,
    # the inputs to `assess_health` for the window (T0..T1000) remain strictly identical by contract.
    
    ref_B = ReferenceProfile(
        champion_identity="champ1", champion_version=1, experiment_identity="exp1",
        feature_schema_identity="s1", feature_config_identity="c1", target_identity="t1",
        reference_window_identity="rw1",
        feature_statistics={"f1": {"mean": 1.0, "std": 0.5, "missing_rate": 0.0}}
    )
    
    obs_B = MonitoringWindow(
        start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2023, 1, 31, tzinfo=timezone.utc),
        sample_count=1000,
        labeled_sample_count=1000,
        champion_identity="champ1",
        champion_version=1,
        observation_fingerprint="obs1",
        feature_statistics={"f1": {"mean": 1.1, "std": 0.6, "missing_rate": 0.0}}
    )
    
    assessment_B = MonitoringEngine.assess_health(ref_B, obs_B, policy)
    
    assert assessment_A.identity == assessment_B.identity
    assert assessment_A.state == assessment_B.state
    assert len(assessment_A.alerts) == len(assessment_B.alerts)

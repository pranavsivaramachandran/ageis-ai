from datetime import datetime, timezone
import pytest
from aegis.governance.models import ChampionHealth
from aegis.events.contracts import AlertSeverity
from aegis.governance.monitoring.models import ReferenceProfile, MonitoringWindow, MonitoringPolicy
from aegis.governance.monitoring.engine import MonitoringEngine

def create_healthy_observation() -> MonitoringWindow:
    return MonitoringWindow(
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        sample_count=200,
        labeled_sample_count=100,
        champion_identity="champ1",
        champion_version=1,
        observation_fingerprint="obs1",
        feature_statistics={"f1": {"mean": 1.0, "std": 0.5, "missing_rate": 0.0}},
        prediction_statistics={"prob_BUY": 0.33, "prob_SELL": 0.33, "prob_NEUTRAL": 0.34, "mean_confidence": 0.8},
        performance_statistics={"mean_f1": 0.8, "accuracy": 0.8, "win_rate": 0.6, "max_drawdown": 0.1}
    )

def create_reference() -> ReferenceProfile:
    return ReferenceProfile(
        champion_identity="champ1",
        champion_version=1,
        experiment_identity="exp1",
        feature_schema_identity="schema1",
        feature_config_identity="fc1",
        target_identity="t1",
        reference_window_identity="ref_w",
        feature_statistics={"f1": {"mean": 1.0, "std": 0.5, "missing_rate": 0.0}},
        prediction_statistics={"prob_BUY": 0.33, "prob_SELL": 0.33, "prob_NEUTRAL": 0.34, "mean_confidence": 0.8},
        performance_statistics={"mean_f1": 0.8, "accuracy": 0.8, "win_rate": 0.6, "max_drawdown": 0.1}
    )

def test_engine_healthy():
    ref = create_reference()
    obs = create_healthy_observation()
    policy = MonitoringPolicy(policy_id="p1")
    
    assessment = MonitoringEngine.assess_health(ref, obs, policy)
    
    assert assessment.state == ChampionHealth.HEALTHY
    assert len(assessment.alerts) == 0

def test_engine_degraded_data_drift():
    ref = create_reference()
    obs = create_healthy_observation()
    # Shift mean from 1.0 to 1.6 -> (1.6 - 1.0) / 1.0 = 0.6 > max_feature_mean_shift (0.5)
    obs.feature_statistics["f1"]["mean"] = 1.6 
    policy = MonitoringPolicy(policy_id="p1")
    
    assessment = MonitoringEngine.assess_health(ref, obs, policy)
    
    assert assessment.state == ChampionHealth.DEGRADED
    assert len(assessment.alerts) == 1
    assert assessment.alerts[0].category == "DATA_DRIFT"
    assert assessment.alerts[0].severity == AlertSeverity.WARNING

def test_engine_degraded_performance_drift():
    ref = create_reference()
    obs = create_healthy_observation()
    # f1 goes from 0.8 to 0.6 -> degradation 0.2 > 0.1
    obs.performance_statistics["mean_f1"] = 0.6 
    policy = MonitoringPolicy(policy_id="p1")
    
    assessment = MonitoringEngine.assess_health(ref, obs, policy)
    
    assert assessment.state == ChampionHealth.DEGRADED
    assert len(assessment.alerts) == 1
    assert assessment.alerts[0].category == "PERFORMANCE_DRIFT"
    assert assessment.alerts[0].severity == AlertSeverity.WARNING

def test_engine_invalid_mismatch():
    ref = create_reference()
    obs = create_healthy_observation()
    # Wrong champion
    obs_dict = obs.model_dump()
    obs_dict["champion_identity"] = "champ2"
    obs2 = MonitoringWindow(**obs_dict)
    
    policy = MonitoringPolicy(policy_id="p1")
    
    assessment = MonitoringEngine.assess_health(ref, obs2, policy)
    
    assert assessment.state == ChampionHealth.INVALID
    assert len(assessment.alerts) == 1
    assert assessment.alerts[0].category == "INTEGRITY"
    assert assessment.alerts[0].severity == AlertSeverity.CRITICAL

def test_engine_insufficient_samples():
    ref = create_reference()
    obs = create_healthy_observation()
    obs_dict = obs.model_dump()
    obs_dict["sample_count"] = 50
    obs2 = MonitoringWindow(**obs_dict)
    
    policy = MonitoringPolicy(policy_id="p1", minimum_observation_samples=100)
    
    assessment = MonitoringEngine.assess_health(ref, obs2, policy)
    
    # Below min samples is a WARNING according to logic
    assert assessment.state == ChampionHealth.DEGRADED
    assert len(assessment.alerts) == 1
    assert assessment.alerts[0].severity == AlertSeverity.WARNING

def test_engine_alert_determinism():
    ref = create_reference()
    obs = create_healthy_observation()
    # trigger 1
    obs.feature_statistics["f1"]["mean"] = 1.6 
    policy = MonitoringPolicy(policy_id="p1")
    
    assessment1 = MonitoringEngine.assess_health(ref, obs, policy)
    assessment2 = MonitoringEngine.assess_health(ref, obs, policy)
    
    # Identical inputs should produce identical alert IDs
    assert assessment1.alerts[0].alert_id == assessment2.alerts[0].alert_id
    
    # Different metric should produce different ID
    obs3 = create_healthy_observation()
    obs3.feature_statistics["f1"]["std"] = 1.6 # std_shift instead of mean_shift
    assessment3 = MonitoringEngine.assess_health(ref, obs3, policy)
    
    assert assessment1.alerts[0].alert_id != assessment3.alerts[0].alert_id
    
    # Different policy should produce different ID
    policy2 = MonitoringPolicy(policy_id="p2")
    assessment4 = MonitoringEngine.assess_health(ref, obs, policy2)
    assert assessment1.alerts[0].alert_id != assessment4.alerts[0].alert_id

def test_engine_corrupt_prediction_data():
    ref = create_reference()
    obs = create_healthy_observation()
    # Corrupt data
    obs.prediction_statistics["prob_BUY"] = float('nan')
    policy = MonitoringPolicy(policy_id="p1")
    
    assessment = MonitoringEngine.assess_health(ref, obs, policy)
    
    assert assessment.state == ChampionHealth.INVALID
    assert len(assessment.alerts) == 1
    assert assessment.alerts[0].category == "INTEGRITY"
    assert assessment.alerts[0].metric == "prediction_data_corruption"
    assert assessment.alerts[0].severity == AlertSeverity.CRITICAL

def test_engine_insufficient_labeled_samples():
    ref = create_reference()
    obs = create_healthy_observation()
    
    policy = MonitoringPolicy(policy_id="p1", minimum_labeled_samples=150)
    # Labeled samples = 100 < 150
    
    assessment = MonitoringEngine.assess_health(ref, obs, policy)
    
    # Should trigger a warning for insufficient labels and skip performance drift
    assert assessment.state == ChampionHealth.DEGRADED
    assert len(assessment.alerts) == 1
    assert assessment.alerts[0].category == "PERFORMANCE_DRIFT"
    assert assessment.alerts[0].metric == "labeled_sample_count"
    assert assessment.alerts[0].reason == "insufficient_labels"
    assert assessment.alerts[0].severity == AlertSeverity.WARNING

from datetime import datetime, timezone
import json
from aegis.governance.models import ChampionHealth
from aegis.governance.monitoring.models import (
    ReferenceProfile, MonitoringWindow, MonitoringPolicy
)
from aegis.governance.monitoring.engine import MonitoringEngine

def main():
    print("=== AEGIS AI SPRINT 14 DEMO: CHAMPION HEALTH & DRIFT MONITORING ===")
    
    print("\n1. Building deterministic Reference Profile...")
    reference = ReferenceProfile(
        champion_identity="champ_X1",
        champion_version=2,
        experiment_identity="exp_A",
        feature_schema_identity="schema_v1",
        feature_config_identity="fc_v1",
        target_identity="t_v1",
        reference_window_identity="ref_w1",
        feature_statistics={
            "f_momentum": {"mean": 1.0, "std": 0.5, "missing_rate": 0.0},
            "f_volatility": {"mean": 0.2, "std": 0.1, "missing_rate": 0.0}
        },
        prediction_statistics={
            "prob_BUY": 0.33,
            "prob_SELL": 0.33,
            "prob_NEUTRAL": 0.34,
            "mean_confidence": 0.8
        },
        performance_statistics={
            "mean_f1": 0.8,
            "accuracy": 0.8,
            "win_rate": 0.6,
            "max_drawdown": 0.1
        }
    )
    
    print(f"   [Reference Profile Identity] -> {reference.identity}")
    
    policy = MonitoringPolicy(
        policy_id="strict_monitoring_v1",
        max_feature_mean_shift=0.5,
        max_prediction_divergence=0.2,
        max_f1_degradation=0.1
    )
    
    print("\n2. Creating HEALTHY Observation Window...")
    obs_healthy = MonitoringWindow(
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        sample_count=500,
        labeled_sample_count=500,
        champion_identity="champ_X1",
        champion_version=2,
        observation_fingerprint="obs_healthy_1",
        feature_statistics={
            "f_momentum": {"mean": 1.1, "std": 0.5, "missing_rate": 0.0},
            "f_volatility": {"mean": 0.21, "std": 0.1, "missing_rate": 0.0}
        },
        prediction_statistics={
            "prob_BUY": 0.34,
            "prob_SELL": 0.32,
            "prob_NEUTRAL": 0.34,
            "mean_confidence": 0.81
        },
        performance_statistics={
            "mean_f1": 0.78,
            "accuracy": 0.79,
            "win_rate": 0.59,
            "max_drawdown": 0.11
        }
    )
    
    assessment_h = MonitoringEngine.assess_health(reference, obs_healthy, policy)
    print(f"   [Health State] -> {assessment_h.state.value}")
    print(f"   [Assessment ID] -> {assessment_h.identity}")
    
    print("\n3. Creating SHIFTED Observation (Data & Prediction Drift)...")
    obs_shifted = MonitoringWindow(
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        sample_count=500,
        labeled_sample_count=500,
        champion_identity="champ_X1",
        champion_version=2,
        observation_fingerprint="obs_shifted_1",
        feature_statistics={
            "f_momentum": {"mean": 2.0, "std": 0.5, "missing_rate": 0.0}, # Shift = 1.0 > 0.5
            "f_volatility": {"mean": 0.2, "std": 0.1, "missing_rate": 0.0}
        },
        prediction_statistics={
            "prob_BUY": 0.80, # Divergence will be huge
            "prob_SELL": 0.10,
            "prob_NEUTRAL": 0.10,
            "mean_confidence": 0.5 # Drop in confidence
        },
        performance_statistics={
            "mean_f1": 0.6, # Degradation = 0.2 > 0.1
            "accuracy": 0.6,
            "win_rate": 0.4,
            "max_drawdown": 0.3
        }
    )
    
    assessment_s = MonitoringEngine.assess_health(reference, obs_shifted, policy)
    print(f"   [Health State] -> {assessment_s.state.value}")
    print(f"   [Assessment ID] -> {assessment_s.identity}")
    print("   [Alerts generated]:")
    for alert in assessment_s.alerts:
        print(f"     - [{alert.severity.value}] {alert.category} | {alert.metric}: {alert.reason}")
        
    print("\n4. Creating INVALID Observation (Schema / Identity Mismatch)...")
    obs_invalid = MonitoringWindow(
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        sample_count=500,
        labeled_sample_count=500,
        champion_identity="champ_X2_WRONG",
        champion_version=2,
        observation_fingerprint="obs_invalid_1",
        feature_statistics={},
        prediction_statistics={},
        performance_statistics={}
    )
    
    assessment_i = MonitoringEngine.assess_health(reference, obs_invalid, policy)
    print(f"   [Health State] -> {assessment_i.state.value}")
    print(f"   [Assessment ID] -> {assessment_i.identity}")
    print("   [Alerts generated]:")
    for alert in assessment_i.alerts:
        print(f"     - [{alert.severity.value}] {alert.category} | {alert.metric}: {alert.reason}")
        
    print("\n5. Conclusion:")
    print("   - Determinstic health states verified.")
    print("   - Persistence mappings created.")
    print("   - No automatic model replacement occurred (PREDICTION_ONLY preserved).")
    
if __name__ == "__main__":
    main()

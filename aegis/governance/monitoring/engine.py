from typing import Tuple, List
import uuid

from aegis.governance.models import ChampionHealth
from aegis.events.contracts import AlertSeverity
from aegis.governance.monitoring.models import (
    ReferenceProfile, MonitoringWindow, MonitoringPolicy, 
    MonitoringAlert, HealthAssessment, DriftReport, ChampionHealthReport
)
from aegis.governance.monitoring.data_drift import calculate_data_drift
from aegis.governance.monitoring.prediction_drift import calculate_prediction_drift
from aegis.governance.monitoring.performance_drift import calculate_performance_drift

class MonitoringEngine:
    @staticmethod
    def assess_health(reference: ReferenceProfile, 
                      observation: MonitoringWindow, 
                      policy: MonitoringPolicy) -> HealthAssessment:
        
        alerts: List[MonitoringAlert] = []
        reasons: List[str] = []
        
        # 1. Structural Checks (INVALID)
        if reference.champion_identity != observation.champion_identity:
            reasons.append("Champion identity mismatch")
            alerts.append(MonitoringEngine._create_alert(
                severity=AlertSeverity.CRITICAL, category="INTEGRITY", metric="champion_identity",
                reference=None, observed=None,
                threshold=None, direction="MISMATCH", reason=f"Champion identity mismatch: expected {reference.champion_identity}, got {observation.champion_identity}",
                ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                champion_identity=reference.champion_identity
            ))
            
        if reference.champion_version != observation.champion_version:
            reasons.append("Champion version mismatch")
            alerts.append(MonitoringEngine._create_alert(
                severity=AlertSeverity.CRITICAL, category="INTEGRITY", metric="champion_version",
                reference=reference.champion_version, observed=observation.champion_version,
                threshold=None, direction="MISMATCH", reason="Champion version mismatch",
                ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                champion_identity=reference.champion_identity
            ))
            
        if observation.sample_count < policy.minimum_observation_samples:
            reasons.append(f"Insufficient samples: {observation.sample_count} < {policy.minimum_observation_samples}")
            alerts.append(MonitoringEngine._create_alert(
                severity=AlertSeverity.WARNING, category="INTEGRITY", metric="sample_count",
                reference=policy.minimum_observation_samples, observed=observation.sample_count,
                threshold=policy.minimum_observation_samples, direction="BELOW", reason="Insufficient sample count",
                ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                champion_identity=reference.champion_identity
            ))
            
        # 2. Data Drift
        data_drift = calculate_data_drift(reference.feature_statistics, observation.feature_statistics)
        for feature, drift in data_drift.items():
            if drift.get("status") == "MISSING_IN_OBSERVATION":
                 reasons.append(f"Feature {feature} missing in observation")
                 alerts.append(MonitoringEngine._create_alert(
                    severity=AlertSeverity.CRITICAL, category="DATA_DRIFT", metric=f"{feature}_missing",
                    reference=0, observed=1, threshold=0, direction="MISMATCH", reason=f"Feature {feature} missing",
                    ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                    champion_identity=reference.champion_identity
                 ))
                 continue
                 
            ms = drift.get("mean_shift")
            if ms is not None and ms > policy.max_feature_mean_shift:
                 reasons.append(f"Feature {feature} mean shift {ms:.4f} > {policy.max_feature_mean_shift}")
                 alerts.append(MonitoringEngine._create_alert(
                    severity=AlertSeverity.WARNING, category="DATA_DRIFT", metric=f"{feature}_mean_shift",
                    reference=None, observed=ms, threshold=policy.max_feature_mean_shift, direction="ABOVE", reason=f"Mean shift triggered for {feature}",
                    ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                    champion_identity=reference.champion_identity
                 ))
                 
            ss = drift.get("std_shift")
            if ss is not None and ss > policy.max_feature_std_shift:
                 reasons.append(f"Feature {feature} std shift {ss:.4f} > {policy.max_feature_std_shift}")
                 alerts.append(MonitoringEngine._create_alert(
                    severity=AlertSeverity.WARNING, category="DATA_DRIFT", metric=f"{feature}_std_shift",
                    reference=None, observed=ss, threshold=policy.max_feature_std_shift, direction="ABOVE", reason=f"Std shift triggered for {feature}",
                    ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                    champion_identity=reference.champion_identity
                 ))
                 
            md = drift.get("missingness_delta")
            if md is not None and md > policy.max_missingness_delta:
                 reasons.append(f"Feature {feature} missingness delta {md:.4f} > {policy.max_missingness_delta}")
                 alerts.append(MonitoringEngine._create_alert(
                    severity=AlertSeverity.WARNING, category="DATA_DRIFT", metric=f"{feature}_missingness_delta",
                    reference=None, observed=md, threshold=policy.max_missingness_delta, direction="ABOVE", reason=f"Missingness delta triggered for {feature}",
                    ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                    champion_identity=reference.champion_identity
                 ))

        # 3. Prediction Drift
        prediction_drift = calculate_prediction_drift(reference.prediction_statistics, observation.prediction_statistics)
        
        if prediction_drift.get("status") == "CORRUPT":
            reasons.append("Prediction data contains NaN or Infinity values")
            alerts.append(MonitoringEngine._create_alert(
                severity=AlertSeverity.CRITICAL, category="INTEGRITY", metric="prediction_data_corruption",
                reference=None, observed=None, threshold=None, direction="CORRUPT", reason="Prediction data contains NaN or Infinity",
                ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                champion_identity=reference.champion_identity
            ))
        else:
            div = prediction_drift.get("prediction_divergence")
            if div is not None and div > policy.max_prediction_divergence:
                reasons.append(f"Prediction divergence {div:.4f} > {policy.max_prediction_divergence}")
                alerts.append(MonitoringEngine._create_alert(
                    severity=AlertSeverity.WARNING, category="PREDICTION_DRIFT", metric="prediction_divergence",
                    reference=None, observed=div, threshold=policy.max_prediction_divergence, direction="ABOVE", reason="Prediction divergence triggered",
                    ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                    champion_identity=reference.champion_identity
                ))
                
            conf_shift = prediction_drift.get("confidence_shift")
            if conf_shift is not None and abs(conf_shift) > policy.max_confidence_shift:
                reasons.append(f"Confidence shift {conf_shift:.4f} exceeded threshold {policy.max_confidence_shift}")
                alerts.append(MonitoringEngine._create_alert(
                    severity=AlertSeverity.WARNING, category="PREDICTION_DRIFT", metric="confidence_shift",
                    reference=None, observed=conf_shift, threshold=policy.max_confidence_shift, direction="ABOVE", reason="Confidence shift triggered",
                    ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                    champion_identity=reference.champion_identity
                ))

        # 4. Performance Drift (if enough labeled samples)
        performance_drift = {}
        if observation.labeled_sample_count >= policy.minimum_labeled_samples:
            performance_drift = calculate_performance_drift(reference.performance_statistics, observation.performance_statistics)
            
            f1_deg = performance_drift.get("mean_f1_degradation")
            if f1_deg is not None and f1_deg > policy.max_f1_degradation:
                reasons.append(f"F1 degradation {f1_deg:.4f} > {policy.max_f1_degradation}")
                alerts.append(MonitoringEngine._create_alert(
                    severity=AlertSeverity.WARNING, category="PERFORMANCE_DRIFT", metric="mean_f1_degradation",
                    reference=None, observed=f1_deg, threshold=policy.max_f1_degradation, direction="ABOVE", reason="F1 degradation triggered",
                    ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                    champion_identity=reference.champion_identity
                ))
                
            acc_deg = performance_drift.get("accuracy_degradation")
            if acc_deg is not None and acc_deg > policy.max_accuracy_degradation:
                reasons.append(f"Accuracy degradation {acc_deg:.4f} > {policy.max_accuracy_degradation}")
                alerts.append(MonitoringEngine._create_alert(
                    severity=AlertSeverity.WARNING, category="PERFORMANCE_DRIFT", metric="accuracy_degradation",
                    reference=None, observed=acc_deg, threshold=policy.max_accuracy_degradation, direction="ABOVE", reason="Accuracy degradation triggered",
                    ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                    champion_identity=reference.champion_identity
                ))
                
            dd_inc = performance_drift.get("max_drawdown_increase")
            if dd_inc is not None and dd_inc > policy.max_drawdown_increase:
                reasons.append(f"Drawdown increase {dd_inc:.4f} > {policy.max_drawdown_increase}")
                alerts.append(MonitoringEngine._create_alert(
                    severity=AlertSeverity.WARNING, category="PERFORMANCE_DRIFT", metric="max_drawdown_increase",
                    reference=None, observed=dd_inc, threshold=policy.max_drawdown_increase, direction="ABOVE", reason="Drawdown increase triggered",
                    ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                    champion_identity=reference.champion_identity
                ))
                
            wr_deg = performance_drift.get("win_rate_degradation")
            if wr_deg is not None and wr_deg > policy.max_win_rate_decrease:
                reasons.append(f"Win rate degradation {wr_deg:.4f} > {policy.max_win_rate_decrease}")
                alerts.append(MonitoringEngine._create_alert(
                    severity=AlertSeverity.WARNING, category="PERFORMANCE_DRIFT", metric="win_rate_degradation",
                    reference=None, observed=wr_deg, threshold=policy.max_win_rate_decrease, direction="ABOVE", reason="Win rate degradation triggered",
                    ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                    champion_identity=reference.champion_identity
                ))
        else:
            reasons.append(f"Insufficient labeled samples for performance drift: {observation.labeled_sample_count} < {policy.minimum_labeled_samples}")
            alerts.append(MonitoringEngine._create_alert(
                severity=AlertSeverity.WARNING, category="PERFORMANCE_DRIFT", metric="labeled_sample_count",
                reference=policy.minimum_labeled_samples, observed=observation.labeled_sample_count,
                threshold=policy.minimum_labeled_samples, direction="BELOW", reason="insufficient_labels",
                ref_id=reference.identity, obs_id=observation.observation_fingerprint, pol_id=policy.identity,
                champion_identity=reference.champion_identity
            ))
        
        # State evaluation
        state = ChampionHealth.HEALTHY
        if any(a.severity == AlertSeverity.WARNING for a in alerts):
            state = ChampionHealth.DEGRADED
        if any(a.severity == AlertSeverity.CRITICAL for a in alerts):
            state = ChampionHealth.INVALID
            
        if not reasons:
            reasons.append("No drift limits exceeded. Champion remains healthy.")
            
        report = DriftReport(
            data_drift=data_drift,
            prediction_drift=prediction_drift,
            performance_drift=performance_drift,
            schema_status="VALID" if state != ChampionHealth.INVALID else "INVALID",
            sample_size=observation.sample_count
        )
        
        return HealthAssessment(
            champion_identity=reference.champion_identity,
            champion_version=reference.champion_version,
            observation_identity=observation.observation_fingerprint,
            reference_identity=reference.identity,
            policy_identity=policy.identity,
            state=state,
            reasons=reasons,
            drift_report=report,
            alerts=alerts
        )

    @staticmethod
    def _create_alert(severity, category, metric, reference, observed, threshold, direction, reason, ref_id, obs_id, pol_id, champion_identity):
        import hashlib
        canonical_str = f"{champion_identity}_{obs_id}_{pol_id}_{category}_{metric}_{direction}"
        alert_id = hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()
        
        return MonitoringAlert(
            alert_id=alert_id,
            severity=severity,
            category=category,
            metric=metric,
            reference_value=float(reference) if reference is not None else None,
            observed_value=float(observed) if observed is not None else None,
            threshold=float(threshold) if threshold is not None else None,
            direction=direction,
            champion_identity=champion_identity,
            observation_identity=obs_id,
            policy_identity=pol_id,
            reason=reason
        )

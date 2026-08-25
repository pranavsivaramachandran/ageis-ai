from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import hashlib
import json

from aegis.governance.models import ChampionHealth
from aegis.events.contracts import AlertSeverity

class ReferenceProfile(BaseModel):
    """Immutable deterministic baseline profile of an approved champion."""
    champion_identity: str
    champion_version: int
    experiment_identity: str
    feature_schema_identity: str
    feature_config_identity: str
    target_identity: str
    calibration_identity: Optional[str] = None
    reference_window_identity: str
    
    feature_statistics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    prediction_statistics: Dict[str, float] = Field(default_factory=dict)
    performance_statistics: Dict[str, float] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = {"frozen": True}

    @property
    def identity(self) -> str:
        # Exclude datetime from hash for determinism
        components = {
            "champion_identity": self.champion_identity,
            "champion_version": self.champion_version,
            "experiment_identity": self.experiment_identity,
            "reference_window_identity": self.reference_window_identity,
            "feature_statistics": self.feature_statistics,
            "prediction_statistics": self.prediction_statistics,
            "performance_statistics": self.performance_statistics
        }
        canonical_str = json.dumps(components, sort_keys=True)
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()[:16]

class MonitoringWindow(BaseModel):
    """Represents an observation period to be evaluated."""
    start_time: datetime
    end_time: datetime
    sample_count: int
    labeled_sample_count: int
    
    champion_identity: str
    champion_version: int
    
    observation_fingerprint: str
    
    feature_statistics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    prediction_statistics: Dict[str, float] = Field(default_factory=dict)
    performance_statistics: Dict[str, float] = Field(default_factory=dict)

    model_config = {"frozen": True}

class MonitoringPolicy(BaseModel):
    """Configurable thresholds for drift detection."""
    policy_id: str
    version: int = 1
    
    max_feature_mean_shift: float = 0.5
    max_feature_std_shift: float = 0.5
    max_missingness_delta: float = 0.1
    
    max_prediction_divergence: float = 0.2
    max_confidence_shift: float = 0.1
    
    max_f1_degradation: float = 0.1
    max_accuracy_degradation: float = 0.1
    max_drawdown_increase: float = 0.1
    max_win_rate_decrease: float = 0.1
    
    minimum_observation_samples: int = 100
    minimum_labeled_samples: int = 50
    
    model_config = {"frozen": True}

    @property
    def identity(self) -> str:
        canonical_str = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()[:16]

class MonitoringAlert(BaseModel):
    """Structured alert indicating a threshold was triggered."""
    alert_id: str
    severity: AlertSeverity
    category: str 
    metric: str
    reference_value: Optional[float] = None
    observed_value: Optional[float] = None
    threshold: Optional[float] = None
    direction: str 
    
    champion_identity: str
    observation_identity: str
    policy_identity: str
    reason: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": True}

class DriftReport(BaseModel):
    """Report detailing drift metrics."""
    data_drift: Dict[str, Any] = Field(default_factory=dict)
    prediction_drift: Dict[str, Any] = Field(default_factory=dict)
    confidence_drift: Dict[str, Any] = Field(default_factory=dict)
    performance_drift: Dict[str, Any] = Field(default_factory=dict)
    schema_status: str = "VALID"
    sample_size: int = 0
    
class HealthAssessment(BaseModel):
    """Result of evaluating a MonitoringWindow against a ReferenceProfile."""
    champion_identity: str
    champion_version: int
    observation_identity: str
    reference_identity: str
    policy_identity: str
    
    state: ChampionHealth
    reasons: List[str] = Field(default_factory=list)
    
    drift_report: DriftReport
    alerts: List[MonitoringAlert] = Field(default_factory=list)
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": True}
    
    @property
    def identity(self) -> str:
        components = {
            "reference_identity": self.reference_identity,
            "observation_identity": self.observation_identity,
            "policy_identity": self.policy_identity,
            "state": self.state.value
        }
        canonical_str = json.dumps(components, sort_keys=True)
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()[:16]

class ChampionHealthReport(BaseModel):
    champion_identity: str
    champion_version: int
    reference_profile: ReferenceProfile
    observation: MonitoringWindow
    health_state: ChampionHealth
    drift_report: DriftReport
    alerts: List[MonitoringAlert] = Field(default_factory=list)
    monitoring_policy: MonitoringPolicy
    recommendation: str 
    
    model_config = {"frozen": True}

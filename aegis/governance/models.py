from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import hashlib
import json

class GovernanceStatus(Enum):
    """Lifecycle states of a model in the registry."""
    CANDIDATE = "CANDIDATE"
    CHALLENGER = "CHALLENGER"
    PROMOTED = "PROMOTED"
    CHAMPION = "CHAMPION"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"
    RETIRED = "RETIRED"


class ChampionHealth(Enum):
    """Health assessment of the current champion."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


class ExperimentStatus(Enum):
    """Explicit lifecycle states of an experiment."""
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    PROMOTABLE = "PROMOTABLE"
    PROMOTED = "PROMOTED"
    SUPERSEDED = "SUPERSEDED"


class PromotionDecisionType(Enum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    HOLD = "HOLD"


class ModelIdentity(BaseModel):
    """Deterministic logical identity of a model."""
    
    model_type: str = Field(..., min_length=1)
    feature_schema_id: str = Field(..., min_length=1)
    feature_config_id: str = Field(..., min_length=1)
    target_config_id: str = Field(..., min_length=1)
    training_config_id: str = Field(..., min_length=1)
    calibration_config_id: Optional[str] = None
    selection_config_id: Optional[str] = None
    seed: int
    
    model_config = {"frozen": True}
    
    @property
    def identity(self) -> str:
        """Reproducible model identity hash based on logical configuration."""
        components = {
            "model_type": self.model_type,
            "feature_schema_id": self.feature_schema_id,
            "feature_config_id": self.feature_config_id,
            "target_config_id": self.target_config_id,
            "training_config_id": self.training_config_id,
            "calibration_config_id": self.calibration_config_id,
            "selection_config_id": self.selection_config_id,
            "seed": self.seed
        }
        canonical_str = json.dumps(components, sort_keys=True)
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()[:16]


class PromotionPolicy(BaseModel):
    """Explicit criteria required for promotion."""
    
    policy_id: str = Field(..., min_length=1)
    version: int = Field(default=1, gt=0)
    
    min_walk_forward_windows: int = Field(default=1, gt=0)
    min_mean_f1: float = Field(default=0.0)
    max_drawdown_pct: float = Field(default=1.0, description="Expressed as positive fraction e.g. 0.20 for 20%")
    min_trades_per_window: int = Field(default=1)
    requires_baseline_beat: bool = Field(default=True)
    require_reproducibility: bool = Field(default=True)
    require_safety_check: bool = Field(default=True)
    
    model_config = {"frozen": True}
    
    @property
    def identity(self) -> str:
        canonical_str = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()[:16]


class PromotionDecision(BaseModel):
    """Immutable record of a promotion evaluation."""
    
    model_identity: str
    model_version: int
    decision: PromotionDecisionType
    reason: str
    policy_identity: str
    evidence_summary: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    previous_champion: Optional[str] = None
    new_champion: Optional[str] = None
    
    model_config = {"frozen": True}


class ExperimentRecord(BaseModel):
    """Structured immutable record of an experiment."""
    
    experiment_id: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_identity: str
    feature_identity: str
    target_identity: str
    candidate_model_ids: List[str] = Field(default_factory=list)
    selected_model_id: Optional[str] = None
    walk_forward_identity: Optional[str] = None
    robustness_identity: Optional[str] = None
    training_config_identity: str
    status: ExperimentStatus = ExperimentStatus.CREATED
    overall_metrics: Dict[str, float] = Field(default_factory=dict)
    failure_reason: Optional[str] = None
    
    model_config = {"frozen": True}


class ArtifactMetadata(BaseModel):
    """Metadata describing a model artifact."""
    
    artifact_format: str
    model_identity: str
    version: int
    fingerprint: str
    training_experiment: str
    feature_schema_version: int
    dataset_identity: str
    training_date: datetime
    dependency_versions: Dict[str, str] = Field(default_factory=dict)
    random_seed: int
    integrity_hash: str
    
    model_config = {"frozen": True}


class GovernanceReport(BaseModel):
    """Human-readable governance state summary."""
    
    champion_identity: Optional[str] = None
    champion_version: Optional[int] = None
    champion_health: Optional[ChampionHealth] = None
    challengers: List[str] = Field(default_factory=list)
    rejected_candidates: List[str] = Field(default_factory=list)
    last_promotion_decision: Optional[PromotionDecision] = None
    safety_status: str = "PREDICTION_ONLY"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = {"frozen": True}

"""
Model registry for AEGIS AI.

Provides deterministic in-process and persistent registration and retrieval of prediction models,
and integrates governance tracking (champion/challenger).
"""

from typing import Dict, Optional, List
import os
from aegis.prediction.model_interface import PredictionModel
from aegis.governance.models import GovernanceStatus, PromotionDecision, PromotionDecisionType, ArtifactMetadata
from aegis.governance.artifact import ArtifactLoader, ArtifactLoadResultStatus, ArtifactVerifier
from aegis.db.session import get_db
from aegis.db.models.governance import ModelRegistration, GovernanceAuditRecord
from datetime import datetime, timezone

class ModelRegistry:
    """
    Persistent deterministic registry for prediction models.
    
    Prevents duplicate registrations and provides safe retrieval.
    Tracks governance status (CANDIDATE, CHALLENGER, CHAMPION, etc.).
    Maintains a single active CHAMPION invariant both in memory and DB.
    """
    
    def __init__(self, artifact_dir: str = "artifacts"):
        self._models: Dict[str, PredictionModel] = {}
        self.artifact_dir = artifact_dir
        
    def register(self, model: PredictionModel, initial_status: GovernanceStatus = GovernanceStatus.CANDIDATE, metadata: Optional[ArtifactMetadata] = None) -> None:
        """
        Register a PredictionModel instance.
        """
        full_id = f"{model.model_id}-v{model.version}"
        
        with get_db() as db:
            existing = db.query(ModelRegistration).filter_by(id=full_id).first()
            if existing:
                raise ValueError(f"Model identity '{full_id}' is already registered")
                
            if initial_status == GovernanceStatus.CHAMPION:
                current_champ = db.query(ModelRegistration).filter_by(governance_status=GovernanceStatus.CHAMPION).first()
                if current_champ:
                    raise ValueError("Cannot register a new CHAMPION directly when an active CHAMPION already exists.")
            
            reg = ModelRegistration(
                id=full_id,
                model_identity=model.model_id,
                version=model.version,
                model_type="Unknown" if not metadata else metadata.artifact_format,
                governance_status=initial_status,
                dataset_identity="Unknown" if not metadata else metadata.dataset_identity,
                feature_identity="Unknown", # Placeholder
                experiment_identity="Unknown" if not metadata else metadata.training_experiment,
                artifact_hash=None if not metadata else metadata.integrity_hash
            )
            db.add(reg)
            
            audit = GovernanceAuditRecord(
                model_identity=model.model_id,
                model_version=model.version,
                experiment_identity=reg.experiment_identity,
                action="REGISTER",
                reason="Initial registration",
                policy_identity="none"
            )
            db.add(audit)
            db.commit()
            
        # Cache in memory
        self._models[full_id] = model
            
    def _load_from_artifact(self, reg: ModelRegistration) -> PredictionModel:
        if not reg.artifact_hash:
            raise ValueError(f"Model {reg.id} has no artifact hash, cannot load from disk.")
            
        filepath = os.path.join(self.artifact_dir, f"{reg.id}.joblib")
        # Construct expected metadata
        expected_meta = ArtifactMetadata(
            artifact_format=reg.model_type,
            model_identity=reg.model_identity,
            version=reg.version,
            fingerprint="unknown",
            training_experiment=reg.experiment_identity,
            feature_schema_version=1,
            dataset_identity=reg.dataset_identity,
            training_date=reg.created_at,
            random_seed=42,
            integrity_hash=reg.artifact_hash
        )
        
        result = ArtifactLoader.load(filepath, expected_meta)
        if result.status != ArtifactLoadResultStatus.VALID:
            raise RuntimeError(f"Artifact verification failed for {reg.id}: {result.reason}")
            
        if not result.model:
            raise RuntimeError(f"Model {reg.id} failed to load.")
            
        self._models[reg.id] = result.model
        return result.model
        
    def get(self, model_id: str, version: int) -> PredictionModel:
        """Retrieve a registered PredictionModel."""
        full_id = f"{model_id}-v{version}"
        
        with get_db() as db:
            reg = db.query(ModelRegistration).filter_by(id=full_id).first()
            if not reg:
                raise KeyError(f"Model identity '{full_id}' not found in registry")
                
            # If in cache, we MUST still verify the physical file hash if one exists
            if full_id in self._models:
                if reg.artifact_hash:
                    filepath = os.path.join(self.artifact_dir, f"{reg.id}.joblib")
                    observed_hash = ArtifactVerifier.compute_file_hash(filepath)
                    if observed_hash is None:
                        raise RuntimeError(f"Artifact missing for cached model {full_id}.")
                    if observed_hash != reg.artifact_hash:
                        raise RuntimeError(f"Artifact integrity hash mismatch for cached model {full_id}. Expected {reg.artifact_hash}, got {observed_hash}")
                return self._models[full_id]
                
            return self._load_from_artifact(reg)
        
    def get_status(self, model_id: str, version: int) -> GovernanceStatus:
        full_id = f"{model_id}-v{version}"
        with get_db() as db:
            reg = db.query(ModelRegistration).filter_by(id=full_id).first()
            if not reg:
                raise KeyError(f"Model identity '{full_id}' not found in registry")
            return reg.governance_status

    def list_models(self) -> list[str]:
        """Return a list of all registered model identities."""
        with get_db() as db:
            regs = db.query(ModelRegistration.id).all()
            return sorted([r.id for r in regs])

    def get_champion(self) -> Optional[PredictionModel]:
        """Return the current active CHAMPION model, if any."""
        with get_db() as db:
            champ_reg = db.query(ModelRegistration).filter_by(governance_status=GovernanceStatus.CHAMPION).first()
            if not champ_reg:
                return None
                
        return self.get(champ_reg.model_identity, champ_reg.version)

    def promote(self, model_id: str, version: int, decision: PromotionDecision) -> None:
        """
        Promote a model to CHAMPION based on a promotion decision.
        Supersedes the previous champion.
        """
        if decision.decision != PromotionDecisionType.PROMOTE:
            raise ValueError("Promotion decision must be PROMOTE")
            
        full_id = f"{model_id}-v{version}"
        
        with get_db() as db:
            target_reg = db.query(ModelRegistration).filter_by(id=full_id).first()
            if not target_reg:
                raise ValueError(f"Cannot promote unknown model '{full_id}'")
                
            # Demote existing champion if any
            champ_reg = db.query(ModelRegistration).filter_by(governance_status=GovernanceStatus.CHAMPION).first()
            if champ_reg:
                champ_reg.governance_status = GovernanceStatus.SUPERSEDED
                champ_reg.superseded_at = datetime.now(timezone.utc)
                db.flush()
                
            target_reg.governance_status = GovernanceStatus.CHAMPION
            target_reg.promoted_at = datetime.now(timezone.utc)
            
            audit = GovernanceAuditRecord(
                model_identity=model_id,
                model_version=version,
                experiment_identity=target_reg.experiment_identity,
                action="PROMOTE",
                reason=decision.reason,
                policy_identity=decision.policy_identity
            )
            db.add(audit)
            db.commit()

    def reject(self, model_id: str, version: int, decision: PromotionDecision) -> None:
        """Mark a model as REJECTED."""
        if decision.decision != PromotionDecisionType.REJECT:
            raise ValueError("Promotion decision must be REJECT")
            
        full_id = f"{model_id}-v{version}"
        
        with get_db() as db:
            target_reg = db.query(ModelRegistration).filter_by(id=full_id).first()
            if not target_reg:
                raise ValueError(f"Cannot reject unknown model '{full_id}'")
                
            if target_reg.governance_status == GovernanceStatus.CHAMPION:
                raise ValueError("Cannot reject the active CHAMPION using this method.")
                
            target_reg.governance_status = GovernanceStatus.REJECTED
            
            audit = GovernanceAuditRecord(
                model_identity=model_id,
                model_version=version,
                experiment_identity=target_reg.experiment_identity,
                action="REJECT",
                reason=decision.reason,
                policy_identity=decision.policy_identity
            )
            db.add(audit)
            db.commit()

    def retire(self, model_id: str, version: int) -> None:
        """Mark a model as RETIRED."""
        full_id = f"{model_id}-v{version}"
        
        with get_db() as db:
            target_reg = db.query(ModelRegistration).filter_by(id=full_id).first()
            if not target_reg:
                raise ValueError(f"Cannot retire unknown model '{full_id}'")
                
            target_reg.governance_status = GovernanceStatus.RETIRED
            
            audit = GovernanceAuditRecord(
                model_identity=model_id,
                model_version=version,
                experiment_identity=target_reg.experiment_identity,
                action="RETIRE",
                reason="Manual retirement",
                policy_identity="none"
            )
            db.add(audit)
            db.commit()

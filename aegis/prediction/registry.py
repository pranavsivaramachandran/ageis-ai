"""
Model registry for AEGIS AI.

Provides deterministic in-process registration and retrieval of prediction models,
and integrates governance tracking (champion/challenger).
"""

from typing import Dict, Optional, List
from aegis.prediction.model_interface import PredictionModel
from aegis.governance.models import GovernanceStatus, PromotionDecision, PromotionDecisionType


class ModelRegistry:
    """
    In-memory deterministic registry for prediction models.
    
    Prevents duplicate registrations and provides safe retrieval.
    Tracks governance status (CANDIDATE, CHALLENGER, CHAMPION, etc.).
    Maintains a single active CHAMPION invariant.
    """
    
    def __init__(self):
        self._models: Dict[str, PredictionModel] = {}
        self._statuses: Dict[str, GovernanceStatus] = {}
        self._champion_id: Optional[str] = None
        
    def register(self, model: PredictionModel, initial_status: GovernanceStatus = GovernanceStatus.CANDIDATE) -> None:
        """
        Register a PredictionModel instance.
        
        Args:
            model: The PredictionModel to register.
            initial_status: GovernanceStatus of the model (default CANDIDATE).
            
        Raises:
            ValueError: If the model_id is already registered.
        """
        full_id = f"{model.model_id}-v{model.version}"
        if full_id in self._models:
            raise ValueError(f"Model identity '{full_id}' is already registered")
            
        self._models[full_id] = model
        self._statuses[full_id] = initial_status
        
        # Invariant check: Do not allow registering a champion directly if one exists
        if initial_status == GovernanceStatus.CHAMPION:
            if self._champion_id is not None:
                raise ValueError("Cannot register a new CHAMPION directly when an active CHAMPION already exists.")
            self._champion_id = full_id
            
    def get(self, model_id: str, version: int) -> PredictionModel:
        """Retrieve a registered PredictionModel."""
        full_id = f"{model_id}-v{version}"
        if full_id not in self._models:
            raise KeyError(f"Model identity '{full_id}' not found in registry")
            
        return self._models[full_id]
        
    def get_status(self, model_id: str, version: int) -> GovernanceStatus:
        full_id = f"{model_id}-v{version}"
        if full_id not in self._statuses:
            raise KeyError(f"Model identity '{full_id}' not found in registry")
        return self._statuses[full_id]

    def list_models(self) -> list[str]:
        """Return a list of all registered model identities."""
        return sorted(list(self._models.keys()))

    def get_champion(self) -> Optional[PredictionModel]:
        """Return the current active CHAMPION model, if any."""
        if self._champion_id is None:
            return None
        return self._models[self._champion_id]

    def promote(self, model_id: str, version: int, decision: PromotionDecision) -> None:
        """
        Promote a model to CHAMPION based on a promotion decision.
        Supersedes the previous champion.
        """
        if decision.decision != PromotionDecisionType.PROMOTE:
            raise ValueError("Promotion decision must be PROMOTE")
            
        full_id = f"{model_id}-v{version}"
        if full_id not in self._models:
            raise ValueError(f"Cannot promote unknown model '{full_id}'")
            
        # Transition previous champion to SUPERSEDED
        if self._champion_id:
            self._statuses[self._champion_id] = GovernanceStatus.SUPERSEDED
            
        self._statuses[full_id] = GovernanceStatus.CHAMPION
        self._champion_id = full_id

    def reject(self, model_id: str, version: int, decision: PromotionDecision) -> None:
        """Mark a model as REJECTED."""
        if decision.decision != PromotionDecisionType.REJECT:
            raise ValueError("Promotion decision must be REJECT")
            
        full_id = f"{model_id}-v{version}"
        if full_id not in self._models:
            raise ValueError(f"Cannot reject unknown model '{full_id}'")
            
        if self._statuses[full_id] == GovernanceStatus.CHAMPION:
            raise ValueError("Cannot reject the active CHAMPION using this method.")
            
        self._statuses[full_id] = GovernanceStatus.REJECTED

    def retire(self, model_id: str, version: int) -> None:
        """Mark a model as RETIRED."""
        full_id = f"{model_id}-v{version}"
        if full_id not in self._models:
            raise ValueError(f"Cannot retire unknown model '{full_id}'")
            
        self._statuses[full_id] = GovernanceStatus.RETIRED
        if self._champion_id == full_id:
            self._champion_id = None

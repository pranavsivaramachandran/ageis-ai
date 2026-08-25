"""
Governance events for AEGIS AI.
"""

from pydantic import Field
from typing import Optional
from aegis.events.contracts import BaseEvent

class ModelRegisteredEvent(BaseEvent):
    event_type: str = Field(default="MODEL_REGISTERED", frozen=True)
    model_identity: str
    version: int

class ExperimentCompletedEvent(BaseEvent):
    event_type: str = Field(default="EXPERIMENT_COMPLETED", frozen=True)
    experiment_id: str
    status: str

class ModelPromotedEvent(BaseEvent):
    event_type: str = Field(default="MODEL_PROMOTED", frozen=True)
    model_identity: str
    version: int
    previous_champion: Optional[str] = None

class ModelRejectedEvent(BaseEvent):
    event_type: str = Field(default="MODEL_REJECTED", frozen=True)
    model_identity: str
    version: int
    reason: str

class ModelRetiredEvent(BaseEvent):
    event_type: str = Field(default="MODEL_RETIRED", frozen=True)
    model_identity: str
    version: int

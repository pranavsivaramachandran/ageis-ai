from sqlalchemy import Column, String, Integer, DateTime, Enum as SAEnum, Index, Float, Text
from sqlalchemy.sql import func
from aegis.db.models.base import Base
from aegis.governance.models import ChampionHealth
from aegis.events.contracts import AlertSeverity
from aegis.db.models.governance import UTCDateTime

class ReferenceProfileRecord(Base):
    __tablename__ = "monitoring_reference_profile"
    
    id = Column(String, primary_key=True) # Identity hash
    champion_identity = Column(String, nullable=False, index=True)
    champion_version = Column(Integer, nullable=False)
    experiment_identity = Column(String, nullable=False)
    reference_window_identity = Column(String, nullable=False)
    
    profile_data = Column(Text, nullable=False) # JSON serialized
    created_at = Column(UTCDateTime(), server_default=func.now())


class MonitoringPolicyRecord(Base):
    __tablename__ = "monitoring_policy"
    
    id = Column(String, primary_key=True) # Identity hash
    policy_id = Column(String, nullable=False, index=True)
    version = Column(Integer, nullable=False)
    
    policy_data = Column(Text, nullable=False) # JSON serialized
    created_at = Column(UTCDateTime(), server_default=func.now())


class HealthAssessmentRecord(Base):
    __tablename__ = "monitoring_health_assessment"
    
    id = Column(String, primary_key=True) # Identity hash
    champion_identity = Column(String, nullable=False, index=True)
    champion_version = Column(Integer, nullable=False)
    observation_identity = Column(String, nullable=False)
    reference_identity = Column(String, nullable=False, index=True)
    policy_identity = Column(String, nullable=False)
    
    state = Column(SAEnum(ChampionHealth), nullable=False, index=True)
    reasons = Column(Text, nullable=False) # JSON serialized list
    drift_report = Column(Text, nullable=False) # JSON serialized
    
    timestamp = Column(UTCDateTime(), server_default=func.now(), index=True)


class MonitoringAlertRecord(Base):
    __tablename__ = "monitoring_alert"
    
    id = Column(String, primary_key=True) # Alert identity
    assessment_identity = Column(String, nullable=False, index=True)
    severity = Column(SAEnum(AlertSeverity), nullable=False, index=True)
    category = Column(String, nullable=False)
    metric = Column(String, nullable=False)
    
    reference_value = Column(Float, nullable=True)
    observed_value = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    direction = Column(String, nullable=False)
    
    champion_identity = Column(String, nullable=False)
    policy_identity = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    
    timestamp = Column(UTCDateTime(), server_default=func.now())

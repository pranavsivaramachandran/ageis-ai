from sqlalchemy import Column, String, Integer, DateTime, Enum as SAEnum, Index, TypeDecorator
from sqlalchemy.sql import func
import datetime

class UTCDateTime(TypeDecorator):
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            if not value.tzinfo:
                raise TypeError("tzinfo is required")
            value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return value
from aegis.db.models.base import Base
from aegis.governance.models import GovernanceStatus

class ModelRegistration(Base):
    __tablename__ = "governance_registry"
    
    __table_args__ = (
        Index('uq_active_champion', 'governance_status', unique=True, sqlite_where=(Column('governance_status') == GovernanceStatus.CHAMPION.name)),
    )

    id = Column(String, primary_key=True) # Format: "{model_id}-v{version}"
    model_identity = Column(String, nullable=False, index=True)
    version = Column(Integer, nullable=False)
    model_type = Column(String, nullable=False)
    
    governance_status = Column(SAEnum(GovernanceStatus), nullable=False, index=True)
    
    dataset_identity = Column(String, nullable=False)
    feature_identity = Column(String, nullable=False)
    experiment_identity = Column(String, nullable=False)
    
    artifact_hash = Column(String, nullable=True)
    
    created_at = Column(UTCDateTime(), server_default=func.now())
    promoted_at = Column(UTCDateTime(), nullable=True)
    superseded_at = Column(UTCDateTime(), nullable=True)
    retired_at = Column(UTCDateTime(), nullable=True)


class GovernanceAuditRecord(Base):
    __tablename__ = "governance_audit"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(UTCDateTime(), server_default=func.now())
    model_identity = Column(String, nullable=False, index=True)
    model_version = Column(Integer, nullable=False)
    experiment_identity = Column(String, nullable=False)
    action = Column(String, nullable=False) # PROMOTE, REJECT, REGISTER, SUPERSEDE, RETIRE, ROLLBACK
    reason = Column(String, nullable=False)
    policy_identity = Column(String, nullable=False)

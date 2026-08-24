from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime
from aegis.db.models.base import Base


class SystemEvent(Base):
    """
    Audit log for system-level events (startup, shutdown, state changes, errors).
    Provides an end-to-end validation of the SQLAlchemy ORM stack and serves
    as the foundation for Sprint 2's audit trail.
    """
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    event_type = Column(String(50), nullable=False, index=True)
    detail = Column(Text, nullable=True)
    trace_id = Column(String(64), nullable=True, index=True)

    def __repr__(self):
        return f"<SystemEvent(id={self.id}, type={self.event_type}, trace_id={self.trace_id})>"

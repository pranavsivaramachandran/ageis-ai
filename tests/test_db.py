"""
Tests for aegis.db — session management, init_db, and SystemEvent CRUD.
Uses in-memory SQLite via conftest fixtures for isolation.
"""
from datetime import datetime, timezone

from sqlalchemy import inspect

from aegis.db.models.base import Base
from aegis.db.models.system_event import SystemEvent


class TestInitDb:

    def test_init_db_creates_system_events_table(self, in_memory_engine):
        """After creating tables, the system_events table should exist."""
        inspector = inspect(in_memory_engine)
        tables = inspector.get_table_names()
        assert "system_events" in tables

    def test_system_events_has_expected_columns(self, in_memory_engine):
        """The system_events table should have the expected columns."""
        inspector = inspect(in_memory_engine)
        columns = {col["name"] for col in inspector.get_columns("system_events")}
        assert columns >= {"id", "timestamp", "event_type", "detail", "trace_id"}


class TestGetDbSession:

    def test_session_can_query(self, db_session):
        """A session from the fixture should be able to query without error."""
        result = db_session.query(SystemEvent).all()
        assert result == []

    def test_session_insert_and_query(self, db_session):
        """Should be able to insert and retrieve a SystemEvent."""
        event = SystemEvent(
            event_type="TEST",
            detail="test detail",
            trace_id="test-trace-123",
        )
        db_session.add(event)
        db_session.flush()

        retrieved = db_session.query(SystemEvent).filter_by(trace_id="test-trace-123").first()
        assert retrieved is not None
        assert retrieved.event_type == "TEST"
        assert retrieved.detail == "test detail"


class TestSystemEventModel:

    def test_timestamp_auto_set(self, db_session):
        """Timestamp should be auto-populated on insert."""
        event = SystemEvent(event_type="STARTUP", trace_id="auto-ts")
        db_session.add(event)
        db_session.flush()

        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)

    def test_repr(self):
        """__repr__ should produce a readable string."""
        event = SystemEvent(id=1, event_type="SHUTDOWN", trace_id="abc")
        assert "SystemEvent" in repr(event)
        assert "SHUTDOWN" in repr(event)

    def test_detail_nullable(self, db_session):
        """detail column should be nullable."""
        event = SystemEvent(event_type="STATE_CHANGE", trace_id="nullable-test")
        db_session.add(event)
        db_session.flush()

        retrieved = db_session.query(SystemEvent).filter_by(trace_id="nullable-test").first()
        assert retrieved.detail is None

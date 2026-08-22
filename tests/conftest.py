"""
Pytest fixtures for AEGIS AI test suite.
Provides test isolation for state machine, database, and configuration.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aegis.state.machine import StateMachine, SystemState, state_machine
from aegis.db.models.base import Base
from aegis.core.config import AegisConfig


@pytest.fixture
def fresh_state_machine():
    """Returns a new StateMachine instance — no global state pollution."""
    return StateMachine()


@pytest.fixture
def reset_global_state_machine():
    """
    Resets the global state_machine singleton to INIT for integration tests
    that must use the global. Restores original state after the test.
    """
    original_state = state_machine._current_state
    original_callbacks = {s: list(cbs) for s, cbs in state_machine._callbacks.items()}

    # Force reset to INIT
    state_machine._current_state = SystemState.INIT
    state_machine._callbacks = {s: [] for s in SystemState}

    yield state_machine

    # Restore
    state_machine._current_state = original_state
    state_machine._callbacks = original_callbacks


@pytest.fixture
def in_memory_engine():
    """Creates an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(in_memory_engine):
    """Provides an isolated database session that rolls back after each test."""
    Session = sessionmaker(bind=in_memory_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def test_config():
    """Returns a fresh AegisConfig instance with default values (no .env file)."""
    return AegisConfig(
        SYSTEM_MODE="PREDICTION_ONLY",
        DATABASE_URL="sqlite:///:memory:",
        LOG_LEVEL="DEBUG",
        _env_file=None,
    )

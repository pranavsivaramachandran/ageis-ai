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
from aegis.db import session as db_session_module
import os
import tempfile

@pytest.fixture(autouse=True)
def init_global_db(request):
    """Replaces global DB engine with in-memory SQLite and initializes schema before every test."""
    if request.node.get_closest_marker("integration"):
        yield
        return
        
    test_engine = create_engine("sqlite:///:memory:")
    db_session_module.engine = test_engine
    db_session_module.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db_session_module.init_db()
    yield
    test_engine.dispose()

@pytest.fixture
def integration_db():
    """Provides a persistent file-backed SQLite database for integration tests."""
    fd, temp_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    test_engine = create_engine(f"sqlite:///{temp_path}")
    db_session_module.engine = test_engine
    db_session_module.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db_session_module.init_db()
    
    yield temp_path
    
    test_engine.dispose()
    os.remove(temp_path)
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

from enum import Enum, auto
from typing import Callable, Dict, List
import threading
from aegis.core.logger import get_logger

logger = get_logger(__name__)

class SystemState(Enum):
    INIT = auto()
    STARTING = auto()
    RUNNING = auto()
    PAUSED = auto()
    SHUTTING_DOWN = auto()
    ERROR = auto()

class StateMachine:
    """
    Core state machine for the AEGIS AI platform.
    Manages transitions and ensures valid system states.
    """
    
    # Valid transitions from a given state to a list of allowed states
    VALID_TRANSITIONS: Dict[SystemState, List[SystemState]] = {
        SystemState.INIT: [SystemState.STARTING, SystemState.ERROR, SystemState.SHUTTING_DOWN],
        SystemState.STARTING: [SystemState.RUNNING, SystemState.ERROR, SystemState.SHUTTING_DOWN],
        SystemState.RUNNING: [SystemState.PAUSED, SystemState.SHUTTING_DOWN, SystemState.ERROR],
        SystemState.PAUSED: [SystemState.RUNNING, SystemState.SHUTTING_DOWN, SystemState.ERROR],
        SystemState.SHUTTING_DOWN: [SystemState.ERROR], # Final state or error during shutdown
        SystemState.ERROR: [SystemState.SHUTTING_DOWN, SystemState.INIT] # Allow recovery to INIT
    }

    def __init__(self):
        self._current_state = SystemState.INIT
        self._lock = threading.Lock()
        self._callbacks: Dict[SystemState, List[Callable]] = {state: [] for state in SystemState}

    @property
    def current_state(self) -> SystemState:
        with self._lock:
            return self._current_state

    def register_callback(self, state: SystemState, callback: Callable):
        """Registers a callback function to be executed when entering a specific state."""
        with self._lock:
            self._callbacks[state].append(callback)

    def transition_to(self, new_state: SystemState):
        """Attempts to transition the system to a new state."""
        with self._lock:
            if new_state not in self.VALID_TRANSITIONS.get(self._current_state, []):
                logger.error(
                    "Invalid state transition attempted",
                    current_state=self._current_state.name,
                    attempted_state=new_state.name
                )
                raise ValueError(f"Cannot transition from {self._current_state.name} to {new_state.name}")
            
            logger.info(
                "State transition",
                from_state=self._current_state.name,
                to_state=new_state.name
            )
            self._current_state = new_state
            
        # Execute callbacks outside the lock to prevent deadlocks
        for callback in self._callbacks[new_state]:
            try:
                callback()
            except Exception as e:
                logger.exception("Error in state transition callback", exc_info=e)

# Global state machine instance
state_machine = StateMachine()

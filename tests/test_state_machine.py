import pytest
from aegis.state.machine import StateMachine, SystemState

def test_initial_state():
    sm = StateMachine()
    assert sm.current_state == SystemState.INIT

def test_valid_transition():
    sm = StateMachine()
    sm.transition_to(SystemState.STARTING)
    assert sm.current_state == SystemState.STARTING

def test_invalid_transition_raises_error():
    sm = StateMachine()
    with pytest.raises(ValueError):
        sm.transition_to(SystemState.RUNNING)

def test_callback_execution():
    sm = StateMachine()
    called = False
    
    def callback():
        nonlocal called
        called = True
        
    sm.register_callback(SystemState.STARTING, callback)
    sm.transition_to(SystemState.STARTING)
    
    assert called is True

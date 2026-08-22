"""
Tests for aegis.core.lifecycle.LifecycleManager.
Covers startup, shutdown, hook ordering, error handling, and idempotency.
"""
import pytest
from aegis.core.lifecycle import LifecycleManager
from aegis.state.machine import SystemState


class TestLifecycleStart:
    """Tests for LifecycleManager.start()"""

    def test_start_transitions_to_running(self, fresh_state_machine, monkeypatch):
        """Happy path: start() should end in RUNNING state."""
        monkeypatch.setattr("aegis.core.lifecycle.state_machine", fresh_state_machine)
        lm = LifecycleManager()
        lm.start()
        assert fresh_state_machine.current_state == SystemState.RUNNING

    def test_start_executes_hooks_in_order(self, fresh_state_machine, monkeypatch):
        """Startup hooks must execute in registration order."""
        monkeypatch.setattr("aegis.core.lifecycle.state_machine", fresh_state_machine)
        lm = LifecycleManager()
        call_order = []

        lm.register_startup_hook(lambda: call_order.append("first"))
        lm.register_startup_hook(lambda: call_order.append("second"))
        lm.register_startup_hook(lambda: call_order.append("third"))

        lm.start()
        assert call_order == ["first", "second", "third"]

    def test_start_failure_transitions_to_error(self, fresh_state_machine, monkeypatch):
        """If a startup hook raises, state should transition to ERROR."""
        monkeypatch.setattr("aegis.core.lifecycle.state_machine", fresh_state_machine)
        lm = LifecycleManager()

        def failing_hook():
            raise RuntimeError("startup failure")

        lm.register_startup_hook(failing_hook)

        with pytest.raises(RuntimeError, match="startup failure"):
            lm.start()

        assert fresh_state_machine.current_state in (SystemState.ERROR, SystemState.SHUTTING_DOWN)

    def test_start_failure_raises_exception(self, fresh_state_machine, monkeypatch):
        """start() must propagate the exception instead of calling sys.exit()."""
        monkeypatch.setattr("aegis.core.lifecycle.state_machine", fresh_state_machine)
        lm = LifecycleManager()

        lm.register_startup_hook(lambda: (_ for _ in ()).throw(ValueError("boom")))

        with pytest.raises(ValueError, match="boom"):
            lm.start()


class TestLifecycleShutdown:
    """Tests for LifecycleManager.shutdown()"""

    def test_shutdown_transitions_to_shutting_down(self, fresh_state_machine, monkeypatch):
        """shutdown() should transition to SHUTTING_DOWN."""
        monkeypatch.setattr("aegis.core.lifecycle.state_machine", fresh_state_machine)
        lm = LifecycleManager()
        # Get to RUNNING first
        lm.start()
        assert fresh_state_machine.current_state == SystemState.RUNNING

        lm.shutdown()
        assert fresh_state_machine.current_state == SystemState.SHUTTING_DOWN

    def test_shutdown_idempotent(self, fresh_state_machine, monkeypatch):
        """Calling shutdown twice should only run hooks once."""
        monkeypatch.setattr("aegis.core.lifecycle.state_machine", fresh_state_machine)
        lm = LifecycleManager()
        call_count = []

        lm.register_shutdown_hook(lambda: call_count.append(1))
        lm.start()
        lm.shutdown()
        lm.shutdown()  # Second call should be a no-op

        assert len(call_count) == 1

    def test_shutdown_hook_error_does_not_prevent_others(self, fresh_state_machine, monkeypatch):
        """If one shutdown hook fails, the remaining hooks should still execute."""
        monkeypatch.setattr("aegis.core.lifecycle.state_machine", fresh_state_machine)
        lm = LifecycleManager()
        executed = []

        lm.register_shutdown_hook(lambda: executed.append("first"))
        lm.register_shutdown_hook(lambda: (_ for _ in ()).throw(RuntimeError("hook error")))
        lm.register_shutdown_hook(lambda: executed.append("third"))

        lm.start()
        lm.shutdown()

        assert "first" in executed
        assert "third" in executed

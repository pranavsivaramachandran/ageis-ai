import signal
from aegis.core.logger import get_logger
from aegis.state.machine import state_machine, SystemState

logger = get_logger(__name__)

class LifecycleManager:
    """
    Manages system startup, shutdown hooks, and signal handling.
    Does NOT call sys.exit() — process termination is the caller's responsibility.
    """
    def __init__(self):
        self._startup_hooks = []
        self._shutdown_hooks = []
        self._signal_shutdown_callback = None

    def register_startup_hook(self, hook):
        self._startup_hooks.append(hook)

    def register_shutdown_hook(self, hook):
        self._shutdown_hooks.append(hook)

    def set_signal_shutdown_callback(self, callback):
        """Register a callback to be invoked after signal-triggered shutdown completes.
        This allows main.py to handle sys.exit() outside the manager."""
        self._signal_shutdown_callback = callback

    def start(self):
        """Executes startup hooks and transitions to STARTING then RUNNING.
        On failure, transitions to ERROR, runs shutdown hooks, then re-raises."""
        try:
            state_machine.transition_to(SystemState.STARTING)
            for hook in self._startup_hooks:
                hook()
            state_machine.transition_to(SystemState.RUNNING)
            logger.info("System successfully started")
        except Exception as e:
            logger.exception("Error during system startup", exc_info=e)
            state_machine.transition_to(SystemState.ERROR)
            self.shutdown()
            raise

    def shutdown(self, signum=None, frame=None):
        """Executes shutdown hooks and transitions to SHUTTING_DOWN."""
        if state_machine.current_state == SystemState.SHUTTING_DOWN:
            return # Already shutting down

        logger.info("Initiating system shutdown", signal=signum)
        try:
            state_machine.transition_to(SystemState.SHUTTING_DOWN)
            for hook in self._shutdown_hooks:
                try:
                    hook()
                except Exception as e:
                    logger.exception("Error during shutdown hook", exc_info=e)
            logger.info("System successfully shut down")
        except Exception as e:
            logger.exception("Error during system shutdown", exc_info=e)
        finally:
            if signum is not None and self._signal_shutdown_callback:
                self._signal_shutdown_callback()

    def setup_signal_handlers(self):
        """Registers handlers for SIGINT and SIGTERM."""
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

# Global lifecycle manager instance
lifecycle_manager = LifecycleManager()

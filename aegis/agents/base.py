from abc import ABC, abstractmethod
from aegis.core.logger import get_logger, set_trace_id, get_trace_id
import uuid

class BaseAgent(ABC):
    """
    Base contract for all AEGIS AI agents.
    Enforces trace_id propagation and health reporting.
    """
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"agent.{self.name}")

    def execute(self, *args, **kwargs):
        """
        Main execution entry point. Ensures trace_id is set.
        Saves and restores the parent trace_id so child agent calls
        don't destroy the parent's trace context.
        """
        previous_trace_id = get_trace_id()
        trace_id = kwargs.pop("trace_id", None) or str(uuid.uuid4())
        set_trace_id(trace_id)
        
        self.logger.info("Starting agent execution", action="execute_start")
        try:
            result = self._run(*args, **kwargs)
            self.logger.info("Finished agent execution", action="execute_end")
            return result
        except Exception as e:
            self.logger.exception("Agent execution failed", action="execute_error", exc_info=e)
            raise
        finally:
            set_trace_id(previous_trace_id)

    @abstractmethod
    def _run(self, *args, **kwargs):
        """
        Internal implementation of the agent's core logic.
        Must be implemented by subclasses.
        """
        pass

    def check_health(self) -> bool:
        """
        Reports the health of the agent.
        Can be overridden by subclasses for specific checks.
        """
        return True


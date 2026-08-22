from aegis.agents.base import BaseAgent
from aegis.core.config import config

class ExecutionAgent(BaseAgent):
    """
    Execution agent responsible for interacting with brokers.
    Currently DORMANT and explicitly disabled in PREDICTION_ONLY mode.
    """
    def __init__(self, name: str = "execution"):
        super().__init__(name)
        
    def _run(self, *args, **kwargs):
        """
        Fails fast if system is in PREDICTION_ONLY mode.
        """
        if config.SYSTEM_MODE == "PREDICTION_ONLY":
            error_msg = "CRITICAL: ExecutionAgent invoked while system is in PREDICTION_ONLY mode."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        self.logger.info("ExecutionAgent would place order here (stub)")
        return {"status": "order_placed"}

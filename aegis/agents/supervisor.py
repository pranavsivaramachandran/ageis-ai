from typing import List
from aegis.agents.base import BaseAgent
from aegis.core.health import health_monitor

class SupervisorAgent(BaseAgent):
    """
    Supervisor agent responsible for monitoring and managing other agents.
    """
    def __init__(self, name: str = "supervisor"):
        super().__init__(name)
        self.child_agents: List[BaseAgent] = []

    def register_agent(self, agent: BaseAgent):
        """Registers a child agent to be monitored."""
        self.child_agents.append(agent)
        health_monitor.register_check(f"agent_{agent.name}", agent.check_health)
        self.logger.info("Registered child agent", child_name=agent.name)

    def _run(self, *args, **kwargs):
        """
        Supervisor main loop/logic. For now, just checks health.
        """
        self.logger.info("Supervisor running health checks on child agents")
        all_healthy = True
        for agent in self.child_agents:
            if not agent.check_health():
                self.logger.error("Child agent unhealthy", child_name=agent.name)
                all_healthy = False
                # In the future, initiate restart procedures here
        return all_healthy

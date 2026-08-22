from typing import Dict, Any, Callable
from aegis.core.logger import get_logger

logger = get_logger(__name__)

class HealthMonitor:
    """
    Manages and reports the health status of system components.
    """
    def __init__(self):
        self._checks: Dict[str, Callable[[], bool]] = {}

    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Registers a health check function for a component."""
        self._checks[name] = check_func
        logger.debug("Registered health check", component=name)

    def check_health(self) -> Dict[str, Any]:
        """Runs all registered health checks."""
        status = {"overall": "healthy", "components": {}}
        for name, check_func in self._checks.items():
            try:
                is_healthy = check_func()
                status["components"][name] = "healthy" if is_healthy else "unhealthy"
                if not is_healthy:
                    status["overall"] = "unhealthy"
            except Exception as e:
                logger.exception("Health check failed", component=name, exc_info=e)
                status["components"][name] = "error"
                status["overall"] = "unhealthy"
                
        return status

# Global health monitor instance
health_monitor = HealthMonitor()

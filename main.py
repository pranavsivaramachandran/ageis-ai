import sys
import time
from aegis.core.config import config
from aegis.core.logger import get_logger, set_trace_id
from aegis.core.lifecycle import lifecycle_manager
from aegis.core.health import health_monitor
from aegis.state.machine import state_machine, SystemState
from aegis.db.session import init_db, engine

# Agents
from aegis.agents.supervisor import SupervisorAgent
from aegis.agents.orchestrator import OrchestratorAgent

# Sprint 2: Infrastructure interfaces (validates import chain, no live providers)
from aegis.interfaces.market_data import MarketDataProvider  # noqa: F401
from aegis.interfaces.broker import BrokerProvider  # noqa: F401
from aegis.events.contracts import BaseEvent  # noqa: F401

logger = get_logger(__name__)

def main():
    set_trace_id("system_startup")
    logger.info("AEGIS AI Platform Starting", mode=config.SYSTEM_MODE)
    logger.info("Infrastructure interfaces loaded",
                market_data_interface="MarketDataProvider",
                broker_interface="BrokerProvider",
                event_contracts="BaseEvent")

    # 1. Setup Signal Handlers
    lifecycle_manager.setup_signal_handlers()
    lifecycle_manager.set_signal_shutdown_callback(lambda: sys.exit(0))

    # 2. Register Startup Hooks
    lifecycle_manager.register_startup_hook(init_db)
    
    # Initialize agents
    supervisor = SupervisorAgent()
    orchestrator = OrchestratorAgent()
    supervisor.register_agent(orchestrator)
    
    def start_supervisor():
        logger.info("Supervisor starting")
        # In a real async/threaded environment, this would start the background loops.
        # For this foundation test, we just run a health check.
        supervisor.execute()

    lifecycle_manager.register_startup_hook(start_supervisor)

    # 3. Register Shutdown Hooks
    def stop_supervisor():
        logger.info("Supervisor stopping")
        # Stop background loops

    lifecycle_manager.register_shutdown_hook(stop_supervisor)
    lifecycle_manager.register_shutdown_hook(engine.dispose)

    # 4. Start the system
    try:
        lifecycle_manager.start()
    except Exception:
        logger.error("System failed to start. Exiting.")
        sys.exit(1)

    # 5. Main Loop (Simulated)
    try:
        if "--dry-run" in sys.argv:
            logger.info("Dry run complete. Shutting down.")
            lifecycle_manager.shutdown()
        else:
            logger.info("System is running. Press Ctrl+C to stop.")
            while state_machine.current_state == SystemState.RUNNING:
                # In a real system, the main thread might just wait or handle top-level events
                time.sleep(1)
                
                # Perform periodic health checks at the top level if needed
                status = health_monitor.check_health()
                if status["overall"] != "healthy":
                    logger.warning("System health degraded", status=status)
    except KeyboardInterrupt:
        # Handled by signal handler, but just in case
        pass # lifecycle_manager.shutdown() should be called by signal handler

if __name__ == "__main__":
    main()


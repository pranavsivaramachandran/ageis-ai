import structlog
import logging
import uuid
import contextvars
from aegis.core.config import config

# Context variable for trace_id
trace_id_var = contextvars.ContextVar("trace_id", default="")

def set_trace_id(trace_id: str = None) -> str:
    """Sets the trace_id in the context and returns it."""
    if not trace_id:
        trace_id = str(uuid.uuid4())
    trace_id_var.set(trace_id)
    return trace_id

def get_trace_id() -> str:
    """Retrieves the current trace_id from the context."""
    return trace_id_var.get()

def add_trace_id(logger, method_name, event_dict):
    """Structlog processor to add trace_id to all log events."""
    trace_id = get_trace_id()
    if trace_id:
        event_dict["trace_id"] = trace_id
    return event_dict

def setup_logging():
    """Configures structured logging for the application."""
    
    # Configure standard logging to intercept and pass to structlog
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            add_trace_id,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer() if config.LOG_LEVEL.upper() == "DEBUG" else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
def get_logger(name: str):
    """Returns a bound structlog logger."""
    return structlog.get_logger(name)

# Initial logging setup based on config
setup_logging()

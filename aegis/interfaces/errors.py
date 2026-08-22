"""
Infrastructure error hierarchy for AEGIS AI.

All infrastructure-level failures inherit from InfrastructureError,
allowing callers to catch broadly or specifically.
"""


class InfrastructureError(Exception):
    """Base exception for all infrastructure-level failures."""
    pass


class ProviderUnavailableError(InfrastructureError):
    """The data or broker provider is down or unreachable."""
    pass


class ConnectionFailureError(InfrastructureError):
    """A network-level connection failure occurred."""
    pass


class AuthenticationError(InfrastructureError):
    """API key, credentials, or authentication token is invalid or expired."""
    pass


class MalformedDataError(InfrastructureError):
    """The response from a provider could not be parsed or validated."""
    pass


class StaleDataError(InfrastructureError):
    """The received data has a timestamp older than the acceptable threshold."""
    pass


class InfrastructureTimeoutError(InfrastructureError):
    """An operation exceeded its allowed time limit.
    Named InfrastructureTimeoutError to avoid shadowing Python's built-in TimeoutError."""
    pass

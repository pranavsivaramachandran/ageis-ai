"""
Tests for aegis.core.health.HealthMonitor.
Covers registration, healthy/unhealthy checks, and exception handling.
"""
from aegis.core.health import HealthMonitor


class TestHealthMonitor:

    def test_no_checks_returns_healthy(self):
        """An empty monitor with no registered checks should report overall healthy."""
        monitor = HealthMonitor()
        status = monitor.check_health()
        assert status["overall"] == "healthy"
        assert status["components"] == {}

    def test_register_and_check_healthy(self):
        """A single healthy check should result in overall healthy."""
        monitor = HealthMonitor()
        monitor.register_check("db", lambda: True)

        status = monitor.check_health()
        assert status["overall"] == "healthy"
        assert status["components"]["db"] == "healthy"

    def test_unhealthy_component(self):
        """One unhealthy check should mark overall as unhealthy."""
        monitor = HealthMonitor()
        monitor.register_check("db", lambda: True)
        monitor.register_check("broker", lambda: False)

        status = monitor.check_health()
        assert status["overall"] == "unhealthy"
        assert status["components"]["db"] == "healthy"
        assert status["components"]["broker"] == "unhealthy"

    def test_check_exception_handled(self):
        """A check that raises an exception should be marked as error, not crash."""
        monitor = HealthMonitor()

        def exploding_check():
            raise ConnectionError("connection refused")

        monitor.register_check("external_api", exploding_check)

        status = monitor.check_health()
        assert status["overall"] == "unhealthy"
        assert status["components"]["external_api"] == "error"

    def test_multiple_healthy_checks(self):
        """All healthy checks should report overall healthy."""
        monitor = HealthMonitor()
        monitor.register_check("db", lambda: True)
        monitor.register_check("cache", lambda: True)
        monitor.register_check("broker", lambda: True)

        status = monitor.check_health()
        assert status["overall"] == "healthy"
        assert all(v == "healthy" for v in status["components"].values())

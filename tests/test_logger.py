"""
Tests for aegis.core.logger — trace_id management and structlog processor.
"""
import uuid
from aegis.core.logger import set_trace_id, get_trace_id, add_trace_id, trace_id_var


class TestTraceId:

    def test_set_and_get_trace_id(self):
        """set_trace_id should store and get_trace_id should retrieve the same value."""
        trace = set_trace_id("test-trace-42")
        assert trace == "test-trace-42"
        assert get_trace_id() == "test-trace-42"

    def test_auto_generated_trace_id(self):
        """set_trace_id() with no argument should generate a valid UUID."""
        trace = set_trace_id()
        assert trace != ""
        # Verify it's a valid UUID
        parsed = uuid.UUID(trace)
        assert str(parsed) == trace

    def test_get_trace_id_default(self):
        """get_trace_id should return empty string when no trace is set."""
        token = trace_id_var.set("")
        try:
            assert get_trace_id() == ""
        finally:
            trace_id_var.reset(token)

    def test_overwrite_trace_id(self):
        """Setting trace_id twice should overwrite the previous value."""
        set_trace_id("first")
        set_trace_id("second")
        assert get_trace_id() == "second"


class TestAddTraceIdProcessor:

    def test_processor_adds_trace_id_to_event_dict(self):
        """The add_trace_id processor should inject trace_id into the event dict."""
        set_trace_id("proc-test-123")
        event_dict = {"event": "something happened"}
        result = add_trace_id(None, None, event_dict)
        assert result["trace_id"] == "proc-test-123"

    def test_processor_skips_when_no_trace_id(self):
        """When trace_id is empty, the processor should not add the key."""
        token = trace_id_var.set("")
        try:
            event_dict = {"event": "no trace"}
            result = add_trace_id(None, None, event_dict)
            assert "trace_id" not in result
        finally:
            trace_id_var.reset(token)

    def test_processor_preserves_existing_keys(self):
        """The processor should not remove existing keys from the event dict."""
        set_trace_id("preserve-test")
        event_dict = {"event": "test", "extra_key": "extra_value"}
        result = add_trace_id(None, None, event_dict)
        assert result["extra_key"] == "extra_value"
        assert result["trace_id"] == "preserve-test"

"""
Tests for AEGIS AI agents — execution safety, supervisor health, orchestrator,
and base agent trace_id scoping.
"""
import pytest
from aegis.agents.execution import ExecutionAgent
from aegis.agents.supervisor import SupervisorAgent
from aegis.agents.orchestrator import OrchestratorAgent
from aegis.core.logger import set_trace_id, get_trace_id


class TestExecutionAgent:

    def test_execution_agent_fails_in_prediction_only(self, monkeypatch):
        import aegis.core.config
        # Explicitly set to PREDICTION_ONLY
        monkeypatch.setattr(aegis.core.config.config, "SYSTEM_MODE", "PREDICTION_ONLY")
        
        agent = ExecutionAgent()
        with pytest.raises(RuntimeError, match="ExecutionAgent invoked while system is in PREDICTION_ONLY mode"):
            agent.execute()


class TestSupervisorAgent:

    def test_supervisor_health_check(self):
        supervisor = SupervisorAgent()
        orchestrator = OrchestratorAgent()
        
        supervisor.register_agent(orchestrator)
        assert len(supervisor.child_agents) == 1
        
        # Run should return True if all child agents are healthy
        assert supervisor.execute() is True

    def test_execution_agent_not_registered_in_supervisor(self):
        """Structural safety: ExecutionAgent should never appear in the supervisor's child list."""
        supervisor = SupervisorAgent()
        orchestrator = OrchestratorAgent()
        supervisor.register_agent(orchestrator)

        for agent in supervisor.child_agents:
            assert not isinstance(agent, ExecutionAgent), \
                "ExecutionAgent must not be registered with the Supervisor in PREDICTION_ONLY mode"


class TestOrchestratorAgent:

    def test_orchestrator_execute_returns_stub(self):
        """Orchestrator should execute without error and return expected stub dict."""
        orchestrator = OrchestratorAgent()
        result = orchestrator.execute()
        assert isinstance(result, dict)
        assert result["status"] == "workflow_executed_stub"

    def test_orchestrator_health_check(self):
        """Orchestrator should report healthy by default."""
        orchestrator = OrchestratorAgent()
        assert orchestrator.check_health() is True


class TestBaseAgentTraceId:

    def test_execute_sets_trace_id(self):
        """execute() should set a trace_id during execution and restore the previous after."""
        orchestrator = OrchestratorAgent()
        set_trace_id("before-execute")
        orchestrator.execute()
        # After execute completes, trace_id should be restored to pre-execution value
        assert get_trace_id() == "before-execute"

    def test_execute_with_custom_trace_id(self):
        """execute(trace_id=...) should use the provided trace_id."""
        orchestrator = OrchestratorAgent()
        set_trace_id("parent-trace")
        orchestrator.execute(trace_id="custom-123")
        # After execution, parent trace should be restored
        assert get_trace_id() == "parent-trace"

    def test_execute_restores_parent_trace_id(self):
        """After child agent execute(), the parent's trace_id must be restored."""
        set_trace_id("parent-abc")

        child = OrchestratorAgent()
        child.execute()

        assert get_trace_id() == "parent-abc"

    def test_execute_restores_trace_on_error(self):
        """Even if _run() raises, the parent trace_id must be restored."""
        set_trace_id("parent-before-error")

        agent = ExecutionAgent()
        with pytest.raises(RuntimeError):
            agent.execute()

        assert get_trace_id() == "parent-before-error"


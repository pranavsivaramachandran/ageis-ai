import pytest
from unittest.mock import MagicMock
from decimal import Decimal

from aegis.evaluation.models import RobustnessScenario
from aegis.evaluation.robustness import RobustnessEvaluator
from aegis.backtest.models import SimulationConfig


def test_robustness_evaluator_scenarios():
    base_evaluator = MagicMock()
    # Mock original simulation config
    original_config = SimulationConfig(
        initial_capital=Decimal("10000"),
        commission_per_trade=Decimal("1"),
        slippage_percent=Decimal("0.001"),
        position_size_percent=Decimal("0.1"),
        holding_period_candles=5,
        stop_loss_atr_multiplier=Decimal("1.0")
    )
    base_evaluator.simulation_config = original_config
    
    # Mock the return value of evaluate
    mock_report = MagicMock()
    base_evaluator.evaluate.return_value = mock_report
    
    robustness = RobustnessEvaluator(base_evaluator=base_evaluator)
    
    scenarios = [
        RobustnessScenario("high_cost", commission_per_trade=Decimal("5")),
        RobustnessScenario("high_slippage", slippage_percent=Decimal("0.005"))
    ]
    
    # Need to verify that the config passed to the internal evaluator had the right parameters
    call_configs = []
    
    def side_effect(history, config, exp_id, simulation_config=None):
        call_configs.append(simulation_config if simulation_config else base_evaluator.simulation_config)
        return mock_report
        
    base_evaluator.evaluate.side_effect = side_effect
    
    report = robustness.evaluate(history=[], config=MagicMock(), experiment_id="test", scenarios=scenarios)
    
    assert len(report.scenarios) == 2
    assert report.scenarios[0].scenario_name == "high_cost"
    assert report.scenarios[1].scenario_name == "high_slippage"
    
    # Check that configs were modified correctly
    assert call_configs[0].commission_per_trade == Decimal("5")
    assert call_configs[0].slippage_percent == Decimal("0.001") # untouched
    
    assert call_configs[1].commission_per_trade == Decimal("1") # original
    assert call_configs[1].slippage_percent == Decimal("0.005")
    
    # Check that it restored correctly
    assert base_evaluator.simulation_config.commission_per_trade == Decimal("1")
    assert base_evaluator.simulation_config.slippage_percent == Decimal("0.001")

"""
Robustness Evaluation Engine.
"""
from typing import Sequence

from aegis.interfaces.market_data import OHLC
from aegis.evaluation.models import (
    WalkForwardConfig,
    RobustnessScenario,
    RobustnessResult,
    RobustnessReport
)
from aegis.evaluation.walk_forward import WalkForwardEvaluator


class RobustnessEvaluator:
    """Runs a series of walk-forward evaluations under different scenarios."""
    
    def __init__(self, base_evaluator: WalkForwardEvaluator):
        self.base_evaluator = base_evaluator

    def evaluate(
        self,
        history: list[OHLC],
        config: WalkForwardConfig,
        experiment_id: str,
        scenarios: Sequence[RobustnessScenario]
    ) -> RobustnessReport:
        """
        Evaluate multiple robustness scenarios.
        """
        results = []
        original_config = self.base_evaluator.simulation_config
        
        for i, scenario in enumerate(scenarios):
            # Apply scenario to a copy of the simulation config
            # (Pydantic models have `.model_copy(update={...})`)
            update_kwargs = {}
            if scenario.commission_per_trade is not None:
                update_kwargs["commission_per_trade"] = scenario.commission_per_trade
            if scenario.slippage_percent is not None:
                update_kwargs["slippage_percent"] = scenario.slippage_percent
                
            scenario_sim_config = original_config.model_copy(update=update_kwargs)
            
            # Run
            scenario_experiment_id = f"{experiment_id}_{scenario.scenario_name}"
            report = self.base_evaluator.evaluate(
                history, 
                config, 
                scenario_experiment_id, 
                simulation_config=scenario_sim_config
            )
            
            results.append(RobustnessResult(
                scenario_name=scenario.scenario_name,
                report=report
            ))
        
        return RobustnessReport(
            base_experiment_id=experiment_id,
            scenarios=results
        )

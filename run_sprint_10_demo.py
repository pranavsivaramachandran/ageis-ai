import sys
import logging
from decimal import Decimal
from datetime import datetime, timezone

from aegis.interfaces.market_data import Timeframe
from aegis.features.builder import FeatureBuilder, FeatureBuilderConfig
from aegis.prediction.model_interface import FeatureSchema
from aegis.ml.labels import TargetConfig, TargetGenerator
from aegis.ml.training import TrainerConfig
from aegis.risk.engine import RiskManagementEngine
from aegis.backtest.models import SimulationConfig
from aegis.evaluation.models import WalkForwardConfig, WalkForwardStrategy
from aegis.evaluation.walk_forward import WalkForwardEvaluator
from aegis.evaluation.robustness import RobustnessEvaluator, RobustnessScenario

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def generate_dummy_history():
    from aegis.interfaces.market_data import OHLC
    from datetime import timedelta
    
    history = []
    base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    
    val = 100.0
    for i in range(500):
        # Create a mock sine wave-ish pattern
        if i % 10 < 5:
            val *= 1.01
        else:
            val *= 0.99
            
        history.append(OHLC(
            symbol="BTC/USD",
            timestamp=base_time + timedelta(hours=i),
            timeframe=Timeframe.H1,
            open=Decimal(str(val)),
            high=Decimal(str(val * 1.02)),
            low=Decimal(str(val * 0.98)),
            close=Decimal(str(val)),
            volume=Decimal("1000")
        ))
    return history

def run_demo():
    print("=" * 60)
    print("AEGIS AI SPRINT 10 CORRECTIONS DEMO")
    print("=" * 60)
    
    history = generate_dummy_history()
    print(f"Generated {len(history)} candles of dummy data.")
    
    # 1. Feature Builder
    fb_config = FeatureBuilderConfig(
        sma_period=10, ema_period=10, rsi_period=14,
        macd_fast=12, macd_slow=26, macd_signal=9,
        atr_period=14, bollinger_period=20, momentum_period=10, volatility_period=10
    )
    feature_builder = FeatureBuilder(fb_config)
    
    # 2. Target Generator
    target_config = TargetConfig(target_horizon_candles=5, threshold=0.005)
    target_generator = TargetGenerator(target_config)
    
    # 3. Schema
    schema = FeatureSchema(
        schema_version=1,
        required_features=["sma_value", "ema_value", "rsi_value", "momentum_value"]
    )
    
    # 4. Trainer Config
    trainer_config = TrainerConfig(random_state=42)
    
    # 5. Risk Engine
    risk_engine = RiskManagementEngine()
    
    # 6. Simulation Config
    sim_config = SimulationConfig(
        initial_capital=Decimal("10000.0"),
        commission_per_trade=Decimal("0.001"),
        slippage_percent=Decimal("0.001")
    )
    
    # 7. Evaluators
    wf_evaluator = WalkForwardEvaluator(
        feature_builder=feature_builder,
        target_generator=target_generator,
        schema=schema,
        trainer_config=trainer_config,
        risk_engine=risk_engine,
        simulation_config=sim_config
    )
    
    rb_evaluator = RobustnessEvaluator(base_evaluator=wf_evaluator)
    
    wf_config = WalkForwardConfig(
        strategy=WalkForwardStrategy.ROLLING,
        train_size=100,
        validation_size=20,
        test_size=50,
        step_size=50,
        minimum_train_samples=50
    )
    
    # A. Test walk forward isolation and threshold validation tuning
    print("\n[A] Running Walk-Forward Evaluation...")
    try:
        wf_report = wf_evaluator.evaluate(history, wf_config, "demo_wf_1")
        print(f"Total Windows: {wf_report.total_windows}")
        print(f"Successful Windows: {wf_report.successful_windows}")
        print(f"Failed Windows: {wf_report.failed_windows}")
        print(f"Mean ML Accuracy: {wf_report.mean_ml_accuracy:.4f}")
        print(f"Mean ML F1: {wf_report.mean_ml_f1:.4f}")
        print(f"Mean ML PnL: {wf_report.mean_ml_pnl}")
        print(f"Mean Baseline PnL: {wf_report.mean_baseline_pnl}")
    except Exception as e:
        print(f"Walk-Forward Error: {e}")
        
    # B. Test robustness scenarios without leaking config state
    print("\n[B] Running Robustness Evaluation...")
    scenarios = [
        RobustnessScenario(
            scenario_name="High_Commission",
            commission_per_trade=Decimal("0.005")
        ),
        RobustnessScenario(
            scenario_name="High_Slippage",
            slippage_percent=Decimal("0.005")
        )
    ]
    
    try:
        rb_report = rb_evaluator.evaluate(history, wf_config, "demo_rb_1", scenarios)
        print(f"Base Config Commission: {wf_evaluator.simulation_config.commission_per_trade}")
        print(f"Base Config Slippage: {wf_evaluator.simulation_config.slippage_percent}")
        for res in rb_report.scenarios:
            print(f"- Scenario: {res.scenario_name} | Total Windows: {res.report.total_windows} | Mean ML PnL: {res.report.mean_ml_pnl}")
    except Exception as e:
        print(f"Robustness Error: {e}")

if __name__ == "__main__":
    run_demo()

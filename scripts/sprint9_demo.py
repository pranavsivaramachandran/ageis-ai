"""
Sprint 9 Demonstration Script.
End-to-End offline deterministic ML pipeline.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
import structlog

from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.features.builder import FeatureBuilder, FeatureBuilderConfig
from aegis.prediction.model_interface import FeatureSchema
from aegis.ml.labels import TargetConfig, TargetGenerator
from aegis.ml.dataset import MLDatasetBuilder
from aegis.ml.training import TrainerConfig, Trainer
from aegis.experiments.dataset import ChronologicalSplitter
from aegis.backtest.engine import BacktestEngine
from aegis.backtest.models import SimulationConfig
from aegis.prediction.engine import BaselinePredictor
from aegis.risk.engine import RiskManagementEngine

logger = structlog.get_logger(__name__)

def build_synthetic_history() -> list[OHLC]:
    """Generate a synthetic historical dataset with some predictable patterns."""
    candles = []
    base_time = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
    val = 1000.0
    
    # 500 hours of synthetic data
    for i in range(500):
        # simple mean reversion / momentum pattern
        if i % 10 < 5:
            val *= 1.002  # uptrend
        else:
            val *= 0.998  # downtrend
            
        candles.append(OHLC(
            symbol="BTC/USD",
            timestamp=base_time + timedelta(hours=i),
            timeframe=Timeframe.H1,
            open=Decimal(str(round(val * 0.999, 2))),
            high=Decimal(str(round(val * 1.01, 2))),
            low=Decimal(str(round(val * 0.99, 2))),
            close=Decimal(str(round(val, 2))),
            volume=Decimal("1000")
        ))
    return candles


def run_demo():
    logger.info("Starting AEGIS AI Sprint 9 Demonstration")
    
    # 1. Historical OHLC
    logger.info("Generating synthetic OHLC history...")
    history = build_synthetic_history()
    
    # 2. Split
    logger.info("Performing chronological split (60/20/20)...")
    split = ChronologicalSplitter.split(history, 0.6, 0.2, 0.2)
    
    # 3. Setup ML Pipeline
    feat_config = FeatureBuilderConfig()
    fb = FeatureBuilder(feat_config)
    target_config = TargetConfig(target_horizon_candles=3, threshold=0.003)
    tg = TargetGenerator(target_config)
    schema = FeatureSchema(
        schema_version=1, 
        required_features=[
            "sma_value", "ema_value", "rsi_value", "macd_line", 
            "macd_signal", "macd_histogram", "atr_value", "momentum_value"
        ]
    )
    
    builder = MLDatasetBuilder(fb, tg, schema)
    
    # 4. Build Datasets
    logger.info("Building Training Dataset...")
    train_dataset = builder.build(split.train)
    logger.info(f"Training samples: {len(train_dataset)}")
    
    logger.info("Building Validation Dataset...")
    val_dataset = builder.build(split.validation)
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    logger.info("Building Test Dataset...")
    test_dataset = builder.build(split.test)
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # 5. Train
    logger.info("Training ML model...")
    trainer = Trainer(TrainerConfig(random_state=42), schema)
    ml_model = trainer.train(train_dataset, model_id="sprint9_ml_model", version=1)
    
    # 6. Evaluate
    from aegis.ml.evaluation import MLEvaluator
    val_metrics = MLEvaluator.evaluate(ml_model, val_dataset)
    test_metrics = MLEvaluator.evaluate(ml_model, test_dataset)
    
    logger.info("--- ML Evaluation Metrics ---")
    logger.info("Validation:", accuracy=val_metrics.accuracy, f1=val_metrics.f1_macro)
    logger.info("Test:", accuracy=test_metrics.accuracy, f1=test_metrics.f1_macro)
    
    # 7. Backtest comparison
    logger.info("--- Running Backtest Engine ---")
    sim_config = SimulationConfig(
        initial_capital=Decimal("100000.0"),
        commission_per_trade=Decimal("2.5"),
        slippage_percent=Decimal("0.0005")
    )
    
    # Setup engines
    pred_engine_base = BaselinePredictor()
    risk_engine = RiskManagementEngine()
    
    # Initialize Engine for Baseline
    bt_baseline = BacktestEngine(fb, pred_engine_base, risk_engine, sim_config)
    
    logger.info("Running Baseline Backtest on Test Split...")
    baseline_report = bt_baseline.run(split.test)
    
    # Initialize Engine for ML Model
    bt_ml = BacktestEngine(fb, ml_model, risk_engine, sim_config)
    logger.info("Running ML Backtest on Test Split...")
    ml_report = bt_ml.run(split.test)
    
    logger.info("===========================================")
    logger.info("       SPRINT 9 FINAL COMPARISON REPORT      ")
    logger.info("===========================================")
    logger.info(f"{'Metric':<20} | {'Baseline':<15} | {'ML Model':<15}")
    logger.info("-" * 55)
    
    # Define fields to compare
    compare = [
        ("Total Trades", str(baseline_report.total_trades), str(ml_report.total_trades)),
        ("Win Rate", f"{float(baseline_report.win_rate)*100:.1f}%", f"{float(ml_report.win_rate)*100:.1f}%"),
        ("PnL", f"${float(baseline_report.total_pnl):.2f}", f"${float(ml_report.total_pnl):.2f}"),
        ("Return", f"{float(baseline_report.total_return)*100:.2f}%", f"{float(ml_report.total_return)*100:.2f}%"),
        ("Profit Factor", f"{float(baseline_report.profit_factor or 0):.2f}", f"{float(ml_report.profit_factor or 0):.2f}"),
        ("Max Drawdown", f"{float(baseline_report.max_drawdown)*100:.2f}%", f"{float(ml_report.max_drawdown)*100:.2f}%")
    ]
    
    for metric, b_val, m_val in compare:
        logger.info(f"{metric:<20} | {b_val:<15} | {m_val:<15}")
        
    logger.info("===========================================")
    logger.info("Demonstration Complete.")


if __name__ == "__main__":
    run_demo()

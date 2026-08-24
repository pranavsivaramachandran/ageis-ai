"""
Walk-Forward Evaluation Engine.
"""
from typing import Sequence, Optional
import logging

from aegis.interfaces.market_data import OHLC
from aegis.evaluation.models import (
    WalkForwardConfig,
    WalkForwardStrategy,
    WindowSplit,
    WindowResult,
    WindowMetrics,
    WalkForwardReport
)
from aegis.evaluation.metrics import (
    aggregate_ml_metrics,
    aggregate_financial_metrics,
    calculate_win_percentage
)
from aegis.ml.dataset import MLDatasetBuilder
from aegis.ml.training import Trainer, ExpectedWindowFailure
from aegis.ml.evaluation import MLEvaluator
from aegis.backtest.engine import BacktestEngine
from aegis.backtest.models import SimulationConfig
from aegis.prediction.engine import PredictionEngine
from aegis.prediction.engine import PredictionEngine, BaselinePredictor


class WindowGenerator:
    """Generates WindowSplits based on history and WalkForwardConfig."""
    
    @staticmethod
    def generate(history: list[OHLC], config: WalkForwardConfig) -> list[WindowSplit]:
        total_len = len(history)
        windows = []
        
        # Validation rules
        window_size_required = config.train_size + config.validation_size + config.test_size
        if total_len < window_size_required:
            return []
            
        current_train_start = 0
        current_train_end = config.train_size
        window_id = 1
        
        while True:
            current_validation_start = current_train_end
            current_validation_end = current_validation_start + config.validation_size
            
            current_test_start = current_validation_end
            current_test_end = current_test_start + config.test_size
            
            if current_test_end > total_len:
                break
                
            windows.append(WindowSplit(
                window_id=window_id,
                train_start=current_train_start,
                train_end=current_train_end,
                validation_start=current_validation_start,
                validation_end=current_validation_end,
                test_start=current_test_start,
                test_end=current_test_end
            ))
            
            # Step forward
            if config.strategy == WalkForwardStrategy.ROLLING:
                current_train_start += config.step_size
            current_train_end += config.step_size
            window_id += 1
            
        return windows


class WalkForwardEvaluator:
    """Orchestrates the walk-forward evaluation process."""
    
    def __init__(
        self,
        feature_builder,
        target_generator,
        schema,
        trainer_config,
        risk_engine,
        simulation_config: SimulationConfig
    ):
        self.feature_builder = feature_builder
        self.target_generator = target_generator
        self.schema = schema
        self.trainer_config = trainer_config
        self.risk_engine = risk_engine
        self.simulation_config = simulation_config
        
        self.dataset_builder = MLDatasetBuilder(
            feature_builder=self.feature_builder,
            target_generator=self.target_generator,
            schema=self.schema
        )
        self.trainer = Trainer(config=self.trainer_config, schema=self.schema)

    def evaluate(self, history: list[OHLC], config: WalkForwardConfig, experiment_id: str, simulation_config: Optional[SimulationConfig] = None) -> WalkForwardReport:
        splits = WindowGenerator.generate(history, config)
        
        results = []
        failed_count = 0
        
        active_sim_config = simulation_config if simulation_config is not None else self.simulation_config
        
        for split in splits:
            try:
                result = self._evaluate_window(history, split, config, active_sim_config)
                results.append(result)
            except ExpectedWindowFailure as e:
                logging.error(f"Window {split.window_id} failed: {e}")
                failed_count += 1
                results.append(WindowResult(
                    window_id=split.window_id,
                    split=split,
                    train_samples=0, validation_samples=0, test_samples=0,
                    metrics=WindowMetrics(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0), # type: ignore
                    ml_model_id="",
                    error=str(e)
                ))

        successful_results = [r for r in results if not r.error]
        
        ml_agg = aggregate_ml_metrics(successful_results)
        fin_agg = aggregate_financial_metrics(successful_results)
        win_pct = calculate_win_percentage(successful_results)

        return WalkForwardReport(
            experiment_id=experiment_id,
            windows=results,
            total_windows=len(splits),
            successful_windows=len(successful_results),
            failed_windows=failed_count,
            mean_ml_accuracy=ml_agg["mean_ml_accuracy"],
            mean_ml_f1=ml_agg["mean_ml_f1"],
            mean_ml_pnl=fin_agg["mean_ml_pnl"],
            mean_baseline_pnl=fin_agg["mean_baseline_pnl"],
            ml_win_percentage=win_pct,
            worst_ml_pnl=fin_agg["worst_ml_pnl"],
            best_ml_pnl=fin_agg["best_ml_pnl"],
            worst_drawdown=fin_agg["worst_drawdown"]
        )
        
    def _evaluate_window(self, history: list[OHLC], split: WindowSplit, config: WalkForwardConfig, sim_config: SimulationConfig) -> WindowResult:
        # 1. Partition Data
        # We need historical data for feature building. The MLDatasetBuilder internally uses
        # history up to `i` to build features for `i`. 
        # To prevent leakage, we provide it with exactly the historical segments required.
        # But `MLDatasetBuilder` expects absolute indices. It's safer to pass the segment.
        
        train_data = history[:split.train_end]
        train_dataset = self.dataset_builder.build(train_data)
        
        # We only want samples in the train window
        train_samples = [s for s in train_dataset.samples 
                         if history[split.train_start].timestamp <= s.timestamp <= history[split.train_end - 1].timestamp]
        from aegis.ml.dataset import MLDataset
        train_dataset = MLDataset(samples=train_samples)

        if len(train_dataset) < config.minimum_train_samples:
            raise ExpectedWindowFailure(f"Insufficient train samples: {len(train_dataset)}")
            
        validation_data = history[:split.validation_end]
        validation_dataset = self.dataset_builder.build(validation_data)
        val_samples = [s for s in validation_dataset.samples 
                         if history[split.validation_start].timestamp <= s.timestamp <= history[split.validation_end - 1].timestamp]
        validation_dataset = MLDataset(samples=val_samples)

        test_data = history[:split.test_end]
        test_dataset = self.dataset_builder.build(test_data)
        t_samples = [s for s in test_dataset.samples 
                         if history[split.test_start].timestamp <= s.timestamp <= history[split.test_end - 1].timestamp]
        test_dataset = MLDataset(samples=t_samples)
        
        if len(test_dataset) == 0:
            raise ExpectedWindowFailure("Test set resulted in 0 samples after feature generation.")

        # 2. Train Model on Train Dataset
        model_id = f"ml_window_{split.window_id}"
        model = self.trainer.train(train_dataset, model_id=model_id)
        
        # 2.5 Validation Tuning: Select confidence threshold
        if len(validation_dataset) > 0:
            best_f1 = -1.0
            best_threshold = 0.5
            # Grid search for the best threshold
            for threshold in [0.4, 0.5, 0.6, 0.7, 0.8]:
                model.confidence_threshold = threshold
                val_metrics = MLEvaluator.evaluate(model, validation_dataset)
                if val_metrics.f1_macro > best_f1:
                    best_f1 = val_metrics.f1_macro
                    best_threshold = threshold
            # Freeze the optimal threshold for the test set
            model.confidence_threshold = best_threshold
        
        # 3. Evaluate ML Metrics on Test Dataset
        ml_metrics = MLEvaluator.evaluate(model, test_dataset)
        
        # 4. Financial Evaluation - ML Model
        # Backtest Engine needs history to build features.
        ml_prediction_engine = PredictionEngine(model)
        ml_backtest = BacktestEngine(
            feature_builder=self.feature_builder,
            prediction_engine=ml_prediction_engine,
            risk_engine=self.risk_engine,
            config=sim_config
        )
        
        # Only run the backtest over the test_data, but engine needs history up to it for features.
        # So we supply test_data, and backtest Engine will naturally trade throughout it.
        # However, to be perfectly chronological, we want the engine to only simulate trades ON the test indices.
        # Let's slice the history but keep enough for feature warmup.
        warmup_required = self.feature_builder.minimum_candles
        backtest_start_idx = max(0, split.test_start - warmup_required)
        backtest_history = history[backtest_start_idx:split.test_end]
        
        test_start_ts = history[split.test_start].timestamp
        ml_report = ml_backtest.run(backtest_history, trading_start_timestamp=test_start_ts)
        
        # 5. Financial Evaluation - Baseline Model
        # For Sprint 10 MVP, if no baseline is provided we can use a dummy baseline model.
        # Actually `BaselinePredictionModel` exists in aegis.prediction.models, returning NEUTRAL always?
        baseline_model = BaselinePredictor()
        # We need a schema for prediction engine
        baseline_prediction_engine = PredictionEngine(baseline_model)
        baseline_backtest = BacktestEngine(
            feature_builder=self.feature_builder,
            prediction_engine=baseline_prediction_engine,
            risk_engine=self.risk_engine,
            config=sim_config
        )
        baseline_report = baseline_backtest.run(backtest_history, trading_start_timestamp=test_start_ts)
        
        metrics = WindowMetrics(
            ml_accuracy=ml_metrics.accuracy,
            ml_precision=ml_metrics.precision_macro,
            ml_recall=ml_metrics.recall_macro,
            ml_f1=ml_metrics.f1_macro,
            ml_pnl=ml_report.total_pnl,
            ml_total_trades=ml_report.total_trades,
            ml_max_drawdown=ml_report.max_drawdown,
            baseline_pnl=baseline_report.total_pnl,
            baseline_total_trades=baseline_report.total_trades,
            baseline_max_drawdown=baseline_report.max_drawdown
        )
        
        return WindowResult(
            window_id=split.window_id,
            split=split,
            train_samples=len(train_dataset),
            validation_samples=len(validation_dataset),
            test_samples=len(test_dataset),
            metrics=metrics,
            ml_model_id=model_id
        )

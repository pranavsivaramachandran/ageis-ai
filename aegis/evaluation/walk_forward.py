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
                    selection_metric=None,
                    selection_reason=None,
                    error=str(e)
                ))

        successful_results = [r for r in results if not r.error]
        
        ml_agg = aggregate_ml_metrics(successful_results)
        fin_agg = aggregate_financial_metrics(successful_results)
        
        from aegis.evaluation.metrics import calculate_ensemble_win_percentage
        
        win_pct = calculate_win_percentage(successful_results)
        ens_win_pct = calculate_ensemble_win_percentage(successful_results)

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
            mean_ensemble_pnl=fin_agg.get("mean_ensemble_pnl"),
            ml_win_percentage=win_pct,
            ensemble_win_percentage=ens_win_pct,
            worst_ml_pnl=fin_agg["worst_ml_pnl"],
            best_ml_pnl=fin_agg["best_ml_pnl"],
            worst_drawdown=fin_agg["worst_drawdown"],
            worst_ensemble_pnl=fin_agg.get("worst_ensemble_pnl"),
            best_ensemble_pnl=fin_agg.get("best_ensemble_pnl"),
            worst_ensemble_drawdown=fin_agg.get("worst_ensemble_drawdown")
        )
        
    def _evaluate_window(self, history: list[OHLC], split: WindowSplit, config: WalkForwardConfig, sim_config: SimulationConfig) -> WindowResult:
        train_data = history[:split.train_end]
        train_dataset = self.dataset_builder.build(train_data)
        
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

        # 2. Train Candidates on Train Dataset
        base_model_id = f"ml_window_{split.window_id}"
        candidates = self.trainer.train_candidates(train_dataset, base_model_id=base_model_id)
        
        if len(validation_dataset) == 0:
            raise ExpectedWindowFailure("Validation set is empty, cannot perform model selection or calibration.")
            
        # 3. Model Selection
        from aegis.ml.selection import ModelSelector
        val_metrics = {}
        for cand in candidates:
            cand_metrics = MLEvaluator.evaluate(cand, validation_dataset)
            val_metrics[cand.model_id] = cand_metrics
            
        selector = ModelSelector(metric_name="f1_macro")
        selection_result = selector.select(candidates, val_metrics)
        best_model = selection_result.selected_model
        
        # 4. Calibration (on validation set)
        from aegis.ml.calibration import IsotonicCalibrator, CalibratedPredictionModel
        import numpy as np
        
        # Get raw probabilities on validation set
        val_X = np.array(validation_dataset.x_matrix)
        if best_model.scaler:
            val_X = best_model.scaler.transform(val_X)
            
        val_y = np.array([s.value for s in validation_dataset.y_vector])
        
        calibrators = {}
        from aegis.prediction.models import PredictionDirection
        raw_val_probs = best_model.classifier.predict_proba(val_X)
        
        for idx, direction in enumerate(best_model.classes_mapping):
            cal = IsotonicCalibrator()
            direction_probs = raw_val_probs[:, idx]
            # true label is 1 if it matches the current class, else 0
            binary_labels = (val_y == direction.value).astype(int)
            cal.fit(direction_probs, binary_labels)
            calibrators[direction] = cal
            
        calibrated_model = CalibratedPredictionModel(best_model, calibrators)
        
        # 5. Ensemble
        from aegis.ml.ensemble import EnsemblePredictionModel
        num_cands = len(candidates)
        ensemble_weights = [1.0 / num_cands] * num_cands
        ensemble_model = EnsemblePredictionModel(candidates, ensemble_weights)
        
        # 6. Evaluate Selected Model on Test Dataset (ML Metrics)
        ml_metrics = MLEvaluator.evaluate(calibrated_model, test_dataset)
        
        # 7. Financial Evaluation
        warmup_required = self.feature_builder.minimum_candles
        backtest_start_idx = max(0, split.test_start - warmup_required)
        backtest_history = history[backtest_start_idx:split.test_end]
        test_start_ts = history[split.test_start].timestamp
        
        # Baseline
        baseline_model = BaselinePredictor()
        baseline_backtest = BacktestEngine(
            feature_builder=self.feature_builder,
            prediction_engine=PredictionEngine(baseline_model),
            risk_engine=self.risk_engine,
            config=sim_config
        )
        baseline_report = baseline_backtest.run(backtest_history, trading_start_timestamp=test_start_ts)
        
        # Selected ML
        ml_backtest = BacktestEngine(
            feature_builder=self.feature_builder,
            prediction_engine=PredictionEngine(calibrated_model),
            risk_engine=self.risk_engine,
            config=sim_config
        )
        ml_report = ml_backtest.run(backtest_history, trading_start_timestamp=test_start_ts)
        
        # Ensemble
        ensemble_backtest = BacktestEngine(
            feature_builder=self.feature_builder,
            prediction_engine=PredictionEngine(ensemble_model),
            risk_engine=self.risk_engine,
            config=sim_config
        )
        ensemble_report = ensemble_backtest.run(backtest_history, trading_start_timestamp=test_start_ts)
        
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
            baseline_max_drawdown=baseline_report.max_drawdown,
            ensemble_pnl=ensemble_report.total_pnl,
            ensemble_total_trades=ensemble_report.total_trades,
            ensemble_max_drawdown=ensemble_report.max_drawdown
        )
        
        return WindowResult(
            window_id=split.window_id,
            split=split,
            train_samples=len(train_dataset),
            validation_samples=len(validation_dataset),
            test_samples=len(test_dataset),
            metrics=metrics,
            ml_model_id=calibrated_model.model_id,
            selection_metric=selection_result.selection_metric,
            selection_reason=selection_result.reason
        )

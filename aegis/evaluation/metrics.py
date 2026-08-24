"""
Statistical aggregation of walk-forward evaluation metrics.
"""

from decimal import Decimal
import statistics
from typing import Sequence

from aegis.evaluation.models import WindowResult


def aggregate_ml_metrics(windows: Sequence[WindowResult]) -> dict[str, float]:
    """
    Aggregates ML metrics across a series of independent walk-forward windows.
    Aggregation semantics:
    - mean_ml_accuracy: Arithmetic mean of window accuracies.
    - mean_ml_f1: Arithmetic mean of window F1 scores.
    """
    if not windows:
        return {
            "mean_ml_accuracy": 0.0,
            "mean_ml_f1": 0.0
        }
        
    accuracies = [w.metrics.ml_accuracy for w in windows]
    f1s = [w.metrics.ml_f1 for w in windows]
    
    return {
        "mean_ml_accuracy": statistics.mean(accuracies) if accuracies else 0.0,
        "mean_ml_f1": statistics.mean(f1s) if f1s else 0.0
    }


def aggregate_financial_metrics(windows: Sequence[WindowResult]) -> dict[str, Decimal]:
    """
    Aggregates financial performance across independent simulated windows.
    Aggregation semantics:
    - mean_ml_pnl / mean_baseline_pnl: Arithmetic mean of per-window independent PnLs.
    - worst_ml_pnl: Minimum (worst) PnL among all independent windows.
    - best_ml_pnl: Maximum (best) PnL among all independent windows.
    - worst_drawdown: Maximum drawdown across all independent windows.
    """
    if not windows:
        return {
            "mean_ml_pnl": Decimal("0.0"),
            "mean_baseline_pnl": Decimal("0.0"),
            "worst_ml_pnl": Decimal("0.0"),
            "best_ml_pnl": Decimal("0.0"),
            "worst_drawdown": Decimal("0.0")
        }
        
    ml_pnls = [w.metrics.ml_pnl for w in windows]
    baseline_pnls = [w.metrics.baseline_pnl for w in windows]
    drawdowns = [w.metrics.ml_max_drawdown for w in windows]
    
    return {
        "mean_ml_pnl": Decimal(statistics.mean([float(p) for p in ml_pnls])),
        "mean_baseline_pnl": Decimal(statistics.mean([float(p) for p in baseline_pnls])),
        "worst_ml_pnl": min(ml_pnls),
        "best_ml_pnl": max(ml_pnls),
        "worst_drawdown": max(drawdowns)  # assuming drawdown is expressed as a positive number or negative, typically max absolute or min
    }


def calculate_win_percentage(windows: Sequence[WindowResult]) -> float:
    """Calculates the percentage of windows where ML outperformed baseline PnL."""
    if not windows:
        return 0.0
        
    wins = sum(1 for w in windows if w.metrics.ml_pnl > w.metrics.baseline_pnl)
    return (wins / len(windows)) * 100.0

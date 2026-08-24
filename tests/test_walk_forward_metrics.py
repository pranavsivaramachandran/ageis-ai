import pytest
from decimal import Decimal
from aegis.evaluation.metrics import aggregate_ml_metrics, aggregate_financial_metrics, calculate_win_percentage
from aegis.evaluation.models import WindowResult, WindowSplit, WindowMetrics

def test_aggregate_ml_metrics():
    windows = [
        WindowResult(
            window_id=1,
            split=WindowSplit(1, 0, 10, 10, 20, 20, 30),
            train_samples=10, validation_samples=10, test_samples=10,
            metrics=WindowMetrics(
                ml_accuracy=0.6, ml_precision=0.5, ml_recall=0.5, ml_f1=0.5,
                ml_pnl=Decimal("100"), ml_total_trades=5, ml_max_drawdown=Decimal("10"),
                baseline_pnl=Decimal("50"), baseline_total_trades=2, baseline_max_drawdown=Decimal("5")
            ),
            ml_model_id="m1"
        ),
        WindowResult(
            window_id=2,
            split=WindowSplit(2, 0, 20, 20, 30, 30, 40),
            train_samples=20, validation_samples=10, test_samples=10,
            metrics=WindowMetrics(
                ml_accuracy=0.8, ml_precision=0.7, ml_recall=0.7, ml_f1=0.7,
                ml_pnl=Decimal("-20"), ml_total_trades=1, ml_max_drawdown=Decimal("20"),
                baseline_pnl=Decimal("10"), baseline_total_trades=1, baseline_max_drawdown=Decimal("10")
            ),
            ml_model_id="m2"
        )
    ]
    
    ml_agg = aggregate_ml_metrics(windows)
    assert pytest.approx(ml_agg["mean_ml_accuracy"]) == 0.7
    assert pytest.approx(ml_agg["mean_ml_f1"]) == 0.6
    
    fin_agg = aggregate_financial_metrics(windows)
    assert fin_agg["mean_ml_pnl"] == Decimal("40.0")
    assert fin_agg["worst_ml_pnl"] == Decimal("-20")
    assert fin_agg["best_ml_pnl"] == Decimal("100")
    assert fin_agg["worst_drawdown"] == Decimal("20")
    
    win_pct = calculate_win_percentage(windows)
    assert win_pct == 50.0  # Window 1 ML > Baseline (100 > 50), Window 2 ML < Baseline (-20 < 10)

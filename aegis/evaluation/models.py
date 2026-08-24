"""
Models for Walk-Forward Evaluation and Robustness testing.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field

from aegis.backtest.models import SimulationConfig


class WalkForwardStrategy(str, Enum):
    """Walk-forward evaluation strategy."""
    EXPANDING = "expanding"
    ROLLING = "rolling"


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward window generation."""
    strategy: WalkForwardStrategy = Field(default=WalkForwardStrategy.EXPANDING)
    train_size: int = Field(..., gt=0, description="Size of training window in candles (if ROLLING) or initial size (if EXPANDING).")
    validation_size: int = Field(..., ge=0, description="Size of validation window in candles.")
    test_size: int = Field(..., gt=0, description="Size of test window in candles.")
    step_size: int = Field(..., gt=0, description="Number of candles to step forward between windows.")
    minimum_train_samples: int = Field(default=10, description="Minimum number of samples required for training.")
    
    model_config = {"frozen": True}


@dataclass(frozen=True)
class WindowSplit:
    """Represents a single walk-forward window's boundaries (by index in history)."""
    window_id: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class WindowMetrics:
    """Performance metrics for a specific window."""
    ml_accuracy: float
    ml_precision: float
    ml_recall: float
    ml_f1: float
    
    ml_pnl: Decimal
    ml_total_trades: int
    ml_max_drawdown: Decimal
    
    baseline_pnl: Decimal
    baseline_total_trades: int
    baseline_max_drawdown: Decimal

    ensemble_pnl: Decimal | None = None
    ensemble_total_trades: int | None = None
    ensemble_max_drawdown: Decimal | None = None


@dataclass(frozen=True)
class WindowResult:
    """Results from evaluating a single walk-forward window."""
    window_id: int
    split: WindowSplit
    
    train_samples: int
    validation_samples: int
    test_samples: int
    
    metrics: WindowMetrics
    
    # Model Selection Info
    ml_model_id: str
    selection_metric: str | None = None
    selection_reason: str | None = None
    
    error: str | None = None


@dataclass(frozen=True)
class WalkForwardReport:
    """Aggregate walk-forward evaluation report."""
    experiment_id: str
    windows: list[WindowResult]
    
    total_windows: int
    successful_windows: int
    failed_windows: int
    
    mean_ml_accuracy: float
    mean_ml_f1: float
    mean_ml_pnl: Decimal
    mean_baseline_pnl: Decimal
    ml_win_percentage: float
    
    worst_ml_pnl: Decimal
    best_ml_pnl: Decimal
    worst_drawdown: Decimal
    
    mean_ensemble_pnl: Decimal | None = None
    ensemble_win_percentage: float | None = None
    worst_ensemble_pnl: Decimal | None = None
    best_ensemble_pnl: Decimal | None = None
    worst_ensemble_drawdown: Decimal | None = None


@dataclass(frozen=True)
class RobustnessScenario:
    """A single robustness test configuration modifier."""
    scenario_name: str
    commission_per_trade: Decimal | None = None
    slippage_percent: Decimal | None = None
    # Can be extended for risk distance, confidence threshold, horizon, etc.


@dataclass(frozen=True)
class RobustnessResult:
    """Result of a walk-forward evaluation under a robustness scenario."""
    scenario_name: str
    report: WalkForwardReport


@dataclass(frozen=True)
class RobustnessReport:
    """Aggregate robustness evaluation report."""
    base_experiment_id: str
    scenarios: list[RobustnessResult]

"""
Backtesting models for AEGIS AI.

Defines canonical representations for historical simulation.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional, Self

from pydantic import BaseModel, Field, model_validator

from aegis.interfaces.market_data import Timeframe
from aegis.prediction.models import PredictionDirection


class SimulationConfig(BaseModel):
    """Configuration for a backtest simulation run."""

    initial_capital: Decimal = Field(
        default=Decimal("10000.0"),
        gt=0,
        description="Starting virtual capital."
    )
    
    commission_per_trade: Decimal = Field(
        default=Decimal("0.0"),
        ge=0,
        description="Fixed commission per trade (absolute currency)."
    )
    
    slippage_percent: Decimal = Field(
        default=Decimal("0.0"),
        ge=0,
        description="Percentage of price to apply as slippage (e.g. 0.001 for 0.1%)."
    )

    holding_period_candles: int = Field(
        default=5,
        gt=0,
        description="Number of candles to hold a trade before exiting."
    )

    model_config = {"frozen": True}


class SimulatedTrade(BaseModel):
    """A completed or open virtual trade."""

    trade_id: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=1)
    direction: PredictionDirection
    
    entry_timestamp: datetime
    entry_price: Decimal = Field(..., gt=0)
    position_size: Decimal = Field(..., gt=0)
    stop_loss_price: Optional[Decimal] = Field(default=None, gt=0)

    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[Decimal] = Field(default=None, gt=0)
    
    realized_pnl: Optional[Decimal] = None
    exit_reason: Optional[str] = None

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_trade(self) -> Self:
        """Validate timestamp ordering and state logic."""
        if self.entry_timestamp.tzinfo is None or self.entry_timestamp.utcoffset() != timedelta(0):
            raise ValueError("Entry timestamp must be timezone-aware UTC")

        if self.exit_timestamp is not None:
            if self.exit_timestamp.tzinfo is None or self.exit_timestamp.utcoffset() != timedelta(0):
                raise ValueError("Exit timestamp must be timezone-aware UTC")
                
            if self.exit_timestamp < self.entry_timestamp:
                raise ValueError("Exit timestamp must be on or after entry timestamp")
                
            if self.exit_price is None or self.realized_pnl is None or self.exit_reason is None:
                raise ValueError("Completed trade must have exit_price, realized_pnl, and exit_reason")
        else:
            if self.exit_price is not None or self.realized_pnl is not None or self.exit_reason is not None:
                raise ValueError("Open trade cannot have exit values")

        return self


class VirtualAccount(BaseModel):
    """Deterministic simulated account."""

    initial_capital: Decimal = Field(..., gt=0)
    current_equity: Decimal = Field(..., ge=0)
    available_cash: Decimal = Field(..., ge=0)
    
    realized_pnl: Decimal = Field(default=Decimal("0.0"))
    peak_equity: Decimal = Field(..., ge=0)
    maximum_drawdown: Decimal = Field(default=Decimal("0.0"), ge=0)
    
    trade_count: int = Field(default=0, ge=0)
    
    @classmethod
    def initialize(cls, capital: Decimal) -> "VirtualAccount":
        """Create a new virtual account with initial capital."""
        if capital <= Decimal("0"):
            raise ValueError("Initial capital must be positive")
            
        return cls(
            initial_capital=capital,
            current_equity=capital,
            available_cash=capital,
            peak_equity=capital
        )
        
    def lock_margin(self, amount: Decimal) -> "VirtualAccount":
        """Lock available cash for a new position."""
        if amount > self.available_cash:
            raise ValueError(f"Insufficient cash. Required {amount}, available {self.available_cash}")
        return VirtualAccount(
            initial_capital=self.initial_capital,
            current_equity=self.current_equity,
            available_cash=self.available_cash - amount,
            realized_pnl=self.realized_pnl,
            peak_equity=self.peak_equity,
            maximum_drawdown=self.maximum_drawdown,
            trade_count=self.trade_count
        )

    def update_mtm(self, floating_pnl: Decimal) -> "VirtualAccount":
        """Update peak equity and max drawdown using floating PnL."""
        floating_equity = self.current_equity + floating_pnl
        new_peak = max(self.peak_equity, floating_equity)
        drawdown = new_peak - floating_equity
        new_max_drawdown = max(self.maximum_drawdown, drawdown)
        
        return VirtualAccount(
            initial_capital=self.initial_capital,
            current_equity=self.current_equity,
            available_cash=self.available_cash,
            realized_pnl=self.realized_pnl,
            peak_equity=new_peak,
            maximum_drawdown=new_max_drawdown,
            trade_count=self.trade_count
        )

    def update_after_trade(self, pnl: Decimal) -> "VirtualAccount":
        """Returns a new VirtualAccount state after a trade completes."""
        new_equity = self.current_equity + pnl
        new_peak = max(self.peak_equity, new_equity)
        
        drawdown = new_peak - new_equity
        new_max_drawdown = max(self.maximum_drawdown, drawdown)
        
        return VirtualAccount(
            initial_capital=self.initial_capital,
            current_equity=new_equity,
            available_cash=new_equity,  # Releases all margin and applies PnL
            realized_pnl=self.realized_pnl + pnl,
            peak_equity=new_peak,
            maximum_drawdown=new_max_drawdown,
            trade_count=self.trade_count + 1
        )


class BacktestReport(BaseModel):
    """Analytical artifact summarizing backtest performance."""

    symbol: str = Field(..., min_length=1)
    timeframe: Timeframe
    start_timestamp: datetime
    end_timestamp: datetime
    
    initial_capital: Decimal
    final_equity: Decimal
    
    total_return: Decimal
    total_pnl: Decimal
    
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    profit_factor: Optional[Decimal] = None
    
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    
    average_trade_pnl: Decimal
    largest_win: Decimal
    largest_loss: Decimal

    model_config = {"frozen": True}

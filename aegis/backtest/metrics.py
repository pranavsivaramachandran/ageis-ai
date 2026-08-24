"""
Backtest evaluation metrics.

Provides deterministic calculation of financial metrics.
"""

from decimal import Decimal

from aegis.backtest.models import VirtualAccount, BacktestReport
from aegis.interfaces.market_data import Timeframe


def calculate_metrics(
    symbol: str,
    timeframe: Timeframe,
    start_timestamp: "datetime",
    end_timestamp: "datetime",
    account: VirtualAccount,
    trades: list["SimulatedTrade"]
) -> BacktestReport:
    """
    Calculate full evaluation metrics from an account state and trade ledger.
    Handles zero-denominators to avoid NaN/Infinity.
    """
    
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.realized_pnl is not None and t.realized_pnl > 0)
    losing_trades = sum(1 for t in trades if t.realized_pnl is not None and t.realized_pnl < 0)
    
    gross_profit = sum((t.realized_pnl for t in trades if t.realized_pnl is not None and t.realized_pnl > 0), Decimal("0"))
    gross_loss = sum((t.realized_pnl for t in trades if t.realized_pnl is not None and t.realized_pnl < 0), Decimal("0"))

    # Win rate
    win_rate = Decimal("0.0")
    if total_trades > 0:
        win_rate = Decimal(winning_trades) / Decimal(total_trades)

    # Profit factor
    profit_factor = Decimal("0.0")
    if abs(gross_loss) > Decimal("0"):
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > Decimal("0"):
        profit_factor = Decimal("999.0")  # Arbitrary high number for 100% win rate w/ profit
        
    # Drawdown pct
    max_drawdown_pct = Decimal("0.0")
    if account.peak_equity > Decimal("0"):
        max_drawdown_pct = account.maximum_drawdown / account.peak_equity
        
    # Total return
    total_return = Decimal("0.0")
    if account.initial_capital > Decimal("0"):
        total_return = (account.current_equity - account.initial_capital) / account.initial_capital
        
    # Averages
    average_trade_pnl = Decimal("0.0")
    if total_trades > 0:
        average_trade_pnl = account.realized_pnl / Decimal(total_trades)
        
    # Largest
    largest_win = Decimal("0.0")
    largest_loss = Decimal("0.0")
    
    if total_trades > 0:
        realized_pnls = [t.realized_pnl for t in trades if t.realized_pnl is not None]
        if realized_pnls:
            largest_win = max(realized_pnls + [Decimal("0.0")])
            largest_loss = min(realized_pnls + [Decimal("0.0")])
            
    return BacktestReport(
        symbol=symbol,
        timeframe=timeframe,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        initial_capital=account.initial_capital,
        final_equity=account.current_equity,
        total_return=total_return,
        total_pnl=account.realized_pnl,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=account.maximum_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        average_trade_pnl=average_trade_pnl,
        largest_win=largest_win,
        largest_loss=largest_loss
    )

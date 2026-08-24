"""
Demonstration script for Sprint 6 backtesting framework.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal

from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.features.builder import FeatureBuilder
from aegis.prediction.engine import BaselinePredictor
from aegis.risk.engine import RiskManagementEngine
from aegis.backtest.models import SimulationConfig
from aegis.backtest.engine import BacktestEngine


def make_candle(dt: datetime, base_price: Decimal) -> OHLC:
    return OHLC(
        symbol="EUR/USD",
        timestamp=dt,
        timeframe=Timeframe.H1,
        open=base_price,
        high=base_price + Decimal("5.0"),
        low=base_price - Decimal("5.0"),
        close=base_price + Decimal("1.0"),
        volume=Decimal("1000")
    )


def run_demo():
    print("Setting up deterministic dataset...")
    
    # Generate 50 candles
    history = []
    base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    current_price = Decimal("100.0")
    
    for i in range(50):
        # A simple trend up then down
        if i < 25:
            current_price += Decimal("2.0")
        else:
            current_price -= Decimal("2.0")
            
        history.append(make_candle(base_time + timedelta(hours=i), current_price))
        
    print(f"Generated {len(history)} historical candles.")

    fb = FeatureBuilder()
    pe = BaselinePredictor()
    re = RiskManagementEngine()
    
    config = SimulationConfig(
        initial_capital=Decimal("10000.0"),
        commission_per_trade=Decimal("2.0"), # $2 each way = $4 total
        slippage_percent=Decimal("0.0005"),  # 0.05%
        holding_period_candles=5
    )
    
    engine = BacktestEngine(fb, pe, re, config)
    
    print("\nRunning backtest simulation...")
    report = engine.run(history)
    
    print("\n================ BACKTEST REPORT ================")
    print(f"Symbol:           {report.symbol}")
    print(f"Timeframe:        {report.timeframe}")
    print(f"Start:            {report.start_timestamp}")
    print(f"End:              {report.end_timestamp}")
    print(f"Initial Capital:  ${report.initial_capital:,.2f}")
    print(f"Final Equity:     ${report.final_equity:,.2f}")
    print(f"Total Return:     {report.total_return * 100:,.2f}%")
    print(f"Total PnL:        ${report.total_pnl:,.2f}")
    print(f"Max Drawdown:     ${report.max_drawdown:,.2f} ({report.max_drawdown_pct * 100:,.2f}%)")
    print("\n--- Trade Stats ---")
    print(f"Total Trades:     {report.total_trades}")
    print(f"Winning Trades:   {report.winning_trades}")
    print(f"Losing Trades:    {report.losing_trades}")
    print(f"Win Rate:         {report.win_rate * 100:,.2f}%")
    print(f"Profit Factor:    {report.profit_factor:,.2f}")
    print(f"Avg Trade PnL:    ${report.average_trade_pnl:,.2f}")
    print(f"Largest Win:      ${report.largest_win:,.2f}")
    print(f"Largest Loss:     ${report.largest_loss:,.2f}")
    print("=================================================")
    

if __name__ == "__main__":
    run_demo()

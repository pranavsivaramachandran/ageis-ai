"""
Tests for backtest metrics.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pytest

from aegis.interfaces.market_data import Timeframe
from aegis.prediction.models import PredictionDirection
from aegis.backtest.models import VirtualAccount, SimulatedTrade
from aegis.backtest.metrics import calculate_metrics

def test_calculate_metrics_empty():
    acc = VirtualAccount.initialize(Decimal("1000.0"))
    dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
    
    report = calculate_metrics("EUR/USD", Timeframe.H1, dt, dt, acc, [])
    
    assert report.total_trades == 0
    assert report.total_pnl == Decimal("0.0")
    assert report.total_return == Decimal("0.0")
    assert report.win_rate == Decimal("0.0")
    assert report.profit_factor is None
    assert report.average_trade_pnl == Decimal("0.0")

def test_calculate_metrics_with_trades():
    acc = VirtualAccount.initialize(Decimal("1000.0"))
    acc = acc.update_after_trade(Decimal("100.0"))
    acc = acc.update_after_trade(Decimal("-50.0"))
    acc = acc.update_after_trade(Decimal("50.0"))
    
    dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
    
    t1 = SimulatedTrade(
        trade_id="1", symbol="E", direction=PredictionDirection.BUY,
        entry_timestamp=dt, entry_price=Decimal("1"), position_size=Decimal("1"),
        exit_timestamp=dt + timedelta(hours=1), exit_price=Decimal("1"), realized_pnl=Decimal("100.0"), exit_reason="c"
    )
    t2 = SimulatedTrade(
        trade_id="2", symbol="E", direction=PredictionDirection.BUY,
        entry_timestamp=dt, entry_price=Decimal("1"), position_size=Decimal("1"),
        exit_timestamp=dt + timedelta(hours=1), exit_price=Decimal("1"), realized_pnl=Decimal("-50.0"), exit_reason="c"
    )
    t3 = SimulatedTrade(
        trade_id="3", symbol="E", direction=PredictionDirection.BUY,
        entry_timestamp=dt, entry_price=Decimal("1"), position_size=Decimal("1"),
        exit_timestamp=dt + timedelta(hours=1), exit_price=Decimal("1"), realized_pnl=Decimal("50.0"), exit_reason="c"
    )
    
    report = calculate_metrics("EUR/USD", Timeframe.H1, dt, dt, acc, [t1, t2, t3])
    
    assert report.final_equity == Decimal("1100.0")
    assert report.total_return == Decimal("0.1")
    assert report.total_trades == 3
    assert report.winning_trades == 2
    assert report.losing_trades == 1
    
    # 2/3 = 0.666...
    assert round(report.win_rate, 4) == Decimal("0.6667")
    
    # Gross profit 150, gross loss 50 => PF 3.0
    assert report.profit_factor == Decimal("3.0")
    
    assert report.largest_win == Decimal("100.0")
    assert report.largest_loss == Decimal("-50.0")
    
    # Avg trade 100/3 = 33.33...
    assert round(report.average_trade_pnl, 2) == Decimal("33.33")

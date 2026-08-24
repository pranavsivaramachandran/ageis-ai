"""
Tests for backtest models.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pytest

from aegis.interfaces.market_data import Timeframe
from aegis.prediction.models import PredictionDirection
from aegis.backtest.models import (
    SimulationConfig,
    SimulatedTrade,
    VirtualAccount,
    BacktestReport
)


def test_virtual_account_initialization():
    acc = VirtualAccount.initialize(Decimal("1000.0"))
    assert acc.initial_capital == Decimal("1000.0")
    assert acc.current_equity == Decimal("1000.0")
    assert acc.available_cash == Decimal("1000.0")
    assert acc.peak_equity == Decimal("1000.0")
    assert acc.maximum_drawdown == Decimal("0.0")
    assert acc.realized_pnl == Decimal("0.0")
    assert acc.trade_count == 0

def test_virtual_account_invalid_capital():
    with pytest.raises(ValueError):
        VirtualAccount.initialize(Decimal("-100.0"))
    with pytest.raises(ValueError):
        VirtualAccount.initialize(Decimal("0.0"))

def test_virtual_account_update_profit():
    acc = VirtualAccount.initialize(Decimal("1000.0"))
    acc = acc.update_after_trade(Decimal("200.0"))
    
    assert acc.current_equity == Decimal("1200.0")
    assert acc.peak_equity == Decimal("1200.0")
    assert acc.maximum_drawdown == Decimal("0.0")
    assert acc.realized_pnl == Decimal("200.0")
    assert acc.trade_count == 1

def test_virtual_account_update_loss_and_drawdown():
    acc = VirtualAccount.initialize(Decimal("1000.0"))
    acc = acc.update_after_trade(Decimal("200.0")) # Peak 1200
    acc = acc.update_after_trade(Decimal("-300.0")) # Equity 900
    
    assert acc.current_equity == Decimal("900.0")
    assert acc.peak_equity == Decimal("1200.0")
    assert acc.maximum_drawdown == Decimal("300.0") # 1200 - 900
    assert acc.realized_pnl == Decimal("-100.0")
    assert acc.trade_count == 2

def test_simulated_trade_valid_open():
    dt = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    trade = SimulatedTrade(
        trade_id="t1",
        symbol="EUR/USD",
        direction=PredictionDirection.BUY,
        entry_timestamp=dt,
        entry_price=Decimal("1.1000"),
        position_size=Decimal("1000")
    )
    assert trade.exit_timestamp is None

def test_simulated_trade_valid_closed():
    dt = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    dt2 = dt + timedelta(hours=1)
    
    trade = SimulatedTrade(
        trade_id="t1",
        symbol="EUR/USD",
        direction=PredictionDirection.BUY,
        entry_timestamp=dt,
        entry_price=Decimal("1.1000"),
        position_size=Decimal("1000"),
        exit_timestamp=dt2,
        exit_price=Decimal("1.1050"),
        realized_pnl=Decimal("5.0"),
        exit_reason="take_profit"
    )
    assert trade.exit_price is not None

def test_simulated_trade_invalid_timestamps():
    dt = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    dt2 = dt - timedelta(hours=1) # Exit before entry
    
    with pytest.raises(ValueError, match="on or after"):
        SimulatedTrade(
            trade_id="t1",
            symbol="EUR/USD",
            direction=PredictionDirection.BUY,
            entry_timestamp=dt,
            entry_price=Decimal("1.1000"),
            position_size=Decimal("1000"),
            exit_timestamp=dt2,
            exit_price=Decimal("1.1050"),
            realized_pnl=Decimal("5.0"),
            exit_reason="take_profit"
        )

def test_simulated_trade_naive_timestamp():
    dt = datetime(2025, 1, 1, 10, 0)
    with pytest.raises(ValueError):
        SimulatedTrade(
            trade_id="t1",
            symbol="EUR/USD",
            direction=PredictionDirection.BUY,
            entry_timestamp=dt,
            entry_price=Decimal("1.1000"),
            position_size=Decimal("1000")
        )

def test_simulated_trade_partial_exit_data():
    dt = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    dt2 = dt + timedelta(hours=1)
    with pytest.raises(ValueError, match="must have exit_price"):
        SimulatedTrade(
            trade_id="t1",
            symbol="EUR/USD",
            direction=PredictionDirection.BUY,
            entry_timestamp=dt,
            entry_price=Decimal("1.1000"),
            position_size=Decimal("1000"),
            exit_timestamp=dt2
        )

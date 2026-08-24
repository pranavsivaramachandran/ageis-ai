from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pytest

from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.evaluation.models import WalkForwardConfig, WalkForwardStrategy
from aegis.evaluation.walk_forward import WindowGenerator


def create_mock_history(size: int = 1000):
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return [
        OHLC(
            timestamp=start + timedelta(hours=i),
            symbol="BTC/USD",
            timeframe=Timeframe.H1,
            open=Decimal("100") + i,
            high=Decimal("105") + i,
            low=Decimal("95") + i,
            close=Decimal("100") + i,
            volume=Decimal("10")
        ) for i in range(size)
    ]


def test_expanding_window_generation():
    history = create_mock_history(100)
    config = WalkForwardConfig(
        strategy=WalkForwardStrategy.EXPANDING,
        train_size=40,
        validation_size=10,
        test_size=10,
        step_size=20
    )
    
    splits = WindowGenerator.generate(history, config)
    
    assert len(splits) == 3
    
    # Window 1
    w1 = splits[0]
    assert w1.window_id == 1
    assert w1.train_start == 0
    assert w1.train_end == 40
    assert w1.validation_start == 40
    assert w1.validation_end == 50
    assert w1.test_start == 50
    assert w1.test_end == 60
    
    # Window 2
    w2 = splits[1]
    assert w2.window_id == 2
    assert w2.train_start == 0
    assert w2.train_end == 60
    assert w2.validation_start == 60
    assert w2.validation_end == 70
    assert w2.test_start == 70
    assert w2.test_end == 80
    
    # Window 3
    w3 = splits[2]
    assert w3.window_id == 3
    assert w3.train_start == 0
    assert w3.train_end == 80
    assert w3.validation_start == 80
    assert w3.validation_end == 90
    assert w3.test_start == 90
    assert w3.test_end == 100


def test_rolling_window_generation():
    history = create_mock_history(100)
    config = WalkForwardConfig(
        strategy=WalkForwardStrategy.ROLLING,
        train_size=40,
        validation_size=10,
        test_size=10,
        step_size=20
    )
    
    splits = WindowGenerator.generate(history, config)
    
    assert len(splits) == 3
    
    # Window 1
    w1 = splits[0]
    assert w1.train_start == 0
    assert w1.train_end == 40
    
    # Window 2
    w2 = splits[1]
    assert w2.train_start == 20
    assert w2.train_end == 60
    
    # Window 3
    w3 = splits[2]
    assert w3.train_start == 40
    assert w3.train_end == 80


def test_insufficient_data():
    history = create_mock_history(50)
    config = WalkForwardConfig(
        strategy=WalkForwardStrategy.EXPANDING,
        train_size=40,
        validation_size=10,
        test_size=10,
        step_size=20
    )
    
    # Total required for first window is 40+10+10 = 60
    splits = WindowGenerator.generate(history, config)
    assert len(splits) == 0

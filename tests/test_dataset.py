"""
Tests for dataset validation and creation.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import math

from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.experiments.dataset import DatasetBuilder


def make_candle(dt: datetime, symbol: str = "EUR/USD", tf: Timeframe = Timeframe.H1, val: str = "1.0") -> OHLC:
    return OHLC(
        symbol=symbol,
        timestamp=dt,
        timeframe=tf,
        open=Decimal(val),
        high=Decimal(val) + Decimal("0.5"),
        low=Decimal(val) - Decimal("0.5"),
        close=Decimal(val),
        volume=Decimal("100")
    )


def test_valid_dataset():
    dt = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    history = [
        make_candle(dt),
        make_candle(dt + timedelta(hours=1)),
        make_candle(dt + timedelta(hours=2))
    ]
    
    metadata, validated = DatasetBuilder.build(history)
    
    assert metadata.symbol == "EUR/USD"
    assert metadata.timeframe == Timeframe.H1
    assert metadata.start_timestamp == dt
    assert metadata.end_timestamp == dt + timedelta(hours=2)
    assert metadata.observation_count == 3
    assert validated == history


def test_empty_dataset_rejection():
    with pytest.raises(ValueError, match="Dataset cannot be empty"):
        DatasetBuilder.build([])


def test_mixed_symbol_rejection():
    dt = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    history = [
        make_candle(dt, symbol="EUR/USD"),
        make_candle(dt + timedelta(hours=1), symbol="GBP/USD")
    ]
    with pytest.raises(ValueError, match="Mixed symbols"):
        DatasetBuilder.build(history)


def test_mixed_timeframe_rejection():
    dt = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    history = [
        make_candle(dt, tf=Timeframe.H1),
        make_candle(dt + timedelta(hours=1), tf=Timeframe.M15)
    ]
    with pytest.raises(ValueError, match="Mixed timeframes"):
        DatasetBuilder.build(history)


def test_out_of_order_rejection():
    dt = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    history = [
        make_candle(dt + timedelta(hours=1)),
        make_candle(dt)
    ]
    with pytest.raises(ValueError, match="Chronological order violated"):
        DatasetBuilder.build(history)


def test_duplicate_timestamp_rejection():
    dt = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    history = [
        make_candle(dt),
        make_candle(dt)
    ]
    with pytest.raises(ValueError, match="Duplicate timestamp"):
        DatasetBuilder.build(history)

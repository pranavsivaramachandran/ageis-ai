"""
Tests for deterministic temporal splitting.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.experiments.dataset import ChronologicalSplitter
from aegis.features.builder import FeatureBuilder


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


def test_valid_split():
    dt = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    history = [make_candle(dt + timedelta(hours=i)) for i in range(10)]
    
    # Split 60 / 20 / 20
    split = ChronologicalSplitter.split(history, 0.6, 0.2, 0.2)
    
    assert len(split.train) == 6
    assert len(split.validation) == 2
    assert len(split.test) == 2
    
    # Verify no overlaps and chronological ordering
    assert split.train[-1].timestamp < split.validation[0].timestamp
    assert split.validation[-1].timestamp < split.test[0].timestamp
    
    # Reconstruct history
    assert split.train + split.validation + split.test == history


def test_invalid_ratios():
    history = [make_candle(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(hours=i)) for i in range(10)]
    
    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        ChronologicalSplitter.split(history, 0.5, 0.5, 0.5)
        
    with pytest.raises(ValueError, match="train_ratio must be strictly between 0 and 1"):
        ChronologicalSplitter.split(history, 0.0, 0.5, 0.5)


def test_empty_test_set():
    history = [make_candle(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(hours=i)) for i in range(10)]
    split = ChronologicalSplitter.split(history, 0.8, 0.2, 0.0)
    
    assert len(split.train) == 8
    assert len(split.validation) == 2
    assert len(split.test) == 0


def test_too_small_dataset():
    history = [make_candle(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))]
    with pytest.raises(ValueError, match="too small"):
        # 1 item cannot be split into train, val, and test safely based on these ratios
        ChronologicalSplitter.split(history, 0.33, 0.33, 0.34)


def test_explicit_leakage_prevention():
    """
    If we append future observations, the split for the ORIGINAL time period 
    and feature calculation for the ORIGINAL time period must not change.
    """
    dt = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    # Original dataset of 100 candles
    dataset_a = [make_candle(dt + timedelta(hours=i), val=str(100+i)) for i in range(100)]
    
    split_a = ChronologicalSplitter.split(dataset_a, 0.6, 0.2, 0.2)
    
    # Calculate a feature on the last train candle
    fb = FeatureBuilder()
    features_a = fb.build(split_a.train)
    
    # Now dataset B adds 50 more future candles
    dataset_b = dataset_a + [make_candle(dt + timedelta(hours=i), val=str(200+i)) for i in range(100, 150)]
    
    # We must be able to split dataset B at the exact same boundaries to avoid leakage,
    # but the generic split by ratio would shift the boundaries.
    # To prove no leakage occurs when generating features, we ensure that if we only 
    # pass the same train subset, the features are identical.
    
    split_b = ChronologicalSplitter.split(dataset_b, 0.6, 0.2, 0.2)
    
    # The boundaries shifted because the ratios applied to a larger dataset.
    # BUT, if we isolate the first 60 candles (which was the original training set),
    # the features generated on them MUST be identical to what they were before.
    train_subset_b = split_b.train[:60]
    features_b = fb.build(train_subset_b)
    
    assert features_a == features_b

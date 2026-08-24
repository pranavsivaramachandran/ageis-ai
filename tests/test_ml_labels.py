"""
Tests for ML label generation, ensuring strict leakage prevention and correct boundaries.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta

from aegis.interfaces.market_data import OHLC, Timeframe
from aegis.prediction.models import PredictionDirection
from aegis.ml.labels import TargetConfig, TargetGenerator

def build_ohlc_sequence(closes: list[float]) -> list[OHLC]:
    """Helper to build a sequence of OHLC candles from closing prices."""
    candles = []
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    for i, c in enumerate(closes):
        candles.append(OHLC(
            symbol="BTC/USD",
            timestamp=base_time + timedelta(hours=i),
            timeframe=Timeframe.H1,
            open=Decimal("100.0"),
            high=Decimal(str(c + 10)),
            low=Decimal(str(c - 10)),
            close=Decimal(str(c)),
            volume=Decimal("1000")
        ))
    return candles

class TestTargetGenerator:
    def test_buy_label(self):
        config = TargetConfig(target_horizon_candles=1, threshold=0.01)
        generator = TargetGenerator(config)
        history = build_ohlc_sequence([100.0, 102.0])  # +2%
        label = generator.get_target(0, history)
        assert label == PredictionDirection.BUY

    def test_sell_label(self):
        config = TargetConfig(target_horizon_candles=1, threshold=0.01)
        generator = TargetGenerator(config)
        history = build_ohlc_sequence([100.0, 98.0])  # -2%
        label = generator.get_target(0, history)
        assert label == PredictionDirection.SELL

    def test_neutral_label(self):
        config = TargetConfig(target_horizon_candles=1, threshold=0.01)
        generator = TargetGenerator(config)
        history = build_ohlc_sequence([100.0, 100.5])  # +0.5%
        label = generator.get_target(0, history)
        assert label == PredictionDirection.NEUTRAL

    def test_threshold_exact_boundary(self):
        config = TargetConfig(target_horizon_candles=1, threshold=0.01)
        generator = TargetGenerator(config)
        # Exactly 1% return. Our code uses strict inequality (> threshold)
        history = build_ohlc_sequence([100.0, 101.0])
        label = generator.get_target(0, history)
        assert label == PredictionDirection.NEUTRAL

    def test_insufficient_future_data(self):
        config = TargetConfig(target_horizon_candles=2, threshold=0.01)
        generator = TargetGenerator(config)
        history = build_ohlc_sequence([100.0, 101.0])
        # Horizon is 2, but we only have index 0 and 1. target index = 2.
        label = generator.get_target(0, history)
        assert label is None

    def test_deterministic_labels(self):
        config = TargetConfig(target_horizon_candles=3, threshold=0.005)
        generator = TargetGenerator(config)
        history1 = build_ohlc_sequence([100.0, 101.0, 102.0, 105.0])
        history2 = build_ohlc_sequence([100.0, 101.0, 102.0, 105.0])
        assert generator.get_target(0, history1) == generator.get_target(0, history2)

    def test_horizon_boundary(self):
        config = TargetConfig(target_horizon_candles=3, threshold=0.01)
        generator = TargetGenerator(config)
        history = build_ohlc_sequence([100.0, 101.0, 102.0, 105.0])
        # index 0 requires index 3. Length is 4.
        assert generator.get_target(0, history) == PredictionDirection.BUY
        # index 1 requires index 4. Length is 4.
        assert generator.get_target(1, history) is None

    def test_timestamp_preservation_implicit(self):
        # TargetGenerator itself doesn't construct samples, it just returns a label.
        # But we ensure it evaluates the exact target index properly.
        config = TargetConfig(target_horizon_candles=1, threshold=0.01)
        generator = TargetGenerator(config)
        history = build_ohlc_sequence([100.0, 102.0, 95.0])
        assert generator.get_target(0, history) == PredictionDirection.BUY
        assert generator.get_target(1, history) == PredictionDirection.SELL
